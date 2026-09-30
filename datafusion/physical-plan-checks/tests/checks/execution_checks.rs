// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Tests that each execution check reports deliberately broken plans and
//! stays quiet for correct ones.

use std::sync::Arc;
use std::time::Duration;

use datafusion_common::stats::Precision;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan_checks::fixtures::{
    BatchLayout, SourceSpec, StatisticsPrecision,
};
use datafusion_physical_plan_checks::{CheckKind, PlanChecker, Report, Severity};

use crate::common::{
    ConfigurableExec, Effect, Transform, checker, checker_of, exact_source,
    inexact_source, messages, schema, source, summary,
};

/// The checks that execute the plan
fn execution_checker() -> PlanChecker {
    checker_of(&[CheckKind::Execution, CheckKind::Variant, CheckKind::Stream])
}

/// Run the checks that execute the plan. `Transform::DropHalf` keeps half of
/// each batch, so its output depends on batch boundaries, which
/// `batch_boundary_invariance` reports; that check is tested in
/// `variant_checks.rs`.
fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    execution_checker()
        .allow("batch_boundary_invariance")
        .check(plan)
        .unwrap()
}

fn ordering_on_a() -> LexOrdering {
    LexOrdering::new(vec![PhysicalSortExpr::new_default(
        col("a", &schema()).unwrap(),
    )])
    .unwrap()
}

/// A source with several sorted partitions, empty batches and many ties
fn sorted_source() -> Arc<dyn ExecutionPlan> {
    SourceSpec::new(schema())
        .with_partition_rows(&[50, 0, 70])
        .with_batch_layout(BatchLayout::Random { max_rows: 8 })
        .with_distinct_values(5)
        .with_ordering(ordering_on_a())
        .build_arc()
        .unwrap()
}

#[test]
fn correct_passthrough_is_clean() {
    for input in [
        exact_source(100),
        inexact_source(100),
        source(&[10, 0, 30], StatisticsPrecision::Exact),
        sorted_source(),
    ] {
        let plan = ConfigurableExec::new(input).build();
        PlanChecker::new().check(&plan).unwrap().assert_clean();
    }
}

#[test]
fn execute_error_is_reported_on_the_failing_node_only() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.execute_error = true;
    let failing = exec.build();
    let report = check(&failing);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "execution_succeeds")]
    );
    assert!(
        messages(&report)[0].contains("ConfigurableExec cannot execute"),
        "{report}"
    );

    let parent = ConfigurableExec::new(failing).build();
    let report = check(&parent);
    assert_eq!(report.violations.len(), 1, "{report}");
    assert_eq!(report.violations[0].path, vec![0]);
}

#[test]
fn panic_is_reported() {
    let plan = ConfigurableExec::new(source(&[10, 10], StatisticsPrecision::Inexact))
        .transform(Transform::Panic)
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "execution_succeeds")]
    );
    assert!(
        messages(&report)[0].contains("panicked: ConfigurableExec panicked"),
        "{report}"
    );
}

#[test]
fn timeout_is_reported() {
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.hang = true;
    let plan = exec.build();
    let report = execution_checker()
        .with_timeout(Duration::from_millis(100))
        .with_stream_timeout(Duration::from_millis(100))
        .check(&plan)
        .unwrap();
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "execution_succeeds")]
    );
    assert!(messages(&report)[0].contains("did not finish"), "{report}");
}

#[test]
fn nulls_in_non_nullable_column() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .transform(Transform::NullFirstColumn)
        .build();
    let report = check(&plan);
    // The batches with nulls are an invariant violation. The empty batches,
    // whose field is marked nullable too, are a lint.
    let mut findings = summary(&report);
    findings.sort();
    assert_eq!(
        findings,
        vec![
            (Severity::Invariant, "batch_schema"),
            (Severity::Lint, "batch_schema")
        ]
    );
    assert!(
        messages(&report).iter().any(|message| message
            .ends_with("column 'a' has nulls, but schema() declares it non-nullable")),
        "{report}"
    );
}

#[test]
fn renamed_column() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .transform(Transform::RenameFirstColumn)
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "batch_schema")]
    );
    assert!(messages(&report)[0].contains("'renamed'"), "{report}");
}

#[test]
fn exact_num_rows_that_is_false() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&plan);
    assert_eq!(
        messages(&report),
        vec![
            "the overall statistics are false: num_rows is Exact(50), but the output \
             has 100 rows",
            "the partition 0 statistics are false: num_rows is Exact(50), but the \
             output has 100 rows",
        ],
        "{report}"
    );
}

#[test]
fn false_exact_statistics_are_listed_per_partition() {
    // The input statistics are passed through, but half of the rows of each
    // batch are dropped, so the exact row counts, and possibly other
    // statistics, are no longer true
    let input = SourceSpec::new(schema())
        .with_partition_rows(&[40, 0, 40])
        .with_batch_layout(BatchLayout::Fixed(10))
        .build_arc()
        .unwrap();
    let plan = ConfigurableExec::new(input)
        .effect(Effect::LowerEqual)
        .transform(Transform::DropHalf)
        .build();
    let report = check(&plan);
    assert!(
        report
            .violations
            .iter()
            .all(|v| v.check == "exact_statistics_hold"),
        "{report}"
    );
    // The statistics of the empty partition hold
    let messages = messages(&report);
    assert_eq!(messages.len(), 3, "{report}");
    for (message, expected) in messages.iter().zip([
        "the overall statistics are false: num_rows is Exact(80), but the output has \
         40 rows",
        "the partition 0 statistics are false: num_rows is Exact(40), but the output \
         has 20 rows",
        "the partition 2 statistics are false: num_rows is Exact(40), but the output \
         has 20 rows",
    ]) {
        assert!(message.starts_with(expected), "{report}");
    }
}

#[test]
fn false_exact_statistics_are_reported_on_the_node_that_makes_them() {
    let check = |plan: &Arc<dyn ExecutionPlan>| {
        checker(&["exact_statistics_hold"]).check(plan).unwrap()
    };
    // A parent that passes statistics through unchanged, as a repartition
    // does, inherits the false row count of its child. It is only reported
    // on the child.
    let lying = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let parent = ConfigurableExec::new(lying).build();
    let report = check(&parent);
    assert_eq!(report.violations.len(), 2, "{report}");
    assert!(
        report.violations.iter().all(|v| v.path == vec![0]),
        "{report}"
    );

    // A node that makes a false claim over a child whose statistics hold is
    // reported, also when the child is not a source
    let correct = ConfigurableExec::new(exact_source(100)).build();
    let lying = ConfigurableExec::new(correct)
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&lying);
    assert_eq!(report.violations.len(), 2, "{report}");
    assert!(
        report.violations.iter().all(|v| v.path.is_empty()),
        "{report}"
    );
}

#[test]
fn inexact_statistics_are_not_checked() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .effect(Effect::LowerEqual)
        .transform(Transform::DropHalf)
        .build();
    check(&plan).assert_clean();
}

#[test]
fn ordering_that_does_not_hold() {
    // The input is sorted, but the node reverses each batch while still
    // reporting the input ordering
    let plan = ConfigurableExec::new(sorted_source())
        .transform(Transform::Reverse)
        .build();
    let report = check(&plan);
    // Partitions 0 and 2 have rows; partition 1 is empty
    assert_eq!(
        summary(&report),
        vec![
            (Severity::Invariant, "orderings_hold"),
            (Severity::Invariant, "orderings_hold"),
        ]
    );

    // An ordering claimed over unsorted input
    let mut exec = ConfigurableExec::new(inexact_source(100));
    exec.claimed_ordering = Some(ordering_on_a());
    let plan = exec.build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "orderings_hold")]
    );
}

#[test]
fn equal_cardinality_that_does_not_hold() {
    for transform in [Transform::DropHalf, Transform::Duplicate] {
        let plan = ConfigurableExec::new(inexact_source(100))
            .transform(transform)
            .build();
        assert_eq!(
            summary(&check(&plan)),
            vec![(Severity::Invariant, "cardinality_effect_holds")],
            "{transform:?}"
        );
    }
}

#[test]
fn lower_and_greater_equal_cardinality() {
    let lower = |transform| {
        ConfigurableExec::new(inexact_source(100))
            .effect(Effect::LowerEqual)
            .transform(transform)
            .build()
    };
    check(&lower(Transform::DropHalf)).assert_clean();
    assert_eq!(
        summary(&check(&lower(Transform::Duplicate))),
        vec![(Severity::Invariant, "cardinality_effect_holds")]
    );

    let greater = |transform| {
        ConfigurableExec::new(inexact_source(100))
            .effect(Effect::GreaterEqual)
            .transform(transform)
            .build()
    };
    check(&greater(Transform::Duplicate)).assert_clean();
    assert_eq!(
        summary(&check(&greater(Transform::DropHalf))),
        vec![(Severity::Invariant, "cardinality_effect_holds")]
    );
}

#[test]
fn static_checks_do_not_execute() {
    // A plan that would hang is never executed when no enabled check needs it
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.hang = true;
    let plan = exec.build();
    checker_of(&[CheckKind::Static])
        .check(&plan)
        .unwrap()
        .assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn check_async_runs_on_the_current_runtime() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .transform(Transform::Duplicate)
        .build();
    let report = execution_checker().check_async(&plan).await.unwrap();
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "cardinality_effect_holds")]
    );
}
