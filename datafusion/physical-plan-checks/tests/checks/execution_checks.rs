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

use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::{ExecutionPlan, StatisticsArgs, StatisticsContext};
use datafusion_physical_plan_checks::fixtures::{
    BatchLayout, SourceSpec, StatisticsPrecision,
};
use datafusion_physical_plan_checks::harness::ROW_ID_RANGE;
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
/// `batch_boundary_invariance` reports, and most plans pass their first rows
/// through without supporting limit pushdown, which `limit_pushdown_missed`
/// reports; those checks are tested in `variant_checks.rs`.
fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    execution_checker()
        .allow("batch_boundary_invariance")
        .allow("limit_pushdown_missed")
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
        row_id_source(&[40, 0, 60]),
    ] {
        let mut exec = ConfigurableExec::new(input);
        exec.limit_pushdown = true;
        PlanChecker::new()
            .check(&exec.build())
            .unwrap()
            .assert_clean();
    }
}

/// A source with one partition per entry of `rows` and a row id column
fn row_id_source(rows: &[usize]) -> Arc<dyn ExecutionPlan> {
    SourceSpec::new(schema())
        .with_partition_rows(rows)
        .with_row_ids(0)
        .build_arc()
        .unwrap()
}

#[test]
fn order_that_is_kept_but_not_reported() {
    let check = |plan: &Arc<dyn ExecutionPlan>| {
        checker(&["maintains_input_order_missed"])
            .check(plan)
            .unwrap()
    };
    let unreported = |input| {
        let mut exec = ConfigurableExec::new(input);
        exec.maintains_order = Some(false);
        exec
    };
    let report = check(&unreported(row_id_source(&[40, 0, 60])).build());
    assert_eq!(
        messages(&report),
        vec![
            "maintains_input_order()[0] is false, but every output partition only has \
             rows of one partition of child 0, in the order of that partition; return \
             true if the node never reorders the rows of this child, and keep the \
             child's orderings in the output equivalence properties"
        ]
    );
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "maintains_input_order_missed")]
    );

    // Not reported for a node that reorders rows, or over input it cannot
    // track or that is already sorted, whether the node reports the input's
    // ordering or not
    let sorted = SourceSpec::new(schema())
        .with_partition_rows(&[40, 0, 60])
        .with_ordering(ordering_on_a())
        .with_row_ids(0)
        .build_arc()
        .unwrap();
    let mut without_ordering = unreported(Arc::clone(&sorted));
    without_ordering.drop_ordering = true;
    for exec in [
        unreported(row_id_source(&[40, 0, 60])).transform(Transform::Reverse),
        unreported(row_id_source(&[40, 0, 60])).transform(Transform::Duplicate),
        unreported(source(&[40, 0, 60], StatisticsPrecision::Exact)),
        unreported(sorted),
        without_ordering,
    ] {
        check(&exec.build()).assert_clean();
    }

    // A coalesce keeps the order of a single input partition, and interleaves
    // several
    let coalesce =
        |input| Arc::new(CoalescePartitionsExec::new(input)) as Arc<dyn ExecutionPlan>;
    assert_eq!(
        summary(&check(&coalesce(row_id_source(&[100])))),
        vec![(Severity::Lint, "maintains_input_order_missed")]
    );
    check(&coalesce(row_id_source(&[40, 0, 60]))).assert_clean();

    // The same partition of a coalesce with a fetch can have rows of one
    // input partition only, which shows nothing about the others
    let coalesce_with_fetch: Arc<dyn ExecutionPlan> = Arc::new(
        CoalescePartitionsExec::new(row_id_source(&[40, 0, 60])).with_fetch(Some(5)),
    );
    check(&coalesce_with_fetch).assert_clean();

    // Each partition of a union has rows of one input only
    let other_input = SourceSpec::new(schema())
        .with_partition_rows(&[30])
        .with_row_ids(ROW_ID_RANGE)
        .build_arc()
        .unwrap();
    let union =
        UnionExec::try_new(vec![row_id_source(&[40, 0, 60]), other_input]).unwrap();
    check(&union).assert_clean();
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

    // A node that makes a false claim of its own over a child with false
    // statistics is reported too, with its statistics computed from the true
    // statistics of the child
    let lying_child = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let lying = ConfigurableExec::new(lying_child)
        .num_rows(Precision::Exact(70))
        .build();
    let report = check(&lying);
    let on_node: Vec<_> = report
        .violations
        .iter()
        .filter(|v| v.path.is_empty())
        .map(|v| v.message.as_str())
        .collect();
    assert_eq!(
        on_node,
        vec![
            "with the false exact statistics of its children replaced by the true \
             values, the overall statistics are false: num_rows is Exact(70), but the \
             output has 100 rows",
            "with the false exact statistics of its children replaced by the true \
             values, the partition 0 statistics are false: num_rows is Exact(70), but \
             the output has 100 rows",
        ],
        "{report}"
    );
    assert_eq!(report.violations.len(), 4, "{report}");
}

#[test]
fn false_exact_sum() {
    // Every row is repeated, but the input statistics are passed through
    let input = exact_source(100);
    let sum = match &StatisticsContext::new()
        .compute(input.as_ref(), &StatisticsArgs::new())
        .unwrap()
        .column_statistics[0]
        .sum_value
    {
        Precision::Exact(ScalarValue::Int64(Some(sum))) => *sum,
        other => panic!("the source reports the sum {other}"),
    };
    let plan = ConfigurableExec::new(input)
        .effect(Effect::GreaterEqual)
        .transform(Transform::Duplicate)
        .build();
    let report = checker(&["exact_statistics_hold"]).check(&plan).unwrap();
    let false_statistics = format!(
        "num_rows is Exact(100), but the output has 200 rows; sum_value of a@0 is \
         Exact(Int64({sum})), but the output has Exact(Int64({}))",
        2 * sum
    );
    assert_eq!(
        messages(&report),
        vec![
            format!("the overall statistics are false: {false_statistics}"),
            format!("the partition 0 statistics are false: {false_statistics}"),
        ],
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
fn memory_that_is_never_released() {
    let check = |plan: &Arc<dyn ExecutionPlan>| {
        checker(&["memory_released"]).check(plan).unwrap()
    };
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.leak_memory = true;
    let leaking = exec.build();
    let report = check(&leaking);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "memory_released")]
    );
    assert!(
        messages(&report)[0].contains("1024 bytes are still reserved"),
        "{report}"
    );

    // The parent inherits the leak, and is not reported
    let parent = ConfigurableExec::new(leaking).build();
    let report = check(&parent);
    assert_eq!(report.violations.len(), 1, "{report}");
    assert_eq!(report.violations[0].path, vec![0]);
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
