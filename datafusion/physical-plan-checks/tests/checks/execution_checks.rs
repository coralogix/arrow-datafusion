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
use datafusion_physical_plan_checks::{PlanChecker, Report, Severity, checks};

use crate::common::{
    ConfigurableExec, Effect, Transform, exact_source, inexact_source, schema, source,
    summary,
};

/// Run only the execution checks
fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    PlanChecker::with_checks(checks::execution_checks())
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
        .with_batch_layout(BatchLayout::Random {
            max_rows: 8,
            empty_batches: true,
        })
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
        report.violations()[0]
            .message
            .contains("ConfigurableExec cannot execute"),
        "{report}"
    );

    let parent = ConfigurableExec::new(failing).build();
    let report = check(&parent);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert_eq!(report.violations()[0].path, vec![0]);
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
        report.violations()[0]
            .message
            .contains("panicked: ConfigurableExec panicked"),
        "{report}"
    );
}

#[test]
fn timeout_is_reported() {
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.hang = true;
    let plan = exec.build();
    let report = PlanChecker::with_checks(checks::execution_checks())
        .with_timeout(Duration::from_millis(100))
        .check(&plan)
        .unwrap();
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "execution_succeeds")]
    );
    assert!(
        report.violations()[0].message.contains("did not finish"),
        "{report}"
    );
}

#[test]
fn nulls_in_non_nullable_column() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .transform(Transform::NullFirstColumn)
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "batch_schema")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("declares it non-nullable"),
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
    assert!(
        report.violations()[0].message.contains("'renamed'"),
        "{report}"
    );
}

#[test]
fn exact_num_rows_that_is_false() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    // Overall and partition 0
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "exact_statistics_hold"),
            (Severity::Invariant, "exact_statistics_hold"),
        ]
    );
}

#[test]
fn exact_column_statistics_that_are_false() {
    // The input statistics are passed through, but half of the rows are
    // dropped, so the exact row count, and possibly the distinct count, minimum
    // and maximum, are no longer true
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .transform(Transform::DropHalf)
        .build();
    let report = check(&plan);
    assert!(
        report
            .violations()
            .iter()
            .all(|v| v.check == "exact_statistics_hold"),
        "{report}"
    );
    assert!(
        report
            .violations()
            .iter()
            .any(|v| v.message.contains("num_rows Exact(100)")),
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
fn allowing_every_execution_check_skips_execution() {
    // A plan that would hang is never executed when no enabled check needs it
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.hang = true;
    let plan = exec.build();
    let mut checker = PlanChecker::new();
    for check in checks::execution_checks() {
        checker = checker.allow(check.name());
    }
    checker.check(&plan).unwrap().assert_clean();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn check_async_runs_on_the_current_runtime() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .transform(Transform::Duplicate)
        .build();
    let report = PlanChecker::with_checks(checks::execution_checks())
        .check_async(&plan)
        .await
        .unwrap();
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "cardinality_effect_holds")]
    );
}
