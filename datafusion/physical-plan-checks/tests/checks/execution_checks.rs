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

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::stats::Precision;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::{
    ExecutionPlan, StatisticsArgs, StatisticsContext, collect_partitioned,
};
use datafusion_physical_plan_checks::fixtures::{
    BatchLayout, SourceSpec, StatisticsPrecision,
};
use datafusion_physical_plan_checks::{PlanChecker, Report, Severity, checks, oracle};

use crate::common::{
    ConfigurableExec, Effect, Transform, exact_source, inexact_source, schema, source,
    summary,
};

/// Run only the execution checks. `Transform::DropHalf` keeps half of each
/// batch, so its output depends on batch boundaries, which
/// `batch_boundary_invariance` reports; that check is tested in
/// `variant_checks.rs`.
fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    PlanChecker::with_checks(checks::execution_checks())
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
        .with_stream_timeout(Duration::from_millis(100))
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
    // One finding for the overall statistics and partition 0
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "exact_statistics_hold")]
    );
    assert_eq!(
        report.violations()[0].message,
        "overall statistics report num_rows Exact(50), but the output has Exact(100) \
         rows (also 1 more false num_rows statistic: in partition 0)"
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

/// The number of false exact statistics of `plan` of each kind, overall and
/// for each partition, counted independently of `exact_statistics_hold`
async fn count_false_statistics(
    plan: &Arc<dyn ExecutionPlan>,
) -> BTreeMap<&'static str, usize> {
    let outputs = collect_partitioned(Arc::clone(plan), Arc::new(TaskContext::default()))
        .await
        .unwrap();
    let mut targets = vec![(None, outputs.concat())];
    targets.extend(outputs.into_iter().enumerate().map(|(p, b)| (Some(p), b)));
    let mut counts = BTreeMap::new();
    for (partition, batches) in targets {
        let args = StatisticsArgs::new().with_partition(partition);
        let claimed = StatisticsContext::new()
            .compute(plan.as_ref(), &args)
            .unwrap();
        let actual = oracle::exact_statistics(&plan.schema(), &batches).unwrap();
        let mut count = |statistic, claimed_exact, contradicted| {
            if claimed_exact && contradicted {
                *counts.entry(statistic).or_insert(0) += 1;
            }
        };
        count(
            "num_rows",
            claimed.num_rows.is_exact() == Some(true),
            claimed.num_rows != actual.num_rows,
        );
        for (claimed, actual) in claimed
            .column_statistics
            .iter()
            .zip(&actual.column_statistics)
        {
            count(
                "null_count",
                claimed.null_count.is_exact() == Some(true),
                claimed.null_count != actual.null_count,
            );
            count(
                "distinct_count",
                claimed.distinct_count.is_exact() == Some(true),
                claimed.distinct_count != actual.distinct_count,
            );
            count(
                "min_value",
                claimed.min_value.is_exact() == Some(true),
                actual.min_value.is_exact() == Some(true)
                    && claimed.min_value != actual.min_value,
            );
            count(
                "max_value",
                claimed.max_value.is_exact() == Some(true),
                actual.max_value.is_exact() == Some(true)
                    && claimed.max_value != actual.max_value,
            );
        }
    }
    counts
}

#[tokio::test(flavor = "current_thread")]
async fn false_exact_statistics_are_grouped_by_statistic() {
    // Several columns and partitions, every other one empty, whose statistics
    // are passed through while half of the rows are dropped. Most statistics
    // become false in most columns and non-empty partitions.
    let schema = Arc::new(Schema::new(
        ["a", "b", "c", "d", "e", "f"]
            .map(|name| Field::new(name, DataType::Int32, true))
            .to_vec(),
    ));
    let input = SourceSpec::new(schema)
        .with_partition_rows(&[40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40])
        .with_distinct_values(30)
        .with_row_id_column("id", 0)
        .build_arc()
        .unwrap();
    let plan = ConfigurableExec::new(input)
        .effect(Effect::LowerEqual)
        .transform(Transform::DropHalf)
        .build();
    let report = PlanChecker::with_checks(vec![Arc::new(checks::ExactStatisticsHold)])
        .check_async(&plan)
        .await
        .unwrap();

    // One finding per kind of false statistic, in catalog order, that keeps
    // the first false statistic and summarizes the others by column and
    // partition
    let messages: Vec<&str> = report
        .violations()
        .iter()
        .map(|v| {
            assert_eq!(v.severity, Severity::Invariant);
            assert!(v.path.is_empty());
            v.message.as_str()
        })
        .collect();
    assert_eq!(
        messages,
        vec![
            "overall statistics report num_rows Exact(240), but the output has \
             Exact(116) rows (also 6 more false num_rows statistics: in partitions 0, \
             2, 4, 6 and 2 more)",
            "overall statistics report null_count Exact(23) for column 'a', but the \
             output has Exact(12) (also 37 more false null_count statistics: 'a' in \
             partitions 0, 2, 6, 8 and 1 more; 'b' overall and in partitions 0, 2, 4, \
             6 and 2 more; 'c' overall and in partitions 0, 2, 4, 6 and 1 more; 'd' \
             overall and in partitions 0, 2, 6, 8 and 1 more; and 13 more in 2 other \
             columns)",
            "overall statistics report distinct_count Exact(30) for column 'a', but \
             the output has Exact(29) (also 45 more false distinct_count statistics: \
             'a' in partitions 0, 2, 4, 6 and 2 more; 'b' overall and in partitions \
             0, 2, 4, 6 and 2 more; 'c' in partitions 0, 2, 4, 6 and 2 more; 'd' \
             overall and in partitions 0, 2, 4, 6 and 2 more; and 19 more in 3 other \
             columns)",
            "partition 0 statistics report min_value Exact(Int32(0)) for column 'a', \
             but the output has Exact(Int32(1)) (also 19 more false min_value \
             statistics: 'a' in partitions 2, 4, 6; 'b' in partitions 0, 2; 'c' in \
             partitions 2, 4; 'd' in partitions 0, 4, 8; and 9 more in 3 other \
             columns)",
            "overall statistics report max_value Exact(UInt64(239)) for column 'id', \
             but the output has Exact(UInt64(237)) (also 19 more false max_value \
             statistics: 'a' in partitions 2, 4; 'b' in partition 8; 'c' in partition \
             10; 'd' in partitions 2, 4, 6; and 12 more in 3 other columns)",
        ],
        "{report}"
    );

    // Every false statistic is represented: the example and the others that
    // the summary counts add up to the number of false statistics of each
    // kind
    let expected = count_false_statistics(&plan).await;
    let mut reported = BTreeMap::new();
    for message in messages {
        let statistic = [
            "num_rows",
            "null_count",
            "distinct_count",
            "min_value",
            "max_value",
        ]
        .into_iter()
        .find(|statistic| message.contains(&format!("report {statistic} ")))
        .unwrap();
        let others = message.split_once(" (also ").map_or(0, |(_, also)| {
            also.split_once(' ').unwrap().0.parse::<usize>().unwrap()
        });
        assert!(
            reported.insert(statistic, 1 + others).is_none(),
            "{message}"
        );
    }
    assert_eq!(reported, expected);
}

#[test]
fn false_exact_statistics_are_reported_on_the_node_that_makes_them() {
    let check = |plan: &Arc<dyn ExecutionPlan>| {
        PlanChecker::with_checks(vec![Arc::new(checks::ExactStatisticsHold)])
            .check(plan)
            .unwrap()
    };
    // A parent that passes statistics through unchanged, as a repartition
    // does, inherits the false row count of its child. It is only reported
    // on the child, in one finding for the overall statistics and partition 0.
    let lying = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let parent = ConfigurableExec::new(lying).build();
    let report = check(&parent);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert!(
        report.violations().iter().all(|v| v.path == vec![0]),
        "{report}"
    );

    // A node that makes a false claim over a child whose statistics hold is
    // reported, also when the child is not a source
    let correct = ConfigurableExec::new(exact_source(100)).build();
    let lying = ConfigurableExec::new(correct)
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&lying);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert!(
        report.violations().iter().all(|v| v.path.is_empty()),
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
