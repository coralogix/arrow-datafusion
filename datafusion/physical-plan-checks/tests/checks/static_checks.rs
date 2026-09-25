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

//! Tests that each static check reports deliberately broken plans and stays
//! quiet for correct ones.

use std::sync::Arc;

use datafusion_common::stats::Precision;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan_checks::fixtures::StatisticsPrecision;
use datafusion_physical_plan_checks::{PlanChecker, Report, Severity};

use crate::common::{
    ConfigurableExec, Effect, exact_source, inexact_source, source, summary,
};

fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    PlanChecker::static_only().check(plan).unwrap()
}

#[test]
fn correct_passthrough_is_clean() {
    let plan = ConfigurableExec::new(exact_source(100)).build();
    check(&plan).assert_clean();

    let plan = ConfigurableExec::new(inexact_source(100)).build();
    check(&plan).assert_clean();
}

#[test]
fn equal_cardinality_with_different_exact_num_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "equal_cardinality_num_rows")]
    );
    assert_eq!(report.violations()[0].node, "ConfigurableExec");
    assert!(report.violations()[0].path.is_empty());
}

#[test]
fn equal_cardinality_upgrades_precision() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .num_rows(Precision::Exact(100))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "equal_cardinality_num_rows")]
    );
}

#[test]
fn equal_cardinality_downgrades_precision() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Inexact(100))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "equal_cardinality_num_rows")]
    );
}

#[test]
fn equal_cardinality_with_different_estimate() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .num_rows(Precision::Inexact(10))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "equal_cardinality_num_rows")]
    );
}

#[test]
fn fetch_with_equal_cardinality() {
    let plan = ConfigurableExec::new(exact_source(100))
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "fetch_not_equal_cardinality")]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn fetch_bounds_overall_num_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(11))
        .build();
    // 11 rows is within the LowerEqual bound of 100, so only the fetch bound
    // is violated
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "fetch_bounds_num_rows")]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Inexact(11))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "fetch_bounds_num_rows")]
    );
}

#[test]
fn fetch_bounds_per_partition_num_rows() {
    let source: Arc<dyn ExecutionPlan> = source(&[50, 50], StatisticsPrecision::Exact);

    // Two partitions with fetch 10 can produce up to 20 rows overall
    let plan = ConfigurableExec::new(Arc::clone(&source))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(20))
        .partition_num_rows(vec![Precision::Exact(10), Precision::Exact(10)])
        .build();
    check(&plan).assert_clean();

    let plan = ConfigurableExec::new(source)
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(20))
        .partition_num_rows(vec![Precision::Exact(5), Precision::Exact(15)])
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "fetch_bounds_num_rows")]
    );
    assert!(
        report.violations()[0].message.contains("partition 1"),
        "{report}"
    );
}

#[test]
fn lower_equal_produces_more_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .num_rows(Precision::Exact(101))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "cardinality_effect_bounds_num_rows")]
    );

    let plan = ConfigurableExec::new(inexact_source(100))
        .effect(Effect::LowerEqual)
        .num_rows(Precision::Inexact(101))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "cardinality_effect_bounds_num_rows")]
    );
}

#[test]
fn greater_equal_produces_fewer_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::GreaterEqual)
        .num_rows(Precision::Exact(99))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "cardinality_effect_bounds_num_rows")]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::GreaterEqual)
        .num_rows(Precision::Exact(1000))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn unknown_cardinality_is_not_checked() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::Unknown)
        .num_rows(Precision::Exact(12345))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn per_child_lengths() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.maintains_input_order_len = Some(2);
    let plan = exec.build();
    // The default `check_invariants` also verifies this length
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "per_child_lengths"),
            (Severity::Invariant, "check_invariants"),
        ]
    );
}

#[test]
fn check_invariants_error() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.invariants_error = true;
    let plan = exec.build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "check_invariants")]
    );
}

#[test]
fn statistics_error() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.stats_error = true;
    let plan = exec.build();
    // One violation for the overall statistics and one for partition 0
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "statistics_shape"),
            (Severity::Invariant, "statistics_shape"),
        ]
    );
}

#[test]
fn statistics_error_is_reported_on_the_failing_node_only() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.stats_error = true;
    let parent = ConfigurableExec::new(exec.build()).build();
    let report = check(&parent);
    assert!(!report.is_empty());
    assert!(
        report.violations().iter().all(|v| v.path == vec![0]),
        "{report}"
    );
}

#[test]
fn statistics_column_count() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.drop_column_stats = 1;
    let plan = exec.build();
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "statistics_shape"),
            (Severity::Invariant, "statistics_shape"),
        ]
    );
}

#[test]
fn partition_statistics_sum() {
    let source: Arc<dyn ExecutionPlan> = source(&[10, 20], StatisticsPrecision::Exact);

    let plan = ConfigurableExec::new(Arc::clone(&source))
        .partition_num_rows(vec![Precision::Exact(10), Precision::Exact(25)])
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "partition_statistics_sum")]
    );

    let plan = ConfigurableExec::new(Arc::clone(&source))
        .effect(Effect::Unknown)
        .num_rows(Precision::Inexact(30))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "partition_statistics_sum")]
    );

    let plan = ConfigurableExec::new(source)
        .effect(Effect::Unknown)
        .num_rows(Precision::Exact(30))
        .partition_num_rows(vec![Precision::Absent, Precision::Exact(31)])
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "partition_statistics_sum")]
    );
}

#[test]
fn statistics_ignore_inputs() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.skip_child_stats = true;
    let plan = exec.build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "statistics_ignore_inputs")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("child_stats_requests() skips the input"),
        "{report}"
    );
}

#[test]
fn allowed_checks_are_skipped() {
    let plan = ConfigurableExec::new(exact_source(100))
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    PlanChecker::static_only()
        .allow("fetch_not_equal_cardinality")
        .check(&plan)
        .unwrap()
        .assert_clean();
}

#[test]
fn violations_are_attributed_to_the_offending_node() {
    let broken = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let plan = ConfigurableExec::new(broken)
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&plan);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert_eq!(report.violations()[0].path, vec![0]);
    assert_eq!(
        report.to_string(),
        "[invariant] A1 equal_cardinality_num_rows at ConfigurableExec (root/0): \
         cardinality_effect() is Equal, but num_rows is Exact(50) for an input with \
         num_rows Exact(100)"
    );

    // `check_node` only checks the root
    PlanChecker::static_only()
        .check_node(&plan)
        .unwrap()
        .assert_clean();
}
