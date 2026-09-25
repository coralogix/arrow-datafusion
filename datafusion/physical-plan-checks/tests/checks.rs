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

//! Tests that each check reports deliberately broken plans and stays quiet
//! for correct ones.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err, not_impl_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::execution_plan::{CardinalityEffect, InvariantLevel};
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};
use datafusion_physical_plan_checks::fixtures::MockSourceExec;
use datafusion_physical_plan_checks::{PlanChecker, Report, Severity};

#[derive(Debug, Clone, Copy)]
enum Effect {
    Unknown,
    Equal,
    LowerEqual,
    GreaterEqual,
}

/// A single-child plan that passes its input statistics through, with knobs
/// to make it break individual parts of the `ExecutionPlan` contract.
#[derive(Debug, Clone)]
struct ConfigurableExec {
    input: Arc<dyn ExecutionPlan>,
    effect: Effect,
    fetch: Option<usize>,
    /// Overrides the overall num_rows
    num_rows: Option<Precision<usize>>,
    /// Overrides the num_rows of each partition
    partition_num_rows: Option<Vec<Precision<usize>>>,
    /// Skip the child in `child_stats_requests`
    skip_child_stats: bool,
    /// Drop this many column statistics entries
    drop_column_stats: usize,
    /// Return an error from `statistics_from_inputs`
    stats_error: bool,
    /// Overrides the length of `maintains_input_order`
    maintains_input_order_len: Option<usize>,
    /// Return an error from `check_invariants`
    invariants_error: bool,
}

impl ConfigurableExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            effect: Effect::Equal,
            fetch: None,
            num_rows: None,
            partition_num_rows: None,
            skip_child_stats: false,
            drop_column_stats: 0,
            stats_error: false,
            maintains_input_order_len: None,
            invariants_error: false,
        }
    }

    fn effect(mut self, effect: Effect) -> Self {
        self.effect = effect;
        self
    }

    fn fetch(mut self, fetch: usize) -> Self {
        self.fetch = Some(fetch);
        self
    }

    fn num_rows(mut self, num_rows: Precision<usize>) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    fn partition_num_rows(mut self, num_rows: Vec<Precision<usize>>) -> Self {
        self.partition_num_rows = Some(num_rows);
        self
    }

    fn build(self) -> Arc<dyn ExecutionPlan> {
        Arc::new(self)
    }
}

impl DisplayAs for ConfigurableExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ConfigurableExec")
    }
}

impl ExecutionPlan for ConfigurableExec {
    fn name(&self) -> &'static str {
        "ConfigurableExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn check_invariants(&self, check: InvariantLevel) -> Result<()> {
        if self.invariants_error {
            return internal_err!("ConfigurableExec invariants are broken");
        }
        datafusion_physical_plan::execution_plan::check_default_invariants(self, check)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.maintains_input_order_len.unwrap_or(1)]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut new = self.as_ref().clone();
        new.input = children.swap_remove(0);
        Ok(Arc::new(new))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        not_impl_err!("ConfigurableExec does not execute")
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        if self.skip_child_stats {
            vec![ChildStats::Skip]
        } else {
            vec![ChildStats::At(partition)]
        }
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        if self.stats_error {
            return internal_err!("ConfigurableExec statistics are broken");
        }
        let mut stats = input_stats[0].as_ref().clone();
        let partitions = self.properties().output_partitioning().partition_count();
        let num_rows = match (args.partition(), &self.partition_num_rows) {
            (None, _) => self.num_rows,
            (Some(p), Some(rows)) => Some(rows[p]),
            // With a single partition, the overall override also applies to
            // the partition, keeping the statistics consistent
            (Some(_), None) if partitions == 1 => self.num_rows,
            (Some(_), None) => None,
        };
        if let Some(num_rows) = num_rows {
            stats.num_rows = num_rows;
        }
        let keep = stats
            .column_statistics
            .len()
            .saturating_sub(self.drop_column_stats);
        stats.column_statistics.truncate(keep);
        Ok(Arc::new(stats))
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        match self.effect {
            Effect::Unknown => CardinalityEffect::Unknown,
            Effect::Equal => CardinalityEffect::Equal,
            Effect::LowerEqual => CardinalityEffect::LowerEqual,
            Effect::GreaterEqual => CardinalityEffect::GreaterEqual,
        }
    }
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
}

/// A source with a single partition and an exact row count
fn exact_source(num_rows: usize) -> Arc<dyn ExecutionPlan> {
    Arc::new(MockSourceExec::new(schema()).with_exact_partition_num_rows(&[num_rows]))
}

/// A source with a single partition and an inexact row count
fn inexact_source(num_rows: usize) -> Arc<dyn ExecutionPlan> {
    Arc::new(MockSourceExec::new(schema()).with_inexact_partition_num_rows(&[num_rows]))
}

fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    PlanChecker::new().check(plan).unwrap()
}

/// `(severity, check name)` of every violation in the report
fn summary(report: &Report) -> Vec<(Severity, &'static str)> {
    report
        .violations()
        .iter()
        .map(|v| (v.severity, v.check))
        .collect()
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
    let source: Arc<dyn ExecutionPlan> =
        Arc::new(MockSourceExec::new(schema()).with_exact_partition_num_rows(&[50, 50]));

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
    let source: Arc<dyn ExecutionPlan> =
        Arc::new(MockSourceExec::new(schema()).with_exact_partition_num_rows(&[10, 20]));

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
    PlanChecker::new()
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
    PlanChecker::new().check_node(&plan).unwrap().assert_clean();
}
