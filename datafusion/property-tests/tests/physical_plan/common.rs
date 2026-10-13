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

//! A plan with knobs for breaking the `ExecutionPlan` contract, and helpers
//! shared by the tests.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::{DynamicFilterPhysicalExpr, lit};
use datafusion_physical_expr::{
    EquivalenceProperties, LexOrdering, OrderingRequirements, PhysicalExpr,
};
use datafusion_physical_plan::execution_plan::{CardinalityEffect, InvariantLevel};
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream,
    StatisticsArgs, apply_expression_roots,
};
use datafusion_property_tests::physical_plan::fixtures::{
    SourceSpec, StatisticsPrecision,
};
use datafusion_property_tests::physical_plan::{PlanChecker, checks};
use datafusion_property_tests::{Report, Severity};

#[derive(Debug, Clone, Copy)]
pub enum Effect {
    Unknown,
    Equal,
    LowerEqual,
    GreaterEqual,
}

/// A single-child plan that reports what it is configured to report, with
/// knobs to make it break individual parts of the `ExecutionPlan` contract.
/// When executed, it passes its input through unchanged.
#[derive(Debug, Clone)]
pub struct ConfigurableExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub effect: Effect,
    /// Overrides `effect` while the node has no fetch
    pub effect_without_fetch: Option<Effect>,
    pub fetch: Option<usize>,
    /// Overrides the overall num_rows
    pub num_rows: Option<Precision<usize>>,
    /// Overrides the num_rows of each partition
    pub partition_num_rows: Option<Vec<Precision<usize>>>,
    /// Skip the child in `child_stats_requests`
    pub skip_child_stats: bool,
    /// Drop this many column statistics entries
    pub drop_column_stats: usize,
    /// Return an error from `statistics_from_inputs`
    pub stats_error: bool,
    /// Overrides the length of `maintains_input_order`
    pub maintains_input_order_len: Option<usize>,
    /// Overrides the values of `maintains_input_order`
    pub maintains_order: Option<bool>,
    /// Require the input to be sorted by this ordering
    pub required_ordering: Option<LexOrdering>,
    /// Return an error from `check_invariants`
    pub invariants_error: bool,
    /// Return an error from `check_invariants(Executable)`
    pub executable_invariants_error: bool,
    /// Report these equivalence properties instead of the input's
    pub claimed_eq_properties: Option<EquivalenceProperties>,
    /// Report this output partitioning instead of the input's
    pub claimed_partitioning: Option<Partitioning>,
    /// Return this from `schema()` instead of the schema of the equivalence
    /// properties
    pub claimed_schema: Option<SchemaRef>,
    /// The expressions `apply_expressions` visits
    pub expressions: Vec<Arc<dyn PhysicalExpr>>,
    /// The expressions `dynamic_expressions_produced` returns
    pub dynamic_expressions: Vec<Arc<dyn PhysicalExpr>>,
    /// Replace every dynamic filter in `expressions` and `dynamic_expressions`
    /// by a new one in `reset_state`
    pub reset_dynamic_expressions: bool,
    /// Return a plan from `with_fetch`, rather than `None`
    pub supports_with_fetch: bool,
    /// Return true from `supports_limit_pushdown`
    pub limit_pushdown: bool,
    /// Report a single output partition, made of every input partition
    pub coalesce: bool,
}

impl ConfigurableExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            effect: Effect::Equal,
            effect_without_fetch: None,
            fetch: None,
            num_rows: None,
            partition_num_rows: None,
            skip_child_stats: false,
            drop_column_stats: 0,
            stats_error: false,
            maintains_input_order_len: None,
            maintains_order: None,
            required_ordering: None,
            invariants_error: false,
            executable_invariants_error: false,
            claimed_eq_properties: None,
            claimed_partitioning: None,
            claimed_schema: None,
            expressions: vec![],
            dynamic_expressions: vec![],
            reset_dynamic_expressions: false,
            supports_with_fetch: false,
            limit_pushdown: false,
            coalesce: false,
        }
    }

    pub fn effect(mut self, effect: Effect) -> Self {
        self.effect = effect;
        self
    }

    pub fn fetch(mut self, fetch: usize) -> Self {
        self.fetch = Some(fetch);
        self
    }

    pub fn num_rows(mut self, num_rows: Precision<usize>) -> Self {
        self.num_rows = Some(num_rows);
        self
    }

    pub fn partition_num_rows(mut self, num_rows: Vec<Precision<usize>>) -> Self {
        self.partition_num_rows = Some(num_rows);
        self
    }

    pub fn build(self) -> Arc<dyn ExecutionPlan> {
        let cache = self.compute_properties();
        Arc::new(Built { exec: self, cache })
    }

    fn compute_properties(&self) -> Arc<PlanProperties> {
        let overridden = self.claimed_eq_properties.is_some()
            || self.claimed_partitioning.is_some()
            || self.coalesce;
        if !overridden {
            return Arc::clone(self.input.properties());
        }
        let mut properties = self.input.properties().as_ref().clone();
        if let Some(eq_properties) = &self.claimed_eq_properties {
            properties = properties.with_eq_properties(eq_properties.clone());
        }
        if self.coalesce {
            properties = properties
                .with_eq_properties(EquivalenceProperties::new(self.input.schema()))
                .with_partitioning(Partitioning::UnknownPartitioning(1));
        }
        if let Some(partitioning) = &self.claimed_partitioning {
            properties = properties.with_partitioning(partitioning.clone());
        }
        Arc::new(properties)
    }
}

/// A [`ConfigurableExec`] with its plan properties computed
#[derive(Debug, Clone)]
struct Built {
    exec: ConfigurableExec,
    cache: Arc<PlanProperties>,
}

impl DisplayAs for Built {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ConfigurableExec")
    }
}

impl ExecutionPlan for Built {
    fn name(&self) -> &'static str {
        "ConfigurableExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn schema(&self) -> SchemaRef {
        match &self.exec.claimed_schema {
            Some(schema) => Arc::clone(schema),
            None => Arc::clone(self.cache.equivalence_properties().schema()),
        }
    }

    fn check_invariants(&self, check: InvariantLevel) -> Result<()> {
        if self.exec.invariants_error {
            return internal_err!("ConfigurableExec invariants are broken");
        }
        if self.exec.executable_invariants_error
            && matches!(check, InvariantLevel::Executable)
        {
            return internal_err!("ConfigurableExec cannot be executed");
        }
        datafusion_physical_plan::execution_plan::check_default_invariants(self, check)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        let maintains = self.exec.maintains_order.unwrap_or(!self.exec.coalesce);
        vec![maintains; self.exec.maintains_input_order_len.unwrap_or(1)]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![
            self.exec
                .required_ordering
                .clone()
                .map(OrderingRequirements::from),
        ]
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.exec.limit_pushdown
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.exec.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(&self.exec.expressions, f)
    }

    fn dynamic_expressions_produced(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.exec.dynamic_expressions.clone()
    }

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let mut exec = self.exec.clone();
        if exec.reset_dynamic_expressions {
            // Each dynamic filter and the new filter that replaces it
            let reset: Vec<_> = exec
                .dynamic_expressions
                .iter()
                .map(|expr| (Arc::clone(expr), new_dynamic_filter(expr)))
                .collect();
            let replace = |expr: &Arc<dyn PhysicalExpr>| {
                reset
                    .iter()
                    .find(|(old, _)| Arc::ptr_eq(old, expr))
                    .map_or_else(|| Arc::clone(expr), |(_, new)| Arc::clone(new))
            };
            exec.expressions = exec.expressions.iter().map(replace).collect();
            exec.dynamic_expressions =
                exec.dynamic_expressions.iter().map(replace).collect();
        }
        Ok(exec.build())
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut exec = self.exec.clone();
        exec.input = children.swap_remove(0);
        Ok(exec.build())
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
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.exec.input.execute(partition, context)
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        if self.exec.skip_child_stats {
            vec![ChildStats::Skip]
        } else if self.exec.coalesce {
            // The only output partition has all input rows
            vec![ChildStats::At(None)]
        } else {
            vec![ChildStats::At(partition)]
        }
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        if self.exec.stats_error {
            return internal_err!("ConfigurableExec statistics are broken");
        }
        let mut stats = input_stats[0].as_ref().clone();
        let partitions = self.properties().output_partitioning().partition_count();
        let num_rows = match (args.partition(), &self.exec.partition_num_rows) {
            (None, _) => self.exec.num_rows,
            (Some(p), Some(rows)) => Some(rows[p]),
            // With a single partition, the overall override also applies to
            // the partition, keeping the statistics consistent
            (Some(_), None) if partitions == 1 => self.exec.num_rows,
            (Some(_), None) => None,
        };
        if let Some(num_rows) = num_rows {
            stats.num_rows = num_rows;
        }
        let keep = stats
            .column_statistics
            .len()
            .saturating_sub(self.exec.drop_column_stats);
        stats.column_statistics.truncate(keep);
        Ok(Arc::new(stats))
    }

    fn fetch(&self) -> Option<usize> {
        self.exec.fetch
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        if !self.exec.supports_with_fetch {
            return None;
        }
        let mut exec = self.exec.clone();
        exec.fetch = fetch;
        Some(exec.build())
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        let effect = match (self.exec.fetch, self.exec.effect_without_fetch) {
            (None, Some(effect)) => effect,
            _ => self.exec.effect,
        };
        match effect {
            Effect::Unknown => CardinalityEffect::Unknown,
            Effect::Equal => CardinalityEffect::Equal,
            Effect::LowerEqual => CardinalityEffect::LowerEqual,
            Effect::GreaterEqual => CardinalityEffect::GreaterEqual,
        }
    }
}

/// A new dynamic filter on the same columns as `expr`, in its initial state,
/// if `expr` is a dynamic filter, and otherwise `expr` itself
fn new_dynamic_filter(expr: &Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
    match expr.downcast_ref::<DynamicFilterPhysicalExpr>() {
        Some(filter) => Arc::new(DynamicFilterPhysicalExpr::new(
            filter.children().into_iter().cloned().collect(),
            lit(true),
        )),
        None => Arc::clone(expr),
    }
}

pub fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]))
}

/// A source with one partition per entry of `rows`
pub fn source(rows: &[usize], precision: StatisticsPrecision) -> Arc<dyn ExecutionPlan> {
    SourceSpec::new(schema())
        .with_partition_rows(rows)
        .with_statistics_precision(precision)
        .build_arc()
        .unwrap()
}

/// A source with a single partition and an exact row count
pub fn exact_source(num_rows: usize) -> Arc<dyn ExecutionPlan> {
    source(&[num_rows], StatisticsPrecision::Exact)
}

/// A source with a single partition and an inexact row count
pub fn inexact_source(num_rows: usize) -> Arc<dyn ExecutionPlan> {
    source(&[num_rows], StatisticsPrecision::Inexact)
}

/// A checker with the built-in checks named `names`
pub fn checker(names: &[&str]) -> PlanChecker {
    let checks: Vec<_> = checks::all_checks()
        .into_iter()
        .filter(|check| names.contains(&check.name))
        .collect();
    assert_eq!(checks.len(), names.len(), "unknown check in {names:?}");
    PlanChecker::with_checks(checks)
}

/// `(severity, check name)` of every violation in the report
pub fn summary(report: &Report) -> Vec<(Severity, &'static str)> {
    report
        .violations
        .iter()
        .map(|v| (v.severity, v.check))
        .collect()
}

/// Messages of every violation in the report
pub fn messages(report: &Report) -> Vec<&str> {
    report
        .violations
        .iter()
        .map(|v| v.message.as_str())
        .collect()
}
