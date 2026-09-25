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

use arrow::array::{RecordBatch, UInt32Array, new_null_array};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{CardinalityEffect, InvariantLevel};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};
use datafusion_physical_plan_checks::fixtures::{SourceSpec, StatisticsPrecision};
use datafusion_physical_plan_checks::{Report, Severity};
use futures::StreamExt;

#[derive(Debug, Clone, Copy)]
pub enum Effect {
    Unknown,
    Equal,
    LowerEqual,
    GreaterEqual,
}

/// How `ConfigurableExec` changes each batch it passes through
#[derive(Debug, Clone, Copy)]
pub enum Transform {
    /// Pass batches through unchanged
    None,
    /// Keep only the first half of the rows of each batch
    DropHalf,
    /// Repeat every batch twice
    Duplicate,
    /// Reverse the rows of each batch
    Reverse,
    /// Replace the first column with nulls, and mark it nullable in the batch
    NullFirstColumn,
    /// Rename the first column in the batch
    RenameFirstColumn,
    /// Panic
    Panic,
}

impl Transform {
    fn apply(self, batch: RecordBatch) -> Result<RecordBatch> {
        let rows = batch.num_rows();
        Ok(match self {
            Transform::None => batch,
            Transform::Panic => panic!("ConfigurableExec panicked"),
            Transform::DropHalf => batch.slice(0, rows / 2),
            Transform::Duplicate => concat_batches(&batch.schema(), [&batch, &batch])?,
            Transform::Reverse => {
                let indices = UInt32Array::from_iter_values((0..rows as u32).rev());
                take_record_batch(&batch, &indices)?
            }
            Transform::NullFirstColumn | Transform::RenameFirstColumn => {
                let schema = batch.schema();
                let first = schema.field(0);
                let (field, column) = match self {
                    Transform::NullFirstColumn => (
                        first.clone().with_nullable(true),
                        new_null_array(first.data_type(), rows),
                    ),
                    _ => (
                        first.clone().with_name("renamed"),
                        Arc::clone(batch.column(0)),
                    ),
                };
                let mut fields: Vec<Field> =
                    schema.fields().iter().map(|f| f.as_ref().clone()).collect();
                fields[0] = field;
                let mut columns = batch.columns().to_vec();
                columns[0] = column;
                RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?
            }
        })
    }
}

/// A single-child plan that passes its input through, with knobs to make it
/// break individual parts of the `ExecutionPlan` contract.
#[derive(Debug, Clone)]
pub struct ConfigurableExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub effect: Effect,
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
    /// Return an error from `check_invariants`
    pub invariants_error: bool,
    /// Report this output ordering instead of the input's orderings
    pub claimed_ordering: Option<LexOrdering>,
    /// Change each output batch
    pub transform: Transform,
    /// Return an error from `execute`
    pub execute_error: bool,
    /// Return a stream that never produces a batch or ends
    pub hang: bool,
}

impl ConfigurableExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
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
            claimed_ordering: None,
            transform: Transform::None,
            execute_error: false,
            hang: false,
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

    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn build(self) -> Arc<dyn ExecutionPlan> {
        let cache = self.compute_properties();
        Arc::new(Built { exec: self, cache })
    }

    fn compute_properties(&self) -> Arc<PlanProperties> {
        match &self.claimed_ordering {
            None => Arc::clone(self.input.properties()),
            Some(ordering) => {
                let eq_properties = EquivalenceProperties::new_with_orderings(
                    self.input.schema(),
                    [ordering.clone()],
                );
                Arc::new(
                    self.input
                        .properties()
                        .as_ref()
                        .clone()
                        .with_eq_properties(eq_properties),
                )
            }
        }
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

    fn check_invariants(&self, check: InvariantLevel) -> Result<()> {
        if self.exec.invariants_error {
            return internal_err!("ConfigurableExec invariants are broken");
        }
        datafusion_physical_plan::execution_plan::check_default_invariants(self, check)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true; self.exec.maintains_input_order_len.unwrap_or(1)]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.exec.input]
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
        if self.exec.execute_error {
            return internal_err!("ConfigurableExec cannot execute");
        }
        if self.exec.hang {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema(),
                futures::stream::pending(),
            )));
        }
        let transform = self.exec.transform;
        let stream = self
            .exec
            .input
            .execute(partition, context)?
            .map(move |batch| transform.apply(batch?));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        if self.exec.skip_child_stats {
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

    fn cardinality_effect(&self) -> CardinalityEffect {
        match self.exec.effect {
            Effect::Unknown => CardinalityEffect::Unknown,
            Effect::Equal => CardinalityEffect::Equal,
            Effect::LowerEqual => CardinalityEffect::LowerEqual,
            Effect::GreaterEqual => CardinalityEffect::GreaterEqual,
        }
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

/// `(severity, check name)` of every violation in the report
pub fn summary(report: &Report) -> Vec<(Severity, &'static str)> {
    report
        .violations()
        .iter()
        .map(|v| (v.severity, v.check))
        .collect()
}
