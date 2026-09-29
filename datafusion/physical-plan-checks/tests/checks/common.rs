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
use std::sync::{Arc, Mutex};

use arrow::array::{
    ArrayRef, AsArray, Float64Array, RecordBatch, UInt32Array, new_null_array,
};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef};
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{
    Boundedness, CardinalityEffect, EmissionType, EvaluationType, InvariantLevel,
};
use datafusion_physical_plan::stream::{
    RecordBatchReceiverStream, RecordBatchStreamAdapter,
};
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    Partitioning, PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream,
    StatisticsArgs,
};
use datafusion_physical_plan_checks::fixtures::{SourceSpec, StatisticsPrecision};
use datafusion_physical_plan_checks::{Report, Severity};
use futures::StreamExt;
use futures::stream::BoxStream;

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

/// What `ConfigurableExec` does when its input returns an error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnInputError {
    /// Return the error
    Propagate,
    /// Drop the error and continue with the next item
    Swallow,
    /// Stay pending forever
    Hang,
    /// Panic
    Panic,
}

/// How `ConfigurableExec` applies its fetch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchMode {
    /// Report the fetch without applying it, and return `None` from
    /// `with_fetch`
    Ignore,
    /// Stop after `fetch` rows per partition, without polling the input again
    Enforce,
    /// Keep one row more than the fetch
    KeepOneMore,
    /// Keep half as many rows as the fetch
    KeepHalf,
    /// Skip the first row of each partition, then keep `fetch` rows
    SkipFirstRow,
    /// Repeat the first row of each partition `fetch` times
    RepeatFirstRow,
}

/// Where `ConfigurableExec` keeps its input streams when its output stream is
/// dropped
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoldInput {
    /// Drop them with the output stream
    No,
    /// Keep them in the plan until the plan is dropped
    InPlan,
    /// Never drop them
    Forever,
    /// Drop them from a spawned task, some time after the output stream is
    /// dropped
    InTask,
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
    /// Overrides the reported boundedness
    pub boundedness: Option<Boundedness>,
    /// Overrides the reported emission type
    pub emission_type: Option<EmissionType>,
    /// Overrides the reported evaluation type
    pub evaluation_type: Option<EvaluationType>,
    /// Buffer all input and emit it when the input ends
    pub buffer_all: bool,
    /// Poll the input from a spawned task
    pub eager: bool,
    /// How to apply the fetch. `with_fetch` returns `None` unless the fetch is
    /// applied in some way.
    pub fetch_mode: FetchMode,
    /// Keep the fetch when `with_fetch(None)` is called
    pub keep_fetch_on_with_fetch_none: bool,
    /// Make the plans returned by `with_fetch` report no ordering
    pub with_fetch_drops_ordering: bool,
    /// Report no ordering
    pub drop_ordering: bool,
    /// Return true from `supports_limit_pushdown`
    pub limit_pushdown: bool,
    /// Skip the first `skip` rows of each partition, as an `OFFSET` would
    pub skip: usize,
    /// Report a single output partition, and produce the rows of every input
    /// partition in it
    pub coalesce: bool,
    /// End each partition after as many rows as the session batch size, as a
    /// node that only emits its first output batch would
    pub stop_after_session_batch: bool,
    /// Multiply the `Float64` values of the `k`-th batch of each partition,
    /// counting from 1, by `1 + k * noise`
    pub float_noise: Option<f64>,
    /// What to do when the input returns an error
    pub on_input_error: OnInputError,
    /// Where to keep input streams when the output stream is dropped
    pub hold_input: HoldInput,
    /// Reserve memory in every partition and never release it
    pub leak_memory: bool,
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
            boundedness: None,
            emission_type: None,
            evaluation_type: None,
            buffer_all: false,
            eager: false,
            fetch_mode: FetchMode::Ignore,
            keep_fetch_on_with_fetch_none: false,
            with_fetch_drops_ordering: false,
            drop_ordering: false,
            limit_pushdown: false,
            skip: 0,
            coalesce: false,
            stop_after_session_batch: false,
            float_noise: None,
            on_input_error: OnInputError::Propagate,
            hold_input: HoldInput::No,
            leak_memory: false,
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
        Arc::new(Built {
            exec: self,
            cache,
            held_inputs: HeldInputs::default(),
        })
    }

    fn compute_properties(&self) -> Arc<PlanProperties> {
        let overridden = self.claimed_ordering.is_some()
            || self.drop_ordering
            || self.coalesce
            || self.boundedness.is_some()
            || self.emission_type.is_some()
            || self.evaluation_type.is_some();
        if !overridden {
            return Arc::clone(self.input.properties());
        }
        let mut properties = self.input.properties().as_ref().clone();
        if let Some(ordering) = &self.claimed_ordering {
            properties =
                properties.with_eq_properties(EquivalenceProperties::new_with_orderings(
                    self.input.schema(),
                    [ordering.clone()],
                ));
        }
        if self.drop_ordering || self.coalesce {
            properties = properties
                .with_eq_properties(EquivalenceProperties::new(self.input.schema()));
        }
        if self.coalesce {
            properties =
                properties.with_partitioning(Partitioning::UnknownPartitioning(1));
        }
        if let Some(boundedness) = self.boundedness {
            properties = properties.with_boundedness(boundedness);
        }
        if let Some(emission_type) = self.emission_type {
            properties = properties.with_emission_type(emission_type);
        }
        if let Some(evaluation_type) = self.evaluation_type {
            properties = properties.with_evaluation_type(evaluation_type);
        }
        Arc::new(properties)
    }
}

/// A [`ConfigurableExec`] with its plan properties computed
#[derive(Debug, Clone)]
struct Built {
    exec: ConfigurableExec,
    cache: Arc<PlanProperties>,
    /// Input streams kept for [`HoldInput::InPlan`]
    held_inputs: HeldInputs,
}

/// Input streams kept alive by the plan
#[derive(Clone, Default)]
struct HeldInputs(Arc<Mutex<Vec<SendableRecordBatchStream>>>);

impl fmt::Debug for HeldInputs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HeldInputs")
    }
}

/// Owns an input stream and keeps it according to a [`HoldInput`] when
/// dropped
struct HeldInput {
    stream: Option<SendableRecordBatchStream>,
    hold: HoldInput,
    held_inputs: HeldInputs,
}

impl Drop for HeldInput {
    fn drop(&mut self) {
        let Some(stream) = self.stream.take() else {
            return;
        };
        match self.hold {
            HoldInput::No => {}
            HoldInput::InPlan => self.held_inputs.0.lock().unwrap().push(stream),
            HoldInput::Forever => {
                Box::leak(Box::new(stream));
            }
            HoldInput::InTask => {
                // A detached task, which must outlive the stream that spawns it
                #[expect(clippy::disallowed_methods)]
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    drop(stream);
                });
            }
        }
    }
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
        vec![!self.exec.coalesce; self.exec.maintains_input_order_len.unwrap_or(1)]
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.exec.limit_pushdown
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
        if self.exec.leak_memory {
            let reservation =
                MemoryConsumer::new(format!("ConfigurableExec[{partition}]"))
                    .register(context.memory_pool());
            reservation.grow(1024);
            Box::leak(Box::new(reservation));
        }
        let mut input = if self.exec.coalesce {
            let partitions = self
                .exec
                .input
                .properties()
                .output_partitioning()
                .partition_count();
            let streams = (0..partitions)
                .map(|p| self.exec.input.execute(p, Arc::clone(&context)))
                .collect::<Result<Vec<_>>>()?;
            Box::pin(RecordBatchStreamAdapter::new(
                self.exec.input.schema(),
                futures::stream::iter(streams).flatten(),
            ))
        } else {
            self.exec.input.execute(partition, Arc::clone(&context))?
        };
        if self.exec.eager {
            let mut builder = RecordBatchReceiverStream::builder(self.schema(), 2);
            let tx = builder.tx();
            builder.spawn(async move {
                while let Some(item) = input.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
                Ok(())
            });
            input = builder.build();
        }
        let mut held = HeldInput {
            stream: Some(input),
            hold: self.exec.hold_input,
            held_inputs: self.held_inputs.clone(),
        };
        let on_input_error = self.exec.on_input_error;
        let mut hung = false;
        let items = futures::stream::poll_fn(move |cx| {
            let stream = held.stream.as_mut().expect("input is present until drop");
            loop {
                if hung {
                    return std::task::Poll::Pending;
                }
                return match stream.poll_next_unpin(cx) {
                    std::task::Poll::Ready(Some(Err(e))) => match on_input_error {
                        OnInputError::Propagate => std::task::Poll::Ready(Some(Err(e))),
                        OnInputError::Swallow => continue,
                        OnInputError::Hang => {
                            hung = true;
                            continue;
                        }
                        OnInputError::Panic => panic!("ConfigurableExec got an error"),
                    },
                    poll => poll,
                };
            }
        });
        let transform = self.exec.transform;
        let mut stream = items.map(move |batch| transform.apply(batch?)).boxed();
        if let Some(noise) = self.exec.float_noise {
            let mut k = 0;
            stream = stream
                .map(move |batch| {
                    k += 1;
                    scale_floats(&batch?, 1.0 + k as f64 * noise)
                })
                .boxed();
        }

        let fetch = self
            .exec
            .fetch
            .filter(|_| self.exec.fetch_mode != FetchMode::Ignore);
        let skip = match (fetch, self.exec.fetch_mode) {
            (Some(_), FetchMode::SkipFirstRow) => self.exec.skip + 1,
            _ => self.exec.skip,
        };
        let fetch_limit = fetch.and_then(|fetch| match self.exec.fetch_mode {
            FetchMode::Ignore | FetchMode::RepeatFirstRow => None,
            FetchMode::Enforce | FetchMode::SkipFirstRow => Some(fetch),
            FetchMode::KeepOneMore => Some(fetch + 1),
            FetchMode::KeepHalf => Some(fetch / 2),
        });
        let batch_limit = self
            .exec
            .stop_after_session_batch
            .then(|| context.session_config().batch_size());
        let limit = match (fetch_limit, batch_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        if skip > 0 || limit.is_some() {
            stream = skip_and_limit(stream, skip, limit);
        }
        if let (Some(fetch), FetchMode::RepeatFirstRow) = (fetch, self.exec.fetch_mode) {
            let schema = self.schema();
            stream = futures::stream::once(async move {
                let mut stream = stream;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    if batch.num_rows() > 0 {
                        let indices = UInt32Array::from(vec![0; fetch]);
                        return Ok(take_record_batch(&batch, &indices)?);
                    }
                }
                Ok(RecordBatch::new_empty(schema))
            })
            .boxed();
        }
        if self.exec.buffer_all {
            let schema = self.schema();
            stream = futures::stream::once(async move {
                let batches = stream.collect::<Vec<_>>().await;
                let batches = batches.into_iter().collect::<Result<Vec<_>>>()?;
                Ok(concat_batches(&schema, &batches)?)
            })
            .boxed();
        }
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
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
        if self.exec.fetch_mode == FetchMode::Ignore {
            return None;
        }
        let mut exec = self.exec.clone();
        if fetch.is_some() || !exec.keep_fetch_on_with_fetch_none {
            exec.fetch = fetch;
        }
        if exec.with_fetch_drops_ordering {
            exec.drop_ordering = true;
        }
        Some(exec.build())
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

/// Skip the first `skip` rows of `stream`, then end after `limit` rows, if
/// set, without polling the input again
fn skip_and_limit(
    mut stream: BoxStream<'static, Result<RecordBatch>>,
    skip: usize,
    limit: Option<usize>,
) -> BoxStream<'static, Result<RecordBatch>> {
    let mut to_skip = skip;
    let mut remaining = limit;
    futures::stream::poll_fn(move |cx| {
        if remaining == Some(0) {
            return std::task::Poll::Ready(None);
        }
        match stream.poll_next_unpin(cx) {
            std::task::Poll::Ready(Some(Ok(batch))) => {
                let skipped = to_skip.min(batch.num_rows());
                to_skip -= skipped;
                let mut batch = batch.slice(skipped, batch.num_rows() - skipped);
                if let Some(remaining) = remaining.as_mut() {
                    let rows = batch.num_rows().min(*remaining);
                    *remaining -= rows;
                    batch = batch.slice(0, rows);
                }
                std::task::Poll::Ready(Some(Ok(batch)))
            }
            poll => poll,
        }
    })
    .boxed()
}

/// Multiply every `Float64` value of `batch` by `factor`
fn scale_floats(batch: &RecordBatch, factor: f64) -> Result<RecordBatch> {
    let columns = batch
        .columns()
        .iter()
        .map(|column| match column.data_type() {
            DataType::Float64 => {
                let values: Float64Array =
                    column.as_primitive::<Float64Type>().unary(|v| v * factor);
                Arc::new(values) as ArrayRef
            }
            _ => Arc::clone(column),
        })
        .collect();
    Ok(RecordBatch::try_new(batch.schema(), columns)?)
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
