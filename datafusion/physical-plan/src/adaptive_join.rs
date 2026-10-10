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

//! Experimental runtime build-side selection for hash joins.
//!
//! [`ThresholdBoundaryExec`] drains its input until EOF or a per-partition
//! byte limit, then reports itself ready for inspection. [`AdaptiveJoinExec`]
//! sits at the plan root and, before executing the plan, resolves hash joins
//! bottom-up: it primes the boundaries under each join whose subtrees are
//! already resolved, and races both sides, doubling the limit, until one side
//! is complete and the other is known to be at least as large. If the
//! complete side is the probe side, it swaps the join inputs. It then releases
//! the boundaries. Buffering is bounded by roughly twice the smaller side, and
//! by `total_max_bytes` per side. Decisions compare row counts.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::DataType;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{
    DataFusionError, Result, assert_eq_or_internal_err, exec_datafusion_err, internal_err,
};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::{Distribution, Partitioning, PhysicalExpr};
use futures::{StreamExt, TryStreamExt, stream};
use log::debug;
use tokio::sync::{OnceCell, watch};

use crate::execution_plan::EvaluationType;
use crate::joins::HashJoinExec;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream, StageBoundary, StageProgress,
};

/// Approximate memory of `batch`. View arrays produced by `take` (e.g. in
/// `RepartitionExec`) share their data buffers with every other batch taken
/// from the same input, so `get_array_memory_size` would count those buffers
/// once per batch; count only the bytes the views reference instead.
fn batch_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|array| {
            let views = match array.data_type() {
                DataType::Utf8View => {
                    Some(array.as_string_view().total_buffer_bytes_used())
                }
                DataType::BinaryView => {
                    Some(array.as_binary_view().total_buffer_bytes_used())
                }
                _ => None,
            };
            match views {
                Some(used) => {
                    used + array.len() * 16
                        + array.nulls().map_or(0, |n| n.buffer().len())
                }
                None => array.get_array_memory_size(),
            }
        })
        .sum()
}

/// What a partition drain produced: buffered batches, plus the unread
/// remainder of the input if the drain stopped before EOF.
struct Buffered {
    batches: Vec<RecordBatch>,
    rest: Option<SendableRecordBatchStream>,
    /// Holds the memory of `batches`; shrunk as they are emitted.
    reservation: MemoryReservation,
}

impl fmt::Debug for Buffered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffered")
            .field("batches", &self.batches.len())
            .finish()
    }
}

#[derive(Debug)]
struct ThresholdPartition {
    /// Live progress of the drain. `complete` is also set on error.
    status: watch::Sender<StageProgress>,
    /// Set once the drain task stops and hands over its output.
    buffered: Arc<Mutex<Option<Result<Buffered>>>>,
    handed_over: watch::Sender<bool>,
    task: Mutex<Option<SpawnedTask<()>>>,
}

/// Boundary that buffers each partition until EOF or a per-partition byte
/// limit, and is ready for inspection at that point. The driver can raise the
/// limit with [`Self::set_limit`] to keep draining.
///
/// After [`StageBoundary::release`], each partition emits its buffered batches
/// followed by the rest of its input. Executing an unprimed partition primes
/// it.
#[derive(Debug)]
pub struct ThresholdBoundaryExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    total_max_bytes: usize,
    max_bytes: usize,
    partitions: Vec<ThresholdPartition>,
    limit: watch::Sender<usize>,
    released: watch::Sender<bool>,
    /// Set when a partition could not reserve memory for a batch, which stops
    /// the race.
    exhausted: Arc<AtomicBool>,
}

/// Per-partition limit the race starts with before doubling.
const INITIAL_LIMIT_BYTES: usize = 1024 * 1024;

impl ThresholdBoundaryExec {
    /// `total_max_bytes` caps how much the boundary may buffer across all
    /// partitions; it is split evenly into a per-partition limit cap.
    pub fn new(input: Arc<dyn ExecutionPlan>, total_max_bytes: usize) -> Self {
        let partition_count = input.properties().output_partitioning().partition_count();
        let max_bytes = total_max_bytes / partition_count.max(1);
        let properties = PlanProperties::clone(input.properties())
            .with_evaluation_type(EvaluationType::Eager);
        let partitions = (0..partition_count)
            .map(|_| ThresholdPartition {
                status: watch::channel(StageProgress {
                    rows: 0,
                    bytes: 0,
                    complete: false,
                })
                .0,
                buffered: Arc::new(Mutex::new(None)),
                handed_over: watch::channel(false).0,
                task: Mutex::new(None),
            })
            .collect();
        Self {
            input,
            properties: Arc::new(properties),
            total_max_bytes,
            max_bytes,
            partitions,
            limit: watch::channel(INITIAL_LIMIT_BYTES.min(max_bytes)).0,
            released: watch::channel(false).0,
            exhausted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether a partition ran out of memory pool space while draining.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted.load(Ordering::Relaxed)
    }

    pub fn is_released(&self) -> bool {
        *self.released.borrow()
    }

    /// Sets the per-partition byte limit, resuming paused drains if raised.
    pub fn set_limit(&self, bytes: usize) {
        self.limit.send_replace(bytes);
    }

    /// Waits until every partition is ready under the current limit.
    pub async fn wait_ready(&self) {
        let limit = *self.limit.borrow();
        for state in &self.partitions {
            // An error means the sender was dropped, which cannot happen while
            // `self` is alive.
            let _ = state
                .status
                .subscribe()
                .wait_for(|p| p.complete || p.bytes >= limit || self.is_exhausted())
                .await;
        }
    }

    /// Sums progress across partitions; complete only if every partition is.
    pub fn total_progress(&self) -> Option<StageProgress> {
        (0..self.partitions.len()).try_fold(
            StageProgress {
                rows: 0,
                bytes: 0,
                complete: true,
            },
            |acc, partition| {
                let p = self.progress(partition)?;
                Some(StageProgress {
                    rows: acc.rows + p.rows,
                    bytes: acc.bytes + p.bytes,
                    complete: acc.complete && p.complete,
                })
            },
        )
    }
}

impl StageBoundary for ThresholdBoundaryExec {
    fn prime(&self, partition: usize, context: Arc<TaskContext>) -> Result<()> {
        let Some(state) = self.partitions.get(partition) else {
            return internal_err!("ThresholdBoundaryExec invalid partition {partition}");
        };
        let mut task = state.task.lock().unwrap();
        if task.is_some() {
            return Ok(());
        }
        let reservation =
            MemoryConsumer::new(format!("ThresholdBoundaryExec[{partition}]"))
                .register(context.memory_pool());
        let mut input = self.input.execute(partition, context)?;
        let exhausted = Arc::clone(&self.exhausted);
        let mut limit = self.limit.subscribe();
        let mut released = self.released.subscribe();
        let status = state.status.clone();
        let buffered = Arc::clone(&state.buffered);
        let handed_over = state.handed_over.clone();
        *task = Some(SpawnedTask::spawn(async move {
            let mut batches = Vec::new();
            let mut progress = *status.borrow();
            let complete = loop {
                if *released.borrow() {
                    break Ok(false);
                }
                if progress.bytes >= *limit.borrow_and_update() {
                    tokio::select! {
                        _ = limit.changed() => continue,
                        _ = released.wait_for(|released| *released) => break Ok(false),
                    }
                }
                match input.next().await {
                    Some(Ok(batch)) => {
                        let size = batch_size(&batch);
                        let reserved = reservation.try_grow(size).is_ok();
                        progress.rows += batch.num_rows();
                        progress.bytes += size;
                        batches.push(batch);
                        if !reserved {
                            // Keep the batch (it is already in memory) but stop
                            // draining; the race gives up on this join.
                            debug!(
                                "ThresholdBoundaryExec: memory pool exhausted after {progress:?}"
                            );
                            exhausted.store(true, Ordering::Relaxed);
                            status.send_replace(progress);
                            break Ok(false);
                        }
                        status.send_replace(progress);
                    }
                    Some(Err(e)) => break Err(e),
                    None => break Ok(true),
                }
            };
            let done = !matches!(complete, Ok(false));
            *buffered.lock().unwrap() = Some(complete.map(|complete| Buffered {
                batches,
                rest: (!complete).then_some(input),
                reservation,
            }));
            if done {
                progress.complete = true;
                status.send_replace(progress);
            }
            handed_over.send_replace(true);
        }));
        Ok(())
    }

    fn is_ready(&self, partition: usize) -> bool {
        let limit = *self.limit.borrow();
        self.partitions.get(partition).is_some_and(|state| {
            let p = *state.status.borrow();
            p.complete || p.bytes >= limit || self.is_exhausted()
        })
    }

    fn progress(&self, partition: usize) -> Option<StageProgress> {
        let state = self.partitions.get(partition)?;
        if matches!(*state.buffered.lock().unwrap(), Some(Err(_))) {
            return None;
        }
        Some(*state.status.borrow())
    }

    fn release(&self) {
        self.released.send_replace(true);
    }
}

impl DisplayAs for ThresholdBoundaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ThresholdBoundaryExec: total_max_bytes={}",
            self.total_max_bytes
        )
    }
}

impl ExecutionPlan for ThresholdBoundaryExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
        assert_eq_or_internal_err!(
            children.len(),
            1,
            "ThresholdBoundaryExec expected one child"
        );
        Ok(Arc::new(Self::new(
            children.swap_remove(0),
            self.total_max_bytes,
        )))
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
        self.prime(partition, context)?;
        let state = &self.partitions[partition];
        let buffered = Arc::clone(&state.buffered);
        let mut handed_over = state.handed_over.subscribe();
        let mut released = self.released.subscribe();
        let output = async move {
            released
                .wait_for(|released| *released)
                .await
                .map_err(|e| exec_datafusion_err!("boundary dropped: {e}"))?;
            handed_over
                .wait_for(|handed_over| *handed_over)
                .await
                .map_err(|e| exec_datafusion_err!("boundary dropped: {e}"))?;
            let Buffered {
                batches,
                rest,
                reservation,
            } = buffered
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| exec_datafusion_err!("partition executed twice"))??;
            let head = stream::iter(batches.into_iter().map(move |batch| {
                reservation.shrink(batch_size(&batch).min(reservation.size()));
                Ok(batch)
            }));
            Ok::<_, DataFusionError>(match rest {
                Some(rest) => head.chain(rest).boxed(),
                None => head.boxed(),
            })
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream::once(output).try_flatten(),
        )))
    }

    fn as_boundary(&self) -> Option<&dyn StageBoundary> {
        Some(self)
    }
}

type Adapted = std::result::Result<Arc<dyn ExecutionPlan>, Arc<DataFusionError>>;

/// Plan root that resolves the build side of every hash join whose inputs are
/// [`ThresholdBoundaryExec`]s before executing the plan.
#[derive(Debug)]
pub struct AdaptiveJoinExec {
    input: Arc<dyn ExecutionPlan>,
    adapted: Arc<OnceCell<Adapted>>,
}

impl AdaptiveJoinExec {
    pub fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            input,
            adapted: Arc::new(OnceCell::new()),
        }
    }
}

/// Visits the nodes of `plan` that this driver owns, skipping the subtrees of
/// nested [`AdaptiveJoinExec`]s (e.g. under a `ScalarSubqueryExec`), which
/// resolve their own joins when executed.
fn apply_in_scope(
    plan: &Arc<dyn ExecutionPlan>,
    mut f: impl FnMut(&Arc<dyn ExecutionPlan>) -> Result<()>,
) -> Result<()> {
    plan.apply(|node| {
        if !Arc::ptr_eq(node, plan) && node.downcast_ref::<AdaptiveJoinExec>().is_some() {
            return Ok(TreeNodeRecursion::Jump);
        }
        f(node)?;
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

fn has_unreleased_boundary(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let mut found = false;
    let _ = apply_in_scope(plan, |n| {
        found |= unreleased_boundary(n).is_some();
        Ok(())
    });
    found
}

fn unreleased_boundary(node: &Arc<dyn ExecutionPlan>) -> Option<&ThresholdBoundaryExec> {
    node.downcast_ref::<ThresholdBoundaryExec>()
        .filter(|b| !b.is_released())
}

/// The `(build, probe)` boundaries of a hash join.
type JoinBoundaries = (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>);

/// The `(build, probe)` boundaries of hash joins whose inputs contain no
/// other unreleased boundaries, i.e. the next stage that can be primed without
/// deadlocking.
fn next_stage(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<JoinBoundaries>> {
    let mut stage = vec![];
    apply_in_scope(plan, |node| {
        if node.downcast_ref::<HashJoinExec>().is_none() {
            return Ok(());
        }
        let children = node.children();
        let resolvable = children.iter().all(|child| {
            unreleased_boundary(child).is_some_and(|b| !has_unreleased_boundary(&b.input))
        });
        if resolvable {
            stage.push((Arc::clone(children[0]), Arc::clone(children[1])));
        }
        Ok(())
    })?;
    Ok(stage)
}

fn boundary(node: &Arc<dyn ExecutionPlan>) -> &ThresholdBoundaryExec {
    node.downcast_ref::<ThresholdBoundaryExec>()
        .expect("next_stage only returns boundaries")
}

/// Drains both sides of a join, doubling the per-partition limit, until one
/// side is complete and the other is known to be at least as large. Returns
/// whether the probe side should become the build side, or `None` if the
/// limit cap was reached or a side failed.
async fn race(
    build: &ThresholdBoundaryExec,
    probe: &ThresholdBoundaryExec,
) -> Option<bool> {
    let max = build.max_bytes.min(probe.max_bytes);
    let mut limit = INITIAL_LIMIT_BYTES.min(max);
    loop {
        build.set_limit(limit);
        probe.set_limit(limit);
        build.wait_ready().await;
        probe.wait_ready().await;
        if build.is_exhausted() || probe.is_exhausted() {
            debug!("AdaptiveJoinExec: race undecided: memory pool exhausted");
            return None;
        }
        let b = build.total_progress()?;
        let p = probe.total_progress()?;
        // Rows, not bytes: building and probing both cost per row.
        if p.complete && (b.complete || b.rows >= p.rows) {
            let swap = !b.complete || p.rows < b.rows;
            debug!("AdaptiveJoinExec: race done: build={b:?} probe={p:?} swap={swap}");
            return Some(swap);
        }
        if b.complete && p.rows >= b.rows {
            debug!("AdaptiveJoinExec: race done: build={b:?} probe={p:?} swap=false");
            return Some(false);
        }
        if limit >= max {
            debug!("AdaptiveJoinExec: race undecided at cap: build={b:?} probe={p:?}");
            return None;
        }
        limit = limit.saturating_mul(2).min(max);
    }
}

/// Returns the swapped join if swapping keeps the partition count and the
/// output ordering, and either keeps the output partitioning or `free` says
/// no ancestor depends on it.
fn swap(
    node: &Arc<dyn ExecutionPlan>,
    join: &HashJoinExec,
    free: bool,
) -> Option<Arc<dyn ExecutionPlan>> {
    if !join.join_type().supports_swap() {
        return None;
    }
    // Dynamic filters are wired to the original build side; drop them.
    let swapped = join
        .builder()
        .reset_state()
        .build()
        .and_then(|join| join.swap_inputs(*join.partition_mode()))
        .ok()?;
    let (old, new) = (node.properties(), swapped.properties());
    let count_ok = old.output_partitioning().partition_count()
        == new.output_partitioning().partition_count();
    // `HashJoinExec` reports hash partitioning on its join keys even when its
    // embedded projection drops them, so this is often false when it does
    // not matter; `free` covers those cases.
    let partitioning_ok = free
        || match old.output_partitioning() {
            #[expect(deprecated)]
            Partitioning::Hash(exprs, _) => new.output_partitioning().satisfy(
                &Distribution::KeyPartitioned(exprs.clone()),
                new.equivalence_properties(),
            ),
            _ => true,
        };
    let ordering_ok =
        old.output_ordering().is_none() || old.output_ordering() == new.output_ordering();
    if !count_ok || !partitioning_ok || !ordering_ok {
        debug!(
            "AdaptiveJoinExec: not swapping {} {} (count_ok={count_ok}, partitioning_ok={partitioning_ok}, ordering_ok={ordering_ok}) old={} new={}",
            join.partition_mode(),
            join.join_type(),
            old.output_partitioning(),
            new.output_partitioning()
        );
        return None;
    }
    debug!(
        "AdaptiveJoinExec: swapped {} {} join",
        join.partition_mode(),
        join.join_type()
    );
    Some(swapped)
}

/// Whether `node`'s `child_idx` child may change its hash partitioning (but
/// not its partition count), given whether `node` itself may (`free`).
fn child_free(node: &Arc<dyn ExecutionPlan>, child_idx: usize, free: bool) -> bool {
    let requirements = node.input_distribution_requirements();
    match requirements.child_distribution(child_idx) {
        #[expect(deprecated)]
        Some(Distribution::HashPartitioned(_) | Distribution::KeyPartitioned(_)) => false,
        Some(Distribution::SinglePartition) => true,
        _ => {
            free || matches!(
                node.name(),
                "RepartitionExec" | "CoalescePartitionsExec" | "SortPreservingMergeExec"
            )
        }
    }
}

/// Swaps every hash join whose build boundary is in `to_swap`, where safe.
fn swap_joins(
    node: Arc<dyn ExecutionPlan>,
    to_swap: &[Arc<dyn ExecutionPlan>],
    free: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    let children = node.children();
    if children.is_empty() || node.downcast_ref::<AdaptiveJoinExec>().is_some() {
        return Ok(node);
    }
    let new_children = children
        .iter()
        .enumerate()
        .map(|(i, child)| {
            swap_joins(Arc::clone(child), to_swap, child_free(&node, i, free))
        })
        .collect::<Result<Vec<_>>>()?;
    let unchanged = children
        .iter()
        .zip(&new_children)
        .all(|(old, new)| Arc::ptr_eq(old, new));
    let node = if unchanged {
        node
    } else {
        node.replace_children(
            new_children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?
    };
    let Some(join) = node.downcast_ref::<HashJoinExec>() else {
        return Ok(node);
    };
    if !to_swap.iter().any(|b| Arc::ptr_eq(b, join.left())) {
        return Ok(node);
    }
    Ok(swap(&node, join, free).unwrap_or(node))
}

async fn adapt(
    mut plan: Arc<dyn ExecutionPlan>,
    context: Arc<TaskContext>,
) -> Result<Arc<dyn ExecutionPlan>> {
    loop {
        let stage = next_stage(&plan)?;
        if stage.is_empty() {
            break;
        }
        for (build, probe) in &stage {
            for b in [boundary(build), boundary(probe)] {
                for partition in 0..b.partitions.len() {
                    b.prime(partition, Arc::clone(&context))?;
                }
            }
        }
        let decisions = futures::future::join_all(
            stage
                .iter()
                .map(|(build, probe)| race(boundary(build), boundary(probe))),
        )
        .await;
        let to_swap = stage
            .iter()
            .zip(decisions)
            .filter(|(_, swap)| *swap == Some(true))
            .map(|((build, _), _)| Arc::clone(build))
            .collect::<Vec<_>>();
        plan = swap_joins(plan, &to_swap, true)?;
        for (build, probe) in &stage {
            boundary(build).release();
            boundary(probe).release();
        }
    }
    // Release anything not under a hash join so it cannot block execution.
    apply_in_scope(&plan, |node| {
        if let Some(boundary) = unreleased_boundary(node) {
            boundary.release();
        }
        Ok(())
    })?;
    Ok(plan)
}

impl DisplayAs for AdaptiveJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "AdaptiveJoinExec")
    }
}

impl ExecutionPlan for AdaptiveJoinExec {
    fn name(&self) -> &'static str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
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
        assert_eq_or_internal_err!(
            children.len(),
            1,
            "AdaptiveJoinExec expected one child"
        );
        Ok(Arc::new(Self::new(children.swap_remove(0))))
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
        let input = Arc::clone(&self.input);
        let adapted = Arc::clone(&self.adapted);
        let output = async move {
            let plan = adapted
                .get_or_init(|| {
                    let context = Arc::clone(&context);
                    async move { adapt(input, context).await.map_err(Arc::new) }
                })
                .await
                .clone()
                .map_err(DataFusionError::Shared)?;
            plan.execute(partition, context)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream::once(output).try_flatten(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joins::{JoinOn, PartitionMode};
    use crate::repartition::RepartitionExec;
    use crate::test::{TestMemoryExec, build_table_i32};
    use datafusion_common::{JoinType, NullEquality};
    use datafusion_physical_expr::expressions::Column;

    fn hashed_input(cols: [&str; 3], key: usize) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = build_table_i32(
            (cols[0], &vec![1, 2, 3]),
            (cols[1], &vec![4, 5, 6]),
            (cols[2], &vec![7, 8, 9]),
        );
        let schema = batch.schema();
        let source = TestMemoryExec::try_new_exec(&[vec![batch]], schema, None)?;
        let key = Arc::new(Column::new(cols[key], key)) as _;
        let repartition = Arc::new(RepartitionExec::try_new(
            source,
            Partitioning::Hash(vec![key], 4),
        )?);
        Ok(Arc::new(ThresholdBoundaryExec::new(
            repartition,
            usize::MAX,
        )))
    }

    fn partitioned_join(
        projection: Option<Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Mirrors TPC-H Q8: orders (build) joined to lineitem (probe) on orderkey.
        let orders = hashed_input(["o_orderkey", "o_custkey", "o_x"], 0)?;
        let lineitem = hashed_input(["l_partkey", "l_x", "l_orderkey"], 2)?;
        let on: JoinOn = vec![(
            Arc::new(Column::new("o_orderkey", 0)) as _,
            Arc::new(Column::new("l_orderkey", 2)) as _,
        )];
        Ok(Arc::new(HashJoinExec::try_new(
            orders,
            lineitem,
            on,
            None,
            &JoinType::Inner,
            projection,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[test]
    fn swap_keeps_equivalent_hash_partitioning() -> Result<()> {
        for projection in [None, Some(vec![0, 1, 5]), Some(vec![1, 5])] {
            let node = partitioned_join(projection.clone())?;
            let join = node.downcast_ref::<HashJoinExec>().unwrap();
            assert!(
                swap(&node, join, false).is_some(),
                "projection={projection:?} old={}",
                node.properties().output_partitioning()
            );
        }
        Ok(())
    }

    #[test]
    fn swap_under_repartition_when_projection_drops_keys() -> Result<()> {
        // Like TPC-H Q8: the join's projection drops both join keys, so its
        // reported hash partitioning cannot be kept, but the parent
        // repartitions anyway.
        let node = partitioned_join(Some(vec![1, 4]))?;
        let join = node.downcast_ref::<HashJoinExec>().unwrap();
        assert!(swap(&node, join, false).is_none());

        let build = Arc::clone(join.left());
        let probe = Arc::clone(join.right());
        let key = Arc::new(Column::new("o_custkey", 0)) as _;
        let plan = Arc::new(RepartitionExec::try_new(
            node,
            Partitioning::Hash(vec![key], 4),
        )?) as Arc<dyn ExecutionPlan>;
        let swapped = swap_joins(Arc::clone(&plan), &[build], false)?;
        assert!(!Arc::ptr_eq(&plan, &swapped));
        assert_eq!(plan.schema(), swapped.schema());
        let mut new_build = None;
        swapped.apply(|n| {
            if let Some(join) = n.downcast_ref::<HashJoinExec>() {
                new_build = Some(Arc::clone(join.left()));
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert!(Arc::ptr_eq(&new_build.unwrap(), &probe));
        Ok(())
    }
}
