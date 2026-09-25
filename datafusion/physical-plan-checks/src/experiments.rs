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

//! Stream experiments: runs of a single node that observe how it drives its
//! input streams, on inputs whose streams stall, fail or never end.
//!
//! Each experiment rebuilds the node from a fresh copy of its subtree made
//! with [`reset_plan_states`]:
//!
//! - The [`MockSourceExec`] leaves below some or all children are replaced by
//!   copies with the same data and a different [`StreamBehavior`].
//! - Every child is wrapped in a [`ProbeExec`], a pass-through node with the
//!   child's own properties that records what the node does with the child's
//!   streams. Observing the node's direct inputs, rather than the leaves,
//!   attributes what is observed to the node itself.
//! - The node is rebuilt on the wrapped children with
//!   [`replace_children_if_necessary`]. When the leaves keep their properties,
//!   the rebuilt node keeps the original node's properties too.
//!
//! The node's output streams are observed as well, and every run has a
//! timeout.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;
use std::time::Duration;

use datafusion_common::instant::Instant;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, Statistics, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation};
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::execution_plan::{
    Boundedness, CardinalityEffect, EmissionType, EvaluationType,
    replace_children_if_necessary, reset_plan_states,
};
use datafusion_physical_plan::{
    ChildStats, ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};
use futures::{FutureExt, StreamExt};

use crate::context::panic_message;
use crate::fixtures::{
    MockSourceExec, PartitionObservation, StreamBehavior, StreamProbe,
};

/// Batch size used by experiments, so that operators that buffer up to a
/// batch before emitting it do so after a few input rows
pub(crate) const EXPERIMENT_BATCH_SIZE: usize = 8;

/// Rows each partition of an unbounded input produces before it stays
/// pending. Enough to fill the buffers of operators that coalesce batches to
/// the default batch size, while bounding the memory an operator that
/// buffers all input can use.
pub(crate) const UNBOUNDED_MAX_ROWS: usize = 1 << 16;

/// How long the cancellation experiment polls the output before dropping it
const DRIVE_BEFORE_DROP: Duration = Duration::from_millis(20);

/// A stream experiment the [`PlanChecker`] can run on each node that has
/// children. Checks request experiments with [`PlanCheck::experiments`], and
/// read the results with [`CheckContext::stream_runs`].
///
/// [`PlanChecker`]: crate::PlanChecker
/// [`PlanCheck::experiments`]: crate::PlanCheck::experiments
/// [`CheckContext::stream_runs`]: crate::CheckContext::stream_runs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Experiment {
    /// Execute every output partition on the normal inputs and poll the output
    /// to the end, counting input polls that are not driven by an output poll
    /// ([`PartitionObservation::polls_without_demand`]). One run.
    Laziness,
    /// For each child in turn, make the [`MockSourceExec`] leaves with rows
    /// below it [`StreamBehavior::Unbounded`], with the other children
    /// unchanged. The output is polled until it ends if the rebuilt node
    /// reports `Bounded`, or has a fetch and is not `Final`; otherwise until
    /// it produces a row if the rebuilt node reports `Incremental` and the
    /// child is not `Final`. The run is skipped when neither applies. One run
    /// per child, with [`StreamRun::varied_child`] set.
    UnboundedInput,
    /// Make every [`MockSourceExec`] leaf return an error after its first
    /// batch ([`StreamBehavior::ErrorAfter`]), and poll every output partition
    /// until it ends or returns an error. One run.
    InputError,
    /// Make every [`MockSourceExec`] leaf stall after its first batch
    /// ([`StreamBehavior::PendingAfter`]), poll the output briefly, then drop
    /// the output streams and wait for the input streams to be dropped. If
    /// some are still alive, drop the rebuilt node too and wait again (see
    /// [`StreamRun::inputs_after_plan_dropped`]). One run.
    Cancellation,
}

/// How a [`StreamRun`] ended
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcome {
    /// Every output partition ended, with `None` or an error
    Ended,
    /// The experiment stopped polling once it had observed what it needed,
    /// such as the first output row of an unbounded node
    Stopped,
    /// The output did not end, or the stop condition was not met, in time: the
    /// checker's execution timeout for runs on finite inputs, and its stream
    /// timeout for runs on inputs that never end
    TimedOut,
    /// Executing or polling the node panicked
    Panicked(String),
    /// The experiment could not be set up, or `execute` returned an error
    Failed(String),
}

/// The result of one run of a stream [`Experiment`] on a node
#[derive(Debug, Clone)]
pub struct StreamRun {
    /// For [`Experiment::UnboundedInput`], the child whose inputs were made
    /// unbounded
    pub varied_child: Option<usize>,
    /// The boundedness the rebuilt node reports
    pub boundedness: Boundedness,
    /// The emission type the rebuilt node reports
    pub emission_type: EmissionType,
    /// The evaluation type the rebuilt node reports
    pub evaluation_type: EvaluationType,
    /// The fetch of the rebuilt node
    pub fetch: Option<usize>,
    /// The boundedness each child of the rebuilt node reports
    pub input_boundedness: Vec<Boundedness>,
    /// Observations of each output partition of the node
    pub output: Vec<PartitionObservation>,
    /// Observations of the streams the node created for each child, one
    /// entry per child partition, at the end of the run. For
    /// [`Experiment::Cancellation`], after the output streams were dropped.
    pub inputs: Vec<Vec<PartitionObservation>>,
    /// For [`Experiment::Cancellation`], when some input streams were still
    /// alive after the output streams were dropped: observations of the input
    /// streams after the rebuilt node was dropped as well
    pub inputs_after_plan_dropped: Option<Vec<Vec<PartitionObservation>>>,
    /// For [`Experiment::UnboundedInput`], true if a partition of an
    /// unbounded leaf served its `max_rows` rows and then stalled. The run
    /// then does not show that the node would never end or produce output:
    /// it may need more rows than the leaf was allowed to serve, for example
    /// a limit with a large fetch or offset.
    pub inputs_exhausted: bool,
    /// How the run ended
    pub outcome: RunOutcome,
}

impl StreamRun {
    /// Total number of rows the node produced, over all output partitions
    pub fn output_rows(&self) -> usize {
        self.output.iter().map(|o| o.rows).sum()
    }

    /// Total number of rows that child `child` delivered to the node
    pub fn input_rows(&self, child: usize) -> usize {
        self.inputs
            .get(child)
            .map(|partitions| partitions.iter().map(|o| o.rows).sum())
            .unwrap_or(0)
    }

    /// Total number of errors that the node's inputs delivered to it
    pub fn input_errors(&self) -> usize {
        self.inputs.iter().flatten().map(|o| o.errors).sum()
    }

    /// Total number of errors the node returned
    pub fn output_errors(&self) -> usize {
        self.output.iter().map(|o| o.errors).sum()
    }

    fn failed(node: &Arc<dyn ExecutionPlan>, message: String) -> Self {
        Self {
            varied_child: None,
            boundedness: node.properties().boundedness,
            emission_type: node.properties().emission_type,
            evaluation_type: node.properties().evaluation_type,
            fetch: node.fetch(),
            input_boundedness: vec![],
            output: vec![],
            inputs: vec![],
            inputs_after_plan_dropped: None,
            inputs_exhausted: false,
            outcome: RunOutcome::Failed(message),
        }
    }
}

/// Settings shared by all experiments
#[derive(Debug, Clone)]
pub(crate) struct ExperimentOptions {
    /// Context used to execute the rebuilt nodes
    pub task_context: Arc<TaskContext>,
    /// Time allowed for a run on finite inputs, which should end like a normal
    /// execution
    pub execution_timeout: Duration,
    /// Time allowed for a run on inputs that never end, and for streams to be
    /// dropped
    pub stream_timeout: Duration,
}

/// Run `experiment` on `node`. Returns no runs for leaves, and for runs that
/// do not apply to the node.
pub(crate) async fn run(
    node: &Arc<dyn ExecutionPlan>,
    experiment: Experiment,
    options: &ExperimentOptions,
) -> Vec<StreamRun> {
    let children = node.children().len();
    if children == 0 {
        return vec![];
    }
    match experiment {
        Experiment::Laziness => run_with(
            node,
            None,
            |_| None,
            |_| Some(StopWhen::Ended),
            options.execution_timeout,
            &options.task_context,
        )
        .await
        .into_iter()
        .collect(),
        Experiment::InputError => run_with(
            node,
            None,
            |_| Some(StreamBehavior::ErrorAfter(1)),
            |_| Some(StopWhen::Ended),
            options.execution_timeout,
            &options.task_context,
        )
        .await
        .into_iter()
        .collect(),
        Experiment::UnboundedInput => {
            let mut runs = vec![];
            for child in 0..children {
                let behavior = |i| {
                    (i == child).then_some(StreamBehavior::Unbounded {
                        max_rows: Some(UNBOUNDED_MAX_ROWS),
                    })
                };
                let stop = |plan: &Arc<dyn ExecutionPlan>| {
                    let properties = plan.properties();
                    let needs_end = properties.boundedness == Boundedness::Bounded
                        || (plan.fetch().is_some()
                            && properties.emission_type != EmissionType::Final);
                    let child_emits = plan.children()[child].properties().emission_type
                        != EmissionType::Final;
                    let needs_rows = properties.emission_type
                        == EmissionType::Incremental
                        && child_emits;
                    if needs_end {
                        Some(StopWhen::Ended)
                    } else if needs_rows {
                        Some(StopWhen::FirstRows)
                    } else {
                        None
                    }
                };
                let run = run_with(
                    node,
                    Some(child),
                    behavior,
                    stop,
                    options.stream_timeout,
                    &options.task_context,
                )
                .await;
                if let Some(run) = run {
                    runs.push(run);
                }
            }
            runs
        }
        Experiment::Cancellation => {
            cancellation(node, options).await.into_iter().collect()
        }
    }
}

/// When a run stops polling the output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StopWhen {
    /// When every output partition has ended
    Ended,
    /// When any output partition produces a batch with rows
    FirstRows,
    /// When every output partition has produced a batch, or ended
    FirstItemEach,
}

/// A node rebuilt for an experiment
struct Instrumented {
    plan: Arc<dyn ExecutionPlan>,
    /// Observes the node's output streams
    output: StreamProbe,
    /// Observes the streams the node creates for each child
    inputs: Vec<StreamProbe>,
    /// Observes each leaf whose behavior was changed to an unbounded one with
    /// a row limit, with that limit
    capped_leaves: Vec<(StreamProbe, usize)>,
}

impl Instrumented {
    /// Rebuild `node` with the leaves below child `i` set to `behavior(i)`, if
    /// any, and every child wrapped in a [`ProbeExec`]. The second value is
    /// false if a behavior was requested but no leaf could be changed.
    fn new(
        node: &Arc<dyn ExecutionPlan>,
        behavior: impl Fn(usize) -> Option<StreamBehavior>,
    ) -> Result<(Self, bool)> {
        let node = reset_plan_states(Arc::clone(node))?;
        let output = StreamProbe::new();
        let mut children: Vec<Arc<dyn ExecutionPlan>> = vec![];
        let mut inputs = vec![];
        let mut capped_leaves = vec![];
        let mut requested = false;
        let mut changed = false;
        for (i, child) in node.children().into_iter().enumerate() {
            let child = match behavior(i) {
                Some(behavior) => {
                    requested = true;
                    let (child, leaves) = with_leaf_behavior(child, behavior)?;
                    changed |= !leaves.is_empty();
                    if let StreamBehavior::Unbounded {
                        max_rows: Some(max_rows),
                    } = behavior
                    {
                        capped_leaves
                            .extend(leaves.into_iter().map(|probe| (probe, max_rows)));
                    }
                    child
                }
                None => Arc::clone(child),
            };
            let probe = StreamProbe::with_consumer(&output);
            children.push(Arc::new(ProbeExec::new(child, probe.clone())));
            inputs.push(probe);
        }
        let plan = replace_children_if_necessary(node, children)?;
        Ok((
            Self {
                plan,
                output,
                inputs,
                capped_leaves,
            },
            !requested || changed,
        ))
    }

    fn output_partitions(&self) -> usize {
        self.plan
            .properties()
            .output_partitioning()
            .partition_count()
    }

    /// Execute every output partition, observed by the output probe
    fn execute(
        &self,
        task_context: &Arc<TaskContext>,
    ) -> Result<Vec<Option<SendableRecordBatchStream>>> {
        (0..self.output_partitions())
            .map(|p| {
                let stream = self.plan.execute(p, Arc::clone(task_context))?;
                Ok(Some(self.output.observe(p, stream)))
            })
            .collect()
    }

    fn output_observations(&self) -> Vec<PartitionObservation> {
        self.output.partitions_up_to(self.output_partitions())
    }

    fn input_observations(&self) -> Vec<Vec<PartitionObservation>> {
        input_observations(&self.plan, &self.inputs)
    }

    fn run(&self, varied_child: Option<usize>, outcome: RunOutcome) -> StreamRun {
        let properties = self.plan.properties();
        StreamRun {
            varied_child,
            boundedness: properties.boundedness,
            emission_type: properties.emission_type,
            evaluation_type: properties.evaluation_type,
            fetch: self.plan.fetch(),
            input_boundedness: self
                .plan
                .children()
                .iter()
                .map(|child| child.properties().boundedness)
                .collect(),
            output: self.output_observations(),
            inputs: self.input_observations(),
            inputs_after_plan_dropped: None,
            inputs_exhausted: self.inputs_exhausted(),
            outcome,
        }
    }

    /// Returns true if a partition of a capped leaf served all the rows it
    /// was allowed to
    fn inputs_exhausted(&self) -> bool {
        self.capped_leaves.iter().any(|(probe, max_rows)| {
            probe
                .partitions()
                .iter()
                .any(|observation| observation.rows >= *max_rows)
        })
    }
}

fn input_observations(
    plan: &Arc<dyn ExecutionPlan>,
    inputs: &[StreamProbe],
) -> Vec<Vec<PartitionObservation>> {
    plan.children()
        .iter()
        .zip(inputs)
        .map(|(child, probe)| {
            let partitions = child.properties().output_partitioning().partition_count();
            probe.partitions_up_to(partitions)
        })
        .collect()
}

/// Replace each [`MockSourceExec`] leaf of `plan`, including `plan` itself,
/// by the source `replace` returns for it, if any, and rebuild the nodes above
/// the replaced leaves with `with_new_children`. A replacement that reports
/// the same properties as the original lets the rebuilt nodes keep theirs.
pub(crate) fn map_mock_leaves(
    plan: &Arc<dyn ExecutionPlan>,
    mut replace: impl FnMut(&MockSourceExec) -> Result<Option<MockSourceExec>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::clone(plan)
        .transform_up(|plan| {
            let Some(source) = plan.downcast_ref::<MockSourceExec>() else {
                return Ok(Transformed::no(plan));
            };
            Ok(match replace(source)? {
                Some(source) => Transformed::yes(Arc::new(source) as _),
                None => Transformed::no(plan),
            })
        })?
        .data)
}

/// Replace the [`MockSourceExec`] leaves of `plan` by copies with `behavior`.
/// Leaves without rows are left unchanged for an unbounded behavior. Returns
/// a probe observing each replaced leaf.
fn with_leaf_behavior(
    plan: &Arc<dyn ExecutionPlan>,
    behavior: StreamBehavior,
) -> Result<(Arc<dyn ExecutionPlan>, Vec<StreamProbe>)> {
    let mut probes = vec![];
    let plan = map_mock_leaves(plan, |source| {
        if matches!(behavior, StreamBehavior::Unbounded { .. }) && !source.has_rows() {
            return Ok(None);
        }
        let probe = StreamProbe::new();
        let source = source
            .clone()
            .try_with_stream_behavior(behavior)?
            .with_probe(probe.clone());
        probes.push(probe);
        Ok(Some(source))
    })?;
    Ok((plan, probes))
}

/// Rebuild `node` with `behavior`, execute it and poll its output until
/// `stop(rebuilt node)` is met. Returns `None` if a behavior was requested but
/// no leaf could be changed, or if `stop` returns `None`.
async fn run_with(
    node: &Arc<dyn ExecutionPlan>,
    varied_child: Option<usize>,
    behavior: impl Fn(usize) -> Option<StreamBehavior>,
    stop: impl FnOnce(&Arc<dyn ExecutionPlan>) -> Option<StopWhen>,
    timeout: Duration,
    task_context: &Arc<TaskContext>,
) -> Option<StreamRun> {
    let instrumented = match Instrumented::new(node, behavior) {
        Ok((instrumented, true)) => instrumented,
        Ok((_, false)) => return None,
        Err(e) => {
            let mut run = StreamRun::failed(
                node,
                format!("rebuilding the node failed: {}", e.strip_backtrace()),
            );
            run.varied_child = varied_child;
            return Some(run);
        }
    };
    let stop = stop(&instrumented.plan)?;
    let outcome = match execute_all(&instrumented, task_context) {
        Ok(mut streams) => drive(&mut streams, stop, timeout).await,
        Err(outcome) => outcome,
    };
    Some(instrumented.run(varied_child, outcome))
}

async fn cancellation(
    node: &Arc<dyn ExecutionPlan>,
    options: &ExperimentOptions,
) -> Option<StreamRun> {
    let instrumented =
        match Instrumented::new(node, |_| Some(StreamBehavior::PendingAfter(1))) {
            Ok((instrumented, true)) => instrumented,
            Ok((_, false)) => return None,
            Err(e) => {
                return Some(StreamRun::failed(
                    node,
                    format!("rebuilding the node failed: {}", e.strip_backtrace()),
                ));
            }
        };
    let mut streams = match execute_all(&instrumented, &options.task_context) {
        Ok(streams) => streams,
        Err(outcome) => return Some(instrumented.run(None, outcome)),
    };
    let outcome =
        match drive(&mut streams, StopWhen::FirstItemEach, DRIVE_BEFORE_DROP).await {
            outcome @ (RunOutcome::Panicked(_) | RunOutcome::Failed(_)) => outcome,
            _ => RunOutcome::Stopped,
        };
    drop(streams);

    let alive = |probes: &[StreamProbe]| probes.iter().any(|p| p.streams_alive() > 0);
    wait_until(|| !alive(&instrumented.inputs), options.stream_timeout).await;
    let mut run = instrumented.run(None, outcome);
    if alive(&instrumented.inputs) {
        let Instrumented { plan, inputs, .. } = instrumented;
        let children: Vec<Arc<dyn ExecutionPlan>> =
            plan.children().into_iter().cloned().collect();
        drop(plan);
        wait_until(|| !alive(&inputs), options.stream_timeout).await;
        run.inputs_after_plan_dropped = Some(
            children
                .iter()
                .zip(&inputs)
                .map(|(child, probe)| {
                    probe.partitions_up_to(
                        child.properties().output_partitioning().partition_count(),
                    )
                })
                .collect(),
        );
    }
    Some(run)
}

/// Wait until `done` returns true, or `timeout` elapses
pub(crate) async fn wait_until(done: impl Fn() -> bool, timeout: Duration) {
    let start = Instant::now();
    while !done() && start.elapsed() < timeout {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// Execute every output partition, catching errors and panics
fn execute_all(
    instrumented: &Instrumented,
    task_context: &Arc<TaskContext>,
) -> std::result::Result<Vec<Option<SendableRecordBatchStream>>, RunOutcome> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| {
        instrumented.execute(task_context)
    })) {
        Ok(Ok(streams)) => Ok(streams),
        Ok(Err(e)) => Err(RunOutcome::Failed(format!(
            "execute failed: {}",
            e.strip_backtrace()
        ))),
        Err(panic) => Err(RunOutcome::Panicked(panic_message(&panic))),
    }
}

/// Poll every stream in `streams` until `stop` is met or `timeout` elapses.
/// Streams that end, with `None` or an error, are dropped and set to `None`.
async fn drive(
    streams: &mut [Option<SendableRecordBatchStream>],
    stop: StopWhen,
    timeout: Duration,
) -> RunOutcome {
    /// Items taken from one stream before giving other streams, and the
    /// runtime, a turn
    const ITEMS_PER_TURN: usize = 16;

    let mut produced = vec![false; streams.len()];
    let future = futures::future::poll_fn(|cx| {
        let mut yielded = false;
        for (slot, produced) in streams.iter_mut().zip(produced.iter_mut()) {
            for item in 0..=ITEMS_PER_TURN {
                let Some(stream) = slot.as_mut() else {
                    break;
                };
                if item == ITEMS_PER_TURN {
                    yielded = true;
                    break;
                }
                match stream.poll_next_unpin(cx) {
                    Poll::Ready(Some(Ok(batch))) => {
                        *produced = true;
                        if stop == StopWhen::FirstRows && batch.num_rows() > 0 {
                            return Poll::Ready(RunOutcome::Stopped);
                        }
                    }
                    Poll::Ready(Some(Err(_)) | None) => {
                        *produced = true;
                        *slot = None;
                    }
                    Poll::Pending => break,
                }
                if stop == StopWhen::FirstItemEach && *produced {
                    break;
                }
            }
        }
        if streams.iter().all(Option::is_none) {
            return Poll::Ready(RunOutcome::Ended);
        }
        if stop == StopWhen::FirstItemEach && produced.iter().all(|p| *p) {
            return Poll::Ready(RunOutcome::Stopped);
        }
        // Streams that returned `Pending` wake the task when they can make
        // progress. Streams that were still ready did not, so wake it to poll
        // them again after the runtime had a turn.
        if yielded {
            cx.waker().wake_by_ref();
        }
        Poll::Pending
    });
    match tokio::time::timeout(timeout, AssertUnwindSafe(future).catch_unwind()).await {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(panic)) => RunOutcome::Panicked(panic_message(&panic)),
        Err(_) => RunOutcome::TimedOut,
    }
}

/// A pass-through node that records, in a [`StreamProbe`], what its parent
/// does with the streams of its input. It reports its input's properties, so
/// the parent computes the same properties on top of it.
#[derive(Debug)]
struct ProbeExec {
    input: Arc<dyn ExecutionPlan>,
    probe: StreamProbe,
}

impl ProbeExec {
    fn new(input: Arc<dyn ExecutionPlan>, probe: StreamProbe) -> Self {
        Self { input, probe }
    }
}

impl DisplayAs for ProbeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ProbeExec")
    }
}

impl ExecutionPlan for ProbeExec {
    fn name(&self) -> &'static str {
        "ProbeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
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
        if children.len() != 1 {
            return internal_err!("ProbeExec has one child");
        }
        Ok(Arc::new(Self::new(
            children.swap_remove(0),
            self.probe.clone(),
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
        let stream = self.input.execute(partition, context)?;
        Ok(self.probe.observe(partition, stream))
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        Ok(Arc::clone(&input_stats[0]))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }
}

/// A [`MemoryPool`] that passes every request to another pool, and tracks
/// the memory reserved through it
#[derive(Debug)]
pub(crate) struct TrackingPool {
    inner: Arc<dyn MemoryPool>,
    reserved: AtomicUsize,
}

impl TrackingPool {
    pub(crate) fn new(inner: Arc<dyn MemoryPool>) -> Self {
        Self {
            inner,
            reserved: AtomicUsize::new(0),
        }
    }

    /// Bytes currently reserved through this pool
    pub(crate) fn tracked(&self) -> usize {
        self.reserved.load(Ordering::Relaxed)
    }
}

impl fmt::Display for TrackingPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrackingPool({})", self.inner)
    }
}

impl MemoryPool for TrackingPool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn register(&self, consumer: &datafusion_execution::memory_pool::MemoryConsumer) {
        self.inner.register(consumer)
    }

    fn unregister(&self, consumer: &datafusion_execution::memory_pool::MemoryConsumer) {
        self.inner.unregister(consumer)
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.reserved.fetch_add(additional, Ordering::Relaxed);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
        let _ = self.reserved.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |reserved| Some(reserved.saturating_sub(shrink)),
        );
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.reserved.fetch_add(additional, Ordering::Relaxed);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

/// A copy of `task_context` whose memory pool is `pool`
pub(crate) fn with_memory_pool(
    task_context: &TaskContext,
    pool: Arc<dyn MemoryPool>,
) -> Result<TaskContext> {
    let runtime = RuntimeEnvBuilder::from_runtime_env(&task_context.runtime_env())
        .with_memory_pool(pool)
        .build_arc()?;
    Ok(copy_task_context(task_context).with_runtime(runtime))
}

/// A copy of `task_context` whose batch size is at most
/// [`EXPERIMENT_BATCH_SIZE`]
pub(crate) fn experiment_task_context(task_context: &TaskContext) -> TaskContext {
    let batch_size = task_context
        .session_config()
        .batch_size()
        .min(EXPERIMENT_BATCH_SIZE);
    with_batch_size(task_context, batch_size)
}

/// A copy of `task_context` whose batch size is `batch_size`
pub(crate) fn with_batch_size(
    task_context: &TaskContext,
    batch_size: usize,
) -> TaskContext {
    let config = task_context
        .session_config()
        .clone()
        .with_batch_size(batch_size);
    copy_task_context(task_context).with_session_config(config)
}

fn copy_task_context(task_context: &TaskContext) -> TaskContext {
    TaskContext::new(
        task_context.task_id(),
        task_context.session_id(),
        task_context.session_config().clone(),
        task_context.scalar_functions().clone(),
        task_context.higher_order_functions().clone(),
        task_context.aggregate_functions().clone(),
        task_context.window_functions().clone(),
        task_context.runtime_env(),
    )
}
