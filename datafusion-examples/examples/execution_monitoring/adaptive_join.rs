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

//! Dynamic build-side selection for hash joins using conditional stage boundaries.
//!
//! Both legs of a join are wrapped in a [`ThresholdBoundaryExec`] which drains
//! its input until either EOF or a byte threshold is reached, and then reports
//! itself ready for inspection. A driver discovers the boundaries through
//! [`ExecutionPlan::as_boundary`], primes them, and once both are ready checks
//! whether either leg completed under the threshold. If only the probe side
//! did, the join is flipped so that the small leg becomes the build side.
//!
//! On release, a boundary emits what it buffered and then streams the rest of
//! its input, so the large leg is never fully materialized by the boundary.
//!
//! This is a demo: no memory accounting, and boundaries are only inserted
//! directly below a [`HashJoinExec`].

use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::instant::Instant;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{
    DataFusionError, Result, assert_eq_or_internal_err, exec_datafusion_err, internal_err,
};
use datafusion::datasource::MemTable;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::EvaluationType;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream, StageBoundary, StageProgress,
    collect, displayable,
};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::{StreamExt, TryStreamExt, stream};
use tokio::sync::watch;

const FACT_ROWS: i64 = 10_000_000;
const DIM_ROWS: i64 = 1_000_000;
const BATCH_ROWS: i64 = 8_192;
const THRESHOLD_BYTES: usize = 1024 * 1024;

/// What a partition drain produced: buffered batches, plus the unread
/// remainder of the input if the drain stopped at the threshold.
struct Buffered {
    batches: Vec<RecordBatch>,
    rest: Option<SendableRecordBatchStream>,
    progress: StageProgress,
}

impl fmt::Debug for Buffered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffered")
            .field("progress", &self.progress)
            .finish()
    }
}

#[derive(Debug)]
struct ThresholdPartition {
    buffered: Arc<Mutex<Option<Result<Buffered>>>>,
    ready: watch::Sender<bool>,
    task: Mutex<Option<SpawnedTask<()>>>,
}

/// Boundary that buffers each partition until EOF or `threshold_bytes`,
/// whichever comes first, and is ready for inspection at that point.
#[derive(Debug)]
struct ThresholdBoundaryExec {
    input: Arc<dyn ExecutionPlan>,
    properties: Arc<PlanProperties>,
    threshold_bytes: usize,
    partitions: Vec<ThresholdPartition>,
    released: watch::Sender<bool>,
}

impl ThresholdBoundaryExec {
    fn new(input: Arc<dyn ExecutionPlan>, threshold_bytes: usize) -> Self {
        let partition_count = input.properties().output_partitioning().partition_count();
        let properties = PlanProperties::clone(input.properties())
            .with_evaluation_type(EvaluationType::Eager);
        let partitions = (0..partition_count)
            .map(|_| ThresholdPartition {
                buffered: Arc::new(Mutex::new(None)),
                ready: watch::channel(false).0,
                task: Mutex::new(None),
            })
            .collect();
        Self {
            input,
            properties: Arc::new(properties),
            threshold_bytes,
            partitions,
            released: watch::channel(false).0,
        }
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
        let mut input = self.input.execute(partition, context)?;
        let threshold = self.threshold_bytes;
        let buffered = Arc::clone(&state.buffered);
        let ready = state.ready.clone();
        *task = Some(SpawnedTask::spawn(async move {
            let mut batches = Vec::new();
            let (mut rows, mut bytes) = (0, 0);
            let complete = loop {
                if bytes >= threshold {
                    break Ok(false);
                }
                match input.next().await {
                    Some(Ok(batch)) => {
                        rows += batch.num_rows();
                        bytes += batch.get_array_memory_size();
                        batches.push(batch);
                    }
                    Some(Err(e)) => break Err(e),
                    None => break Ok(true),
                }
            };
            *buffered.lock().unwrap() = Some(complete.map(|complete| Buffered {
                batches,
                rest: (!complete).then_some(input),
                progress: StageProgress {
                    rows,
                    bytes,
                    complete,
                },
            }));
            ready.send_replace(true);
        }));
        Ok(())
    }

    fn is_ready(&self, partition: usize) -> bool {
        self.partitions
            .get(partition)
            .is_some_and(|state| *state.ready.borrow())
    }

    fn progress(&self, partition: usize) -> Option<StageProgress> {
        let state = self.partitions.get(partition)?;
        let buffered = state.buffered.lock().unwrap();
        buffered.as_ref()?.as_ref().ok().map(|b| b.progress)
    }

    fn release(&self) {
        self.released.send_replace(true);
    }
}

impl DisplayAs for ThresholdBoundaryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ThresholdBoundaryExec: threshold_bytes={}",
            self.threshold_bytes
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
            self.threshold_bytes,
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
        // Executing an unprimed boundary primes it, so it degrades to a
        // pass-through with a small buffer once released.
        self.prime(partition, context)?;
        let state = &self.partitions[partition];
        let buffered = Arc::clone(&state.buffered);
        let mut ready = state.ready.subscribe();
        let mut released = self.released.subscribe();
        let output = async move {
            released
                .wait_for(|released| *released)
                .await
                .map_err(|e| exec_datafusion_err!("boundary dropped: {e}"))?;
            ready
                .wait_for(|ready| *ready)
                .await
                .map_err(|e| exec_datafusion_err!("boundary dropped: {e}"))?;
            let Buffered { batches, rest, .. } = buffered
                .lock()
                .unwrap()
                .take()
                .ok_or_else(|| exec_datafusion_err!("partition executed twice"))??;
            let head = stream::iter(batches.into_iter().map(Ok));
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

/// Wraps both inputs of every hash join in a [`ThresholdBoundaryExec`].
fn insert_boundaries(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_up(|node| {
        if node.downcast_ref::<HashJoinExec>().is_none() {
            return Ok(Transformed::no(node));
        }
        let children = node
            .children()
            .into_iter()
            .map(|child| {
                Arc::new(ThresholdBoundaryExec::new(
                    Arc::clone(child),
                    THRESHOLD_BYTES,
                )) as Arc<dyn ExecutionPlan>
            })
            .collect();
        Ok(Transformed::yes(node.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?))
    })
    .map(|t| t.data)
}

fn for_each_boundary(
    plan: &Arc<dyn ExecutionPlan>,
    mut f: impl FnMut(&dyn StageBoundary) -> Result<()>,
) -> Result<()> {
    plan.apply(|node| {
        if let Some(boundary) = node.as_boundary() {
            f(boundary)?;
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

fn partition_count(boundary: &dyn StageBoundary) -> usize {
    boundary
        .properties()
        .output_partitioning()
        .partition_count()
}

/// Sums progress across partitions; complete only if every partition is.
fn total_progress(boundary: &dyn StageBoundary) -> Option<StageProgress> {
    (0..partition_count(boundary)).try_fold(
        StageProgress {
            rows: 0,
            bytes: 0,
            complete: true,
        },
        |acc, partition| {
            let p = boundary.progress(partition)?;
            Some(StageProgress {
                rows: acc.rows + p.rows,
                bytes: acc.bytes + p.bytes,
                complete: acc.complete && p.complete,
            })
        },
    )
}

/// Primes every boundary and waits until all are ready for inspection.
async fn prime_all(plan: &Arc<dyn ExecutionPlan>, ctx: &Arc<TaskContext>) -> Result<()> {
    for_each_boundary(plan, |boundary| {
        for partition in 0..partition_count(boundary) {
            boundary.prime(partition, Arc::clone(ctx))?;
        }
        Ok(())
    })?;
    loop {
        let mut all_ready = true;
        for_each_boundary(plan, |boundary| {
            all_ready &= (0..partition_count(boundary)).all(|p| boundary.is_ready(p));
            Ok(())
        })?;
        if all_ready {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

/// Flips any hash join whose probe side completed under the threshold while
/// its build side did not (or is larger).
fn choose_build_sides(plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_up(|node| {
        let Some(join) = node.downcast_ref::<HashJoinExec>() else {
            return Ok(Transformed::no(node));
        };
        let progress =
            |side: &Arc<dyn ExecutionPlan>| side.as_boundary().and_then(total_progress);
        let (Some(build), Some(probe)) = (progress(join.left()), progress(join.right()))
        else {
            return Ok(Transformed::no(node));
        };
        println!("  build side: {build:?}");
        println!("  probe side: {probe:?}");
        let swap = probe.complete && (!build.complete || probe.bytes < build.bytes);
        if !swap || !join.join_type().supports_swap() {
            println!("  keeping build side");
            return Ok(Transformed::no(node));
        }
        println!("  probe side is smaller and complete: swapping build side");
        Ok(Transformed::yes(join.swap_inputs(*join.partition_mode())?))
    })
    .map(|t| t.data)
}

fn release_all(plan: &Arc<dyn ExecutionPlan>) -> Result<()> {
    for_each_boundary(plan, |boundary| {
        boundary.release();
        Ok(())
    })
}

async fn register_tables(ctx: &SessionContext) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let table = |rows: i64, modulo: i64| -> Result<MemTable> {
        let batches = (0..rows)
            .step_by(BATCH_ROWS as usize)
            .map(|start| {
                let range = start..(start + BATCH_ROWS).min(rows);
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from_iter_values(
                            range.clone().map(|i| i % modulo),
                        )),
                        Arc::new(Int64Array::from_iter_values(range)),
                    ],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        MemTable::try_new(Arc::clone(&schema), vec![batches])
    };
    ctx.register_table("fact", Arc::new(table(FACT_ROWS, DIM_ROWS)?))?;
    ctx.register_table("dim", Arc::new(table(DIM_ROWS, DIM_ROWS)?))?;
    Ok(())
}

/// Runs a join whose static plan puts the larger input on the build side
/// because of a filter selectivity misestimate, and fixes it at runtime by
/// inspecting both legs.
pub async fn adaptive_join() -> Result<()> {
    // Dynamic join filters are disabled because they are wired up for the
    // original build side, and `swap_inputs` refuses to swap once they exist.
    let config = SessionConfig::new().with_target_partitions(1).set_bool(
        "datafusion.optimizer.enable_join_dynamic_filter_pushdown",
        false,
    );
    let ctx = SessionContext::new_with_config(config);
    register_tables(&ctx).await?;
    // The planner cannot analyze `f.v % 1000 = 0`, so it assumes the default
    // 20% selectivity: 2M estimated rows from `fact` vs 1M from `dim`, so `dim`
    // becomes the build side. In reality the filter keeps 0.1% (10K rows).
    let sql = "SELECT count(*), sum(f.v + d.v) FROM fact f JOIN dim d ON f.k = d.k \
               WHERE f.v % 1000 = 0";
    let task_ctx = ctx.task_ctx();

    let static_plan = ctx.sql(sql).await?.create_physical_plan().await?;
    println!(
        "Static plan:\n{}",
        displayable(static_plan.as_ref()).indent(true)
    );
    let start = Instant::now();
    let expected = collect(static_plan, Arc::clone(&task_ctx)).await?;
    println!("Static plan finished in {:?}\n", start.elapsed());

    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let start = Instant::now();
    let plan = insert_boundaries(plan)?;
    prime_all(&plan, &task_ctx).await?;
    println!("Boundaries ready after {:?}", start.elapsed());
    let plan = choose_build_sides(plan)?;
    release_all(&plan)?;
    let actual = collect(Arc::clone(&plan), task_ctx).await?;
    println!("Adaptive plan finished in {:?}", start.elapsed());
    println!(
        "Adaptive plan:\n{}",
        displayable(plan.as_ref()).indent(true)
    );

    assert_eq!(expected, actual);
    println!(
        "Results match:\n{}",
        arrow::util::pretty::pretty_format_batches(&actual)?
    );
    Ok(())
}
