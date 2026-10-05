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

//! Variant runs: executions of a rewritten copy of a node, or of a copy run
//! under different settings, whose complete output is collected so that
//! checks can compare it with the node's normal output.
//!
//! Unlike stream [`Experiment`]s, variant runs do not observe how a node
//! drives its streams. Each one executes a fresh copy of the node's subtree,
//! made with [`reset_plan_states`], to completion on finite inputs, within
//! the checker's execution timeout, like the normal execution.
//!
//! [`Experiment`]: crate::Experiment

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::{
    replace_children_if_necessary, reset_plan_states,
};
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};

use crate::exec::{self, collect_output};
use crate::fixtures::{BatchLayout, map_mock_leaves};
use crate::{CheckContext, NodeOutput};

/// Fetches and input limits tried for every node, besides one larger than the
/// node's output and inputs. A limit of 0 is not tried: `LIMIT 0` is replaced by
/// an empty relation during logical optimization, so plans never see a fetch
/// of 0.
const SMALL_LIMITS: [usize; 2] = [1, 7];

/// Batch sizes tried by [`Variant::BatchSize`]
const BATCH_SIZES: [usize; 4] = [1, 2, 7, 8192];

/// Batch layouts tried by [`Variant::BatchLayout`]
const LEAF_LAYOUTS: [BatchLayout; 3] = [
    BatchLayout::Fixed(1),
    BatchLayout::Random { max_rows: 3 },
    BatchLayout::Single,
];

/// Seed for random leaf layouts. Every node uses the same seed, so a node and
/// its children see the same batches in their leaves.
const LAYOUT_SEED: u64 = 42;

/// One variant run of a node. The [`PlanChecker`] runs every variant that
/// applies to each node when a [`CheckKind::Variant`] check is enabled, and
/// checks read the results with [`CheckContext::variant_runs`].
///
/// [`PlanChecker`]: crate::PlanChecker
/// [`CheckKind::Variant`]: crate::CheckKind::Variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The plan returned by `with_fetch(None)`, for nodes where it returns a
    /// plan. Its output is the **unfetched output** of a node with a fetch.
    WithoutFetch,
    /// The plan returned by `with_fetch(Some(n))`, for `n` of 1, 7 and one
    /// more than the rows of the node's output (with and without its fetch)
    /// and of each of its inputs
    WithFetch(usize),
    /// The node rebuilt with every child limited to its first `n` rows per
    /// partition, for the same values as [`Self::WithFetch`], if the node has
    /// children. A child with one partition is limited with a
    /// `GlobalLimitExec`, and any other child with a `LocalLimitExec`, as the
    /// `LimitPushdown` optimizer rule limits them when
    /// `supports_limit_pushdown()` is true.
    LimitedInputs(usize),
    /// The node executed with the session batch size set to 1, 2, 7 and 8192
    BatchSize(usize),
    /// The node rebuilt with the rows of each partition of its
    /// [`MockSourceExec`] leaves split into one row per batch, random batches
    /// of up to 3 rows, and one batch per partition. Every leaf keeps the same
    /// rows in the same order and partitions, and the same `PlanProperties`.
    /// Only for nodes with a `MockSourceExec` leaf.
    ///
    /// [`MockSourceExec`]: crate::fixtures::MockSourceExec
    BatchLayout(BatchLayout),
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Variant::WithoutFetch => write!(f, "with_fetch(None)"),
            Variant::WithFetch(n) => write!(f, "with_fetch(Some({n}))"),
            Variant::LimitedInputs(n) => {
                write!(f, "every child limited to its first {n} rows per partition")
            }
            Variant::BatchSize(n) => write!(f, "batch size {n}"),
            Variant::BatchLayout(layout) => write!(f, "input rows in {layout}"),
        }
    }
}

/// The result of one variant run of a node
#[derive(Debug, Clone)]
pub struct VariantRun {
    /// What was run
    pub variant: Variant,
    /// The output of every partition, or why building or executing the
    /// variant failed, panicked or timed out
    pub output: std::result::Result<NodeOutput, String>,
}

/// Run every variant that applies to `node`, whose normal output and the
/// normal outputs of its children are in `context`
pub(crate) async fn run(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    timeout: Duration,
) -> Vec<VariantRun> {
    // The output without the fetch sizes the other variants, so it runs first
    let mut runs: Vec<VariantRun> = run_variant(node, Variant::WithoutFetch, timeout)
        .await
        .into_iter()
        .collect();

    // A limit larger than the output of the node, with and without its fetch,
    // and of each of its children, which removes nothing
    let unfetched = runs.first().and_then(|run| run.output.as_ref().ok());
    let larger = std::iter::once(node)
        .chain(node.children())
        .filter_map(|plan| context.output(plan))
        .chain(unfetched)
        .map(NodeOutput::num_rows)
        .max()
        .unwrap_or(0)
        + 1;
    let mut limits = SMALL_LIMITS.to_vec();
    if !limits.contains(&larger) {
        limits.push(larger);
    }

    let mut variants: Vec<Variant> =
        limits.iter().map(|n| Variant::WithFetch(*n)).collect();
    if !node.children().is_empty() {
        variants.extend(limits.iter().map(|n| Variant::LimitedInputs(*n)));
    }
    variants.extend(BATCH_SIZES.map(Variant::BatchSize));
    variants.extend(LEAF_LAYOUTS.map(Variant::BatchLayout));
    for variant in variants {
        runs.extend(run_variant(node, variant, timeout).await);
    }
    runs
}

/// Run `variant` on `node`, or return `None` if it does not apply to the node
async fn run_variant(
    node: &Arc<dyn ExecutionPlan>,
    variant: Variant,
    timeout: Duration,
) -> Option<VariantRun> {
    let output = match build(node, variant) {
        Ok(None) => return None,
        Ok(Some(plan)) => {
            let task_context = match variant {
                Variant::BatchSize(batch_size) => exec::task_context(batch_size),
                _ => Arc::new(TaskContext::default()),
            };
            collect_output(plan, task_context, timeout).await
        }
        Err(e) => Err(format!("building the plan failed: {}", e.strip_backtrace())),
    };
    Some(VariantRun { variant, output })
}

/// Build the plan that `variant` executes from a fresh copy of `node`, or
/// `None` if the variant does not apply to the node
fn build(
    node: &Arc<dyn ExecutionPlan>,
    variant: Variant,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let node = reset_plan_states(Arc::clone(node))?;
    Ok(match variant {
        Variant::WithoutFetch => node.with_fetch(None),
        Variant::WithFetch(n) => node.with_fetch(Some(n)),
        Variant::LimitedInputs(n) => {
            let children = node
                .children()
                .into_iter()
                .map(|child| {
                    let partitions =
                        child.properties().output_partitioning().partition_count();
                    let child = Arc::clone(child);
                    if partitions == 1 {
                        Arc::new(GlobalLimitExec::new(child, 0, Some(n)))
                            as Arc<dyn ExecutionPlan>
                    } else {
                        Arc::new(LocalLimitExec::new(child, n))
                    }
                })
                .collect();
            Some(replace_children_if_necessary(node, children)?)
        }
        Variant::BatchSize(_) => Some(node),
        Variant::BatchLayout(layout) => {
            let mut changed = false;
            let plan = map_mock_leaves(&node, |source| {
                changed = true;
                Ok(Some(Arc::new(
                    source.with_batch_layout(layout, LAYOUT_SEED)?,
                )))
            })?;
            changed.then_some(plan)
        }
    })
}
