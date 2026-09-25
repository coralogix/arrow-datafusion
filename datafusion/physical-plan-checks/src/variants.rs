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

use crate::context::{NodeOutput, collect_output};
use crate::experiments::{map_mock_leaves, with_batch_size};
use crate::fixtures::BatchLayout;

/// Fetches and input limits tried for every node, besides one larger than the
/// node's output and inputs. A limit of 0 is not tried: `LIMIT 0` is replaced by
/// an empty relation during logical optimization, so plans never see a fetch
/// of 0.
const SMALL_LIMITS: [usize; 2] = [1, 7];

/// Batch sizes tried by [`VariantKind::BatchSize`]
const BATCH_SIZES: [usize; 4] = [1, 2, 7, 8192];

/// Batch layouts tried by [`VariantKind::BatchLayout`]
const LEAF_LAYOUTS: [BatchLayout; 3] = [
    BatchLayout::Fixed(1),
    BatchLayout::Random {
        max_rows: 3,
        empty_batches: true,
    },
    BatchLayout::Single,
];

/// Seed for random leaf layouts. Every node uses the same seed, so a node and
/// its children see the same batches in their leaves.
const LAYOUT_SEED: u64 = 42;

/// A family of variant runs the [`PlanChecker`] can collect for each node.
/// Checks request them with [`PlanCheck::variants`], and read the results
/// with [`CheckContext::variant_runs`]. See [`Variant`] for what each run
/// executes.
///
/// [`PlanChecker`]: crate::PlanChecker
/// [`PlanCheck::variants`]: crate::PlanCheck::variants
/// [`CheckContext::variant_runs`]: crate::CheckContext::variant_runs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VariantKind {
    /// [`Variant::WithoutFetch`], for nodes where `with_fetch(None)` returns a
    /// plan. Collected first, so that the other kinds can size their runs by
    /// its output.
    WithoutFetch,
    /// [`Variant::WithFetch`] for fetches of 1, 7 and one more than the
    /// rows of the node's output and of each of its inputs, for nodes where
    /// `with_fetch` returns a plan
    WithFetch,
    /// [`Variant::LimitedInputs`] for the same values as
    /// [`Self::WithFetch`], for nodes with children for which
    /// `supports_limit_pushdown()` is true
    LimitedInputs,
    /// [`Variant::BatchSize`] with batch sizes 1, 2, 7 and 8192, for every
    /// node
    BatchSize,
    /// [`Variant::BatchLayout`] with one row per batch, random batches of up
    /// to 3 rows and empty batches, and one batch per partition, for nodes
    /// with at least one [`MockSourceExec`] leaf
    ///
    /// [`MockSourceExec`]: crate::fixtures::MockSourceExec
    BatchLayout,
}

/// One variant run of a node
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The plan returned by `with_fetch(None)`
    WithoutFetch,
    /// The plan returned by `with_fetch(Some(n))`
    WithFetch(usize),
    /// The node rebuilt with every child limited to its first `n` rows per
    /// partition: a `GlobalLimitExec` above a child with one partition, and a
    /// `LocalLimitExec` above any other child. This is how the
    /// `LimitPushdown` optimizer rule limits the children of a node that
    /// supports limit pushdown.
    LimitedInputs(usize),
    /// The node executed with the session batch size set to `n`
    BatchSize(usize),
    /// The node rebuilt with the rows of each partition of its
    /// [`MockSourceExec`] leaves split into batches with this layout. Every
    /// leaf keeps the same rows in the same order and partitions, and the
    /// same `PlanProperties`.
    ///
    /// [`MockSourceExec`]: crate::fixtures::MockSourceExec
    BatchLayout(BatchLayout),
}

impl Variant {
    /// The family this variant belongs to
    pub fn kind(&self) -> VariantKind {
        match self {
            Variant::WithoutFetch => VariantKind::WithoutFetch,
            Variant::WithFetch(_) => VariantKind::WithFetch,
            Variant::LimitedInputs(_) => VariantKind::LimitedInputs,
            Variant::BatchSize(_) => VariantKind::BatchSize,
            Variant::BatchLayout(_) => VariantKind::BatchLayout,
        }
    }
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

/// Settings shared by all variant runs
#[derive(Debug, Clone)]
pub(crate) struct VariantOptions {
    /// Context of the normal execution, which variants other than
    /// [`Variant::BatchSize`] use unchanged
    pub task_context: Arc<TaskContext>,
    /// Time allowed for each run
    pub timeout: Duration,
}

/// Run the variants of `kind` that apply to `node`. `larger_than_output` is
/// a row count larger than the output of the node and of each of its
/// children, used as a fetch or limit that removes nothing.
pub(crate) async fn run(
    node: &Arc<dyn ExecutionPlan>,
    kind: VariantKind,
    larger_than_output: usize,
    options: &VariantOptions,
) -> Vec<VariantRun> {
    let mut limits = SMALL_LIMITS.to_vec();
    if !limits.contains(&larger_than_output) {
        limits.push(larger_than_output);
    }
    let variants: Vec<Variant> = match kind {
        VariantKind::WithoutFetch => vec![Variant::WithoutFetch],
        VariantKind::WithFetch => limits.into_iter().map(Variant::WithFetch).collect(),
        VariantKind::LimitedInputs => {
            if node.supports_limit_pushdown() && !node.children().is_empty() {
                limits.into_iter().map(Variant::LimitedInputs).collect()
            } else {
                vec![]
            }
        }
        VariantKind::BatchSize => BATCH_SIZES.map(Variant::BatchSize).to_vec(),
        VariantKind::BatchLayout => LEAF_LAYOUTS.map(Variant::BatchLayout).to_vec(),
    };
    let mut runs = vec![];
    for variant in variants {
        let plan = match build(node, variant) {
            Ok(Some(plan)) => plan,
            Ok(None) => continue,
            Err(e) => {
                runs.push(VariantRun {
                    variant,
                    output: Err(format!(
                        "building the plan failed: {}",
                        e.strip_backtrace()
                    )),
                });
                continue;
            }
        };
        let task_context = match variant {
            Variant::BatchSize(batch_size) => {
                Arc::new(with_batch_size(&options.task_context, batch_size))
            }
            _ => Arc::clone(&options.task_context),
        };
        let output = collect_output(plan, task_context, options.timeout).await;
        runs.push(VariantRun { variant, output });
    }
    runs
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
                source.with_batch_layout(layout, LAYOUT_SEED).map(Some)
            })?;
            changed.then_some(plan)
        }
    })
}
