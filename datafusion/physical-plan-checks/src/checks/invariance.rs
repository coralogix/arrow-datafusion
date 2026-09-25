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

//! Checks that a node produces the same results under settings and input
//! layouts that should not change them.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;

use super::RunProblems;
use super::rewrite::{FetchExpectation, all_batches};
use crate::context::NodeOutput;
use crate::{CheckContext, Finding, PlanCheck, Variant, VariantKind, VariantRun, oracle};

/// Returns true if every child of `node` produced the same rows, in the same
/// order and partitions, under `variant` as normally. The node then saw the
/// same input, only split into batches differently.
///
/// Batch sizes and leaf layouts apply to the whole subtree, so a child can
/// produce different results under a variant, either because it is broken
/// (reported on the child) or legitimately, for example rows that tie on a
/// sort key in another order, or rows interleaved differently by a
/// repartition. A node whose output depends on the order of its input rows
/// can then legitimately differ too, so it is not compared under that variant.
fn inputs_unchanged(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    variant: Variant,
) -> bool {
    node.children().into_iter().all(|child| {
        let Some(normal) = context.output(child) else {
            return false;
        };
        match context.variant_run(child, variant) {
            // The variant does not apply to the child, which then runs as
            // it does normally, such as a child with no `MockSourceExec` leaf
            // to split differently
            None => true,
            Some(VariantRun {
                output: Ok(output), ..
            }) => {
                output.partitions().len() == normal.partitions().len()
                    && output.partitions().iter().zip(normal.partitions()).all(
                        |(a, b)| matches!(oracle::same_rows_in_order(a, b), Ok(true)),
                    )
            }
            Some(_) => false,
        }
    })
}

/// How the output of `node` under a variant differs from its normal output
/// in ways the variant should not change, each with a kind to group
/// differences by
fn differences(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    normal: &NodeOutput,
    output: &NodeOutput,
) -> Vec<(&'static str, String)> {
    let mut differences = vec![];
    if let Some(fetch) = node.fetch() {
        // Which rows a fetch keeps can depend on timing and batch boundaries,
        // such as how the partitions below a `CoalescePartitionsExec`
        // interleave, so only compare what the fetch guarantees. With one
        // output partition this includes the exact number of rows.
        let unfetched = context.unfetched_output(node);
        let expectation = FetchExpectation {
            limit: fetch,
            available_rows: unfetched.unwrap_or(normal).num_rows(),
            available_name: if unfetched.is_some() {
                "the output without the fetch"
            } else {
                "the normal output"
            },
            reference: unfetched,
            reference_name: "the output without the fetch",
            ordering: node.properties().output_ordering(),
        };
        differences.extend(expectation.problems(output));
    } else {
        let (batches, normal_batches) = (all_batches(output), all_batches(normal));
        if matches!(oracle::same_rows(&batches, &normal_batches), Ok(false)) {
            let unmatched =
                oracle::unmatched_rows(&batches, &normal_batches).unwrap_or(0);
            differences.push((
                "different rows",
                format!(
                    "the node produced {} rows, {unmatched} of which do not appear in \
                     the normal output, instead of the {} rows it produces normally",
                    output.num_rows(),
                    normal.num_rows()
                ),
            ));
        }
    }
    for ordering in node
        .properties()
        .equivalence_properties()
        .oeq_class()
        .iter()
    {
        for (p, batches) in output.partitions().iter().enumerate() {
            // An ordering that does not hold normally is reported by
            // `orderings_hold`
            let sorted_normally = normal.partitions().get(p).is_some_and(|normal| {
                matches!(oracle::first_unsorted_row(normal, ordering), Ok(None))
            });
            if sorted_normally
                && let Ok(Some(row)) = oracle::first_unsorted_row(batches, ordering)
            {
                differences.push((
                    "ordering",
                    format!(
                        "partition {p} is not sorted by [{ordering}] at row {row}, \
                         although the node reports that ordering and its normal output \
                         is sorted by it"
                    ),
                ));
                break;
            }
        }
    }
    differences
}

/// Compare the output of `node` under each variant of `kind` with its normal
/// output
fn check_invariance(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    kind: VariantKind,
) -> Vec<Finding> {
    // A failing node is reported by `execution_succeeds`
    let Some(normal) = context.output(node) else {
        return vec![];
    };
    let mut problems = RunProblems::default();
    for run in context.variant_runs(node, kind) {
        if !inputs_unchanged(node, context, run.variant) {
            continue;
        }
        let label = format!("with {}", run.variant);
        let output = match &run.output {
            Ok(output) => output,
            Err(error) => {
                problems.add(
                    "failed",
                    label,
                    Finding::invariant(format!("executing the node failed: {error}")),
                );
                continue;
            }
        };
        for (kind, message) in differences(node, context, normal, output) {
            problems.add(kind, &label, Finding::invariant(message));
        }
    }
    problems.into_findings()
}

/// E1: results do not depend on the session batch size.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchSizeInvariance;

impl PlanCheck for BatchSizeInvariance {
    fn code(&self) -> &'static str {
        "E1"
    }

    fn name(&self) -> &'static str {
        "batch_size_invariance"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn variants(&self) -> &'static [VariantKind] {
        &[VariantKind::WithoutFetch, VariantKind::BatchSize]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        Ok(check_invariance(node, context, VariantKind::BatchSize))
    }
}

/// E2: results do not depend on how input rows are split into batches.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchBoundaryInvariance;

impl PlanCheck for BatchBoundaryInvariance {
    fn code(&self) -> &'static str {
        "E2"
    }

    fn name(&self) -> &'static str {
        "batch_boundary_invariance"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn variants(&self) -> &'static [VariantKind] {
        &[VariantKind::WithoutFetch, VariantKind::BatchLayout]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        Ok(check_invariance(node, context, VariantKind::BatchLayout))
    }
}
