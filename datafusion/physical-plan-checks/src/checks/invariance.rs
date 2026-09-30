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

use super::rewrite::FetchExpectation;
use crate::{CheckContext, Finding, NodeOutput, Variant, VariantRun, oracle};

/// E1: results do not depend on the session batch size.
pub(super) fn batch_size_invariance(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    Ok(check_invariance(node, context, |variant| {
        matches!(variant, Variant::BatchSize(_))
    }))
}

/// E2: results do not depend on how input rows are split into batches.
pub(super) fn batch_boundary_invariance(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    Ok(check_invariance(node, context, |variant| {
        matches!(variant, Variant::BatchLayout(_))
    }))
}

/// Compare the output of `node` under each variant for which `compared`
/// returns true with its normal output
fn check_invariance(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    compared: fn(&Variant) -> bool,
) -> Vec<Finding> {
    // A failing node is reported by `execution_succeeds`
    let Some(normal) = context.output(node) else {
        return vec![];
    };
    let mut findings = vec![];
    for run in context.variant_runs(node) {
        if !compared(&run.variant) || !inputs_unchanged(node, context, run.variant) {
            continue;
        }
        match &run.output {
            Ok(output) => {
                for difference in differences(node, context, normal, output) {
                    findings.push(Finding::invariant(format!(
                        "with {}: {difference}",
                        run.variant
                    )));
                }
            }
            Err(error) => findings.push(Finding::invariant(format!(
                "with {}: executing the node failed: {error}",
                run.variant
            ))),
        }
    }
    findings
}

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
/// in ways the variant should not change
fn differences(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
    normal: &NodeOutput,
    output: &NodeOutput,
) -> Vec<String> {
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
            unlimited: unfetched,
            ordering: node.properties().output_ordering(),
        };
        differences.extend(expectation.problems(output));
    } else if matches!(
        oracle::same_rows(&output.batches(), &normal.batches()),
        Ok(false)
    ) {
        let unmatched =
            oracle::unmatched_rows(&output.batches(), &normal.batches()).unwrap_or(0);
        differences.push(format!(
            "the node produced {} rows, {unmatched} of which do not appear in the \
             normal output, instead of the {} rows it produces normally",
            output.num_rows(),
            normal.num_rows()
        ));
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
                differences.push(format!(
                    "partition {p} is not sorted by [{ordering}] at row {row}, although \
                     the node reports that ordering and its normal output is sorted by it"
                ));
                break;
            }
        }
    }
    differences
}
