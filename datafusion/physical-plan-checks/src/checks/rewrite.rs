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

//! Checks that plans returned by rewrite hooks, and nodes rewritten the way
//! optimizer rules rewrite them, produce what the original node promises.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion_common::Result;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

use super::{RunProblems, partition_count};
use crate::context::NodeOutput;
use crate::{CheckContext, Finding, PlanCheck, Variant, VariantKind, oracle};

/// All batches of `output`, in partition order
pub(super) fn all_batches(output: &NodeOutput) -> Vec<RecordBatch> {
    output.batches().cloned().collect()
}

/// What the output of a node must satisfy when at most `limit` rows per
/// partition are wanted from it: the output of a node with a fetch, or of a
/// node whose inputs were limited.
///
/// A fetch does not have to keep exactly `limit` rows per partition: the
/// partitions of a TopK `SortExec` share one threshold and can keep fewer rows,
/// and the `LimitPushdown` rule only relies on the total.
pub(super) struct FetchExpectation<'a> {
    /// Largest number of rows allowed in each partition
    pub limit: usize,
    /// The number of rows of the output the limit applies to. The whole
    /// output must have at least `min(limit, available_rows)` rows.
    pub available_rows: usize,
    /// How to name the output `available_rows` counts, in messages
    pub available_name: &'static str,
    /// The output without the limit, which every row must appear in, if known
    pub reference: Option<&'a NodeOutput>,
    /// How to name `reference` in messages
    pub reference_name: &'static str,
    /// The ordering the node reports, if any. The output must then keep the
    /// first rows of `reference` in that order; see [`Self::order_problem`].
    pub ordering: Option<&'a LexOrdering>,
}

impl FetchExpectation<'_> {
    /// Problems with `output`, each with a kind to group problems by.
    /// Comparisons that cannot be computed, for example because the output
    /// does not match the schema (reported by `batch_schema`), are skipped.
    pub(super) fn problems(&self, output: &NodeOutput) -> Vec<(&'static str, String)> {
        let mut problems = vec![];
        let rows = output.partition_num_rows();
        if let Some((p, n)) = rows.iter().enumerate().find(|(_, n)| **n > self.limit) {
            problems.push((
                "too many rows",
                format!("partition {p} has {n} rows, more than {}", self.limit),
            ));
        }
        let total = output.num_rows();
        let min_rows = self.limit.min(self.available_rows);
        if total < min_rows {
            problems.push((
                "too few rows",
                format!(
                    "the output has {total} rows, but limiting {} ({} rows) to {} \
                     rows gives {min_rows}",
                    self.available_name, self.available_rows, self.limit
                ),
            ));
        }
        let Some(reference) = self.reference else {
            return problems;
        };
        if let Ok(unmatched) =
            oracle::unmatched_rows(&all_batches(output), &all_batches(reference))
            && unmatched > 0
        {
            problems.push((
                "rows not in the reference",
                format!(
                    "{unmatched} of its {total} rows do not appear in {}",
                    self.reference_name
                ),
            ));
        }
        if let Some(ordering) = self.ordering {
            problems.extend(self.order_problem(output, reference, ordering));
        }
        problems
    }

    /// Whether `output` keeps the first rows of `reference` in sort order.
    ///
    /// With one partition, the output must be a prefix of the reference,
    /// apart from rows that tie on the ordering. With several, each partition
    /// must be sorted, and the partitions together must contain the first
    /// `min(limit, rows)` rows of the whole reference in the same sense, which
    /// is what a `SortPreservingMergeExec` with the same fetch above them
    /// needs. Each partition does not have to be a prefix of its own
    /// partition: the partitions of a TopK `SortExec` share a threshold and
    /// reject rows that tie with it, so a partition can keep rows it saw before
    /// another partition set the threshold instead of its own first rows.
    fn order_problem(
        &self,
        output: &NodeOutput,
        reference: &NodeOutput,
        ordering: &LexOrdering,
    ) -> Option<(&'static str, String)> {
        if let ([batches], [reference]) = (output.partitions(), reference.partitions()) {
            let row =
                oracle::first_non_prefix_row(reference, batches, ordering).ok()??;
            return Some((
                "not a prefix",
                format!(
                    "the output is not a prefix of {} sorted by [{ordering}], apart \
                     from the order of rows that tie on it, from row {row}",
                    self.reference_name
                ),
            ));
        }
        for (p, batches) in output.partitions().iter().enumerate() {
            if let Ok(Some(row)) = oracle::first_unsorted_row(batches, ordering) {
                return Some((
                    "not sorted",
                    format!("partition {p} is not sorted by [{ordering}] at row {row}"),
                ));
            }
        }
        // The first `n` rows of all partitions of `output`, in sort order
        let sorted_first_rows =
            |output: &NodeOutput, n: usize| -> Result<Vec<RecordBatch>> {
                let sorted = oracle::sort_rows(&all_batches(output), ordering)?;
                Ok(first_rows(&NodeOutput::new(vec![sorted]), n).partitions()[0].clone())
            };
        let n = self.limit.min(reference.num_rows());
        let expected = sorted_first_rows(reference, reference.num_rows()).ok()?;
        let kept = sorted_first_rows(output, n).ok()?;
        let row = oracle::first_non_prefix_row(&expected, &kept, ordering).ok()??;
        Some((
            "not a prefix",
            format!(
                "the first {n} rows of all partitions together, in sort order, are not \
                 the first rows of {} sorted by [{ordering}], apart from the order of \
                 rows that tie on it, from row {row}",
                self.reference_name
            ),
        ))
    }
}

/// D1: the plan returned by `with_fetch` produces a valid limit of the node's
/// output, and the node's own fetch does too.
#[derive(Debug, Default, Clone, Copy)]
pub struct WithFetchEquivalent;

impl WithFetchEquivalent {
    /// Check what the plan that `variant` executes reports about itself
    fn check_plan(
        node: &Arc<dyn ExecutionPlan>,
        variant: Variant,
        problems: &mut RunProblems,
    ) {
        let (plan, fetch) = match variant {
            Variant::WithoutFetch => (node.with_fetch(None), None),
            Variant::WithFetch(n) => (node.with_fetch(Some(n)), Some(n)),
            _ => return,
        };
        let Some(plan) = plan else {
            return;
        };
        let run = variant.to_string();
        if plan.fetch() != fetch {
            problems.add(
                "fetch",
                &run,
                Finding::invariant(format!(
                    "the plan reports fetch() {:?} instead of {fetch:?}",
                    plan.fetch()
                )),
            );
        }
        if plan.schema() != node.schema() {
            let fields = |plan: &Arc<dyn ExecutionPlan>| {
                plan.schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}: {}", f.name(), f.data_type()))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            problems.add(
                "schema",
                &run,
                Finding::invariant(format!(
                    "the plan has the schema [{}], but the node has [{}]",
                    fields(&plan),
                    fields(node)
                )),
            );
        }
        let (partitions, expected) = (
            partition_count(plan.as_ref()),
            partition_count(node.as_ref()),
        );
        if partitions != expected {
            problems.add(
                "partitions",
                &run,
                Finding::invariant(format!(
                    "the plan has {partitions} output partitions, but the node has \
                     {expected}"
                )),
            );
        }
        let eq_properties = plan.properties().equivalence_properties();
        for ordering in node
            .properties()
            .equivalence_properties()
            .oeq_class()
            .iter()
        {
            if matches!(eq_properties.ordering_satisfy(ordering.clone()), Ok(false)) {
                problems.add(
                    "ordering",
                    &run,
                    Finding::lint(format!(
                        "the plan does not report the ordering [{ordering}] that the \
                         node reports"
                    )),
                );
            }
        }
    }

    /// What the output of the node with a fetch of `limit` must satisfy
    fn expectation<'a>(
        node: &'a Arc<dyn ExecutionPlan>,
        limit: usize,
        normal: &'a NodeOutput,
        unfetched: Option<&'a NodeOutput>,
    ) -> FetchExpectation<'a> {
        match unfetched {
            Some(unfetched) => FetchExpectation {
                limit,
                available_rows: unfetched.num_rows(),
                available_name: "the output without the fetch",
                reference: Some(unfetched),
                reference_name: "the output without the fetch",
                ordering: node.properties().output_ordering(),
            },
            // A node with a fetch whose `with_fetch(None)` returns `None`. Its
            // output without the fetch has at least as many rows as its
            // normal output, but which rows it has is unknown.
            None => FetchExpectation {
                limit,
                available_rows: normal.num_rows(),
                available_name: "the normal output",
                reference: None,
                reference_name: "the output without the fetch",
                ordering: None,
            },
        }
    }
}

impl PlanCheck for WithFetchEquivalent {
    fn code(&self) -> &'static str {
        "D1"
    }

    fn name(&self) -> &'static str {
        "with_fetch_equivalent"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn variants(&self) -> &'static [VariantKind] {
        &[VariantKind::WithoutFetch, VariantKind::WithFetch]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let runs: Vec<_> = context
            .variant_runs(node, VariantKind::WithoutFetch)
            .iter()
            .chain(context.variant_runs(node, VariantKind::WithFetch))
            .collect();
        let mut problems = RunProblems::default();
        for run in &runs {
            Self::check_plan(node, run.variant, &mut problems);
        }
        // A failing node is reported by `execution_succeeds`
        let Some(normal) = context.output(node) else {
            return Ok(problems.into_findings());
        };
        let unfetched = context.unfetched_output(node);

        // The normal output of a node with a fetch is the output of that fetch
        if let Some(fetch) = node.fetch()
            && !runs.is_empty()
        {
            let expectation = Self::expectation(node, fetch, normal, unfetched);
            for (kind, message) in expectation.problems(normal) {
                problems.add(
                    kind,
                    format!("the node's own fetch of {fetch}"),
                    Finding::invariant(message),
                );
            }
        }

        for run in runs {
            let label = run.variant.to_string();
            let output = match &run.output {
                Ok(output) => output,
                Err(error) => {
                    problems.add(
                        "failed",
                        label,
                        Finding::invariant(format!("executing the plan failed: {error}")),
                    );
                    continue;
                }
            };
            match run.variant {
                Variant::WithoutFetch => {
                    let same =
                        oracle::same_rows(&all_batches(output), &all_batches(normal));
                    if node.fetch().is_none() && matches!(same, Ok(false)) {
                        problems.add(
                            "not the normal output",
                            label,
                            Finding::invariant(format!(
                                "the node has no fetch, but the plan produced {} rows \
                                 that differ from the {} rows of the normal output",
                                output.num_rows(),
                                normal.num_rows()
                            )),
                        );
                    }
                }
                Variant::WithFetch(n) => {
                    let expectation = Self::expectation(node, n, normal, unfetched);
                    for (kind, message) in expectation.problems(output) {
                        problems.add(kind, &label, Finding::invariant(message));
                    }
                }
                _ => {}
            }
        }
        Ok(problems.into_findings())
    }
}

/// The first `n` rows of each partition of `output`
fn first_rows(output: &NodeOutput, n: usize) -> NodeOutput {
    let partitions = output
        .partitions()
        .iter()
        .map(|batches| {
            let mut remaining = n;
            let mut kept = vec![];
            for batch in batches {
                if remaining == 0 {
                    break;
                }
                let rows = batch.num_rows().min(remaining);
                kept.push(batch.slice(0, rows));
                remaining -= rows;
            }
            kept
        })
        .collect();
    NodeOutput::new(partitions)
}

/// D2: a node that supports limit pushdown produces a valid limit of its
/// output from inputs limited the way `LimitPushdown` limits them.
#[derive(Debug, Default, Clone, Copy)]
pub struct LimitPushdownEquivalent;

impl PlanCheck for LimitPushdownEquivalent {
    fn code(&self) -> &'static str {
        "D2"
    }

    fn name(&self) -> &'static str {
        "limit_pushdown_equivalent"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn variants(&self) -> &'static [VariantKind] {
        &[VariantKind::WithoutFetch, VariantKind::LimitedInputs]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        if !node.supports_limit_pushdown() || node.children().is_empty() {
            return Ok(vec![]);
        }
        // `LimitPushdown` merges limit nodes into the limit it pushes down
        // before it asks any node whether it supports limit pushdown, so it
        // never relies on their answer
        if node.is::<GlobalLimitExec>() || node.is::<LocalLimitExec>() {
            return Ok(vec![]);
        }
        let Some(normal) = context.output(node) else {
            return Ok(vec![]);
        };
        // For these nodes `LimitPushdown` does not remove the limit above
        // them: it gives them a fetch, or keeps the limit, so only their first
        // `n` rows count. It removes the limit above any other node that
        // supports limit pushdown and limits each child instead.
        let combines_partitions =
            node.is::<CoalescePartitionsExec>() || node.is::<SortPreservingMergeExec>();
        let reference = context.unfetched_output(node);
        let reference_name = if node.fetch().is_some() {
            "the output without the fetch"
        } else {
            "the normal output"
        };

        let mut problems = RunProblems::default();
        for run in context.variant_runs(node, VariantKind::LimitedInputs) {
            let Variant::LimitedInputs(n) = run.variant else {
                continue;
            };
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
            let candidate = if combines_partitions {
                first_rows(output, n)
            } else {
                output.clone()
            };
            let expectation = FetchExpectation {
                limit: n,
                available_rows: normal.num_rows(),
                available_name: "the normal output",
                reference,
                reference_name,
                ordering: node.properties().output_ordering(),
            };
            for (kind, message) in expectation.problems(&candidate) {
                problems.add(
                    kind,
                    &label,
                    Finding::invariant(format!(
                        "{message}, so a limit above the node cannot be pushed to its \
                         children; return false from supports_limit_pushdown"
                    )),
                );
            }
        }
        Ok(problems.into_findings())
    }
}
