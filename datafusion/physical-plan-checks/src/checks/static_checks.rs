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

//! `CheckKind::Static` checks: they compare the statistics, cardinality effect,
//! fetch and per-child metadata a node reports with each other and with what
//! its children report, without executing it.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::stats::Precision;
use datafusion_physical_plan::execution_plan::{CardinalityEffect, InvariantLevel};
use datafusion_physical_plan::{ChildStats, ExecutionPlan};

use super::{overall_statistics, partition_count, partition_statistics};
use crate::{CheckContext, Finding};

/// A1: a single-child node with `CardinalityEffect::Equal` and no fetch reports
/// the same `num_rows` as its input, with the same precision.
pub(super) fn equal_cardinality_num_rows(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    if !matches!(node.cardinality_effect(), CardinalityEffect::Equal)
        || node.fetch().is_some()
    {
        return Ok(vec![]);
    }
    let children = node.children();
    let [child] = children.as_slice() else {
        return Ok(vec![]);
    };
    // Errors are reported by `statistics_shape`
    let (Ok(input), Ok(output)) = (
        overall_statistics(child.as_ref()),
        overall_statistics(node.as_ref()),
    ) else {
        return Ok(vec![]);
    };

    let input_rows = input.num_rows;
    let output_rows = output.num_rows;
    let finding = match (input_rows, output_rows) {
        (Precision::Exact(n), Precision::Exact(m)) if n != m => {
            Some(Finding::invariant(format!(
                "cardinality_effect() is Equal, but num_rows is {output_rows} \
                 for an input with num_rows {input_rows}"
            )))
        }
        (Precision::Inexact(_) | Precision::Absent, Precision::Exact(_)) => {
            Some(Finding::invariant(format!(
                "cardinality_effect() is Equal, but num_rows is {output_rows} \
                 for an input with num_rows {input_rows}; an Equal node cannot \
                 know its row count more precisely than its input does"
            )))
        }
        (Precision::Exact(_), Precision::Inexact(_)) => Some(Finding::lint(format!(
            "cardinality_effect() is Equal and the input has num_rows \
             {input_rows}, but the node reports {output_rows}; it could report \
             {input_rows}"
        ))),
        (Precision::Inexact(n), Precision::Inexact(m)) if n != m => {
            Some(Finding::lint(format!(
                "cardinality_effect() is Equal, but the num_rows estimate \
                 {output_rows} differs from the input estimate {input_rows}"
            )))
        }
        // Absent output for a known input is reported by
        // `statistics_ignore_inputs`
        _ => None,
    };
    Ok(finding.into_iter().collect())
}

/// A2: a node with a fetch does not claim `CardinalityEffect::Equal`.
pub(super) fn fetch_not_equal_cardinality(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    match (node.fetch(), node.cardinality_effect()) {
        (Some(fetch), CardinalityEffect::Equal) => Ok(vec![Finding::invariant(format!(
            "fetch() is Some({fetch}), but cardinality_effect() is Equal; a fetch \
             can drop input rows, so the effect should be LowerEqual while a fetch \
             is set"
        ))]),
        _ => Ok(vec![]),
    }
}

/// A3: a node with `fetch() == Some(n)` reports at most `n` rows per partition,
/// and at most `n * partition_count` rows overall.
pub(super) fn fetch_bounds_num_rows(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    let Some(fetch) = node.fetch() else {
        return Ok(vec![]);
    };
    let check_bound = |target: &str, num_rows: Precision<usize>, bound: usize| {
        match num_rows {
            Precision::Exact(n) if n > bound => Some(Finding::invariant(format!(
                "fetch() is Some({fetch}), but {target} num_rows is {num_rows}, which is \
             more than the fetch allows ({bound})"
            ))),
            Precision::Inexact(n) if n > bound => Some(Finding::lint(format!(
                "fetch() is Some({fetch}), but the {target} num_rows estimate {num_rows} is \
             more than the fetch allows ({bound})"
            ))),
            _ => None,
        }
    };
    let partitions = partition_count(node.as_ref());
    let mut findings = vec![];
    // A fetch limits each partition, so the overall bound grows with the
    // partition count. Nodes with a single output partition are bounded by
    // `fetch` exactly.
    if let Ok(stats) = overall_statistics(node.as_ref()) {
        let bound = fetch.saturating_mul(partitions.max(1));
        findings.extend(check_bound("overall", stats.num_rows, bound));
    }
    if partitions > 1 {
        for partition in 0..partitions {
            if let Ok(stats) = partition_statistics(node.as_ref(), partition) {
                let target = format!("partition {partition}");
                findings.extend(check_bound(&target, stats.num_rows, fetch));
            }
        }
    }
    Ok(findings)
}

/// A4: `num_rows` is consistent with `CardinalityEffect::LowerEqual` and
/// `CardinalityEffect::GreaterEqual`.
pub(super) fn cardinality_effect_bounds_num_rows(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    let effect = node.cardinality_effect();
    if !matches!(
        effect,
        CardinalityEffect::LowerEqual | CardinalityEffect::GreaterEqual
    ) {
        return Ok(vec![]);
    }
    let Ok(output) = overall_statistics(node.as_ref()) else {
        return Ok(vec![]);
    };
    let output = output.num_rows;
    let children = node.children();
    let mut findings = vec![];
    match effect {
        // Only single-child nodes: which input a multi-child node's
        // LowerEqual refers to is not defined
        CardinalityEffect::LowerEqual => {
            if let [child] = children.as_slice()
                && let Ok(input) = overall_statistics(child.as_ref())
            {
                let input = input.num_rows;
                findings.extend(match (input, output) {
                    (Precision::Exact(n), Precision::Exact(m)) if m > n => {
                        Some(Finding::invariant(format!(
                            "cardinality_effect() is LowerEqual, but num_rows is \
                             {output} for an input with num_rows {input}"
                        )))
                    }
                    (
                        Precision::Exact(n) | Precision::Inexact(n),
                        Precision::Inexact(m),
                    ) if m > n => Some(Finding::lint(format!(
                        "cardinality_effect() is LowerEqual, but the num_rows \
                             estimate {output} is larger than the input num_rows {input}"
                    ))),
                    _ => None,
                });
            }
        }
        // Producing at least as many rows as all inputs together implies
        // producing at least as many rows as each input
        CardinalityEffect::GreaterEqual => {
            for (i, child) in children.iter().enumerate() {
                let Ok(input) = overall_statistics(child.as_ref()) else {
                    continue;
                };
                let input = input.num_rows;
                findings.extend(match (input, output) {
                    (Precision::Exact(n), Precision::Exact(m)) if m < n => {
                        Some(Finding::invariant(format!(
                            "cardinality_effect() is GreaterEqual, but num_rows is \
                             {output} while input {i} has num_rows {input}"
                        )))
                    }
                    (Precision::Exact(n), Precision::Inexact(m)) if m < n => {
                        Some(Finding::lint(format!(
                            "cardinality_effect() is GreaterEqual, but the num_rows \
                             estimate {output} is smaller than the num_rows {input} of \
                             input {i}"
                        )))
                    }
                    _ => None,
                });
            }
        }
        CardinalityEffect::Equal | CardinalityEffect::Unknown => {}
    }
    Ok(findings)
}

/// A7: every method that returns one entry per child returns exactly
/// `children().len()` entries.
pub(super) fn per_child_lengths(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    let children = node.children();
    let wrong_len = |method: &str, actual: usize| {
        (actual != children.len()).then(|| {
            Finding::invariant(format!(
                "{method} returned {actual} entries, but the node has {} children",
                children.len()
            ))
        })
    };
    let distributions = node.input_distribution_requirements();
    let mut findings: Vec<Finding> = [
        (
            "maintains_input_order()",
            node.maintains_input_order().len(),
        ),
        (
            "required_input_ordering()",
            node.required_input_ordering().len(),
        ),
        (
            "benefits_from_input_partitioning()",
            node.benefits_from_input_partitioning().len(),
        ),
        (
            "input_distribution_requirements()",
            distributions.per_child_distributions().len(),
        ),
    ]
    .into_iter()
    .filter_map(|(method, len)| wrong_len(method, len))
    .collect();
    let partitions = (0..partition_count(node.as_ref())).map(Some);
    for partition in std::iter::once(None).chain(partitions) {
        let method = format!("child_stats_requests({partition:?})");
        let requests = node.child_stats_requests(partition);
        findings.extend(wrong_len(&method, requests.len()));
        for (i, (child, request)) in children.iter().zip(&requests).enumerate() {
            if let ChildStats::At(Some(p)) = request {
                let child_partitions = partition_count(child.as_ref());
                if *p >= child_partitions {
                    findings.push(Finding::invariant(format!(
                        "{method} requested partition {p} of child {i}, which has \
                         {child_partitions} partitions"
                    )));
                }
            }
        }
    }
    Ok(findings)
}

/// A7: the node's own [`ExecutionPlan::check_invariants`] passes at
/// [`InvariantLevel::Always`].
pub(super) fn check_invariants(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    match node.check_invariants(InvariantLevel::Always) {
        Ok(()) => Ok(vec![]),
        Err(e) => Ok(vec![Finding::invariant(format!(
            "check_invariants(Always) failed: {}",
            e.strip_backtrace()
        ))]),
    }
}

/// A8: statistics can be computed, overall and for every partition, and have
/// one column statistics entry per field.
pub(super) fn statistics_shape(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    // Errors from a child are reported on the child, not again on its parent
    let child_fails = || {
        node.children().into_iter().any(|child| {
            overall_statistics(child.as_ref()).is_err()
                || (0..partition_count(child.as_ref()))
                    .any(|p| partition_statistics(child.as_ref(), p).is_err())
        })
    };
    let fields = node.schema().fields().len();
    let partitions = (0..partition_count(node.as_ref())).map(Some);
    let mut findings = vec![];
    for partition in std::iter::once(None).chain(partitions) {
        let (target, result) = match partition {
            None => ("overall".to_string(), overall_statistics(node.as_ref())),
            Some(p) => (
                format!("partition {p}"),
                partition_statistics(node.as_ref(), p),
            ),
        };
        match result {
            Ok(stats) => {
                let columns = stats.column_statistics.len();
                if columns != fields {
                    findings.push(Finding::invariant(format!(
                        "{target} statistics have {columns} column statistics \
                         entries, but the schema has {fields} fields"
                    )));
                }
            }
            Err(e) if !child_fails() => findings.push(Finding::invariant(format!(
                "computing {target} statistics returned an error: {}; return \
                 Statistics::new_unknown when statistics are not available",
                e.strip_backtrace()
            ))),
            Err(_) => {}
        }
    }
    Ok(findings)
}

/// A8: per-partition row counts are consistent with the overall row count.
pub(super) fn partition_statistics_sum(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    let partitions = partition_count(node.as_ref());
    if partitions == 0 {
        return Ok(vec![]);
    }
    // Errors are reported by `statistics_shape`
    let Ok(overall) = overall_statistics(node.as_ref()) else {
        return Ok(vec![]);
    };
    let Ok(per_partition) = (0..partitions)
        .map(|p| partition_statistics(node.as_ref(), p).map(|s| s.num_rows))
        .collect::<Result<Vec<_>>>()
    else {
        return Ok(vec![]);
    };

    let overall_rows = overall.num_rows;
    let exact_rows = per_partition
        .iter()
        .map(|rows| match rows {
            Precision::Exact(n) => Some(*n),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();

    let mut findings = vec![];
    if let Some(exact_rows) = exact_rows {
        let sum: usize = exact_rows.iter().sum();
        match overall_rows {
            Precision::Exact(n) if n != sum => {
                findings.push(Finding::invariant(format!(
                    "overall num_rows is {overall_rows}, but the exact per-partition \
                     num_rows {exact_rows:?} sum to {sum}"
                )));
            }
            Precision::Exact(_) => {}
            Precision::Inexact(_) | Precision::Absent => {
                findings.push(Finding::lint(format!(
                    "every partition has an exact num_rows ({exact_rows:?}), but \
                     overall num_rows is {overall_rows}; it could be Exact({sum})"
                )));
            }
        }
    } else if let Precision::Exact(n) = overall_rows {
        for (p, rows) in per_partition.iter().enumerate() {
            if let Precision::Exact(m) = rows
                && *m > n
            {
                findings.push(Finding::invariant(format!(
                    "partition {p} num_rows is {rows}, which is more than the overall \
                     num_rows {overall_rows}"
                )));
            }
        }
    }
    Ok(findings)
}

/// A9: a single-child `CardinalityEffect::Equal` node without a fetch whose
/// input has a known row count also reports a row count.
pub(super) fn statistics_ignore_inputs(
    node: &Arc<dyn ExecutionPlan>,
    _context: &CheckContext,
) -> Result<Vec<Finding>> {
    if !matches!(node.cardinality_effect(), CardinalityEffect::Equal)
        || node.fetch().is_some()
    {
        return Ok(vec![]);
    }
    let children = node.children();
    let [child] = children.as_slice() else {
        return Ok(vec![]);
    };
    let (Ok(input), Ok(output)) = (
        overall_statistics(child.as_ref()),
        overall_statistics(node.as_ref()),
    ) else {
        return Ok(vec![]);
    };
    if input.num_rows == Precision::Absent || output.num_rows != Precision::Absent {
        return Ok(vec![]);
    }

    let hint = if node
        .child_stats_requests(None)
        .iter()
        .all(|r| *r == ChildStats::Skip)
    {
        "; child_stats_requests() skips the input, so statistics_from_inputs only \
         receives unknown input statistics"
    } else {
        ""
    };
    Ok(vec![Finding::lint(format!(
        "cardinality_effect() is Equal and the input has num_rows {}, but the node \
         reports Absent{hint}",
        input.num_rows
    ))])
}
