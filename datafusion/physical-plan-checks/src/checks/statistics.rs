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

//! Checks on the statistics a node reports, independent of its cardinality
//! effect.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::stats::Precision;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::{ChildStats, ExecutionPlan};

use super::{overall_statistics, partition_count, partition_statistics};
use crate::{CheckContext, Finding};

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
