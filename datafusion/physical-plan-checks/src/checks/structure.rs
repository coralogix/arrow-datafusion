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

//! Checks on the shape of per-child metadata and the node's own invariants.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::execution_plan::InvariantLevel;
use datafusion_physical_plan::{ChildStats, ExecutionPlan};

use super::partition_count;
use crate::{CheckContext, Finding};

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
