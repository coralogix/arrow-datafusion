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
use crate::{CheckContext, Finding, PlanCheck};

/// A7: every method that returns one entry per child returns exactly
/// `children().len()` entries.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerChildLengths;

impl PerChildLengths {
    fn check_len(
        method: &str,
        actual: usize,
        expected: usize,
        findings: &mut Vec<Finding>,
    ) {
        if actual != expected {
            findings.push(Finding::invariant(format!(
                "{method} returned {actual} entries, but the node has {expected} children"
            )));
        }
    }

    fn check_child_stats_requests(
        node: &Arc<dyn ExecutionPlan>,
        partition: Option<usize>,
        findings: &mut Vec<Finding>,
    ) -> bool {
        let children = node.children();
        let requests = node.child_stats_requests(partition);
        let method = format!("child_stats_requests({partition:?})");
        let before = findings.len();
        Self::check_len(&method, requests.len(), children.len(), findings);
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
        findings.len() == before
    }
}

impl PlanCheck for PerChildLengths {
    fn code(&self) -> &'static str {
        "A7"
    }

    fn name(&self) -> &'static str {
        "per_child_lengths"
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        _context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let expected = node.children().len();
        let mut findings = vec![];
        Self::check_len(
            "maintains_input_order()",
            node.maintains_input_order().len(),
            expected,
            &mut findings,
        );
        Self::check_len(
            "required_input_ordering()",
            node.required_input_ordering().len(),
            expected,
            &mut findings,
        );
        Self::check_len(
            "benefits_from_input_partitioning()",
            node.benefits_from_input_partitioning().len(),
            expected,
            &mut findings,
        );
        Self::check_len(
            "input_distribution_requirements()",
            node.input_distribution_requirements()
                .per_child_distributions()
                .len(),
            expected,
            &mut findings,
        );
        Self::check_child_stats_requests(node, None, &mut findings);
        // Report only the first failing partition, since the same mistake
        // usually repeats for every partition
        for partition in 0..partition_count(node.as_ref()) {
            if !Self::check_child_stats_requests(node, Some(partition), &mut findings) {
                break;
            }
        }
        Ok(findings)
    }
}

/// A7: the node's own [`ExecutionPlan::check_invariants`] passes at
/// [`InvariantLevel::Always`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CheckInvariants;

impl PlanCheck for CheckInvariants {
    fn code(&self) -> &'static str {
        "A7"
    }

    fn name(&self) -> &'static str {
        "check_invariants"
    }

    fn check_node(
        &self,
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
}
