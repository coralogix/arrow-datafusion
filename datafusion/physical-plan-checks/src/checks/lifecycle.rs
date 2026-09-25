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

//! Checks on the lifecycle of execution: releasing resources and propagating
//! errors.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;

use crate::fixtures::PartitionObservation;
use crate::{CheckContext, Experiment, Finding, PlanCheck, RunOutcome};

/// Partitions, per child, that still have live streams
fn alive_partitions(inputs: &[Vec<PartitionObservation>]) -> Vec<(usize, Vec<usize>)> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(child, partitions)| {
            let alive: Vec<usize> = partitions
                .iter()
                .enumerate()
                .filter(|(_, observation)| observation.streams_alive() > 0)
                .map(|(p, _)| p)
                .collect();
            (!alive.is_empty()).then_some((child, alive))
        })
        .collect()
}

/// F1: memory reservations and input streams are released when they are no
/// longer needed.
#[derive(Debug, Default, Clone, Copy)]
pub struct ResourcesReleased;

impl ResourcesReleased {
    fn check_memory(
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
        findings: &mut Vec<Finding>,
    ) {
        let Some(reserved) = context.memory_reserved_after_execution(node) else {
            return;
        };
        // A child that leaks makes its parents leak too; report it on the child
        let child_leaks = node.children().into_iter().any(|child| {
            context
                .memory_reserved_after_execution(child)
                .is_some_and(|bytes| bytes > 0)
        });
        if reserved > 0 && !child_leaks {
            findings.push(Finding::invariant(format!(
                "after executing the node to completion and dropping its streams and \
                 the plan, {reserved} bytes are still reserved in the memory pool"
            )));
        }
    }

    fn check_cancellation(
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
        findings: &mut Vec<Finding>,
    ) {
        for run in context.stream_runs(node, Experiment::Cancellation) {
            if matches!(run.outcome, RunOutcome::Failed(_) | RunOutcome::Panicked(_)) {
                continue;
            }
            let alive = alive_partitions(&run.inputs);
            if alive.is_empty() {
                continue;
            }
            let after_plan = run
                .inputs_after_plan_dropped
                .as_deref()
                .map(alive_partitions)
                .unwrap_or_else(|| alive.clone());
            if after_plan.is_empty() {
                for (child, partitions) in alive {
                    findings.push(Finding::lint(format!(
                        "after the output streams were dropped part way, the streams \
                         of partitions {partitions:?} of child {child} were only \
                         dropped once the plan itself was dropped; the plan keeps \
                         input streams, or tasks that poll them, alive"
                    )));
                }
            } else {
                for (child, partitions) in after_plan {
                    findings.push(Finding::invariant(format!(
                        "after the output streams were dropped part way, and then the \
                         plan itself, the streams of partitions {partitions:?} of \
                         child {child} were still alive; tie spawned tasks and input \
                         streams to the output stream"
                    )));
                }
            }
        }
    }
}

impl PlanCheck for ResourcesReleased {
    fn code(&self) -> &'static str {
        "F1"
    }

    fn name(&self) -> &'static str {
        "resources_released"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn experiments(&self) -> &'static [Experiment] {
        &[Experiment::Cancellation]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let mut findings = vec![];
        Self::check_memory(node, context, &mut findings);
        Self::check_cancellation(node, context, &mut findings);
        Ok(findings)
    }
}

/// F2: an error from an input makes the output return an error.
#[derive(Debug, Default, Clone, Copy)]
pub struct ErrorsPropagate;

impl PlanCheck for ErrorsPropagate {
    fn code(&self) -> &'static str {
        "F2"
    }

    fn name(&self) -> &'static str {
        "errors_propagate"
    }

    fn experiments(&self) -> &'static [Experiment] {
        &[Experiment::InputError]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let mut findings = vec![];
        for run in context.stream_runs(node, Experiment::InputError) {
            // An input error the node never received does not have to be
            // returned. Neither does one received after the node had enough
            // rows for its fetch: an eager node can read the error into a
            // buffer while its output is still serving earlier batches.
            let fetch_satisfied = run
                .fetch
                .is_some_and(|fetch| run.output.iter().any(|o| o.rows >= fetch));
            if run.input_errors() == 0 || fetch_satisfied {
                continue;
            }
            let message = match &run.outcome {
                RunOutcome::Ended if run.output_errors() == 0 => {
                    "an input returned an error, but every output partition ended \
                     without returning an error"
                        .to_string()
                }
                RunOutcome::TimedOut if run.output_errors() == 0 => {
                    "an input returned an error, but the output neither returned an \
                     error nor ended before timing out"
                        .to_string()
                }
                RunOutcome::Panicked(panic) => {
                    format!("an input returned an error, and the node panicked: {panic}")
                }
                _ => continue,
            };
            findings.push(Finding::invariant(message));
        }
        Ok(findings)
    }
}
