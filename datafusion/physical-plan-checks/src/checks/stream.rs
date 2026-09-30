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

//! Checks that compare how a node reports its streams behave (boundedness,
//! emission type, evaluation type) with how they behave in stream
//! experiments.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::{
    Boundedness, CardinalityEffect, EmissionType, EvaluationType,
};

use crate::{CheckContext, Experiment, Finding, RunOutcome};

/// B10: a node that reports `Bounded` ends on unbounded input.
pub(super) fn boundedness_holds(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    // True if the node, without its fetch, passes every input row through. Its
    // output then ends as soon as `fetch` rows reach it, which is the
    // guarantee `GlobalLimitExec` reports as `Bounded`. A node that can drop
    // rows, such as a filter, only ends if enough rows survive.
    let fetch_is_only_limit = || {
        node.with_fetch(None).is_some_and(|plan| {
            matches!(plan.cardinality_effect(), CardinalityEffect::Equal)
        })
    };
    let mut findings = vec![];
    for run in context.stream_runs(node, Experiment::UnboundedInput) {
        let Some(child) = run.varied_child else {
            continue;
        };
        let child_unbounded = run
            .input_boundedness
            .get(child)
            .is_some_and(Boundedness::is_unbounded);
        match (&run.outcome, run.boundedness) {
            // A child that reports Bounded is responsible for its own claim,
            // and a child that delivered nothing gave the node no chance to
            // end. A node with a fetch may need more rows than the leaves were
            // allowed to serve (a large fetch or offset), so a run in which
            // they reached their row limit shows nothing. Without a fetch,
            // nothing justifies needing a particular number of rows before
            // ending.
            (RunOutcome::TimedOut, Boundedness::Bounded)
                if child_unbounded
                    && run.input_rows(child) > 0
                    && !(run.inputs_exhausted && run.fetch.is_some()) =>
            {
                findings.push(Finding::invariant(format!(
                    "rebuilt with child {child} unbounded, the node reports \
                     Boundedness::Bounded, but its output did not end while the child \
                     kept delivering rows"
                )));
            }
            (RunOutcome::Ended, Boundedness::Unbounded { .. })
                if run.fetch.is_some()
                    && run.output_errors() == 0
                    && fetch_is_only_limit() =>
            {
                findings.push(Finding::lint(format!(
                    "rebuilt with child {child} unbounded, the node reports \
                     Boundedness::Unbounded, but it has fetch {:?} and otherwise passes \
                     every row through, so like GlobalLimitExec its output ends once the \
                     fetch is reached",
                    run.fetch
                )));
            }
            _ => {}
        }
    }
    Ok(findings)
}

/// B11: a node that reports `EmissionType::Incremental` produces output
/// before its input ends.
pub(super) fn emission_type_holds(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    // A node that produces no rows from the finite input cannot be expected
    // to produce rows from a repetition of it
    if context
        .output(node)
        .is_none_or(|output| output.num_rows() == 0)
    {
        return Ok(vec![]);
    }
    let mut findings = vec![];
    for run in context.stream_runs(node, Experiment::UnboundedInput) {
        let Some(child) = run.varied_child else {
            continue;
        };
        // Runs where the leaves reached their row limit are not skipped: a
        // node that buffers all of its input always reaches it, and that is
        // the main problem this check finds
        if run.emission_type != EmissionType::Incremental
            || run.outcome != RunOutcome::TimedOut
            || run.output_rows() > 0
            || run.input_rows(child) == 0
        {
            continue;
        }
        findings.push(Finding::invariant(format!(
            "rebuilt with child {child} unbounded, the node reports \
             EmissionType::Incremental, but it produced no rows before timing out \
             while the child kept delivering rows; report EmissionType::Final if the \
             node waits for the end of its input"
        )));
    }
    Ok(findings)
}

/// B12: a node that reports `EvaluationType::Lazy` only polls its inputs
/// while its own output is being polled.
pub(super) fn lazy_evaluation_holds(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    if node.properties().evaluation_type != EvaluationType::Lazy {
        return Ok(vec![]);
    }
    let mut findings = vec![];
    for run in context.stream_runs(node, Experiment::Laziness) {
        if matches!(run.outcome, RunOutcome::Failed(_)) {
            continue;
        }
        for (child, partitions) in run.inputs.iter().enumerate() {
            let undemanded: Vec<usize> = partitions
                .iter()
                .enumerate()
                .filter(|(_, observation)| observation.polls_without_demand > 0)
                .map(|(p, _)| p)
                .collect();
            if !undemanded.is_empty() {
                findings.push(Finding::invariant(format!(
                    "the node reports EvaluationType::Lazy, but polled partitions \
                     {undemanded:?} of child {child} while its own output was not being \
                     polled, for example from a spawned task or in execute(); report \
                     EvaluationType::Eager"
                )));
            }
        }
    }
    Ok(findings)
}
