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

//! Built-in [`PlanCheck`]s. See `CHECKS.md` for what each check verifies.

use std::sync::Arc;

use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::{ExecutionPlan, StatisticsArgs, StatisticsContext};

use crate::{Finding, PlanCheck};

mod cardinality;
mod invariance;
mod lifecycle;
mod rewrite;
mod runtime;
mod statistics;
mod stream;
mod structure;

pub use cardinality::{
    CardinalityEffectBoundsNumRows, EqualCardinalityNumRows, FetchBoundsNumRows,
    FetchNotEqualCardinality,
};
pub use invariance::{BatchBoundaryInvariance, BatchSizeInvariance};
pub use lifecycle::{ErrorsPropagate, ResourcesReleased};
pub use rewrite::{LimitPushdownEquivalent, WithFetchEquivalent};
pub use runtime::{
    BatchSchema, CardinalityEffectHolds, ExactStatisticsHold, ExecutionSucceeds,
    OrderingsHold,
};
pub use statistics::{PartitionStatisticsSum, StatisticsIgnoreInputs, StatisticsShape};
pub use stream::{BoundednessHolds, EmissionTypeHolds, LazyEvaluationHolds};
pub use structure::{CheckInvariants, PerChildLengths};

/// All built-in checks, in catalog order
pub fn all_checks() -> Vec<Arc<dyn PlanCheck>> {
    let mut checks = static_checks();
    checks.extend(execution_checks());
    checks
}

/// Built-in checks that execute the plan (sections B to F), in catalog order
pub fn execution_checks() -> Vec<Arc<dyn PlanCheck>> {
    vec![
        Arc::new(ExecutionSucceeds),
        Arc::new(BatchSchema),
        Arc::new(ExactStatisticsHold),
        Arc::new(OrderingsHold),
        Arc::new(CardinalityEffectHolds),
        Arc::new(BoundednessHolds),
        Arc::new(EmissionTypeHolds),
        Arc::new(LazyEvaluationHolds),
        Arc::new(WithFetchEquivalent),
        Arc::new(LimitPushdownEquivalent),
        Arc::new(BatchSizeInvariance),
        Arc::new(BatchBoundaryInvariance),
        Arc::new(ResourcesReleased),
        Arc::new(ErrorsPropagate),
    ]
}

/// Built-in checks that do not execute the plan (section A), in catalog order
pub fn static_checks() -> Vec<Arc<dyn PlanCheck>> {
    vec![
        Arc::new(EqualCardinalityNumRows),
        Arc::new(FetchNotEqualCardinality),
        Arc::new(FetchBoundsNumRows),
        Arc::new(CardinalityEffectBoundsNumRows),
        Arc::new(PerChildLengths),
        Arc::new(CheckInvariants),
        Arc::new(StatisticsShape),
        Arc::new(PartitionStatisticsSum),
        Arc::new(StatisticsIgnoreInputs),
    ]
}

/// Statistics of `plan` for all partitions, computed with a fresh
/// [`StatisticsContext`] and no statistics providers
fn overall_statistics(plan: &dyn ExecutionPlan) -> Result<Arc<Statistics>> {
    StatisticsContext::new().compute(plan, &StatisticsArgs::new())
}

/// Statistics of `plan` for a single partition, computed with a fresh
/// [`StatisticsContext`] and no statistics providers
fn partition_statistics(
    plan: &dyn ExecutionPlan,
    partition: usize,
) -> Result<Arc<Statistics>> {
    StatisticsContext::new()
        .compute(plan, &StatisticsArgs::new().with_partition(Some(partition)))
}

/// Number of output partitions of `plan`
fn partition_count(plan: &dyn ExecutionPlan) -> usize {
    plan.properties().output_partitioning().partition_count()
}

/// Problems found in several runs of one node, such as the variant runs of
/// one kind. Each kind of problem is reported once, for the first run that
/// has it, naming the other runs that have it too, so that one cause does not
/// produce a finding per run.
#[derive(Debug, Default)]
struct RunProblems {
    problems: Vec<RunProblem>,
}

#[derive(Debug)]
struct RunProblem {
    kind: &'static str,
    run: String,
    finding: Finding,
    other_runs: Vec<String>,
}

impl RunProblems {
    /// Record a problem of `kind`, described by `finding`, in the run
    /// described by `run`
    fn add(&mut self, kind: &'static str, run: impl Into<String>, finding: Finding) {
        let run = run.into();
        match self
            .problems
            .iter_mut()
            .find(|problem| problem.kind == kind)
        {
            Some(problem) => problem.other_runs.push(run),
            None => self.problems.push(RunProblem {
                kind,
                run,
                finding,
                other_runs: vec![],
            }),
        }
    }

    fn into_findings(self) -> Vec<Finding> {
        self.problems
            .into_iter()
            .map(|problem| {
                let mut finding = problem.finding;
                let also = if problem.other_runs.is_empty() {
                    String::new()
                } else {
                    format!(" (also {})", problem.other_runs.join(", "))
                };
                finding.message = format!("{}: {}{also}", problem.run, finding.message);
                finding
            })
            .collect()
    }
}
