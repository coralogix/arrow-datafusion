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

/// Problems of one node found in several places, such as the variant runs of
/// one kind, or the statistics of each partition. Each kind of problem is
/// reported once, with the finding of the first place that has it, followed
/// by a summary of the other places that have it too, so that one cause does
/// not produce a finding per place. Kinds are reported in the order they are
/// first added, and places in the order they are added.
#[derive(Debug)]
struct Problems<P> {
    problems: Vec<Problem<P>>,
}

#[derive(Debug)]
struct Problem<P> {
    kind: &'static str,
    place: P,
    finding: Finding,
    other_places: Vec<P>,
}

impl<P> Default for Problems<P> {
    fn default() -> Self {
        Self { problems: vec![] }
    }
}

impl<P> Problems<P> {
    /// Record a problem of `kind`, described by `finding`, at `place`. The
    /// finding is only kept for the first place with a problem of `kind`.
    fn add(&mut self, kind: &'static str, place: impl Into<P>, finding: Finding) {
        let place = place.into();
        match self
            .problems
            .iter_mut()
            .find(|problem| problem.kind == kind)
        {
            Some(problem) => problem.other_places.push(place),
            None => self.problems.push(Problem {
                kind,
                place,
                finding,
                other_places: vec![],
            }),
        }
    }

    /// One finding per kind of problem, made by `describe` from the finding
    /// of the first place, that place, and the other places
    fn into_findings_with(
        self,
        mut describe: impl FnMut(&'static str, Finding, P, Vec<P>) -> Finding,
    ) -> Vec<Finding> {
        self.problems
            .into_iter()
            .map(|problem| {
                describe(
                    problem.kind,
                    problem.finding,
                    problem.place,
                    problem.other_places,
                )
            })
            .collect()
    }
}

/// Problems found in several runs of one node, each described by a label
type RunProblems = Problems<String>;

impl RunProblems {
    /// Prefix each finding with its run and name the other runs, as in
    /// `<first run>: <message> (also <run>, <run>)`
    fn into_findings(self) -> Vec<Finding> {
        self.into_findings_with(|_, mut finding, run, other_runs| {
            let also = if other_runs.is_empty() {
                String::new()
            } else {
                format!(" (also {})", other_runs.join(", "))
            };
            finding.message = format!("{run}: {}{also}", finding.message);
            finding
        })
    }
}
