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

//! [`Case`]: the inputs of one test of a factory, and how the harness derives
//! them by probing the plan's input requirements.

use std::fmt::Write;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_expr::{
    Distribution, LexOrdering, LexRequirement, OrderingRequirements, PhysicalExpr,
    PhysicalSortExpr,
};
use datafusion_physical_plan::distribution_requirements::ChildSatisfactionOptions;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use super::report::HarnessProblem;
use super::{PlanFactory, Profile};
use crate::context::panic_message;
use crate::fixtures::{SourceSpec, StatisticsPrecision};

/// How many times the harness builds a plan to find inputs that meet its
/// requirements, which can change with the inputs
pub const MAX_PROBES: usize = 5;

/// Row ids of input `i` start at `i * ROW_ID_RANGE`
pub const ROW_ID_RANGE: u64 = 1_000_000_000;

/// One test of a [`PlanFactory`]: a fully specified input for each of its
/// inputs, derived from the factory's base specs, a [`Profile`], and the
/// plan's input requirements.
///
/// A case can be reproduced from its specs alone: build each of
/// [`Self::inputs`], call [`PlanFactory::create`] on them, and run a
/// [`PlanChecker`] on the result, or use [`PlanHarness::run_case`] with
/// [`Self::index`].
///
/// [`PlanChecker`]: crate::PlanChecker
/// [`PlanHarness::run_case`]: crate::harness::PlanHarness::run_case
#[derive(Debug, Clone)]
pub struct Case {
    index: usize,
    profile: Profile,
    inputs: Vec<SourceSpec>,
    problem: Option<HarnessProblem>,
}

impl Case {
    /// The position of the case among the cases of its factory
    pub fn index(&self) -> usize {
        self.index
    }

    /// The profile the case was derived from
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The name of the case, which is the name of its profile
    pub fn name(&self) -> &str {
        self.profile.name()
    }

    /// The spec of each input
    pub fn inputs(&self) -> &[SourceSpec] {
        &self.inputs
    }

    /// Why the case could not be tested, if it could not: `create` failed,
    /// an input could not be generated, or the plan's input requirements
    /// could not be met
    pub fn problem(&self) -> Option<&HarnessProblem> {
        self.problem.as_ref()
    }

    /// Generate the inputs of the case
    pub fn build_inputs(&self) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
        self.inputs.iter().map(SourceSpec::build_arc).collect()
    }

    /// A short description of the inputs, derived from their specs, such as
    /// `default: input 0 rows [20, 0, 40]; input 1 hash [r_k@0] into 3
    /// partitions, 90 rows, sorted by [r_k@0 ASC]; exact statistics; seeds 0,
    /// 1`. The batch layout is the profile's and is not repeated.
    pub fn description(&self) -> String {
        let mut description = format!("{}: ", self.name());
        if self.inputs.is_empty() {
            description.push_str("no inputs");
            return description;
        }
        for (i, spec) in self.inputs.iter().enumerate() {
            if i > 0 {
                description.push_str("; ");
            }
            write!(description, "input {i} {}", describe_layout(spec)).unwrap();
        }
        let precision = match self.profile.statistics_precision() {
            StatisticsPrecision::Exact => "exact",
            StatisticsPrecision::Inexact => "inexact",
            StatisticsPrecision::Absent => "absent",
        };
        let seeds: Vec<String> = self
            .inputs
            .iter()
            .map(|spec| spec.seed().to_string())
            .collect();
        let seeds_label = if seeds.len() == 1 { "seed" } else { "seeds" };
        write!(
            description,
            "; {precision} statistics; {seeds_label} {}",
            seeds.join(", ")
        )
        .unwrap();
        description
    }
}

/// The partitioning and ordering of an input, as in `rows [100, 0, 200]` or
/// `hash [a@0] into 3 partitions, 300 rows, sorted by [a@0 ASC]`
fn describe_layout(spec: &SourceSpec) -> String {
    let mut layout = match (spec.partition_rows(), spec.hash_partitioning()) {
        (Some(rows), _) => format!("rows {rows:?}"),
        (None, Some((exprs, partitions))) => {
            let exprs: Vec<String> = exprs.iter().map(ToString::to_string).collect();
            format!(
                "hash [{}] into {partitions} partitions, {} rows",
                exprs.join(", "),
                spec.num_rows()
            )
        }
        (None, None) => unreachable!("a spec has rows or hash partitioning"),
    };
    if let Some(ordering) = spec.ordering() {
        write!(layout, ", sorted by [{ordering}]").unwrap();
    }
    layout
}

/// The cases of `factory` for `profiles`, in profile order. A factory without
/// inputs has one case, for the first profile, since every profile gives the
/// same plan. A profile that gives the same inputs as an earlier one, for
/// example `single partition` for a plan that requires every input in one
/// partition, adds no case, unless it runs checks the earlier one does not.
pub(crate) fn derive_cases(
    factory: &PlanFactory,
    profiles: &[Profile],
) -> Vec<(Case, Option<Arc<dyn ExecutionPlan>>)> {
    let profiles = if factory.inputs().is_empty() {
        &profiles[..profiles.len().min(1)]
    } else {
        profiles
    };
    let mut cases: Vec<(Case, Option<Arc<dyn ExecutionPlan>>)> = vec![];
    let mut seen: Vec<(String, bool)> = vec![];
    for profile in profiles {
        let (case, plan) = derive_case(factory, profile, cases.len());
        let key = format!("{:?}", case.inputs);
        let duplicate = seen.iter().any(|(seen_key, stream_experiments)| {
            *seen_key == key && (*stream_experiments || !profile.stream_experiments())
        });
        if duplicate {
            continue;
        }
        seen.push((key, profile.stream_experiments()));
        cases.push((case, plan));
    }
    cases
}

/// A change to the spec of an input that meets a requirement
#[derive(Debug, Clone)]
enum Fix {
    /// Sort every partition
    Ordering(LexOrdering),
    /// Hash partition on these expressions
    Hash(Vec<Arc<dyn PhysicalExpr>>),
    /// Put all rows in one partition
    SinglePartition,
}

/// A requirement of a node in the plan that its child does not meet
#[derive(Debug)]
struct Unmet {
    /// The input the child is, if it is one
    input: Option<usize>,
    fix: Fix,
    description: String,
}

/// The specs of `factory`'s inputs for `profile`, before requirements
fn initial_specs(factory: &PlanFactory, profile: &Profile) -> Vec<SourceSpec> {
    factory
        .inputs()
        .iter()
        .enumerate()
        .map(|(i, base)| {
            let rows = base.num_rows() * profile.row_multiplier();
            base.clone()
                .with_partition_rows(&profile.partition_rows(rows))
                .with_batch_layout(profile.batch_layout())
                .with_statistics_precision(profile.statistics_precision())
                .with_seed(profile.seed().wrapping_mul(1000).wrapping_add(i as u64))
                .with_row_id_column(factory.row_id_name(i), i as u64 * ROW_ID_RANGE)
        })
        .collect()
}

/// Derive the case of `factory` for `profile`: build the plan on inputs laid
/// out by the profile, change the inputs to meet the requirements of the
/// nodes directly above them, and repeat until every requirement is met, at
/// most [`MAX_PROBES`] times. Also returns the plan built on the final
/// inputs, if the case has no problem.
pub(crate) fn derive_case(
    factory: &PlanFactory,
    profile: &Profile,
    index: usize,
) -> (Case, Option<Arc<dyn ExecutionPlan>>) {
    let mut specs = initial_specs(factory, profile);
    let case = |inputs: Vec<SourceSpec>, problem| Case {
        index,
        profile: profile.clone(),
        inputs,
        problem,
    };
    let mut last_unmet: Vec<String> = vec![];
    for _ in 0..MAX_PROBES {
        let inputs = match build_inputs(&specs) {
            Ok(inputs) => inputs,
            Err(problem) => return (case(specs, Some(problem)), None),
        };
        let plan = match create(factory, inputs.clone()) {
            Ok(plan) => plan,
            Err(problem) => return (case(specs, Some(problem)), None),
        };
        let unmet = unmet_requirements(&plan, &inputs);
        if unmet.is_empty() {
            return (case(specs, None), Some(plan));
        }
        let (fixable, not_inputs): (Vec<Unmet>, Vec<Unmet>) =
            unmet.into_iter().partition(|unmet| unmet.input.is_some());
        // A requirement on a child the factory built can only be met by the
        // factory, unless meeting the requirements on the inputs meets it too
        if fixable.is_empty() {
            let descriptions: Vec<String> = not_inputs
                .into_iter()
                .map(|unmet| unmet.description)
                .collect();
            let problem = HarnessProblem::requirements(format!(
                "{}; the child is not an input, so the factory must build it so \
                 that it meets the requirement",
                descriptions.join("; ")
            ));
            return (case(specs, Some(problem)), None);
        }
        let partitions = profile.partition_count().max(1);
        let before = format!("{specs:?}");
        for unmet in &fixable {
            let input = unmet.input.expect("fixable requirements are on inputs");
            specs[input] = apply(specs[input].clone(), &unmet.fix, partitions);
        }
        last_unmet = fixable
            .into_iter()
            .chain(not_inputs)
            .map(|unmet| unmet.description)
            .collect();
        if format!("{specs:?}") == before {
            let problem = HarnessProblem::requirements(format!(
                "{}; the inputs already have what the requirements ask for",
                last_unmet.join("; ")
            ));
            return (case(specs, Some(problem)), None);
        }
    }
    let problem = HarnessProblem::requirements(format!(
        "the input requirements were still not met after building the plan \
         {MAX_PROBES} times, changing the inputs each time to meet them: {}",
        last_unmet.join("; ")
    ));
    (case(specs, Some(problem)), None)
}

fn build_inputs(
    specs: &[SourceSpec],
) -> std::result::Result<Vec<Arc<dyn ExecutionPlan>>, HarnessProblem> {
    specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            spec.build_arc().map_err(|e| {
                HarnessProblem::inputs(format!(
                    "generating input {i} failed: {}",
                    e.strip_backtrace()
                ))
            })
        })
        .collect()
}

/// Call `create`, catching errors and panics
fn create(
    factory: &PlanFactory,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
) -> std::result::Result<Arc<dyn ExecutionPlan>, HarnessProblem> {
    match std::panic::catch_unwind(AssertUnwindSafe(|| factory.create(inputs))) {
        Ok(Ok(plan)) => Ok(plan),
        Ok(Err(e)) => Err(HarnessProblem::create(format!(
            "create returned an error: {}",
            e.strip_backtrace()
        ))),
        Err(panic) => Err(HarnessProblem::create(format!(
            "create panicked: {}",
            panic_message(&panic)
        ))),
    }
}

/// `spec` changed to meet `fix`, with hash partitioning into `partitions`
/// partitions
fn apply(spec: SourceSpec, fix: &Fix, partitions: usize) -> SourceSpec {
    let rows = spec.num_rows();
    match fix {
        Fix::Ordering(ordering) => spec.with_ordering(ordering.clone()),
        Fix::Hash(exprs) => spec.with_hash_partitioning(exprs.clone(), partitions, rows),
        Fix::SinglePartition => spec.with_partition_rows(&[rows]),
    }
}

/// The index of `child` in `inputs`, if it is one of them
fn input_index(
    child: &Arc<dyn ExecutionPlan>,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Option<usize> {
    inputs
        .iter()
        .position(|input| std::ptr::addr_eq(Arc::as_ptr(input), Arc::as_ptr(child)))
}

/// `root/0/1` for the path `[0, 1]`
fn path_label(path: &[usize]) -> String {
    std::iter::once("root".to_string())
        .chain(path.iter().map(ToString::to_string))
        .collect::<Vec<_>>()
        .join("/")
}

/// Every hard ordering requirement and distribution requirement of a node of
/// `plan` that its child does not meet. Soft ordering requirements, which a
/// node can work without, are not imposed.
fn unmet_requirements(
    plan: &Arc<dyn ExecutionPlan>,
    inputs: &[Arc<dyn ExecutionPlan>],
) -> Vec<Unmet> {
    let mut unmet = vec![];
    let mut stack = vec![(Arc::clone(plan), vec![])];
    while let Some((node, path)) = stack.pop() {
        let children = node.children();
        let orderings = node.required_input_ordering();
        let distributions = node.input_distribution_requirements();
        let at = |child: usize| {
            format!(
                "{} at {} requires child {child}",
                node.name(),
                path_label(&path)
            )
        };
        for (c, child) in children.iter().enumerate() {
            let input = input_index(child, inputs);
            if let Some(Some(OrderingRequirements::Hard(alternatives))) = orderings.get(c)
                && !alternatives.iter().any(|requirement| {
                    matches!(
                        child
                            .equivalence_properties()
                            .ordering_satisfy_requirement(requirement.clone()),
                        Ok(true)
                    )
                })
                && let Some(ordering) = alternatives.first().and_then(to_ordering)
            {
                unmet.push(Unmet {
                    input,
                    description: format!("{} to be sorted by [{ordering}]", at(c)),
                    fix: Fix::Ordering(ordering),
                });
            }
            let satisfied = distributions
                .child_satisfaction(c, child.as_ref(), ChildSatisfactionOptions::new())
                .is_ok_and(|satisfaction| satisfaction.is_satisfied());
            if !satisfied && let Some(distribution) = distributions.child_distribution(c)
            {
                unmet.extend(distribution_fix(distribution).map(|fix| Unmet {
                    input,
                    description: format!("{} to have {distribution}", at(c)),
                    fix,
                }));
            }
        }
        // Children that must be co-partitioned get the same number of hash
        // partitions. Those that are hash partitioned already meet this;
        // the others are hash partitioned on their keys.
        let child_refs: Vec<&dyn ExecutionPlan> =
            children.iter().map(|child| child.as_ref()).collect();
        if let Ok(unsatisfied) =
            distributions.unsatisfied_co_partitioned_children(node.name(), &child_refs)
        {
            for c in unsatisfied {
                let child = children[c];
                let hash_partitioned = matches!(
                    child.output_partitioning(),
                    datafusion_physical_expr::Partitioning::Hash(..)
                );
                if hash_partitioned {
                    continue;
                }
                if let Some(distribution) = distributions.child_distribution(c)
                    && let Some(Fix::Hash(exprs)) = distribution_fix(distribution)
                {
                    unmet.push(Unmet {
                        input: input_index(child, inputs),
                        description: format!(
                            "{} to be co-partitioned with the other children",
                            at(c)
                        ),
                        fix: Fix::Hash(exprs),
                    });
                }
            }
        }
        for (c, child) in children.into_iter().enumerate().rev() {
            if input_index(child, inputs).is_none() {
                let mut child_path = path.clone();
                child_path.push(c);
                stack.push((Arc::clone(child), child_path));
            }
        }
    }
    unmet
}

/// The fix that meets `distribution`
#[expect(
    deprecated,
    reason = "HashPartitioned is accepted during the KeyPartitioned migration"
)]
fn distribution_fix(distribution: &Distribution) -> Option<Fix> {
    match distribution {
        Distribution::UnspecifiedDistribution => None,
        Distribution::SinglePartition => Some(Fix::SinglePartition),
        Distribution::HashPartitioned(exprs) | Distribution::KeyPartitioned(exprs) => {
            Some(Fix::Hash(exprs.clone()))
        }
    }
}

/// The ordering that meets `requirement`, with default sort options where
/// the requirement has none
fn to_ordering(requirement: &LexRequirement) -> Option<LexOrdering> {
    LexOrdering::new(requirement.iter().map(|requirement| {
        PhysicalSortExpr::new(
            Arc::clone(&requirement.expr),
            requirement.options.unwrap_or_default(),
        )
    }))
}
