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

//! Testing a plan from a description of how to build it.
//!
//! A [`PlanFactory`] says how to build the plan under test from its inputs.
//! The [`PlanHarness`] generates the inputs, derives the cases worth testing
//! by probing the plan's input requirements, runs a [`PlanChecker`] on the
//! plan of every case, and returns a [`FactoryReport`] that groups the
//! findings of all cases.
//!
//! For each [`Profile`], the harness lays out the inputs as the profile
//! says, builds the plan, and reads what each node directly above an input
//! requires of it (`required_input_ordering` and
//! `input_distribution_requirements`). It then changes the inputs to meet
//! the requirements: an input that must be sorted is sorted, one that must
//! be hash partitioned is hash partitioned into the profile's number of
//! partitions (the same number for every input), and one that must be a
//! single partition gets one. A plan's requirements can depend on its
//! inputs, so the harness builds the plan again and repeats until the
//! requirements are met, at most [`MAX_PROBES`] times. A case whose
//! requirements cannot be met is reported, not tested.
//!
//! [`PlanChecker`] remains the way to check a plan built by hand.

mod case;
mod factory;
mod profile;
mod report;

use std::collections::HashSet;
use std::sync::Arc;

use datafusion_common::{DataFusionError, Result, plan_err};
use datafusion_physical_plan::ExecutionPlan;

pub use case::{Case, MAX_PROBES, ROW_ID_RANGE};
pub use factory::{AllowedCheck, PlanFactory};
pub use profile::Profile;
pub use report::{CaseFinding, CaseRun, FactoryReport, FindingGroup, HarnessProblem};

use crate::PlanChecker;

/// Checks a [`PlanFactory`] on every case derived from its [`Profile`]s.
///
/// See [`PlanFactory`] for an example, and the [module documentation](self)
/// for how cases are derived.
///
/// Plans are executed on a current-thread Tokio runtime, one case after
/// another, so that the findings are deterministic: some operators produce
/// output that depends on how tasks interleave.
#[derive(Debug, Clone)]
pub struct PlanHarness {
    checker: PlanChecker,
    profiles: Vec<Profile>,
}

impl Default for PlanHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanHarness {
    /// A harness that runs every built-in check on the cases of
    /// [`Profile::defaults`]
    pub fn new() -> Self {
        Self {
            checker: PlanChecker::new(),
            profiles: Profile::defaults(),
        }
    }

    /// Run the checks of `checker`, with its settings such as timeouts,
    /// instead of every built-in check. The harness adds the checks allowed
    /// by each factory, and leaves out checks that need stream experiments
    /// in profiles that do not run them.
    pub fn with_checker(mut self, checker: PlanChecker) -> Self {
        self.checker = checker;
        self
    }

    /// Derive cases from `profiles` instead of [`Profile::defaults`], for
    /// factories that do not set their own
    pub fn with_profiles(mut self, profiles: Vec<Profile>) -> Self {
        self.profiles = profiles;
        self
    }

    /// The profiles cases are derived from, for factories that do not set
    /// their own
    pub fn profiles(&self) -> &[Profile] {
        &self.profiles
    }

    /// The profiles for `factory`
    fn profiles_for<'a>(&'a self, factory: &'a PlanFactory) -> &'a [Profile] {
        factory.profiles().unwrap_or(&self.profiles)
    }

    /// The cases of `factory`, in order. Deriving them builds the plan, but
    /// does not execute it.
    pub fn cases(&self, factory: &PlanFactory) -> Vec<Case> {
        case::derive_cases(factory, self.profiles_for(factory))
            .into_iter()
            .map(|(case, _)| case)
            .collect()
    }

    /// Check every case of `factory`.
    ///
    /// Starts a current-thread Tokio runtime to execute the plans, and so
    /// panics if called from within a Tokio runtime. Use
    /// [`Self::check_async`] in async code.
    ///
    /// Returns an error if a check could not run (see [`PlanCheck::check_node`]).
    /// Problems with the factory or its cases are reported in the
    /// [`FactoryReport`].
    ///
    /// [`PlanCheck::check_node`]: crate::PlanCheck::check_node
    pub fn check(&self, factory: &PlanFactory) -> Result<FactoryReport> {
        runtime()?.block_on(self.check_async(factory))
    }

    /// Check every case of `factory` on the current Tokio runtime
    pub async fn check_async(&self, factory: &PlanFactory) -> Result<FactoryReport> {
        let mut runs = vec![];
        for (case, plan) in case::derive_cases(factory, self.profiles_for(factory)) {
            runs.push(self.run(factory, case, plan).await?);
        }
        Ok(FactoryReport::new(
            factory.name().to_string(),
            factory.allowed().to_vec(),
            self.factory_problems(factory),
            runs,
        ))
    }

    /// Derive the case of `factory` at `index` again and check it, for
    /// example to reproduce a finding from a [`FactoryReport`]. See
    /// [`Self::check`] for the runtime.
    pub fn run_case(&self, factory: &PlanFactory, index: usize) -> Result<CaseRun> {
        runtime()?.block_on(self.run_case_async(factory, index))
    }

    /// [`Self::run_case`] on the current Tokio runtime
    pub async fn run_case_async(
        &self,
        factory: &PlanFactory,
        index: usize,
    ) -> Result<CaseRun> {
        let mut cases = case::derive_cases(factory, self.profiles_for(factory));
        if index >= cases.len() {
            return plan_err!(
                "factory '{}' has {} cases, there is no case {index}",
                factory.name(),
                cases.len()
            );
        }
        let (case, plan) = cases.swap_remove(index);
        self.run(factory, case, plan).await
    }

    /// Check the plan of `case`, if it has one
    async fn run(
        &self,
        factory: &PlanFactory,
        case: Case,
        plan: Option<Arc<dyn ExecutionPlan>>,
    ) -> Result<CaseRun> {
        let plan = match plan {
            Some(plan) if case.problem().is_none() => plan,
            _ => {
                return Ok(CaseRun {
                    case,
                    plan: None,
                    report: None,
                });
            }
        };
        let report = self.checker_for(factory, &case).check_async(&plan).await?;
        Ok(CaseRun {
            case,
            plan: Some(plan),
            report: Some(report),
        })
    }

    /// The checker the harness runs on the plan of `case`: the harness's
    /// checker without the checks that need stream experiments, unless the
    /// case's profile runs them, and with the checks `factory` allows
    /// skipped. Use it to check a case rebuilt by hand from its specs.
    pub fn checker_for(&self, factory: &PlanFactory, case: &Case) -> PlanChecker {
        let stream_experiments = case.profile().stream_experiments();
        factory.allowed().iter().fold(
            self.checker.clone().retain_checks(|check| {
                stream_experiments || check.experiments().is_empty()
            }),
            |checker, allowed| checker.allow(allowed.check.clone()),
        )
    }

    /// Problems of `factory` that do not depend on a case
    fn factory_problems(&self, factory: &PlanFactory) -> Vec<HarnessProblem> {
        let names: HashSet<&str> = self
            .checker
            .checks()
            .iter()
            .map(|check| check.name())
            .collect();
        factory
            .allowed()
            .iter()
            .filter(|allowed| !names.contains(allowed.check.as_str()))
            .map(|allowed| {
                HarnessProblem::allowed(format!(
                    "the factory allows '{}', but the checker has no check with that \
                     name",
                    allowed.check
                ))
            })
            .collect()
    }
}

/// A current-thread runtime, so that execution is deterministic
fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}
