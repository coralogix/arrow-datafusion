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

//! [`PlanCheck`] and the [`PlanChecker`] that runs checks over a plan.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;

use super::checks;
use crate::report::{Finding, Report, Violation};

/// A property that every [`ExecutionPlan`] node should satisfy.
///
/// Checks run on one node at a time. A check may inspect the node's children
/// (for example to compare statistics), but must only report problems with
/// the node itself: the [`PlanChecker`] visits every node of the tree, so each
/// child is checked separately. Checks only call the node's own methods and
/// those of its children; they do not execute plans.
///
/// See `docs/physical_plan/CHECKS.md` in this crate for the catalog of checks.
///
/// # Example
/// ```
/// use datafusion_property_tests::Finding;
/// use datafusion_property_tests::physical_plan::{PlanCheck, PlanChecker};
///
/// let no_empty_names = PlanCheck {
///     name: "no_empty_names",
///     check: |node| {
///         Ok(if node.name().is_empty() {
///             vec![Finding::invariant("name() is empty")]
///         } else {
///             vec![]
///         })
///     },
/// };
/// let checker = PlanChecker::new().with_check(no_empty_names);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PlanCheck {
    /// Stable, unique name, such as `equal_cardinality_num_rows`, used in
    /// reports and to skip the check with [`PlanChecker::allow`]
    pub name: &'static str,
    /// Check one node and return the problems found.
    ///
    /// Returning an `Err` means the check itself could not run, and aborts the
    /// whole [`PlanChecker::check`] call. Problems with the plan, including
    /// errors returned by the plan's own methods, are [`Finding`]s.
    pub check: fn(&Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>>,
}

/// Runs a set of [`PlanCheck`]s over every node of an [`ExecutionPlan`] tree.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_physical_plan::ExecutionPlan;
/// # use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
/// use datafusion_property_tests::physical_plan::PlanChecker;
/// use datafusion_property_tests::physical_plan::fixtures::SourceSpec;
///
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
/// let source = SourceSpec::new(schema).with_partition_rows(&[10, 20]).build_arc()?;
/// let plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(source));
///
/// let report = PlanChecker::new().check(&plan)?;
/// report.assert_no_invariant_violations();
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
#[derive(Debug, Clone)]
pub struct PlanChecker {
    checks: Vec<PlanCheck>,
}

impl Default for PlanChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanChecker {
    /// Create a checker with all built-in checks
    pub fn new() -> Self {
        Self::with_checks(checks::all_checks())
    }

    /// Create a checker that runs `checks`
    pub fn with_checks(checks: Vec<PlanCheck>) -> Self {
        Self { checks }
    }

    /// Add a check, such as a check specific to a user defined plan
    pub fn with_check(mut self, check: PlanCheck) -> Self {
        self.checks.push(check);
        self
    }

    /// Skip the check named `name`.
    ///
    /// # Panics
    ///
    /// If the checker has no check named `name`, which is usually a typo.
    pub fn allow(mut self, name: &str) -> Self {
        assert!(
            self.checks.iter().any(|check| check.name == name),
            "the checker has no check named '{name}'"
        );
        self.checks.retain(|check| check.name != name);
        self
    }

    /// Run every check against every node in `plan`
    pub fn check(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        let mut violations = vec![];
        self.check_recursive(plan, &mut vec![], &mut violations)?;
        Ok(Report { violations })
    }

    fn check_recursive(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        path: &mut Vec<usize>,
        violations: &mut Vec<Violation>,
    ) -> Result<()> {
        for check in &self.checks {
            for finding in (check.check)(plan)? {
                violations.push(Violation {
                    check: check.name,
                    severity: finding.severity,
                    node: plan.name().to_string(),
                    path: path.clone(),
                    message: finding.message,
                });
            }
        }
        for (i, child) in plan.children().into_iter().enumerate() {
            path.push(i);
            self.check_recursive(child, path, violations)?;
            path.pop();
        }
        Ok(())
    }
}
