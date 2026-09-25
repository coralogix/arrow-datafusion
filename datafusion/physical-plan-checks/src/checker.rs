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

//! [`PlanCheck`] trait and the [`PlanChecker`] that runs checks over a plan.

use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;

use crate::checks;
use crate::report::{Finding, Report, Violation};

/// A single property that every [`ExecutionPlan`] node should satisfy.
///
/// Checks are run on one node at a time. A check may inspect the node's
/// children (for example to compare statistics), but must only report problems
/// with the node itself: the [`PlanChecker`] visits every node of the tree, so
/// each child is checked separately.
///
/// See `CHECKS.md` in this crate for the catalog of checks.
pub trait PlanCheck: Debug + Send + Sync {
    /// Catalog code of the check, such as `A1`
    fn code(&self) -> &'static str;

    /// Stable, unique name of the check, such as `equal_cardinality_num_rows`.
    /// Used to allow (skip) a check with [`PlanChecker::allow`].
    fn name(&self) -> &'static str;

    /// Check a single node and return any problems found.
    ///
    /// Returning an `Err` means the check itself could not run and aborts the
    /// whole [`PlanChecker::check`] call. Problems with the plan, including
    /// errors returned by the plan's own methods, should be reported as
    /// [`Finding`]s instead.
    fn check_node(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>>;
}

/// Runs a set of [`PlanCheck`]s over every node of an [`ExecutionPlan`] tree.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_physical_plan::ExecutionPlan;
/// # use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
/// use datafusion_physical_plan_checks::PlanChecker;
/// use datafusion_physical_plan_checks::fixtures::MockSourceExec;
///
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
/// let source = MockSourceExec::new(schema).with_exact_partition_num_rows(&[10, 20]);
/// let plan: Arc<dyn ExecutionPlan> =
///     Arc::new(CoalescePartitionsExec::new(Arc::new(source)));
///
/// let report = PlanChecker::new().check(&plan).unwrap();
/// report.assert_no_invariant_violations();
/// ```
#[derive(Debug)]
pub struct PlanChecker {
    checks: Vec<Arc<dyn PlanCheck>>,
    allowed: HashSet<String>,
}

impl Default for PlanChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanChecker {
    /// Create a checker with all built-in checks
    pub fn new() -> Self {
        Self {
            checks: checks::all_checks(),
            allowed: HashSet::new(),
        }
    }

    /// Create a checker with no checks. Add checks with [`Self::with_check`].
    pub fn empty() -> Self {
        Self {
            checks: vec![],
            allowed: HashSet::new(),
        }
    }

    /// Add a check, such as a check specific to a user defined plan
    pub fn with_check(mut self, check: Arc<dyn PlanCheck>) -> Self {
        self.checks.push(check);
        self
    }

    /// Skip the check with the given [`PlanCheck::name`]
    pub fn allow(mut self, check_name: impl Into<String>) -> Self {
        self.allowed.insert(check_name.into());
        self
    }

    /// The checks this checker runs, including allowed (skipped) checks
    pub fn checks(&self) -> &[Arc<dyn PlanCheck>] {
        &self.checks
    }

    /// Run all checks that are not allowed against every node in `plan`
    pub fn check(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        let mut violations = vec![];
        let mut path = vec![];
        self.check_recursive(plan, &mut path, &mut violations)?;
        Ok(Report::new(violations))
    }

    /// Run all checks that are not allowed against the root node of `plan`
    /// only, without visiting its children
    pub fn check_node(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        let mut violations = vec![];
        self.check_single(plan, &[], &mut violations)?;
        Ok(Report::new(violations))
    }

    fn check_recursive(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        path: &mut Vec<usize>,
        violations: &mut Vec<Violation>,
    ) -> Result<()> {
        self.check_single(plan, path, violations)?;
        for (i, child) in plan.children().into_iter().enumerate() {
            path.push(i);
            self.check_recursive(child, path, violations)?;
            path.pop();
        }
        Ok(())
    }

    fn check_single(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        path: &[usize],
        violations: &mut Vec<Violation>,
    ) -> Result<()> {
        for check in &self.checks {
            if self.allowed.contains(check.name()) {
                continue;
            }
            for finding in check.check_node(plan)? {
                violations.push(Violation {
                    code: check.code(),
                    check: check.name(),
                    severity: finding.severity,
                    node: plan.name().to_string(),
                    path: path.to_vec(),
                    message: finding.message,
                });
            }
        }
        Ok(())
    }
}
