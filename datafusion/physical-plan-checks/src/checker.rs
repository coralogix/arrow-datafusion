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

use std::collections::{BTreeSet, HashSet};
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::ExecutionPlan;

use crate::checks;
use crate::context::{CheckContext, ContextOptions};
use crate::experiments::Experiment;
use crate::report::{Finding, Report, Violation};

/// A single property that every [`ExecutionPlan`] node should satisfy.
///
/// Checks are run on one node at a time. A check may inspect the node's
/// children (for example to compare statistics), but must only report problems
/// with the node itself: the [`PlanChecker`] visits every node of the tree, so
/// each child is checked separately.
///
/// Checks do not execute plans themselves. A check that needs a node's output
/// returns true from [`Self::requires_execution`], and the [`PlanChecker`]
/// executes every node of the plan before running checks and makes the output
/// available through [`CheckContext::output`]. A check that needs to observe
/// how a node drives its input streams lists the [`Experiment`]s it needs in
/// [`Self::experiments`], and reads their results with
/// [`CheckContext::stream_runs`].
///
/// See `CHECKS.md` in this crate for the catalog of checks.
pub trait PlanCheck: Debug + Send + Sync {
    /// Catalog code of the check, such as `A1`
    fn code(&self) -> &'static str;

    /// Stable, unique name of the check, such as `equal_cardinality_num_rows`.
    /// Used to allow (skip) a check with [`PlanChecker::allow`].
    fn name(&self) -> &'static str;

    /// Whether the check reads the output of executing nodes from the
    /// [`CheckContext`]
    fn requires_execution(&self) -> bool {
        false
    }

    /// The stream experiments whose results the check reads from the
    /// [`CheckContext`]. The [`PlanChecker`] runs each experiment requested by
    /// an enabled check on every node with children.
    fn experiments(&self) -> &'static [Experiment] {
        &[]
    }

    /// Check a single node and return any problems found.
    ///
    /// Returning an `Err` means the check itself could not run and aborts the
    /// whole [`PlanChecker::check`] call. Problems with the plan, including
    /// errors returned by the plan's own methods, should be reported as
    /// [`Finding`]s instead.
    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>>;
}

/// Runs a set of [`PlanCheck`]s over every node of an [`ExecutionPlan`] tree.
///
/// If any enabled check requires execution, every node of the plan is executed
/// first, each on its own, and its output is collected. Stream experiments
/// requested by enabled checks run on every node too. Plans under test should
/// therefore be built on inputs that can be executed repeatedly, such as
/// [`MockSourceExec`]. Experiments that control how inputs behave only change
/// `MockSourceExec` leaves.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_physical_plan::ExecutionPlan;
/// # use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
/// use datafusion_physical_plan_checks::PlanChecker;
/// use datafusion_physical_plan_checks::fixtures::SourceSpec;
///
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
/// let source = SourceSpec::new(schema).with_partition_rows(&[10, 20]).build_arc()?;
/// let plan: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(source));
///
/// let report = PlanChecker::new().check(&plan)?;
/// report.assert_no_invariant_violations();
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
///
/// [`MockSourceExec`]: crate::fixtures::MockSourceExec
#[derive(Debug)]
pub struct PlanChecker {
    checks: Vec<Arc<dyn PlanCheck>>,
    allowed: HashSet<String>,
    task_context: Arc<TaskContext>,
    timeout: Duration,
    stream_timeout: Duration,
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

    /// Create a checker with the built-in checks that do not execute the plan
    pub fn static_only() -> Self {
        Self::with_checks(checks::static_checks())
    }

    /// Create a checker with no checks. Add checks with [`Self::with_check`].
    pub fn empty() -> Self {
        Self::with_checks(vec![])
    }

    /// Create a checker that runs `checks`
    pub fn with_checks(checks: Vec<Arc<dyn PlanCheck>>) -> Self {
        Self {
            checks,
            allowed: HashSet::new(),
            task_context: Arc::new(TaskContext::default()),
            timeout: Duration::from_secs(30),
            stream_timeout: Duration::from_secs(2),
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

    /// Set the [`TaskContext`] used to execute plans, for example to change the
    /// batch size
    pub fn with_task_context(mut self, task_context: Arc<TaskContext>) -> Self {
        self.task_context = task_context;
        self
    }

    /// Set the time allowed for executing a single node. A node that takes
    /// longer is reported as failing to execute. Stream experiments on finite
    /// inputs, which should end like a normal execution, use it too. Defaults
    /// to 30 seconds.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the time a stream experiment on inputs that never end may take, and
    /// the time allowed for a node to release streams and memory it no longer
    /// needs. A node that is expected to end or produce output on such inputs,
    /// but does not, is reported after this long. Defaults to 2 seconds.
    pub fn with_stream_timeout(mut self, timeout: Duration) -> Self {
        self.stream_timeout = timeout;
        self
    }

    /// The checks this checker runs, including allowed (skipped) checks
    pub fn checks(&self) -> &[Arc<dyn PlanCheck>] {
        &self.checks
    }

    /// Run all checks that are not allowed against every node in `plan`.
    ///
    /// If any check requires execution, this starts a Tokio runtime to execute
    /// the plan, and so it panics if called from within a Tokio runtime. Use
    /// [`Self::check_async`] in async code.
    pub fn check(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        if !self.requires_execution() {
            return self.run_checks(plan, &CheckContext::default(), true);
        }
        let runtime = runtime()?;
        runtime.block_on(self.check_async(plan))
    }

    /// Run all checks that are not allowed against every node in `plan`,
    /// executing the plan on the current Tokio runtime if needed
    pub async fn check_async(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        let context = self.context(plan).await;
        self.run_checks(plan, &context, true)
    }

    /// Run all checks that are not allowed against the root node of `plan`
    /// only, without reporting problems in its children
    pub fn check_node(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<Report> {
        if !self.requires_execution() {
            return self.run_checks(plan, &CheckContext::default(), false);
        }
        let runtime = runtime()?;
        runtime.block_on(async {
            let context = self.context(plan).await;
            self.run_checks(plan, &context, false)
        })
    }

    fn active_checks(&self) -> impl Iterator<Item = &Arc<dyn PlanCheck>> {
        self.checks
            .iter()
            .filter(|check| !self.allowed.contains(check.name()))
    }

    fn requires_execution(&self) -> bool {
        self.active_checks()
            .any(|check| check.requires_execution() || !check.experiments().is_empty())
    }

    async fn context(&self, plan: &Arc<dyn ExecutionPlan>) -> CheckContext {
        if !self.requires_execution() {
            return CheckContext::default();
        }
        let options = ContextOptions {
            execute: self.active_checks().any(|check| check.requires_execution()),
            experiments: self
                .active_checks()
                .flat_map(|check| check.experiments().iter().copied())
                .collect::<BTreeSet<_>>(),
            task_context: Arc::clone(&self.task_context),
            timeout: self.timeout,
            stream_timeout: self.stream_timeout,
        };
        CheckContext::gather(plan, &options).await
    }

    fn run_checks(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
        recursive: bool,
    ) -> Result<Report> {
        let mut violations = vec![];
        let mut path = vec![];
        self.check_recursive(plan, context, recursive, &mut path, &mut violations)?;
        Ok(Report::new(violations))
    }

    fn check_recursive(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
        recursive: bool,
        path: &mut Vec<usize>,
        violations: &mut Vec<Violation>,
    ) -> Result<()> {
        for check in self.active_checks() {
            for finding in check.check_node(plan, context)? {
                violations.push(Violation {
                    code: check.code(),
                    check: check.name(),
                    severity: finding.severity,
                    node: plan.name().to_string(),
                    path: path.clone(),
                    message: finding.message,
                });
            }
        }
        if recursive {
            for (i, child) in plan.children().into_iter().enumerate() {
                path.push(i);
                self.check_recursive(child, context, recursive, path, violations)?;
                path.pop();
            }
        }
        Ok(())
    }
}

/// A runtime for executing plans from synchronous code
fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}
