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

//! Types describing the outcome of running checks against a plan.

use std::fmt;

/// How serious a [`Violation`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// The plan breaks a guarantee that DataFusion relies on. Optimizer rules
    /// or parent operators that trust the reported property can produce wrong
    /// results.
    Invariant,
    /// The plan is correct but reports weaker properties than it could, so
    /// DataFusion misses optimizations.
    Lint,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Invariant => write!(f, "invariant"),
            Severity::Lint => write!(f, "lint"),
        }
    }
}

/// A single problem found by a check, before it is attributed to a plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// How serious the problem is
    pub severity: Severity,
    /// Human readable description of the problem
    pub message: String,
}

impl Finding {
    /// Create a [`Severity::Invariant`] finding
    pub fn invariant(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Invariant,
            message: message.into(),
        }
    }

    /// Create a [`Severity::Lint`] finding
    pub fn lint(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Lint,
            message: message.into(),
        }
    }
}

/// A [`Finding`] attributed to a check and a node in the plan tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Catalog code of the check, such as `A1` (see `CHECKS.md`)
    pub code: &'static str,
    /// Stable name of the check, such as `equal_cardinality_num_rows`
    pub check: &'static str,
    /// How serious the problem is
    pub severity: Severity,
    /// [`ExecutionPlan::name`] of the offending node
    ///
    /// [`ExecutionPlan::name`]: datafusion_physical_plan::ExecutionPlan::name
    pub node: String,
    /// Child indexes from the root to the offending node. Empty for the root.
    pub path: Vec<usize>,
    /// Human readable description of the problem
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = if self.path.is_empty() {
            "root".to_string()
        } else {
            let parts: Vec<String> = self.path.iter().map(ToString::to_string).collect();
            format!("root/{}", parts.join("/"))
        };
        write!(
            f,
            "[{}] {} {} at {} ({path}): {}",
            self.severity, self.code, self.check, self.node, self.message
        )
    }
}

/// The result of running a [`PlanChecker`] against a plan.
///
/// [`PlanChecker`]: crate::PlanChecker
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    violations: Vec<Violation>,
}

impl Report {
    /// Create a report from a list of violations
    pub fn new(violations: Vec<Violation>) -> Self {
        Self { violations }
    }

    /// All violations, in plan traversal order
    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }

    /// Violations with [`Severity::Invariant`]
    pub fn invariant_violations(&self) -> impl Iterator<Item = &Violation> {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Invariant)
    }

    /// Violations with [`Severity::Lint`]
    pub fn lints(&self) -> impl Iterator<Item = &Violation> {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Lint)
    }

    /// Violations reported by the check with the given name
    pub fn for_check<'a>(
        &'a self,
        check: &'a str,
    ) -> impl Iterator<Item = &'a Violation> {
        self.violations.iter().filter(move |v| v.check == check)
    }

    /// Returns true if no violations were found
    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }

    /// Panics if the report contains any [`Severity::Invariant`] violation
    pub fn assert_no_invariant_violations(&self) {
        assert!(
            self.invariant_violations().next().is_none(),
            "ExecutionPlan invariant violations found:\n{self}"
        );
    }

    /// Panics if the report contains any violation, including lints
    pub fn assert_clean(&self) {
        assert!(
            self.is_empty(),
            "ExecutionPlan check violations found:\n{self}"
        );
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.violations.is_empty() {
            return write!(f, "no violations");
        }
        for (i, violation) in self.violations.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{violation}")?;
        }
        Ok(())
    }
}
