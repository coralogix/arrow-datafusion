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

//! [`FactoryReport`]: the findings of every case of a factory, grouped.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use datafusion_physical_plan::ExecutionPlan;

use super::{AllowedCheck, Case};
use crate::{Report, Severity, Violation};

/// A problem that kept the harness from testing a case, or from testing it
/// as the factory asked. These are problems of the test setup, not of the
/// plan, but they mean the plan was not checked, so
/// [`FactoryReport::assert_no_invariant_violations`] fails on them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessProblem {
    /// Code, such as `H1`, in the same style as the catalog codes of checks
    pub code: &'static str,
    /// Stable name, such as `create_succeeds`
    pub name: &'static str,
    /// Human readable description of the problem
    pub message: String,
}

impl HarnessProblem {
    /// H1: [`PlanFactory::create`] returned an error or panicked
    ///
    /// [`PlanFactory::create`]: crate::harness::PlanFactory::create
    pub(crate) fn create(message: String) -> Self {
        Self {
            code: "H1",
            name: "create_succeeds",
            message,
        }
    }

    /// H2: the input requirements of the plan could not be met
    pub(crate) fn requirements(message: String) -> Self {
        Self {
            code: "H2",
            name: "input_requirements_met",
            message,
        }
    }

    /// H3: an input could not be generated from its spec
    pub(crate) fn inputs(message: String) -> Self {
        Self {
            code: "H3",
            name: "inputs_generated",
            message,
        }
    }

    /// H4: an allowed check does not exist
    pub(crate) fn allowed(message: String) -> Self {
        Self {
            code: "H4",
            name: "allowed_check_exists",
            message,
        }
    }
}

/// What a [`FindingGroup`] found
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaseFinding {
    /// A violation the checker found in the plan of a case
    Violation(Violation),
    /// A problem that kept the harness from testing a case
    Harness(HarnessProblem),
}

/// Findings that are the same in several cases of a factory: the same
/// severity, check, node path and node name, and the same message apart from
/// numbers (see [`FactoryReport`]). Holds the finding of the first case it
/// occurred in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindingGroup {
    /// The finding, with the message of the first case
    pub finding: CaseFinding,
    /// The indexes of the cases it occurred in, in order
    pub cases: Vec<usize>,
}

impl FindingGroup {
    /// The violation, if the finding is one
    pub fn violation(&self) -> Option<&Violation> {
        match &self.finding {
            CaseFinding::Violation(violation) => Some(violation),
            CaseFinding::Harness(_) => None,
        }
    }

    /// The harness problem, if the finding is one
    pub fn harness_problem(&self) -> Option<&HarnessProblem> {
        match &self.finding {
            CaseFinding::Harness(problem) => Some(problem),
            CaseFinding::Violation(_) => None,
        }
    }
}

/// The result of one case: its plan and the checker's report, or nothing if
/// the case has a [`Case::problem`]
#[derive(Debug, Clone)]
pub struct CaseRun {
    /// The case
    pub case: Case,
    /// The plan built for the case
    pub plan: Option<Arc<dyn ExecutionPlan>>,
    /// The checker's report on the plan
    pub report: Option<Report>,
}

/// The result of checking every case of a [`PlanFactory`] with a
/// [`PlanHarness`].
///
/// Findings are grouped across cases: a finding of one case is the same as
/// a finding of another if both have the same severity, check, node path
/// and node name, and the same message once every number is replaced, lists
/// and ranges of numbers are collapsed, and a trailing summary of other
/// places or runs (`(also ...)`) is removed. The first case's message is
/// kept. When a case has several findings that are the same in this sense,
/// such as one per partition, the first of them is grouped with the first
/// of another case, and so on. Groups are ordered by node, in plan traversal
/// order, and then by the first case they occurred in. Findings that only
/// occur in some cases are often the most interesting ones, so the report
/// names the cases of every group that does not occur in every case.
///
/// [`PlanFactory`]: crate::harness::PlanFactory
/// [`PlanHarness`]: crate::harness::PlanHarness
#[derive(Debug, Clone)]
pub struct FactoryReport {
    name: String,
    cases: Vec<Case>,
    plans: Vec<Option<String>>,
    allowed: Vec<AllowedCheck>,
    groups: Vec<FindingGroup>,
}

impl FactoryReport {
    /// Group the findings of `runs`, and the harness problems of the factory
    /// that do not belong to a case
    pub(crate) fn new(
        name: String,
        allowed: Vec<AllowedCheck>,
        factory_problems: Vec<HarnessProblem>,
        runs: Vec<CaseRun>,
    ) -> Self {
        let all_cases: Vec<usize> = runs.iter().map(|run| run.case.index()).collect();
        let mut groups: Vec<FindingGroup> = factory_problems
            .into_iter()
            .map(|problem| FindingGroup {
                finding: CaseFinding::Harness(problem),
                cases: all_cases.clone(),
            })
            .collect();
        // The first position among the findings of their case, for ordering
        let mut first_seen: Vec<(usize, usize)> = vec![(0, 0); groups.len()];
        let mut index: HashMap<(String, usize), usize> = HashMap::new();
        for run in &runs {
            let findings: Vec<CaseFinding> = match (run.case.problem(), &run.report) {
                (Some(problem), _) => vec![CaseFinding::Harness(problem.clone())],
                (None, Some(report)) => report
                    .violations()
                    .iter()
                    .cloned()
                    .map(CaseFinding::Violation)
                    .collect(),
                (None, None) => vec![],
            };
            let mut occurrences: HashMap<String, usize> = HashMap::new();
            for (position, finding) in findings.into_iter().enumerate() {
                let key = group_key(&finding);
                let occurrence = occurrences.entry(key.clone()).or_default();
                let slot = (key, *occurrence);
                *occurrence += 1;
                match index.get(&slot) {
                    Some(&group) => groups[group].cases.push(run.case.index()),
                    None => {
                        index.insert(slot, groups.len());
                        first_seen.push((run.case.index(), position));
                        groups.push(FindingGroup {
                            finding,
                            cases: vec![run.case.index()],
                        });
                    }
                }
            }
        }
        // Harness problems first, then by node path (plan traversal order),
        // then by first appearance
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by(|&a, &b| {
            let rank = |group: &FindingGroup| match &group.finding {
                CaseFinding::Harness(_) => (0, vec![]),
                CaseFinding::Violation(violation) => (1, violation.path.clone()),
            };
            rank(&groups[a])
                .cmp(&rank(&groups[b]))
                .then(first_seen[a].cmp(&first_seen[b]))
        });
        let mut slots: Vec<Option<FindingGroup>> = groups.into_iter().map(Some).collect();
        let groups = order
            .into_iter()
            .map(|i| slots[i].take().expect("each group is taken once"))
            .collect();

        let plans = runs
            .iter()
            .map(|run| {
                run.plan.as_ref().map(|plan| {
                    datafusion_physical_plan::displayable(plan.as_ref())
                        .indent(true)
                        .to_string()
                })
            })
            .collect();
        Self {
            name,
            cases: runs.into_iter().map(|run| run.case).collect(),
            plans,
            allowed,
            groups,
        }
    }

    /// The name of the factory
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The cases that were derived, in order
    pub fn cases(&self) -> &[Case] {
        &self.cases
    }

    /// The display of the plan built for the case at `index`, if it was built
    pub fn plan(&self, index: usize) -> Option<&str> {
        self.plans.get(index).and_then(|plan| plan.as_deref())
    }

    /// The checks skipped for every case, with the reasons
    pub fn allowed(&self) -> &[AllowedCheck] {
        &self.allowed
    }

    /// Every group of findings, harness problems first, then in plan
    /// traversal order
    pub fn groups(&self) -> &[FindingGroup] {
        &self.groups
    }

    /// Groups whose finding is a violation
    pub fn violations(&self) -> impl Iterator<Item = &FindingGroup> {
        self.groups
            .iter()
            .filter(|group| group.violation().is_some())
    }

    /// Groups whose finding is a [`Severity::Invariant`] violation
    pub fn invariant_violations(&self) -> impl Iterator<Item = &FindingGroup> {
        self.groups.iter().filter(|group| {
            group
                .violation()
                .is_some_and(|v| v.severity == Severity::Invariant)
        })
    }

    /// Groups whose finding is a [`Severity::Lint`] violation
    pub fn lints(&self) -> impl Iterator<Item = &FindingGroup> {
        self.groups.iter().filter(|group| {
            group
                .violation()
                .is_some_and(|v| v.severity == Severity::Lint)
        })
    }

    /// Groups whose finding is a harness problem
    pub fn harness_problems(&self) -> impl Iterator<Item = &FindingGroup> {
        self.groups
            .iter()
            .filter(|group| group.harness_problem().is_some())
    }

    /// Groups of violations reported by the check with the given name
    pub fn for_check<'a>(
        &'a self,
        check: &'a str,
    ) -> impl Iterator<Item = &'a FindingGroup> {
        self.groups
            .iter()
            .filter(move |group| group.violation().is_some_and(|v| v.check == check))
    }

    /// Returns true if there are no findings at all
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Panics if any case has a [`Severity::Invariant`] violation, or could
    /// not be tested because of a harness problem
    pub fn assert_no_invariant_violations(&self) {
        assert!(
            self.invariant_violations().next().is_none()
                && self.harness_problems().next().is_none(),
            "ExecutionPlan invariant violations or harness problems found:\n{self}"
        );
    }

    /// Panics if there is any finding, including lints
    pub fn assert_clean(&self) {
        assert!(
            self.is_empty(),
            "ExecutionPlan check violations found:\n{self}"
        );
    }

    /// `all cases`, or the names of the cases at `indexes`
    fn case_names(&self, indexes: &[usize]) -> String {
        if indexes.len() == self.cases.len() {
            return "all cases".to_string();
        }
        let names: Vec<&str> = indexes
            .iter()
            .filter_map(|i| self.cases.get(*i))
            .map(Case::name)
            .collect();
        format!("cases: {}", names.join(", "))
    }
}

/// What makes two findings the same across cases. See [`FactoryReport`].
fn group_key(finding: &CaseFinding) -> String {
    match finding {
        CaseFinding::Violation(v) => format!(
            "{} {} {} {:?} {}: {}",
            v.severity,
            v.code,
            v.check,
            v.path,
            v.node,
            normalize(&v.message)
        ),
        CaseFinding::Harness(problem) => {
            format!(
                "{} {}: {}",
                problem.code,
                problem.name,
                normalize(&problem.message)
            )
        }
    }
}

/// `message` without a trailing `(also ...)` summary, with every number
/// replaced by `#`, and lists (`#, #`) and ranges (`#-#`) of numbers
/// collapsed to one `#`. Digits inside single quotes are kept, since findings
/// quote names such as columns (`'c1'`), and findings about different names
/// are different findings.
fn normalize(message: &str) -> String {
    let message = match message.find(" (also ") {
        Some(end) => &message[..end],
        None => message,
    };
    let mut normalized = String::with_capacity(message.len());
    let mut quoted = false;
    for c in message.chars() {
        if c == '\'' {
            quoted = !quoted;
        }
        if c.is_ascii_digit() && !quoted {
            if !normalized.ends_with('#') {
                normalized.push('#');
            }
        } else {
            normalized.push(c);
        }
    }
    loop {
        let collapsed = normalized.replace("#, #", "#").replace("#-#", "#");
        if collapsed == normalized {
            return normalized;
        }
        normalized = collapsed;
    }
}

impl fmt::Display for FactoryReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.name)?;
        for case in &self.cases {
            writeln!(f, "  case {} {}", case.index(), case.description())?;
        }
        if let Some(plan) = self.plans.iter().flatten().next() {
            let first = self
                .plans
                .iter()
                .position(Option::is_some)
                .expect("a plan exists");
            writeln!(f, "  plan of case {first}:")?;
            for line in plan.lines() {
                writeln!(f, "    {line}")?;
            }
        }
        for allowed in &self.allowed {
            writeln!(f, "  allowed {}: {}", allowed.check, allowed.reason)?;
        }
        if self.groups.is_empty() {
            return write!(f, "no findings");
        }
        for (i, group) in self.groups.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            let cases = self.case_names(&group.cases);
            match &group.finding {
                CaseFinding::Violation(v) => {
                    let path = if v.path.is_empty() {
                        "root".to_string()
                    } else {
                        let parts: Vec<String> =
                            v.path.iter().map(ToString::to_string).collect();
                        format!("root/{}", parts.join("/"))
                    };
                    write!(
                        f,
                        "[{}] {} {} at {} ({path}) [{cases}]: {}",
                        v.severity, v.code, v.check, v.node, v.message
                    )?;
                }
                CaseFinding::Harness(problem) => write!(
                    f,
                    "[harness] {} {} [{cases}]: {}",
                    problem.code, problem.name, problem.message
                )?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::normalize;

    #[test]
    fn normalize_messages() {
        assert_eq!(
            normalize(
                "partition 12 statistics report null_count Exact(9) for column 'l_k' \
                 (also 6 more false null_count statistics: 'l_k' in partitions 0-2)"
            ),
            "partition # statistics report null_count Exact(#) for column 'l_k'"
        );
        assert_eq!(
            normalize("streams of partitions [0, 1, 2] of child 0"),
            normalize("streams of partitions [0] of child 1")
        );
        assert_eq!(normalize("Exact(Int32(8))"), "Exact(Int#(#))");
        // Words are kept, so findings about different columns or
        // statistics differ
        assert_ne!(
            normalize("num_rows Exact(10) for column 'a'"),
            normalize("num_rows Exact(10) for column 'b'")
        );
        // Including names that differ only in their digits
        assert_ne!(
            normalize("num_rows Exact(10) for column 'c1'"),
            normalize("num_rows Exact(10) for column 'c2'")
        );
        assert_eq!(
            normalize("column '__row_id_0' has Exact(Utf8(\"s012\"))"),
            "column '__row_id_0' has Exact(Utf#(\"s#\"))"
        );
    }
}
