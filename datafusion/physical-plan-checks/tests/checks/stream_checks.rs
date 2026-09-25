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

//! Tests of the checks that run stream experiments (boundedness, emission,
//! laziness, resource release and error propagation), using plans that break
//! those properties on purpose.

use std::sync::Arc;
use std::time::Duration;

use datafusion_common::Result;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, EvaluationType,
};
use datafusion_physical_plan::limit::GlobalLimitExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{ExecutionPlan, Partitioning};
use datafusion_physical_plan_checks::fixtures::StatisticsPrecision;
use datafusion_physical_plan_checks::{
    CheckContext, Experiment, Finding, PlanCheck, PlanChecker, Report, Severity, checks,
};

use crate::common::{
    ConfigurableExec, Effect, FetchMode, HoldInput, OnInputError, exact_source, schema,
    source, summary,
};

/// A checker with short timeouts, so that plans that never end are reported
/// quickly
fn checker(checks: Vec<Arc<dyn PlanCheck>>) -> PlanChecker {
    PlanChecker::with_checks(checks)
        .with_timeout(Duration::from_millis(300))
        .with_stream_timeout(Duration::from_millis(300))
}

/// Run only `check`
fn check_with(check: Arc<dyn PlanCheck>, plan: &Arc<dyn ExecutionPlan>) -> Report {
    checker(vec![check]).check(plan).unwrap()
}

fn multi_partition_source() -> Arc<dyn ExecutionPlan> {
    source(&[40, 0, 60], StatisticsPrecision::Exact)
}

#[test]
fn correct_plans_are_clean() {
    let mut plans = vec![];
    for input in [exact_source(100), multi_partition_source()] {
        plans.push(ConfigurableExec::new(Arc::clone(&input)).build());

        // Eager and buffering nodes that report it
        let mut eager = ConfigurableExec::new(Arc::clone(&input));
        eager.eager = true;
        eager.evaluation_type = Some(EvaluationType::Eager);
        plans.push(eager.build());

        let mut buffering = ConfigurableExec::new(Arc::clone(&input));
        buffering.buffer_all = true;
        buffering.emission_type = Some(EmissionType::Final);
        plans.push(buffering.build());

        // A fetch that makes the output bounded, and is reported as such
        let mut limited = ConfigurableExec::new(Arc::clone(&input))
            .fetch(5)
            .effect(Effect::LowerEqual);
        limited.fetch_mode = FetchMode::Enforce;
        limited.boundedness = Some(Boundedness::Bounded);
        plans.push(limited.build());
    }
    // The fetch plan does not apply its fetch to its statistics, so only run
    // the checks that use stream experiments
    let stream_checks: Vec<Arc<dyn PlanCheck>> = vec![
        Arc::new(checks::ExecutionSucceeds),
        Arc::new(checks::BoundednessHolds),
        Arc::new(checks::EmissionTypeHolds),
        Arc::new(checks::LazyEvaluationHolds),
        Arc::new(checks::ResourcesReleased),
        Arc::new(checks::ErrorsPropagate),
    ];
    for plan in plans {
        checker(stream_checks.clone())
            .check(&plan)
            .unwrap()
            .assert_clean();
    }
}

#[test]
fn bounded_claim_that_does_not_hold() {
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.boundedness = Some(Boundedness::Bounded);
    let lying = exec.build();
    let report = check_with(Arc::new(checks::BoundednessHolds), &lying);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "boundedness_holds")]
    );
    assert!(
        report.violations()[0].message.contains("did not end"),
        "{report}"
    );

    // A parent that trusts the child's claim is not reported
    let parent = ConfigurableExec::new(lying).build();
    let report = check_with(Arc::new(checks::BoundednessHolds), &parent);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert_eq!(report.violations()[0].path, vec![0]);
}

#[test]
fn fetch_that_makes_the_output_bounded_is_a_lint() {
    let mut exec = ConfigurableExec::new(multi_partition_source()).fetch(5);
    exec.fetch_mode = FetchMode::Enforce;
    let plan = exec.build();
    let report = check_with(Arc::new(checks::BoundednessHolds), &plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "boundedness_holds")]
    );

    // Not for a node that can drop rows, since its output only ends if enough
    // rows remain
    let mut exec = ConfigurableExec::new(multi_partition_source())
        .fetch(5)
        .effect(Effect::LowerEqual);
    exec.fetch_mode = FetchMode::Enforce;
    let plan = exec.build();
    check_with(Arc::new(checks::BoundednessHolds), &plan).assert_clean();
}

#[test]
fn incremental_claim_that_does_not_hold() {
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.buffer_all = true;
    let plan = exec.build();
    let report = check_with(Arc::new(checks::EmissionTypeHolds), &plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "emission_type_holds")]
    );
    assert!(
        report.violations()[0].message.contains("produced no rows"),
        "{report}"
    );
}

#[test]
fn incremental_node_over_a_final_child_is_not_checked() {
    // The child waits for all of its input, and says so, so the parent
    // cannot produce output before the input ends either
    let mut child = ConfigurableExec::new(multi_partition_source());
    child.buffer_all = true;
    child.emission_type = Some(EmissionType::Final);
    let mut parent = ConfigurableExec::new(child.build());
    parent.emission_type = Some(EmissionType::Incremental);
    let plan = parent.build();
    check_with(Arc::new(checks::EmissionTypeHolds), &plan).assert_clean();
}

#[test]
fn lazy_claim_that_does_not_hold() {
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.eager = true;
    let eager = exec.build();
    let report = check_with(Arc::new(checks::LazyEvaluationHolds), &eager);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "lazy_evaluation_holds")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("partitions [0, 1, 2] of child 0"),
        "{report}"
    );

    // A lazy parent of an eager child only polls the child on demand, even
    // though the child's own input is polled by a spawned task
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.eager = true;
    exec.evaluation_type = Some(EvaluationType::Eager);
    let mut parent = ConfigurableExec::new(exec.build());
    parent.evaluation_type = Some(EvaluationType::Lazy);
    let plan = parent.build();
    check_with(Arc::new(checks::LazyEvaluationHolds), &plan).assert_clean();
}

#[test]
fn input_streams_held_by_the_plan_are_a_lint() {
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.hold_input = HoldInput::InPlan;
    let plan = exec.build();
    let report = check_with(Arc::new(checks::ResourcesReleased), &plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "resources_released")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("only dropped once the plan itself was dropped"),
        "{report}"
    );
}

#[test]
fn input_streams_that_are_never_dropped() {
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.hold_input = HoldInput::Forever;
    let plan = exec.build();
    let report = check_with(Arc::new(checks::ResourcesReleased), &plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "resources_released")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("partitions [0, 1, 2] of child 0 were still alive"),
        "{report}"
    );
}

#[test]
fn memory_that_is_never_released() {
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.leak_memory = true;
    let leaking = exec.build();
    let report = check_with(Arc::new(checks::ResourcesReleased), &leaking);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "resources_released")]
    );
    assert!(
        report.violations()[0]
            .message
            .contains("1024 bytes are still reserved"),
        "{report}"
    );

    // The parent inherits the leak, and is not reported
    let parent = ConfigurableExec::new(leaking).build();
    let report = check_with(Arc::new(checks::ResourcesReleased), &parent);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert_eq!(report.violations()[0].path, vec![0]);
}

#[test]
fn errors_that_do_not_propagate() {
    for (on_error, expected) in [
        (OnInputError::Swallow, "ended without returning an error"),
        (OnInputError::Hang, "neither returned an error nor ended"),
        (
            OnInputError::Panic,
            "panicked: ConfigurableExec got an error",
        ),
    ] {
        let mut exec = ConfigurableExec::new(multi_partition_source());
        exec.on_input_error = on_error;
        let plan = exec.build();
        let report = check_with(Arc::new(checks::ErrorsPropagate), &plan);
        assert_eq!(
            summary(&report),
            vec![(Severity::Invariant, "errors_propagate")],
            "{on_error:?}"
        );
        assert!(
            report.violations()[0].message.contains(expected),
            "{on_error:?}: {report}"
        );
    }

    // A parent of a node that swallows errors never sees one
    let mut exec = ConfigurableExec::new(multi_partition_source());
    exec.on_input_error = OnInputError::Swallow;
    let parent = ConfigurableExec::new(exec.build()).build();
    let report = check_with(Arc::new(checks::ErrorsPropagate), &parent);
    assert_eq!(report.violations().len(), 1, "{report}");
    assert_eq!(report.violations()[0].path, vec![0]);
}

#[test]
fn error_after_the_fetch_is_satisfied_does_not_have_to_propagate() {
    // The spawned task reads the error into its buffer, but the output has
    // enough rows for its fetch before it gets there
    let mut exec = ConfigurableExec::new(exact_source(100))
        .fetch(1)
        .effect(Effect::LowerEqual);
    exec.eager = true;
    exec.fetch_mode = FetchMode::Enforce;
    exec.on_input_error = OnInputError::Swallow;
    let plan = exec.build();
    check_with(Arc::new(checks::ErrorsPropagate), &plan).assert_clean();
}

/// A user defined check that reads the laziness experiment and reports how
/// many input polls were not driven by the node's output
#[derive(Debug)]
struct CountUndemandedPolls;

impl PlanCheck for CountUndemandedPolls {
    fn code(&self) -> &'static str {
        "X1"
    }

    fn name(&self) -> &'static str {
        "count_undemanded_polls"
    }

    fn experiments(&self) -> &'static [Experiment] {
        &[Experiment::Laziness]
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        Ok(context
            .stream_runs(node, Experiment::Laziness)
            .iter()
            .flat_map(|run| run.inputs.iter().flatten())
            .filter(|observation| observation.polls_without_demand > 0)
            .map(|_| Finding::lint("undemanded"))
            .collect())
    }
}

#[test]
fn laziness_experiment_tells_eager_from_lazy_builtin_plans() {
    let schema = schema();
    let repartition: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(
            multi_partition_source(),
            Partitioning::RoundRobinBatch(2),
        )
        .unwrap(),
    );
    let projection: Arc<dyn ExecutionPlan> = Arc::new(
        ProjectionExec::try_new(
            vec![(col("a", &schema).unwrap(), "a".to_string())],
            multi_partition_source(),
        )
        .unwrap(),
    );
    // The repartition polls every input partition from a spawned task
    let report = check_with(Arc::new(CountUndemandedPolls), &repartition);
    assert_eq!(report.violations().len(), 3, "{report}");
    check_with(Arc::new(CountUndemandedPolls), &projection).assert_clean();
}

#[test]
fn limit_that_needs_more_rows_than_the_input_serves_is_not_reported() {
    // Unbounded leaves stall after a limit on the rows they serve. A limit
    // that needs more rows than that to end cannot be shown not to end.
    let large_fetch: Arc<dyn ExecutionPlan> =
        Arc::new(GlobalLimitExec::new(exact_source(40), 0, Some(10_000_000)));
    let large_skip: Arc<dyn ExecutionPlan> =
        Arc::new(GlobalLimitExec::new(exact_source(40), 10_000_000, Some(10)));
    for plan in [large_fetch, large_skip] {
        check_with(Arc::new(checks::BoundednessHolds), &plan).assert_clean();
        check_with(Arc::new(checks::EmissionTypeHolds), &plan).assert_clean();
    }

    // Without a fetch, the limit never ends, whatever the row limit
    let no_fetch: Arc<dyn ExecutionPlan> =
        Arc::new(GlobalLimitExec::new(exact_source(40), 5, None));
    assert_eq!(
        summary(&check_with(Arc::new(checks::BoundednessHolds), &no_fetch)),
        vec![(Severity::Invariant, "boundedness_holds")]
    );
}
