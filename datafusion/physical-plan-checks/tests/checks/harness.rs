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

//! Tests of the harness: deriving cases from factories by probing their
//! input requirements, and grouping the findings of all cases.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UInt64Type};
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{JoinType, NullEquality, Result, internal_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{
    Distribution, LexOrdering, LexRequirement, OrderingRequirements, PhysicalExpr,
    PhysicalSortExpr, PhysicalSortRequirement,
};
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::limit::GlobalLimitExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, InputDistributionRequirements, Partitioning, PlanProperties,
    ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs, StatisticsContext,
    collect_partitioned,
};
use datafusion_physical_plan_checks::fixtures::{SourceSpec, StatisticsPrecision};
use datafusion_physical_plan_checks::harness::{
    MAX_PROBES, PlanFactory, PlanHarness, Profile, ROW_ID_RANGE,
};
use datafusion_physical_plan_checks::{PlanChecker, Severity};

use crate::common::{ConfigurableExec, Effect, FetchMode};

type Plan = Arc<dyn ExecutionPlan>;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
    ]))
}

fn spec() -> SourceSpec {
    SourceSpec::new(schema()).with_num_rows(60)
}

fn ordering_on(column: &str, input: &Plan) -> Result<LexOrdering> {
    Ok(LexOrdering::new(vec![PhysicalSortExpr::new_default(col(
        column,
        &input.schema(),
    )?)])
    .unwrap())
}

/// A factory for a plan with one input generated from [`spec`]
fn one_input<F>(name: &str, create: F) -> PlanFactory
where
    F: Fn(Plan) -> Result<Plan> + Send + Sync + 'static,
{
    PlanFactory::new(name, vec![spec()], move |mut inputs| {
        create(inputs.remove(0))
    })
}

/// A harness that only derives cases, with the static checks
fn static_harness() -> PlanHarness {
    PlanHarness::new().with_checker(PlanChecker::static_only())
}

/// The ids of the row id column `name` of `input`
fn row_ids(input: &Plan, name: &str) -> Vec<u64> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let partitions = runtime
        .block_on(collect_partitioned(
            Arc::clone(input),
            Arc::new(TaskContext::default()),
        ))
        .unwrap();
    partitions
        .iter()
        .flatten()
        .flat_map(|batch: &RecordBatch| {
            let index = batch.schema().index_of(name).unwrap();
            batch
                .column(index)
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

/// What a [`RequirementExec`] requires of its input
#[derive(Debug, Clone, Copy)]
enum Rule {
    /// Hash partitioning on `a`, and once the input is hash partitioned,
    /// ordering on `a` too, as a node whose requirements depend on its input
    HashThenOrdering,
    /// Ordering on `b` while the input is sorted on `a`, and on `a` otherwise,
    /// which no input meets
    Alternating,
    /// A soft ordering requirement on `a`, which the node can work without
    SoftOrdering,
}

/// A pass-through node with the input requirements of a [`Rule`]
#[derive(Debug)]
struct RequirementExec {
    input: Plan,
    rule: Rule,
}

impl RequirementExec {
    fn plan(input: Plan, rule: Rule) -> Plan {
        Arc::new(Self { input, rule })
    }

    fn input_sorted_on(&self, column: &str) -> bool {
        let ordering = ordering_on(column, &self.input).unwrap();
        self.input
            .equivalence_properties()
            .ordering_satisfy(ordering)
            .unwrap()
    }

    fn requirement_on(&self, column: &str) -> LexRequirement {
        let expr = col(column, &self.input.schema()).unwrap();
        LexRequirement::new(vec![PhysicalSortRequirement::new(expr, None)]).unwrap()
    }
}

impl DisplayAs for RequirementExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RequirementExec: {:?}", self.rule)
    }
}

impl ExecutionPlan for RequirementExec {
    fn name(&self) -> &'static str {
        "RequirementExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Plan> {
        vec![&self.input]
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        let hashed = matches!(self.input.output_partitioning(), Partitioning::Hash(..));
        vec![match self.rule {
            Rule::HashThenOrdering => {
                hashed.then(|| OrderingRequirements::new(self.requirement_on("a")))
            }
            Rule::Alternating => {
                let column = if self.input_sorted_on("a") { "b" } else { "a" };
                Some(OrderingRequirements::new(self.requirement_on(column)))
            }
            Rule::SoftOrdering => {
                Some(OrderingRequirements::new_soft(self.requirement_on("a")))
            }
        }]
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        let distribution = match self.rule {
            Rule::HashThenOrdering => Distribution::KeyPartitioned(vec![
                col("a", &self.input.schema()).unwrap(),
            ]),
            Rule::Alternating | Rule::SoftOrdering => {
                Distribution::UnspecifiedDistribution
            }
        };
        InputDistributionRequirements::new(vec![distribution])
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Plan>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Plan> {
        Ok(Self::plan(children.swap_remove(0), self.rule))
    }

    fn with_new_children(self: Arc<Self>, children: Vec<Plan>) -> Result<Plan> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.input.execute(partition, context)
    }
}

#[test]
fn ordering_requirement_sorts_the_input() {
    let factory = one_input("SortPreservingMergeExec", |input| {
        Ok(Arc::new(SortPreservingMergeExec::new(
            ordering_on("a", &input)?,
            input,
        )))
    });
    let cases = static_harness().cases(&factory);
    assert_eq!(cases.len(), 5);
    for case in &cases {
        assert!(case.problem().is_none(), "{:?}", case.problem());
        assert_eq!(case.inputs()[0].ordering().unwrap().to_string(), "a@0 ASC");
    }
    // The input keeps the profile's partitions
    assert_eq!(
        cases[0].inputs()[0].partition_rows(),
        Some(&[20, 0, 40][..])
    );
}

#[test]
fn compatible_base_ordering_is_kept() {
    // The base spec is sorted on (a, b), which meets the requirement on a
    let base_ordering = LexOrdering::new(vec![
        PhysicalSortExpr::new_default(col("a", &schema()).unwrap()),
        PhysicalSortExpr::new_default(col("b", &schema()).unwrap()),
    ])
    .unwrap();
    let factory = PlanFactory::new(
        "SortPreservingMergeExec",
        vec![spec().with_ordering(base_ordering)],
        |inputs| {
            let input = Arc::clone(&inputs[0]);
            Ok(Arc::new(SortPreservingMergeExec::new(
                ordering_on("a", &input)?,
                input,
            )) as Plan)
        },
    );
    let case = &static_harness().cases(&factory)[0];
    assert_eq!(
        case.inputs()[0].ordering().unwrap().to_string(),
        "a@0 ASC, b@1 ASC"
    );

    // A base ordering that does not meet the requirement is replaced
    let factory = PlanFactory::new(
        "SortPreservingMergeExec",
        vec![
            spec().with_ordering(ordering_on("b", &spec().build_arc().unwrap()).unwrap()),
        ],
        |inputs| {
            let input = Arc::clone(&inputs[0]);
            Ok(Arc::new(SortPreservingMergeExec::new(
                ordering_on("a", &input)?,
                input,
            )) as Plan)
        },
    );
    let case = &static_harness().cases(&factory)[0];
    assert_eq!(case.inputs()[0].ordering().unwrap().to_string(), "a@0 ASC");
}

#[test]
fn single_partition_requirement() {
    let factory = one_input("GlobalLimitExec", |input| {
        Ok(Arc::new(GlobalLimitExec::new(input, 0, Some(5))))
    });
    let cases = static_harness().cases(&factory);
    let names: Vec<&str> = cases.iter().map(|case| case.name()).collect();
    // `single partition` gives the same inputs as `default`, so it adds no
    // case
    assert_eq!(
        names,
        [
            "default",
            "inexact statistics",
            "absent statistics",
            "empty input"
        ]
    );
    for case in &cases {
        assert_eq!(case.inputs()[0].partition_count(), 1);
    }
    assert_eq!(cases[0].inputs()[0].partition_rows(), Some(&[60][..]));
}

#[test]
fn duplicate_cases_are_kept_when_they_run_other_checks() {
    // A profile that gives the same inputs as the default profile, but runs
    // different checks, is not dropped
    let factory = one_input("GlobalLimitExec", |input| {
        Ok(Arc::new(GlobalLimitExec::new(input, 0, Some(5))))
    });
    let harness = static_harness().with_profiles(vec![
        Profile::new("without stream experiments").with_partition_weights(&[1]),
        Profile::default_profile(),
    ]);
    assert_eq!(harness.cases(&factory).len(), 2);
    // But one that runs no more checks than an earlier one is
    let harness = static_harness().with_profiles(vec![
        Profile::default_profile(),
        Profile::new("without stream experiments").with_partition_weights(&[1]),
    ]);
    assert_eq!(harness.cases(&factory).len(), 1);
}

fn hash_join(mode: PartitionMode) -> PlanFactory {
    let right = SourceSpec::new(Arc::new(Schema::new(vec![
        Field::new("c", DataType::Int32, false),
        Field::new("d", DataType::Int32, true),
    ])))
    .with_num_rows(90);
    PlanFactory::new("HashJoinExec", vec![spec(), right], move |inputs| {
        let on = vec![(
            col("a", &inputs[0].schema())?,
            col("c", &inputs[1].schema())?,
        )];
        Ok(Arc::new(HashJoinExec::try_new(
            Arc::clone(&inputs[0]),
            Arc::clone(&inputs[1]),
            on,
            None,
            &JoinType::Inner,
            None,
            mode,
            NullEquality::NullEqualsNothing,
            false,
        )?) as Plan)
    })
}

#[test]
fn hash_requirement_and_co_partitioning() {
    // Both inputs are hash partitioned on their keys, into as many
    // partitions as the profile has
    let harness = static_harness().with_profiles(vec![
        Profile::default_profile(),
        Profile::new("5 partitions").with_partition_weights(&[1, 1, 0, 1, 1]),
    ]);
    let cases = harness.cases(&hash_join(PartitionMode::Partitioned));
    for (case, partitions) in cases.iter().zip([3, 5]) {
        assert!(case.problem().is_none(), "{:?}", case.problem());
        let [left, right] = case.inputs() else {
            panic!("two inputs");
        };
        let (left_exprs, left_n) = left.hash_partitioning().unwrap();
        let (right_exprs, right_n) = right.hash_partitioning().unwrap();
        assert_eq!(left_exprs[0].to_string(), "a@0");
        assert_eq!(right_exprs[0].to_string(), "c@0");
        assert_eq!((left_n, right_n), (partitions, partitions));
        assert_eq!((left.num_rows(), right.num_rows()), (60, 90));
    }
    assert_eq!(
        cases[0].description(),
        "default: input 0 hash [a@0] into 3 partitions, 60 rows; input 1 hash [c@0] \
         into 3 partitions, 90 rows; exact statistics; seeds 0, 1"
    );

    // A collect left join puts its build side in one partition, and leaves
    // the probe side as the profile lays it out
    let cases = static_harness().cases(&hash_join(PartitionMode::CollectLeft));
    assert_eq!(cases[0].inputs()[0].partition_rows(), Some(&[60][..]));
    assert_eq!(
        cases[0].inputs()[1].partition_rows(),
        Some(&[30, 0, 60][..])
    );
}

#[test]
fn requirements_that_depend_on_the_input_are_probed_again() {
    let creates = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&creates);
    let factory = one_input("RequirementExec", move |input| {
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(RequirementExec::plan(input, Rule::HashThenOrdering))
    });
    let harness = static_harness().with_profiles(vec![Profile::default_profile()]);
    let case = &harness.cases(&factory)[0];
    assert!(case.problem().is_none(), "{:?}", case.problem());
    let input = &case.inputs()[0];
    assert_eq!(input.hash_partitioning().unwrap().0[0].to_string(), "a@0");
    assert_eq!(input.ordering().unwrap().to_string(), "a@0 ASC");
    // The first plan asks for hash partitioning, the second, on hash
    // partitioned input, for an ordering, and the third is satisfied
    assert_eq!(creates.load(Ordering::Relaxed), 3);
}

#[test]
fn soft_ordering_requirements_are_not_imposed() {
    let factory = one_input("RequirementExec", |input| {
        Ok(RequirementExec::plan(input, Rule::SoftOrdering))
    });
    let case = &static_harness().cases(&factory)[0];
    assert!(case.problem().is_none());
    assert!(case.inputs()[0].ordering().is_none());
}

#[test]
fn unsatisfiable_requirements_are_reported() {
    let factory = one_input("RequirementExec", |input| {
        Ok(RequirementExec::plan(input, Rule::Alternating))
    });
    let report = static_harness().check(&factory).unwrap();
    let problems: Vec<_> = report.harness_problems().collect();
    assert_eq!(problems.len(), 1, "{report}");
    let problem = problems[0].harness_problem().unwrap();
    assert_eq!(
        (problem.code, problem.name),
        ("H2", "input_requirements_met")
    );
    assert!(
        problem
            .message
            .contains(&format!("after building the plan {MAX_PROBES} times")),
        "{}",
        problem.message
    );
    assert_eq!(problems[0].cases.len(), report.cases().len());
    // The plan was not checked
    assert_eq!(report.violations().count(), 0);
    assert!(report.plan(0).is_none());
    assert!(
        report
            .to_string()
            .contains("[harness] H2 input_requirements_met [all cases]")
    );
}

#[test]
fn requirements_on_children_that_are_not_inputs_are_reported() {
    // The projection keeps the ordering of its input, but the harness only
    // sorts inputs to meet the requirements of the nodes directly above them
    let factory = one_input("SortPreservingMergeExec over a projection", |input| {
        let schema = input.schema();
        let exprs = vec![
            (col("a", &schema)?, "a".to_string()),
            (col("b", &schema)?, "b".to_string()),
        ];
        let projection: Plan = Arc::new(ProjectionExec::try_new(exprs, input)?);
        Ok(Arc::new(SortPreservingMergeExec::new(
            ordering_on("a", &projection)?,
            projection,
        )))
    });
    let case = &static_harness().cases(&factory)[0];
    let problem = case.problem().unwrap();
    assert_eq!(problem.code, "H2");
    assert!(
        problem.message.contains(
            "SortPreservingMergeExec at root requires child 0 to be sorted by [a@0 ASC]; \
             the child is not an input"
        ),
        "{}",
        problem.message
    );
}

#[test]
fn case_descriptions() {
    let factory = one_input("FilterExec", |input| {
        let b = col("b", &input.schema())?;
        let predicate =
            Arc::new(datafusion_physical_expr::expressions::IsNotNullExpr::new(b));
        Ok(Arc::new(FilterExec::try_new(predicate, input)?))
    });
    let descriptions: Vec<String> = static_harness()
        .cases(&factory)
        .iter()
        .map(|case| format!("{} {}", case.index(), case.description()))
        .collect();
    assert_eq!(
        descriptions,
        [
            "0 default: input 0 rows [20, 0, 40]; exact statistics; seed 0",
            "1 single partition: input 0 rows [60]; exact statistics; seed 0",
            "2 inexact statistics: input 0 rows [20, 0, 40]; inexact statistics; seed 0",
            "3 absent statistics: input 0 rows [20, 0, 40]; absent statistics; seed 0",
            "4 empty input: input 0 rows [0, 0, 0]; exact statistics; seed 0",
        ]
    );
}

#[test]
fn case_inputs_follow_the_profile() {
    let factory = one_input("FilterExec", |input| {
        let b = col("b", &input.schema())?;
        let predicate =
            Arc::new(datafusion_physical_expr::expressions::IsNotNullExpr::new(b));
        Ok(Arc::new(FilterExec::try_new(predicate, input)?))
    });
    let cases = static_harness().cases(&factory);
    for (case, precision) in cases.iter().zip([
        StatisticsPrecision::Exact,
        StatisticsPrecision::Exact,
        StatisticsPrecision::Inexact,
        StatisticsPrecision::Absent,
        StatisticsPrecision::Exact,
    ]) {
        let input = &case.build_inputs().unwrap()[0];
        let statistics = StatisticsContext::new()
            .compute(input.as_ref(), &StatisticsArgs::new())
            .unwrap();
        let rows = case.inputs()[0].num_rows();
        let expected = match precision {
            StatisticsPrecision::Exact => Precision::Exact(rows),
            StatisticsPrecision::Inexact => Precision::Inexact(rows),
            StatisticsPrecision::Absent => Precision::Absent,
        };
        assert_eq!(statistics.num_rows, expected, "{}", case.description());
    }
    // The extended profiles add more seeds, partition counts and rows
    let extended = static_harness().with_profiles(Profile::extended());
    let descriptions: Vec<String> = extended
        .cases(&factory)
        .iter()
        .skip(5)
        .map(|case| case.description())
        .collect();
    assert_eq!(
        descriptions,
        [
            "seed 1: input 0 rows [20, 0, 40]; exact statistics; seed 1000",
            "seed 2: input 0 rows [20, 0, 40]; exact statistics; seed 2000",
            "2 partitions: input 0 rows [30, 30]; exact statistics; seed 0",
            "5 partitions: input 0 rows [17, 8, 0, 25, 10]; exact statistics; seed 0",
            "large input: input 0 rows [160, 0, 320]; exact statistics; seed 0",
        ]
    );
}

#[test]
fn row_id_columns() {
    let factory = hash_join(PartitionMode::CollectLeft);
    let case = &static_harness().cases(&factory)[0];
    let inputs = case.build_inputs().unwrap();
    let left_schema = inputs[0].schema();
    let right_schema = inputs[1].schema();
    let names = |schema: &SchemaRef| -> Vec<String> {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    };
    assert_eq!(names(&left_schema), ["a", "b", "__row_id_0"]);
    assert_eq!(names(&right_schema), ["c", "d", "__row_id_1"]);

    // Ids are unique, and each input has its own range
    let left = row_ids(&inputs[0], "__row_id_0");
    let right = row_ids(&inputs[1], "__row_id_1");
    assert_eq!(left, (0..60).collect::<Vec<u64>>());
    assert_eq!(
        right,
        (ROW_ID_RANGE..ROW_ID_RANGE + 90).collect::<Vec<u64>>()
    );

    // The plan binds its keys by name, so they refer to the key columns of
    // the inputs that have row ids
    let plan = factory.create(inputs).unwrap();
    assert_eq!(
        plan.schema().fields().len(),
        left_schema.fields().len() + right_schema.fields().len()
    );
}

#[test]
fn shared_row_id_name() {
    let factory = PlanFactory::new("UnionExec", vec![spec(), spec()], |inputs| {
        datafusion_physical_plan::union::UnionExec::try_new(inputs)
    })
    .with_shared_row_id_name();
    let case = &static_harness().cases(&factory)[0];
    let inputs = case.build_inputs().unwrap();
    assert_eq!(inputs[0].schema(), inputs[1].schema());
    assert_eq!(
        case.inputs()[1].row_id_column(),
        Some(("__row_id", ROW_ID_RANGE))
    );

    let plan = factory.create(inputs).unwrap();
    assert!(
        plan.children()
            .iter()
            .all(|child| child.name() == "MockSourceExec")
    );

    // Without it, `UnionExec::try_new` renames the row id column of the
    // second input with a projection, which is then part of the plan
    let factory = PlanFactory::new("UnionExec", vec![spec(), spec()], |inputs| {
        datafusion_physical_plan::union::UnionExec::try_new(inputs)
    });
    let case = &static_harness().cases(&factory)[0];
    let plan = factory.create(case.build_inputs().unwrap()).unwrap();
    assert_eq!(plan.children()[1].name(), "ProjectionExec");
}

#[test]
fn factories_without_inputs() {
    let factory = PlanFactory::new("EmptyExec", vec![], |_| {
        Ok(Arc::new(EmptyExec::new(schema())) as Plan)
    });
    let report = PlanHarness::new().check(&factory).unwrap();
    assert_eq!(report.cases().len(), 1);
    assert_eq!(report.cases()[0].description(), "default: no inputs");
    assert!(report.cases()[0].profile().stream_experiments());
    report.assert_clean();
}

/// A pass-through node over one input that reports `fetch()` 5 and
/// `CardinalityEffect::Equal` (A2 in every case), and `num_rows`
/// `Exact(60)`, which is false when the input is empty (B2 in the empty
/// input case only)
fn fetch_with_equal_effect() -> PlanFactory {
    one_input("ConfigurableExec", |input| {
        let mut exec = ConfigurableExec::new(input)
            .effect(Effect::Equal)
            .num_rows(Precision::Exact(60));
        exec.fetch = Some(5);
        exec.fetch_mode = FetchMode::Ignore;
        Ok(exec.build())
    })
}

#[test]
fn findings_are_grouped_across_cases() {
    let report = static_harness().check(&fetch_with_equal_effect()).unwrap();
    let a2: Vec<_> = report.for_check("fetch_not_equal_cardinality").collect();
    assert_eq!(a2.len(), 1, "{report}");
    assert_eq!(a2[0].cases, [0, 1, 2, 3, 4]);

    let display = report.to_string();
    assert!(
        display.contains(
            "[invariant] A2 fetch_not_equal_cardinality at ConfigurableExec (root) \
             [all cases]: fetch() is Some(5)"
        ),
        "{display}"
    );

    // Findings that only occur in some cases name them. The overall
    // num_rows claim is only false for empty input.
    let report = PlanHarness::new()
        .check(&fetch_with_equal_effect())
        .unwrap();
    let b2: Vec<_> = report.for_check("exact_statistics_hold").collect();
    assert_eq!(b2.len(), 1, "{report}");
    assert_eq!(b2[0].cases, [4]);
    assert!(
        report.to_string().contains(
            "[invariant] B2 exact_statistics_hold at ConfigurableExec (root) [cases: empty \
             input]: overall statistics report num_rows Exact(60), but the output has \
             Exact(0) rows"
        ),
        "{report}"
    );
    // The same claim is an invariant violation when exact and a lint when
    // an estimate, so the two are grouped separately, although the numbers
    // in the messages are the same
    let a3: Vec<(Severity, Vec<usize>)> = report
        .for_check("fetch_bounds_num_rows")
        .filter(|group| group.violation().unwrap().message.contains("partition 0"))
        .map(|group| (group.violation().unwrap().severity, group.cases.clone()))
        .collect();
    assert_eq!(
        a3,
        [(Severity::Invariant, vec![0]), (Severity::Lint, vec![2])],
        "{report}"
    );
}

#[test]
fn findings_in_different_places_are_not_merged() {
    // A3 reports the overall estimate and the estimate of every partition
    // with rows; they are separate findings, grouped separately
    let factory = one_input("FilterExec with fetch", |input| {
        let b = col("b", &input.schema())?;
        let predicate =
            Arc::new(datafusion_physical_expr::expressions::IsNotNullExpr::new(b));
        FilterExec::try_new(predicate, input)?
            .with_fetch(Some(1))
            .ok_or_else(|| datafusion_common::internal_datafusion_err!("no fetch"))
    });
    let report = static_harness().check(&factory).unwrap();
    let a3: Vec<(String, Vec<usize>)> = report
        .for_check("fetch_bounds_num_rows")
        .map(|group| {
            (
                group.violation().unwrap().message.clone(),
                group.cases.clone(),
            )
        })
        .collect();
    assert_eq!(a3.len(), 3, "{report}");
    assert!(a3[0].0.contains("overall"), "{a3:?}");
    assert!(a3[1].0.contains("partition 0"), "{a3:?}");
    assert!(a3[2].0.contains("partition 2"), "{a3:?}");
    // The overall estimate is too large in every case with rows and
    // statistics, the partition estimates only with several partitions
    assert_eq!(a3[0].1, [0, 1, 2]);
    assert_eq!(a3[1].1, [0, 2]);
}

#[test]
fn stream_checks_run_in_the_default_case() {
    // A node that polls its input from a spawned task but reports `Lazy` is
    // reported in the default case, whose profile runs stream experiments,
    // and not in the others, which do not
    let factory = one_input("ConfigurableExec", |input| {
        let mut exec = ConfigurableExec::new(input);
        exec.eager = true;
        Ok(exec.build())
    });
    let report = PlanHarness::new().check(&factory).unwrap();
    let b12: Vec<_> = report.for_check("lazy_evaluation_holds").collect();
    assert_eq!(b12.len(), 1, "{report}");
    assert_eq!(b12[0].cases, [0]);
}

#[test]
fn allowed_checks_with_reasons() {
    let factory = fetch_with_equal_effect()
        .allow("fetch_not_equal_cardinality", "the fetch is only reported");
    let report = static_harness().check(&factory).unwrap();
    assert_eq!(report.for_check("fetch_not_equal_cardinality").count(), 0);
    assert_eq!(report.allowed()[0].reason, "the fetch is only reported");
    assert!(
        report
            .to_string()
            .contains("allowed fetch_not_equal_cardinality: the fetch is only reported")
    );

    // Allowing a check that does not exist is reported
    let factory = fetch_with_equal_effect().allow("no_such_check", "a typo");
    let report = static_harness().check(&factory).unwrap();
    let problems: Vec<_> = report.harness_problems().collect();
    assert_eq!(problems.len(), 1, "{report}");
    assert_eq!(problems[0].harness_problem().unwrap().code, "H4");
}

#[test]
#[should_panic(expected = "allowing a check needs a reason")]
fn allowing_a_check_needs_a_reason() {
    let _ = fetch_with_equal_effect().allow("fetch_not_equal_cardinality", " ");
}

#[test]
fn reproduce_a_single_case() {
    let factory = fetch_with_equal_effect();
    let harness = PlanHarness::new();
    let report = harness.check(&factory).unwrap();
    let empty = report
        .cases()
        .iter()
        .find(|case| case.name() == "empty input")
        .unwrap()
        .index();

    // `run_case` derives the case again and checks it, and finds what the
    // report says about the case
    let run = harness.run_case(&factory, empty).unwrap();
    assert_eq!(run.case.description(), report.cases()[empty].description());
    let violations = run.report.as_ref().unwrap().violations();
    let in_case: Vec<_> = report
        .groups()
        .iter()
        .filter(|group| group.cases.contains(&empty))
        .filter_map(|group| group.violation())
        .collect();
    assert!(!in_case.is_empty());
    assert_eq!(in_case.len(), violations.len());
    for grouped in in_case {
        assert!(
            violations
                .iter()
                .any(|v| v.check == grouped.check && v.path == grouped.path),
            "{grouped}"
        );
    }

    // The specs of the case are enough to rebuild and check it by hand
    let case = &report.cases()[empty];
    let plan = factory.create(case.build_inputs().unwrap()).unwrap();
    let by_hand = harness.checker_for(&factory, case).check(&plan).unwrap();
    assert_eq!(by_hand.violations(), run.report.unwrap().violations());

    assert!(harness.run_case(&factory, 99).is_err());
}

#[test]
fn failing_create_is_reported() {
    let factory = one_input("fails", |_| internal_err!("no plan today"));
    let report = static_harness().check(&factory).unwrap();
    let groups: Vec<_> = report.groups().iter().collect();
    assert_eq!(groups.len(), 1, "{report}");
    let problem = groups[0].harness_problem().unwrap();
    assert_eq!((problem.code, problem.name), ("H1", "create_succeeds"));
    assert!(
        problem.message.contains("no plan today"),
        "{}",
        problem.message
    );
    assert_eq!(groups[0].cases, [0, 1, 2, 3, 4]);

    // A panic is reported the same way
    let factory = one_input("panics", |_| panic!("no plan at all"));
    let report = static_harness().check(&factory).unwrap();
    let problem = report.groups()[0].harness_problem().unwrap();
    assert!(problem.message.contains("create panicked: no plan at all"));

    // So is a factory that only fails for some inputs
    let factory = one_input("fails on several partitions", |input| {
        if input.output_partitioning().partition_count() > 1 {
            return internal_err!("one partition only");
        }
        Ok(input)
    });
    let report = static_harness().check(&factory).unwrap();
    let h1: Vec<_> = report.harness_problems().collect();
    assert_eq!(h1.len(), 1, "{report}");
    assert_eq!(h1[0].cases, [0, 2, 3, 4]);
}

#[test]
#[should_panic(expected = "harness problems found")]
fn harness_problems_fail_the_assertion() {
    let factory = one_input("fails", |_| internal_err!("no plan today"));
    static_harness()
        .check(&factory)
        .unwrap()
        .assert_no_invariant_violations();
}

#[test]
fn invariant_violations_and_lints() {
    let report = static_harness().check(&fetch_with_equal_effect()).unwrap();
    assert!(
        report
            .invariant_violations()
            .all(|group| group.violation().unwrap().severity == Severity::Invariant)
    );
    assert!(report.invariant_violations().next().is_some());
    assert_eq!(report.name(), "ConfigurableExec");
}
