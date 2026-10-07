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

//! Tests that each static check reports deliberately broken plans and stays
//! quiet for correct ones.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::stats::Precision;
use datafusion_functions_aggregate::min_max::min_udaf;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::{
    Column, DynamicFilterPhysicalExpr, col, lit,
};
use datafusion_physical_expr::{
    ConstExpr, EquivalenceProperties, LexOrdering, Partitioning, PhysicalExpr,
    PhysicalSortExpr,
};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::{DisplayFormatType, ExecutionPlan};
use datafusion_physical_plan_checks::fixtures::{SourceSpec, StatisticsPrecision};
use datafusion_physical_plan_checks::{CheckKind, Report, Severity};

use crate::common::{
    ConfigurableExec, Effect, checker, checker_of, exact_source, inexact_source,
    messages, schema, source, summary,
};

fn check(plan: &Arc<dyn ExecutionPlan>) -> Report {
    checker_of(&[CheckKind::Static]).check(plan).unwrap()
}

/// Run only the check named `name`
fn check_with(name: &str, plan: &Arc<dyn ExecutionPlan>) -> Report {
    checker(&[name]).check(plan).unwrap()
}

fn ordering_on_a() -> LexOrdering {
    LexOrdering::new(vec![PhysicalSortExpr::new_default(
        col("a", &schema()).unwrap(),
    )])
    .unwrap()
}

#[test]
fn correct_passthrough_is_clean() {
    let plan = ConfigurableExec::new(exact_source(100)).build();
    check(&plan).assert_clean();

    let plan = ConfigurableExec::new(inexact_source(100)).build();
    check(&plan).assert_clean();
}

#[test]
fn equal_cardinality_with_different_exact_num_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&plan);
    // The node has one partition, which reports the overall row count too
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "equal_cardinality_num_rows"); 2]
    );
    assert_eq!(report.violations[0].node, "ConfigurableExec");
    assert!(report.violations[0].path.is_empty());
}

#[test]
fn equal_cardinality_upgrades_precision() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .num_rows(Precision::Exact(100))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "equal_cardinality_num_rows"); 2]
    );
}

#[test]
fn equal_cardinality_downgrades_precision() {
    let plan = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Inexact(100))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "equal_cardinality_num_rows"); 2]
    );
}

#[test]
fn equal_cardinality_with_different_estimate() {
    let plan = ConfigurableExec::new(inexact_source(100))
        .num_rows(Precision::Inexact(10))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "equal_cardinality_num_rows"); 2]
    );
}

#[test]
fn equal_cardinality_per_partition() {
    let source: Arc<dyn ExecutionPlan> = source(&[40, 0, 60], StatisticsPrecision::Exact);

    // The node computes the statistics of each partition from the same input
    // partition, so partition 2 must have the 60 rows of input partition 2
    let plan = ConfigurableExec::new(Arc::clone(&source))
        .num_rows(Precision::Exact(90))
        .partition_num_rows(vec![
            Precision::Exact(40),
            Precision::Exact(0),
            Precision::Exact(50),
        ])
        .build();
    assert_eq!(
        messages(&check_with("equal_cardinality_num_rows", &plan)),
        vec![
            "cardinality_effect() is Equal, but the overall num_rows is Exact(90) for \
             the input with num_rows Exact(100)",
            "cardinality_effect() is Equal, but the partition 2 num_rows is Exact(50) \
             for input partition 2 with num_rows Exact(60)",
        ]
    );

    // The only partition of a node with one output partition has every input
    // row
    let mut exec =
        ConfigurableExec::new(Arc::clone(&source)).num_rows(Precision::Exact(90));
    exec.coalesce = true;
    assert_eq!(
        messages(&check_with("equal_cardinality_num_rows", &exec.build())),
        vec![
            "cardinality_effect() is Equal, but the overall num_rows is Exact(90) for \
             the input with num_rows Exact(100)",
            "cardinality_effect() is Equal, but the partition 0 num_rows is Exact(90) \
             for the input with num_rows Exact(100)",
        ]
    );

    // A repartition computes the statistics of each partition from all input
    // rows, so which input rows a partition has is unknown
    let repartition: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(source, Partitioning::RoundRobinBatch(4)).unwrap(),
    );
    check(&repartition).assert_clean();
}

#[test]
fn fetch_with_equal_cardinality() {
    let plan = ConfigurableExec::new(exact_source(100))
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "fetch_not_equal_cardinality")]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn fetch_bounds_overall_num_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(11))
        .build();
    // 11 rows is within the LowerEqual bound of 100, so only the fetch bound
    // is violated
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "fetch_bounds_num_rows")]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Inexact(11))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "fetch_bounds_num_rows")]
    );
}

#[test]
fn fetch_bounds_per_partition_num_rows() {
    let source: Arc<dyn ExecutionPlan> = source(&[50, 50], StatisticsPrecision::Exact);

    // Two partitions with fetch 10 can produce up to 20 rows overall
    let plan = ConfigurableExec::new(Arc::clone(&source))
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(20))
        .partition_num_rows(vec![Precision::Exact(10), Precision::Exact(10)])
        .build();
    check(&plan).assert_clean();

    let plan = ConfigurableExec::new(source)
        .effect(Effect::LowerEqual)
        .fetch(10)
        .num_rows(Precision::Exact(20))
        .partition_num_rows(vec![Precision::Exact(5), Precision::Exact(15)])
        .build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "fetch_bounds_num_rows")]
    );
    assert!(
        report.violations[0].message.contains("partition 1"),
        "{report}"
    );
}

#[test]
fn lower_equal_produces_more_rows() {
    // Reported for the overall statistics and for the only partition
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::LowerEqual)
        .num_rows(Precision::Exact(101))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "cardinality_effect_bounds_num_rows"); 2]
    );

    let plan = ConfigurableExec::new(inexact_source(100))
        .effect(Effect::LowerEqual)
        .num_rows(Precision::Inexact(101))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "cardinality_effect_bounds_num_rows"); 2]
    );
}

#[test]
fn greater_equal_produces_fewer_rows() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::GreaterEqual)
        .num_rows(Precision::Exact(99))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "cardinality_effect_bounds_num_rows"); 2]
    );

    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::GreaterEqual)
        .num_rows(Precision::Exact(1000))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn cardinality_effect_bounds_per_partition() {
    let source: Arc<dyn ExecutionPlan> = source(&[40, 0, 60], StatisticsPrecision::Exact);

    // Partition 1 has more rows than input partition 1, although the overall
    // row count and the sum of the partitions are consistent
    let plan = ConfigurableExec::new(Arc::clone(&source))
        .effect(Effect::LowerEqual)
        .num_rows(Precision::Exact(100))
        .partition_num_rows(vec![
            Precision::Exact(30),
            Precision::Exact(10),
            Precision::Exact(60),
        ])
        .build();
    let report = check(&plan);
    assert_eq!(
        messages(&report),
        vec![
            "cardinality_effect() is LowerEqual, but the partition 1 num_rows is \
             Exact(10) for input partition 1 with num_rows Exact(0)"
        ],
        "{report}"
    );

    let plan = ConfigurableExec::new(source)
        .effect(Effect::GreaterEqual)
        .num_rows(Precision::Exact(100))
        .partition_num_rows(vec![
            Precision::Exact(50),
            Precision::Exact(0),
            Precision::Exact(50),
        ])
        .build();
    let report = check(&plan);
    assert_eq!(
        messages(&report),
        vec![
            "cardinality_effect() is GreaterEqual, but the partition 2 num_rows is \
             Exact(50) while partition 2 of input 0 has num_rows Exact(60)"
        ],
        "{report}"
    );

    // Each partition of a union is a partition of one of its inputs
    let union = UnionExec::try_new(vec![
        source_with_rows(&[40, 0, 60]),
        source_with_rows(&[30]),
    ])
    .unwrap();
    check(&union).assert_clean();
}

fn source_with_rows(rows: &[usize]) -> Arc<dyn ExecutionPlan> {
    source(rows, StatisticsPrecision::Exact)
}

#[test]
fn unknown_cardinality_is_not_checked() {
    let plan = ConfigurableExec::new(exact_source(100))
        .effect(Effect::Unknown)
        .num_rows(Precision::Exact(12345))
        .build();
    check(&plan).assert_clean();
}

#[test]
fn per_child_lengths() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.maintains_input_order_len = Some(2);
    let plan = exec.build();
    // The default `check_invariants` also verifies this length
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "per_child_lengths"),
            (Severity::Invariant, "check_invariants"),
        ]
    );
}

#[test]
fn check_invariants_error() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.invariants_error = true;
    let plan = exec.build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "check_invariants")]
    );
}

#[test]
fn statistics_error() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.stats_error = true;
    let plan = exec.build();
    // One violation for the overall statistics and one for partition 0
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "statistics_shape"),
            (Severity::Invariant, "statistics_shape"),
        ]
    );
}

#[test]
fn statistics_error_is_reported_on_the_failing_node_only() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.stats_error = true;
    let parent = ConfigurableExec::new(exec.build()).build();
    let report = check(&parent);
    assert!(!report.violations.is_empty());
    assert!(
        report.violations.iter().all(|v| v.path == vec![0]),
        "{report}"
    );
}

#[test]
fn statistics_column_count() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.drop_column_stats = 1;
    let plan = exec.build();
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "statistics_shape"),
            (Severity::Invariant, "statistics_shape"),
        ]
    );
}

#[test]
fn partition_statistics_sum() {
    let source: Arc<dyn ExecutionPlan> = source(&[10, 20], StatisticsPrecision::Exact);

    // Partition 1 also has more rows than input partition 1, although the node
    // reports cardinality_effect() Equal
    let plan = ConfigurableExec::new(Arc::clone(&source))
        .partition_num_rows(vec![Precision::Exact(10), Precision::Exact(25)])
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![
            (Severity::Invariant, "equal_cardinality_num_rows"),
            (Severity::Invariant, "partition_statistics_sum")
        ]
    );

    let plan = ConfigurableExec::new(Arc::clone(&source))
        .effect(Effect::Unknown)
        .num_rows(Precision::Inexact(30))
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Lint, "partition_statistics_sum")]
    );

    let plan = ConfigurableExec::new(source)
        .effect(Effect::Unknown)
        .num_rows(Precision::Exact(30))
        .partition_num_rows(vec![Precision::Absent, Precision::Exact(31)])
        .build();
    assert_eq!(
        summary(&check(&plan)),
        vec![(Severity::Invariant, "partition_statistics_sum")]
    );
}

#[test]
fn statistics_ignore_inputs() {
    let mut exec = ConfigurableExec::new(exact_source(100));
    exec.skip_child_stats = true;
    let plan = exec.build();
    let report = check(&plan);
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "statistics_ignore_inputs")]
    );
    assert!(
        report.violations[0]
            .message
            .contains("child_stats_requests() skips the input"),
        "{report}"
    );
}

#[test]
fn allowed_checks_are_skipped() {
    let plan = ConfigurableExec::new(exact_source(100))
        .fetch(10)
        .num_rows(Precision::Exact(10))
        .build();
    checker_of(&[CheckKind::Static])
        .allow("fetch_not_equal_cardinality")
        .check(&plan)
        .unwrap()
        .assert_clean();
}

#[test]
#[should_panic(expected = "the checker has no check named 'no_such_check'")]
fn allowing_an_unknown_check_panics() {
    let _ = checker_of(&[CheckKind::Static]).allow("no_such_check");
}

#[test]
fn violations_are_attributed_to_the_offending_node() {
    let broken = ConfigurableExec::new(exact_source(100))
        .num_rows(Precision::Exact(50))
        .build();
    let plan = ConfigurableExec::new(broken)
        .num_rows(Precision::Exact(50))
        .build();
    let report = check(&plan);
    assert_eq!(
        report.to_string(),
        "[invariant] equal_cardinality_num_rows at ConfigurableExec (root/0): \
         cardinality_effect() is Equal, but the overall num_rows is Exact(50) for the \
         input with num_rows Exact(100)
[invariant] equal_cardinality_num_rows at ConfigurableExec (root/0): \
         cardinality_effect() is Equal, but the partition 0 num_rows is Exact(50) for \
         the input with num_rows Exact(100)"
    );
}

#[test]
fn check_invariants_when_the_plan_can_be_executed() {
    let not_executable = |input| {
        let mut exec = ConfigurableExec::new(input);
        exec.executable_invariants_error = true;
        exec.required_ordering = Some(ordering_on_a());
        exec.build()
    };
    let sorted = SourceSpec::new(schema())
        .with_ordering(ordering_on_a())
        .build_arc()
        .unwrap();
    let report = check(&not_executable(sorted));
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "check_invariants")]
    );
    let message = messages(&report)[0];
    assert!(
        message.starts_with(
            "check_invariants(Executable) failed, although the children meet the \
             node's ordering, distribution and co-partitioning requirements:"
        ) && message.contains("ConfigurableExec cannot be executed"),
        "{report}"
    );

    // The input does not meet the required ordering, so the plan cannot be
    // executed whatever the node does
    check(&not_executable(exact_source(100))).assert_clean();
}

#[test]
fn schema_that_differs_from_the_equivalence_properties() {
    let with_schema = |schema: Schema| {
        let mut exec = ConfigurableExec::new(exact_source(10));
        exec.claimed_schema = Some(Arc::new(schema));
        exec.build()
    };
    let renamed = with_schema(Schema::new(vec![Field::new("b", DataType::Int32, false)]));
    let report = check_with("schema_consistency", &renamed);
    assert_eq!(
        messages(&report),
        vec![
            "field 0 of schema() is 'b' Int32 NOT NULL, but the equivalence properties \
             have 'a' Int32 NOT NULL"
        ]
    );
    assert_eq!(
        summary(&report),
        vec![(Severity::Invariant, "schema_consistency")]
    );

    let wider = with_schema(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
    ]));
    assert_eq!(
        messages(&check_with("schema_consistency", &wider)),
        vec!["schema() has 2 fields, but the schema of the equivalence properties has 1"]
    );

    // Metadata the equivalence properties do not have is a lint
    let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
    let with_metadata = with_schema(Schema::new_with_metadata(
        vec![Field::new("a", DataType::Int32, false)],
        metadata,
    ));
    let report = check_with("schema_consistency", &with_metadata);
    assert_eq!(
        summary(&report),
        vec![(Severity::Lint, "schema_consistency")]
    );

    // A node that builds its schema from the schema of its input, and its
    // equivalence properties from the input's, inherits the difference. It is
    // only reported on the input.
    let mut parent = ConfigurableExec::new(Arc::clone(&renamed));
    parent.claimed_schema = Some(renamed.schema());
    let report = check_with("schema_consistency", &parent.build());
    assert_eq!(report.violations.len(), 1, "{report}");
    assert_eq!(report.violations[0].path, vec![0]);
}

#[test]
fn columns_that_do_not_refer_to_the_schema() {
    let check = |exec: ConfigurableExec| {
        messages(&check_with("expression_column_refs", &exec.build()))
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>()
    };
    let column = |name: &str, index: usize| {
        Arc::new(Column::new(name, index)) as Arc<dyn PhysicalExpr>
    };
    let schema = schema();

    // An ordering on a column past the end of the schema, and on a column
    // whose index is another field's
    let mut exec = ConfigurableExec::new(exact_source(10));
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    eq_properties.add_ordering([PhysicalSortExpr::new_default(column("a", 1))]);
    eq_properties.add_ordering([PhysicalSortExpr::new_default(column("b", 0))]);
    exec.claimed_eq_properties = Some(eq_properties);
    assert_eq!(
        check(exec),
        vec![
            "the ordering [a@1 ASC] refers to column 'a'@1, but schema() has 1 fields",
            "the ordering [b@0 ASC] refers to column 'b'@0, but field 0 of schema() is \
             'a'",
        ]
    );

    // An equivalence class, a constant and a hash partitioning
    let mut exec = ConfigurableExec::new(exact_source(10));
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(&schema));
    eq_properties
        .add_equal_conditions(column("a", 0), column("x", 0))
        .unwrap();
    eq_properties
        .add_constants([ConstExpr::from(column("c", 5))])
        .unwrap();
    exec.claimed_eq_properties = Some(eq_properties);
    exec.claimed_partitioning = Some(Partitioning::Hash(vec![column("a", 2)], 1));
    let problems = check(exec);
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(
        problems[0].starts_with("the equivalence class")
            && problems[0]
                .ends_with("refers to column 'x'@0, but field 0 of schema() is 'a'"),
        "{problems:?}"
    );
    assert!(
        problems[1].starts_with("the equivalence class")
            && problems[1].ends_with("refers to column 'c'@5, but schema() has 1 fields"),
        "{problems:?}"
    );
    assert_eq!(
        problems[2],
        "the output partitioning Hash([a@2], 1) refers to column 'a'@2, but schema() \
         has 1 fields"
    );

    // An expression the node evaluates on its input
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.expressions = vec![column("a", 0), column("z", 0)];
    assert_eq!(
        check(exec),
        vec![
            "the expression z@0 visited by apply_expressions refers to column 'z'@0, but \
             field 0 of the input schema is 'a'"
        ]
    );
}

#[test]
fn final_aggregate_expressions_refer_to_the_input_of_the_partial_aggregate() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let input = SourceSpec::new(Arc::clone(&schema))
        .with_partition_rows(&[10, 20])
        .build_arc()
        .unwrap();
    let min_v = Arc::new(
        AggregateExprBuilder::new(min_udaf(), vec![col("v", &schema).unwrap()])
            .schema(Arc::clone(&schema))
            .alias("min(v)")
            .build()
            .unwrap(),
    );
    let group_by =
        PhysicalGroupBy::new_single(vec![(col("g", &schema).unwrap(), "g".to_string())]);
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        group_by,
        vec![Arc::clone(&min_v)],
        vec![None],
        input,
        Arc::clone(&schema),
    )
    .unwrap();
    let final_group_by = partial.group_expr().as_final();
    let coalesce = Arc::new(CoalescePartitionsExec::new(Arc::new(partial)));
    let plan: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Final,
            final_group_by,
            vec![min_v],
            vec![None],
            coalesce,
            schema,
        )
        .unwrap(),
    );
    check_with("expression_column_refs", &plan).assert_clean();
}

#[test]
fn dynamic_filters_that_apply_expressions_does_not_visit() {
    let filter = || {
        Arc::new(DynamicFilterPhysicalExpr::new(
            vec![col("a", &schema()).unwrap()],
            lit(true),
        )) as Arc<dyn PhysicalExpr>
    };
    let dynamic_filter = filter();
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.dynamic_expressions = vec![Arc::clone(&dynamic_filter)];
    let report = check_with("dynamic_expressions_visited", &exec.clone().build());
    assert_eq!(
        messages(&report),
        vec![
            "dynamic_expressions_produced() returned DynamicFilter [ empty ], but no \
             expression visited by apply_expressions has its expression id; include \
             produced dynamic filters in apply_expressions"
        ]
    );

    // Visited inside another expression, as a filter whose predicate includes
    // it: a copy with new children keeps the expression id
    let remapped = Arc::clone(&dynamic_filter)
        .with_new_children(vec![col("a", &schema()).unwrap()])
        .unwrap();
    exec.expressions = vec![Arc::new(
        datafusion_physical_expr::expressions::NotExpr::new(remapped),
    )];
    check_with("dynamic_expressions_visited", &exec.build()).assert_clean();

    // An expression without an expression id, which the default
    // check_invariants reports too
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.dynamic_expressions = vec![col("a", &schema()).unwrap()];
    exec.expressions = exec.dynamic_expressions.clone();
    assert_eq!(
        summary(&check(&exec.build())),
        vec![
            (Severity::Invariant, "check_invariants"),
            (Severity::Invariant, "dynamic_expressions_visited"),
        ]
    );
}

#[test]
fn dynamic_filters_that_survive_reset_state() {
    let mut exec = ConfigurableExec::new(exact_source(10));
    exec.dynamic_expressions = vec![Arc::new(DynamicFilterPhysicalExpr::new(
        vec![col("a", &schema()).unwrap()],
        lit(true),
    ))];
    exec.expressions = exec.dynamic_expressions.clone();
    let report = check_with("dynamic_expressions_reset", &exec.clone().build());
    assert_eq!(
        messages(&report),
        vec![
            "after reset_state, dynamic_expressions_produced() still returns \
             DynamicFilter [ empty ], with the expression id of a dynamic expression the \
             node produced before the reset, so it shares the state that executing the \
             node updates; create new dynamic filters in reset_state"
        ]
    );

    exec.reset_dynamic_expressions = true;
    check(&exec.build()).assert_clean();

    // Built-in plans that produce dynamic filters
    let input = SourceSpec::new(schema())
        .with_partition_rows(&[10, 20])
        .build_arc()
        .unwrap();
    let topk: Arc<dyn ExecutionPlan> =
        Arc::new(SortExec::new(ordering_on_a(), Arc::clone(&input)).with_fetch(Some(5)));
    let min_a = AggregateExprBuilder::new(min_udaf(), vec![col("a", &schema()).unwrap()])
        .schema(schema())
        .alias("min(a)")
        .build()
        .unwrap();
    let min: Arc<dyn ExecutionPlan> = Arc::new(
        AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(vec![]),
            vec![Arc::new(min_a)],
            vec![None],
            input,
            schema(),
        )
        .unwrap(),
    );
    for plan in [topk, min] {
        assert_eq!(plan.dynamic_expressions_produced().len(), 1);
        checker(&["dynamic_expressions_visited", "dynamic_expressions_reset"])
            .check(&plan)
            .unwrap()
            .assert_clean();
    }
}

#[test]
fn displays_that_fail() {
    let display = |exec: ConfigurableExec| {
        let report = check_with("display_no_panic", &exec.build());
        assert!(
            report
                .violations
                .iter()
                .all(|v| v.severity == Severity::Invariant),
            "{report}"
        );
        messages(&report)
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let exec = || ConfigurableExec::new(exact_source(10));

    let mut unnamed = exec();
    unnamed.name = "";
    assert_eq!(
        display(unnamed),
        vec!["name() is empty; return the name of the plan, such as \"FilterExec\""]
    );

    let mut verbose_panics = exec();
    verbose_panics.display_panics = Some(DisplayFormatType::Verbose);
    assert_eq!(
        display(verbose_panics),
        vec![
            "displaying the node with fmt_as(DisplayFormatType::Verbose) panicked: \
             ConfigurableExec cannot be displayed as Verbose"
        ]
    );

    let mut default_fails = exec();
    default_fails.display_error = Some(DisplayFormatType::Default);
    assert_eq!(
        display(default_fails),
        vec![
            "displaying the node with fmt_as(DisplayFormatType::Default) returned \
             fmt::Error, which makes to_string() panic, although writing to a String \
             cannot fail"
        ]
    );

    // The tree renderer formats every node with `TreeRender`, so it fails for
    // the node and for its parents. Only the node is reported.
    let mut tree_panics = exec();
    tree_panics.display_panics = Some(DisplayFormatType::TreeRender);
    let parent = ConfigurableExec::new(tree_panics.build()).build();
    let report = check_with("display_no_panic", &parent);
    assert_eq!(
        messages(&report),
        vec![
            "displaying the node with fmt_as(DisplayFormatType::TreeRender) panicked: \
             ConfigurableExec cannot be displayed as TreeRender",
            "displaying the node with the tree renderer panicked: ConfigurableExec \
             cannot be displayed as TreeRender",
        ]
    );
    assert!(
        report.violations.iter().all(|v| v.path == vec![0]),
        "{report}"
    );
}
