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

//! Runs all checks against the `ExecutionPlan`s defined in DataFusion and
//! records the findings in snapshots, one per family of operators: simple
//! operators, aggregates and joins.
//!
//! The snapshots are the list of known violations in the built-in plans. When
//! a plan is fixed, or a new check finds a new problem, a snapshot changes
//! and must be reviewed and updated with `cargo insta review` (or by running
//! the test with `INSTA_UPDATE=always`).
//!
//! Aggregates and joins are built the way the physical planner and the
//! optimizer build them, on inputs that meet their `required_input_ordering`
//! and `input_distribution_requirements`.
//!
//! Known findings are also listed, with their causes, in
//! `IMPLEMENTATION_STATUS.md`, as are checks allowed for specific plans.

use std::fmt::Write;
use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{JoinSide, JoinType, NullEquality, Result, ScalarValue};
use datafusion_expr::{AggregateUDF, Operator};
use datafusion_functions_aggregate::average::avg_udaf;
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::min_max::{max_udaf, min_udaf};
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::aggregate::{AggregateExprBuilder, AggregateFunctionExpr};
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal, col};
use datafusion_physical_expr::{
    LexOrdering, Partitioning, PhysicalExpr, PhysicalSortExpr,
};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, LimitOptions, PhysicalGroupBy,
};
use datafusion_physical_plan::buffer::BufferExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter, JoinOn};
use datafusion_physical_plan::joins::{
    AsOfJoinExec, AsOfMatchExpr, CrossJoinExec, HashJoinExec, NestedLoopJoinExec,
    PartitionMode, PiecewiseMergeJoinExec, SortMergeJoinExec, StreamJoinPartitionMode,
    SymmetricHashJoinExec,
};
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::{ExecutionPlan, displayable};
use datafusion_physical_plan_checks::PlanChecker;
use datafusion_physical_plan_checks::fixtures::{BatchLayout, SourceSpec};

const FETCH: usize = 10;

/// A plan under test
struct Case {
    name: String,
    plan: Arc<dyn ExecutionPlan>,
    /// Checks skipped for this plan. Each one is a known limitation of the
    /// check, explained where the case is built and in
    /// `IMPLEMENTATION_STATUS.md`, never a way to hide a real problem.
    allowed: Vec<&'static str>,
}

impl Case {
    fn new(name: impl Into<String>, plan: Arc<dyn ExecutionPlan>) -> Self {
        Self {
            name: name.into(),
            plan,
            allowed: vec![],
        }
    }

    /// Skip `check` for this plan
    fn allow(mut self, check: &'static str) -> Self {
        self.allowed.push(check);
        self
    }
}

/// Run every check that is not allowed against each case and render the
/// plans and findings.
///
/// Runs on a current-thread runtime (see the tests) so that execution, and
/// therefore the snapshot, is deterministic. Some plans, such as a
/// partitioned TopK `SortExec`, produce output that depends on how partitions
/// interleave.
async fn audit(cases: Vec<Case>) -> Result<String> {
    let mut output = String::new();
    for case in cases {
        let checker = case
            .allowed
            .iter()
            .fold(PlanChecker::new(), |checker, check| checker.allow(*check));
        let report = checker.check_async(&case.plan).await?;
        writeln!(output, "## {}", case.name).unwrap();
        let display = displayable(case.plan.as_ref()).indent(true).to_string();
        for line in display.lines() {
            writeln!(output, "    {line}").unwrap();
        }
        if !case.allowed.is_empty() {
            writeln!(output, "allowed: {}", case.allowed.join(", ")).unwrap();
        }
        writeln!(output, "{report}\n").unwrap();
    }
    Ok(output)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Boolean, false),
        Field::new("c", DataType::Utf8, true),
    ]))
}

fn ordering_on_a() -> Result<LexOrdering> {
    Ok(
        LexOrdering::new(vec![PhysicalSortExpr::new_default(col("a", &schema())?)])
            .unwrap(),
    )
}

/// Random batches of up to 16 rows, including empty batches
const BATCH_LAYOUT: BatchLayout = BatchLayout::Random {
    max_rows: 16,
    empty_batches: true,
};

/// The spec every source starts from: random batches of up to 16 rows,
/// including empty batches, with exact statistics
fn spec() -> SourceSpec {
    SourceSpec::new(schema()).with_batch_layout(BATCH_LAYOUT)
}

/// Three partitions with exact row counts
fn multi_partition_source() -> Result<Arc<dyn ExecutionPlan>> {
    spec().with_partition_rows(&[100, 200, 300]).build_arc()
}

/// Three partitions with exact row counts, each sorted on `a`
fn sorted_multi_partition_source() -> Result<Arc<dyn ExecutionPlan>> {
    spec()
        .with_partition_rows(&[100, 200, 300])
        .with_ordering(ordering_on_a()?)
        .build_arc()
}

/// One partition with an exact row count
fn single_partition_source() -> Result<Arc<dyn ExecutionPlan>> {
    spec().with_partition_rows(&[600]).build_arc()
}

/// One partition with an exact row count, sorted on `a`
fn sorted_single_partition_source() -> Result<Arc<dyn ExecutionPlan>> {
    spec()
        .with_partition_rows(&[600])
        .with_ordering(ordering_on_a()?)
        .build_arc()
}

#[expect(deprecated)]
fn coalesce_batches(
    input: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Arc<dyn ExecutionPlan> {
    use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
    Arc::new(CoalesceBatchesExec::new(input, 8192).with_fetch(fetch))
}

/// The simple operators under test, each with a descriptive name
fn builtin_plans() -> Result<Vec<Case>> {
    let schema = schema();
    let a = col("a", &schema)?;
    let b = col("b", &schema)?;

    let plans: Vec<(&'static str, Arc<dyn ExecutionPlan>)> = vec![
        ("EmptyExec", Arc::new(EmptyExec::new(Arc::clone(&schema)))),
        (
            "PlaceholderRowExec",
            Arc::new(PlaceholderRowExec::new(Arc::clone(&schema))),
        ),
        (
            "ProjectionExec",
            Arc::new(ProjectionExec::try_new(
                vec![(Arc::clone(&a), "a".to_string())],
                multi_partition_source()?,
            )?),
        ),
        (
            "FilterExec",
            Arc::new(FilterExec::try_new(
                Arc::clone(&b),
                multi_partition_source()?,
            )?),
        ),
        (
            "FilterExec with fetch",
            FilterExec::try_new(Arc::clone(&b), multi_partition_source()?)?
                .with_fetch(Some(FETCH))
                .expect("FilterExec supports fetch"),
        ),
        (
            "CoalesceBatchesExec",
            coalesce_batches(multi_partition_source()?, None),
        ),
        (
            "CoalesceBatchesExec with fetch",
            coalesce_batches(multi_partition_source()?, Some(FETCH)),
        ),
        (
            "CoalescePartitionsExec",
            Arc::new(CoalescePartitionsExec::new(multi_partition_source()?)),
        ),
        (
            "CoalescePartitionsExec with fetch",
            Arc::new(
                CoalescePartitionsExec::new(multi_partition_source()?)
                    .with_fetch(Some(FETCH)),
            ),
        ),
        (
            "SortExec",
            Arc::new(SortExec::new(ordering_on_a()?, single_partition_source()?)),
        ),
        (
            "SortExec with fetch",
            Arc::new(
                SortExec::new(ordering_on_a()?, single_partition_source()?)
                    .with_fetch(Some(FETCH)),
            ),
        ),
        (
            "SortExec on a nullable column",
            Arc::new(SortExec::new(
                LexOrdering::new(vec![PhysicalSortExpr::new_default(col("c", &schema)?)])
                    .unwrap(),
                single_partition_source()?,
            )),
        ),
        (
            "SortExec on sorted input",
            Arc::new(SortExec::new(
                ordering_on_a()?,
                sorted_single_partition_source()?,
            )),
        ),
        (
            "SortExec with fetch on sorted input",
            Arc::new(
                SortExec::new(ordering_on_a()?, sorted_single_partition_source()?)
                    .with_fetch(Some(FETCH)),
            ),
        ),
        (
            "SortExec with preserve_partitioning",
            Arc::new(
                SortExec::new(ordering_on_a()?, multi_partition_source()?)
                    .with_preserve_partitioning(true),
            ),
        ),
        (
            "SortExec with preserve_partitioning and fetch",
            Arc::new(
                SortExec::new(ordering_on_a()?, multi_partition_source()?)
                    .with_preserve_partitioning(true)
                    .with_fetch(Some(FETCH)),
            ),
        ),
        (
            "SortPreservingMergeExec",
            Arc::new(SortPreservingMergeExec::new(
                ordering_on_a()?,
                sorted_multi_partition_source()?,
            )),
        ),
        (
            "SortPreservingMergeExec with fetch",
            Arc::new(
                SortPreservingMergeExec::new(
                    ordering_on_a()?,
                    sorted_multi_partition_source()?,
                )
                .with_fetch(Some(FETCH)),
            ),
        ),
        (
            "RepartitionExec round robin",
            Arc::new(RepartitionExec::try_new(
                multi_partition_source()?,
                Partitioning::RoundRobinBatch(4),
            )?),
        ),
        (
            "RepartitionExec hash",
            Arc::new(RepartitionExec::try_new(
                multi_partition_source()?,
                Partitioning::Hash(vec![Arc::clone(&a)], 4),
            )?),
        ),
        (
            "GlobalLimitExec",
            Arc::new(GlobalLimitExec::new(
                single_partition_source()?,
                5,
                Some(FETCH),
            )),
        ),
        (
            "GlobalLimitExec without fetch",
            Arc::new(GlobalLimitExec::new(single_partition_source()?, 5, None)),
        ),
        (
            "LocalLimitExec",
            Arc::new(LocalLimitExec::new(multi_partition_source()?, FETCH)),
        ),
        (
            "UnionExec",
            UnionExec::try_new(vec![
                multi_partition_source()?,
                single_partition_source()?,
            ])?,
        ),
        (
            "BufferExec",
            Arc::new(BufferExec::new(multi_partition_source()?, 1024)),
        ),
        (
            "CooperativeExec",
            Arc::new(CooperativeExec::new(multi_partition_source()?)),
        ),
    ];
    Ok(plans
        .into_iter()
        .map(|(name, plan)| Case::new(name, plan))
        .collect())
}

/// Partitions of the hash repartition between two aggregation phases, and of
/// hash partitioned inputs
const HASH_PARTITIONS: usize = 3;

/// Two grouping keys, one of them nullable, and two value columns
fn aggregate_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int32, false),
        Field::new("h", DataType::Utf8, true),
        Field::new("v", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
    ]))
}

fn aggregate_spec() -> SourceSpec {
    SourceSpec::new(aggregate_schema()).with_batch_layout(BATCH_LAYOUT)
}

fn ordering_on_g() -> Result<LexOrdering> {
    Ok(LexOrdering::new(vec![PhysicalSortExpr::new_default(col(
        "g",
        &aggregate_schema(),
    )?)])
    .unwrap())
}

/// `udaf(arg) AS alias`, with `arg` a column of `schema`
fn aggregate(
    udaf: Arc<AggregateUDF>,
    arg: &str,
    alias: &str,
    schema: &SchemaRef,
) -> Result<Arc<AggregateFunctionExpr>> {
    Ok(Arc::new(
        AggregateExprBuilder::new(udaf, vec![col(arg, schema)?])
            .schema(Arc::clone(schema))
            .alias(alias)
            .build()?,
    ))
}

/// `count(v), sum(v), min(v), max(v), avg(f), sum(f)`
fn all_aggregates() -> Result<Vec<Arc<AggregateFunctionExpr>>> {
    let schema = aggregate_schema();
    Ok(vec![
        aggregate(count_udaf(), "v", "count(v)", &schema)?,
        aggregate(sum_udaf(), "v", "sum(v)", &schema)?,
        aggregate(min_udaf(), "v", "min(v)", &schema)?,
        aggregate(max_udaf(), "v", "max(v)", &schema)?,
        aggregate(avg_udaf(), "f", "avg(f)", &schema)?,
        aggregate(sum_udaf(), "f", "sum(f)", &schema)?,
    ])
}

/// `GROUP BY` the given columns of the aggregate schema
fn group_by(columns: &[&str]) -> Result<PhysicalGroupBy> {
    let schema = aggregate_schema();
    Ok(PhysicalGroupBy::new_single(
        columns
            .iter()
            .map(|name| Ok((col(name, &schema)?, name.to_string())))
            .collect::<Result<_>>()?,
    ))
}

/// No `GROUP BY`, as the physical planner builds it
fn no_group_by() -> PhysicalGroupBy {
    PhysicalGroupBy::new(vec![], vec![], vec![], false)
}

/// `GROUP BY ROLLUP (g, h)`, as the physical planner expands it
fn rollup_g_h() -> Result<PhysicalGroupBy> {
    let schema = aggregate_schema();
    let null = |data_type: &DataType| -> Result<Arc<dyn PhysicalExpr>> {
        Ok(Arc::new(Literal::new(ScalarValue::try_from(data_type)?)))
    };
    Ok(PhysicalGroupBy::new(
        vec![
            (col("g", &schema)?, "g".to_string()),
            (col("h", &schema)?, "h".to_string()),
        ],
        vec![
            (null(&DataType::Int32)?, "g".to_string()),
            (null(&DataType::Utf8)?, "h".to_string()),
        ],
        vec![vec![true, true], vec![false, true], vec![false, false]],
        true,
    ))
}

/// A one-phase aggregate in `mode` (`Single` or `SinglePartitioned`)
fn single_phase(
    mode: AggregateMode,
    group_by: PhysicalGroupBy,
    aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
    input: Arc<dyn ExecutionPlan>,
    limit: Option<LimitOptions>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let filters = vec![None; aggr_expr.len()];
    let input_schema = input.schema();
    Ok(Arc::new(
        AggregateExec::try_new(mode, group_by, aggr_expr, filters, input, input_schema)?
            .with_limit_options(limit),
    ))
}

/// How the output of a partial aggregate reaches the final aggregate
#[derive(Debug, Clone, Copy)]
enum Exchange {
    /// `CoalescePartitionsExec` and a `Final` aggregate
    Coalesce,
    /// `RepartitionExec` hashing on the grouping keys, optionally preserving
    /// the order of its input, and a `FinalPartitioned` aggregate
    Hash { preserve_order: bool },
}

/// A two-phase aggregate, as the physical planner builds it and
/// `EnforceDistribution` connects the phases. `limit` is set on both phases,
/// as `TopKAggregation` does.
fn two_phase(
    group_by: PhysicalGroupBy,
    aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
    input: Arc<dyn ExecutionPlan>,
    exchange: Exchange,
    limit: Option<LimitOptions>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let filters = vec![None; aggr_expr.len()];
    let input_schema = input.schema();
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        group_by,
        aggr_expr,
        filters.clone(),
        input,
        Arc::clone(&input_schema),
    )?
    .with_limit_options(limit);
    let final_group_by = partial.group_expr().as_final();
    let final_aggr_expr = partial.aggr_expr().to_vec();
    let (mode, exchanged): (AggregateMode, Arc<dyn ExecutionPlan>) = match exchange {
        Exchange::Coalesce => (
            AggregateMode::Final,
            Arc::new(CoalescePartitionsExec::new(Arc::new(partial))),
        ),
        Exchange::Hash { preserve_order } => {
            let keys = partial.output_group_expr();
            let repartition = RepartitionExec::try_new(
                Arc::new(partial),
                Partitioning::Hash(keys, HASH_PARTITIONS),
            )?;
            let repartition = if preserve_order {
                repartition.with_preserve_order()
            } else {
                repartition
            };
            (AggregateMode::FinalPartitioned, Arc::new(repartition))
        }
    };
    Ok(Arc::new(
        AggregateExec::try_new(
            mode,
            final_group_by,
            final_aggr_expr,
            filters,
            exchanged,
            input_schema,
        )?
        .with_limit_options(limit),
    ))
}

/// The aggregates under test
fn aggregate_plans() -> Result<Vec<Case>> {
    let schema = aggregate_schema();
    let multi_partition = || aggregate_spec().with_partition_rows(&[100, 200, 300]);
    let single_partition = || aggregate_spec().with_partition_rows(&[600]);
    let hash_on_g = || -> Result<SourceSpec> {
        Ok(aggregate_spec().with_hash_partitioning(
            vec![col("g", &schema)?],
            HASH_PARTITIONS,
            600,
        ))
    };
    let min_max = vec![
        aggregate(min_udaf(), "v", "min(v)", &schema)?,
        aggregate(max_udaf(), "v", "max(v)", &schema)?,
    ];
    // Many distinct values, so that few groups tie on max(v) and the groups
    // a TopK aggregate keeps are well defined
    let many_values = || aggregate_spec().with_distinct_values(1000);

    Ok(vec![
        // SELECT count(v), ... FROM t
        Case::new(
            "AggregateExec Partial and Final without GROUP BY",
            two_phase(
                no_group_by(),
                all_aggregates()?,
                multi_partition().build_arc()?,
                Exchange::Coalesce,
                None,
            )?,
        ),
        // SELECT min(v), max(v) FROM t: the partial aggregate creates a
        // dynamic filter
        Case::new(
            "AggregateExec Partial and Final without GROUP BY, min and max only",
            two_phase(
                no_group_by(),
                min_max,
                multi_partition().build_arc()?,
                Exchange::Coalesce,
                None,
            )?,
        ),
        // SELECT g, count(v), ... FROM t GROUP BY g, with one target
        // partition over a multi-partition input
        Case::new(
            "AggregateExec Partial and Final",
            two_phase(
                group_by(&["g"])?,
                all_aggregates()?,
                multi_partition().build_arc()?,
                Exchange::Coalesce,
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Partial and FinalPartitioned",
            two_phase(
                group_by(&["g", "h"])?,
                all_aggregates()?,
                multi_partition().build_arc()?,
                Exchange::Hash {
                    preserve_order: false,
                },
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Partial and FinalPartitioned on sorted input",
            two_phase(
                group_by(&["g"])?,
                all_aggregates()?,
                multi_partition()
                    .with_ordering(ordering_on_g()?)
                    .build_arc()?,
                Exchange::Hash {
                    preserve_order: true,
                },
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Single",
            single_phase(
                AggregateMode::Single,
                group_by(&["g"])?,
                all_aggregates()?,
                single_partition().build_arc()?,
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Single without GROUP BY",
            single_phase(
                AggregateMode::Single,
                no_group_by(),
                all_aggregates()?,
                single_partition().build_arc()?,
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Single on sorted input",
            single_phase(
                AggregateMode::Single,
                group_by(&["g"])?,
                all_aggregates()?,
                single_partition()
                    .with_ordering(ordering_on_g()?)
                    .build_arc()?,
                None,
            )?,
        ),
        Case::new(
            "AggregateExec Single on partially sorted input",
            single_phase(
                AggregateMode::Single,
                group_by(&["h", "g"])?,
                all_aggregates()?,
                single_partition()
                    .with_ordering(ordering_on_g()?)
                    .build_arc()?,
                None,
            )?,
        ),
        Case::new(
            "AggregateExec SinglePartitioned",
            single_phase(
                AggregateMode::SinglePartitioned,
                group_by(&["g"])?,
                all_aggregates()?,
                hash_on_g()?.build_arc()?,
                None,
            )?,
        ),
        // SELECT g, max(v) FROM t GROUP BY g ORDER BY max(v) DESC LIMIT 5,
        // after `TopKAggregation` pushed the limit into both phases
        Case::new(
            "AggregateExec Partial and FinalPartitioned with TopK limit",
            two_phase(
                group_by(&["g"])?,
                vec![aggregate(max_udaf(), "v", "max(v)", &schema)?],
                many_values()
                    .with_partition_rows(&[100, 200, 300])
                    .build_arc()?,
                Exchange::Hash {
                    preserve_order: false,
                },
                Some(LimitOptions::new_with_order(5, true)),
            )?,
        ),
        // SELECT DISTINCT g FROM t LIMIT 5, after `LimitedDistinctAggregation`
        Case::new(
            "AggregateExec Single with DISTINCT limit",
            single_phase(
                AggregateMode::Single,
                group_by(&["g"])?,
                vec![],
                single_partition().build_arc()?,
                Some(LimitOptions::new(5)),
            )?,
        )
        // The limit is a soft limit: the aggregate stops after the input
        // batch in which it has seen 5 groups, so how many and which groups
        // it emits depends on batch boundaries, and only the `LIMIT` above it
        // makes the result well defined (`aggregates/mod.rs:859-864`). It
        // reports no fetch, so the check cannot know that.
        .allow("batch_boundary_invariance"),
        // SELECT g, h, count(v), ... FROM t GROUP BY ROLLUP (g, h), on few
        // rows with many distinct values, so that there are more groups
        // than input rows
        Case::new(
            "AggregateExec Partial and FinalPartitioned with ROLLUP",
            two_phase(
                rollup_g_h()?,
                all_aggregates()?,
                many_values()
                    .with_partition_rows(&[20, 30, 40])
                    .build_arc()?,
                Exchange::Hash {
                    preserve_order: false,
                },
                None,
            )?,
        ),
    ])
}

/// Rows of the left input of a join
const LEFT_ROWS: usize = 60;
/// Rows of the right input of a join
const RIGHT_ROWS: usize = 90;

/// A nullable join key and a value, with a row id so that every row is
/// unique. The left key has more distinct values than the right key, so that
/// both sides have rows without a match: left rows with keys the right side
/// does not have, and rows with null keys on both sides.
fn left_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("l_k", DataType::Int32, true),
        Field::new("l_v", DataType::Int32, false),
    ]))
}

fn right_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("r_k", DataType::Int32, true),
        Field::new("r_v", DataType::Int32, false),
    ]))
}

fn left_spec() -> SourceSpec {
    SourceSpec::new(left_schema())
        .with_batch_layout(BATCH_LAYOUT)
        .with_distinct_values(12)
        .with_row_id_column("l_id", 0)
}

fn right_spec() -> SourceSpec {
    SourceSpec::new(right_schema())
        .with_batch_layout(BATCH_LAYOUT)
        .with_distinct_values(8)
        .with_row_id_column("r_id", 1_000_000)
        .with_seed(1)
}

/// `name` in the schema generated by `spec`, which includes its row id
fn column(spec: &SourceSpec, name: &str) -> Result<Arc<dyn PhysicalExpr>> {
    col(name, &spec.schema())
}

/// `(l_k, r_k)`
fn join_on() -> Result<JoinOn> {
    Ok(vec![(
        column(&left_spec(), "l_k")?,
        column(&right_spec(), "r_k")?,
    )])
}

/// `spec` with every partition sorted on `sort_exprs`, which refer to the
/// schema generated by `spec`
fn sorted_on(spec: SourceSpec, sort_exprs: Vec<PhysicalSortExpr>) -> SourceSpec {
    spec.with_ordering(LexOrdering::new(sort_exprs).unwrap())
}

/// `spec` with `rows` rows, hash partitioned on its column `key`
fn hash_on_key(spec: SourceSpec, key: &str, rows: usize) -> Result<SourceSpec> {
    let key = column(&spec, key)?;
    Ok(spec.with_hash_partitioning(vec![key], HASH_PARTITIONS, rows))
}

/// A join filter comparing `left_column` with `right_column` using `op`. Its
/// intermediate schema copies the input fields, with their nullability, as
/// the physical planner does.
fn join_filter(
    left_column: &str,
    op: Operator,
    right_column: &str,
) -> Result<JoinFilter> {
    let left = left_spec().schema();
    let right = right_spec().schema();
    let left_index = left.index_of(left_column)?;
    let right_index = right.index_of(right_column)?;
    let schema = Arc::new(Schema::new(vec![
        left.field(left_index).clone(),
        right.field(right_index).clone(),
    ]));
    let expression = Arc::new(BinaryExpr::new(
        Arc::new(Column::new(left_column, 0)),
        op,
        Arc::new(Column::new(right_column, 1)),
    ));
    Ok(JoinFilter::new(
        expression,
        vec![
            ColumnIndex {
                index: left_index,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: right_index,
                side: JoinSide::Right,
            },
        ],
        schema,
    ))
}

/// `l_v < r_v`, which keeps about half of the pairs with matching keys
fn value_filter() -> Result<JoinFilter> {
    join_filter("l_v", Operator::Lt, "r_v")
}

/// Options of a hash join case
struct HashJoin {
    mode: PartitionMode,
    join_type: JoinType,
    filter: bool,
    null_equality: NullEquality,
    fetch: Option<usize>,
}

impl HashJoin {
    fn new(mode: PartitionMode, join_type: JoinType) -> Self {
        Self {
            mode,
            join_type,
            filter: false,
            null_equality: NullEquality::NullEqualsNothing,
            fetch: None,
        }
    }

    fn with_filter(mut self) -> Self {
        self.filter = true;
        self
    }

    /// A hash join as the physical planner and `JoinSelection` build it:
    /// in `CollectLeft` mode on a single-partition build side and a
    /// multi-partition probe side, and in `Partitioned` mode on inputs hash
    /// partitioned on the join keys
    fn build(self) -> Result<Arc<dyn ExecutionPlan>> {
        let (left, right) = match self.mode {
            PartitionMode::Partitioned => (
                hash_on_key(left_spec(), "l_k", LEFT_ROWS)?,
                hash_on_key(right_spec(), "r_k", RIGHT_ROWS)?,
            ),
            _ => (
                left_spec().with_partition_rows(&[LEFT_ROWS]),
                right_spec().with_partition_rows(&[30, 0, 60]),
            ),
        };
        let filter = self.filter.then(value_filter).transpose()?;
        let join = HashJoinExec::try_new(
            left.build_arc()?,
            right.build_arc()?,
            join_on()?,
            filter,
            &self.join_type,
            None,
            self.mode,
            self.null_equality,
            false,
        )?;
        Ok(match self.fetch {
            Some(fetch) => join
                .with_fetch(Some(fetch))
                .expect("hash join supports fetch"),
            None => Arc::new(join),
        })
    }
}

/// A sort merge join on inputs sorted on the join keys and hash partitioned
/// on them, as `EnforceDistribution` and `EnforceSorting` arrange them
fn sort_merge_join(
    join_type: JoinType,
    filter: Option<JoinFilter>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let options = SortOptions::default();
    let left = hash_on_key(left_spec(), "l_k", LEFT_ROWS)?;
    let right = hash_on_key(right_spec(), "r_k", RIGHT_ROWS)?;
    let left_key = column(&left, "l_k")?;
    let right_key = column(&right, "r_k")?;
    let left = sorted_on(left, vec![PhysicalSortExpr::new(left_key, options)]);
    let right = sorted_on(right, vec![PhysicalSortExpr::new(right_key, options)]);
    Ok(Arc::new(SortMergeJoinExec::try_new(
        left.build_arc()?,
        right.build_arc()?,
        join_on()?,
        filter,
        join_type,
        vec![options],
        NullEquality::NullEqualsNothing,
    )?))
}

/// A nested loop join with the filter `l_k < r_k`, which is false for null
/// keys, on a single-partition left input
fn nested_loop_join(join_type: JoinType) -> Result<Arc<dyn ExecutionPlan>> {
    Ok(Arc::new(NestedLoopJoinExec::try_new(
        left_spec().with_partition_rows(&[LEFT_ROWS]).build_arc()?,
        right_spec().with_partition_rows(&[30, 0, 60]).build_arc()?,
        Some(join_filter("l_k", Operator::Lt, "r_k")?),
        &join_type,
        None,
    )?))
}

/// A piecewise merge join on `l_k < r_k`, with the buffered (left) side in
/// one partition and sorted as the join requires. Right existence joins
/// require neither, and get a buffered side with several unsorted
/// partitions.
fn piecewise_merge_join(join_type: JoinType) -> Result<Arc<dyn ExecutionPlan>> {
    let left_key = column(&left_spec(), "l_k")?;
    let right_key = column(&right_spec(), "r_k")?;
    let left = if matches!(
        join_type,
        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
    ) {
        left_spec().with_partition_rows(&[20, 0, 40])
    } else {
        // `<` requires the buffered side in descending order, nulls first
        sorted_on(
            left_spec().with_partition_rows(&[LEFT_ROWS]),
            vec![PhysicalSortExpr::new(
                Arc::clone(&left_key),
                SortOptions::new(true, true),
            )],
        )
    };
    Ok(Arc::new(PiecewiseMergeJoinExec::try_new(
        left.build_arc()?,
        right_spec().with_partition_rows(&[30, 0, 60]).build_arc()?,
        (left_key, right_key),
        Operator::Lt,
        join_type,
        HASH_PARTITIONS,
    )?))
}

/// A symmetric hash join, which `JoinSelection` uses when both inputs are
/// unbounded, on finite inputs
fn symmetric_hash_join(
    mode: StreamJoinPartitionMode,
    join_type: JoinType,
    filter: Option<JoinFilter>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let (left, right) = match mode {
        StreamJoinPartitionMode::Partitioned => (
            hash_on_key(left_spec(), "l_k", LEFT_ROWS)?,
            hash_on_key(right_spec(), "r_k", RIGHT_ROWS)?,
        ),
        StreamJoinPartitionMode::SinglePartition => (
            left_spec().with_partition_rows(&[LEFT_ROWS]),
            right_spec().with_partition_rows(&[RIGHT_ROWS]),
        ),
    };
    Ok(Arc::new(SymmetricHashJoinExec::try_new(
        left.build_arc()?,
        right.build_arc()?,
        join_on()?,
        filter,
        &join_type,
        NullEquality::NullEqualsNothing,
        None,
        None,
        mode,
    )?))
}

/// `ASOF JOIN ... ON l_k = r_k MATCH_CONDITION (l_v >= r_v)`, on inputs
/// sorted as the join requires: on the equality keys, ascending with nulls
/// first, then on the match expression, ascending for `>=`
fn asof_join() -> Result<Arc<dyn ExecutionPlan>> {
    let options = SortOptions::new(false, true);
    let sorted = |spec: SourceSpec, key: &str, value: &str| -> Result<SourceSpec> {
        let exprs = vec![
            PhysicalSortExpr::new(column(&spec, key)?, options),
            PhysicalSortExpr::new(column(&spec, value)?, options),
        ];
        Ok(sorted_on(spec, exprs))
    };
    let left = sorted(left_spec().with_partition_rows(&[20, 0, 40]), "l_k", "l_v")?;
    let right = sorted(
        right_spec().with_partition_rows(&[RIGHT_ROWS]),
        "r_k",
        "r_v",
    )?;
    let match_condition = AsOfMatchExpr::new(
        column(&left_spec(), "l_v")?,
        Operator::GtEq,
        column(&right_spec(), "r_v")?,
    );
    Ok(Arc::new(AsOfJoinExec::try_new(
        left.build_arc()?,
        right.build_arc()?,
        join_on()?,
        match_condition,
        None,
    )?))
}

/// The joins under test. Join types are chosen to cover the different code
/// paths of each operator rather than every combination: types that emit
/// unmatched rows of the build side at the end, types that emit probe rows
/// as they arrive, existence and mark joins, and filters.
fn join_plans() -> Result<Vec<Case>> {
    use JoinType::*;
    use PartitionMode::{CollectLeft, Partitioned};

    let mut cases = vec![];
    for join_type in [
        Inner, Left, Right, Full, LeftSemi, RightSemi, LeftAnti, RightAnti, LeftMark,
        RightMark,
    ] {
        cases.push((
            format!("HashJoinExec CollectLeft {join_type}"),
            HashJoin::new(CollectLeft, join_type).build()?,
        ));
    }
    for join_type in [Inner, Full, RightSemi, LeftAnti] {
        cases.push((
            format!("HashJoinExec CollectLeft {join_type} with filter"),
            HashJoin::new(CollectLeft, join_type)
                .with_filter()
                .build()?,
        ));
    }
    let mut nulls_equal = HashJoin::new(CollectLeft, Inner);
    nulls_equal.null_equality = NullEquality::NullEqualsNull;
    cases.push((
        "HashJoinExec CollectLeft Inner with nulls equal".to_string(),
        nulls_equal.build()?,
    ));
    let mut with_fetch = HashJoin::new(CollectLeft, Inner);
    with_fetch.fetch = Some(FETCH);
    cases.push((
        "HashJoinExec CollectLeft Inner with fetch".to_string(),
        with_fetch.build()?,
    ));
    for join_type in [Inner, Left, Full, RightAnti, LeftMark] {
        cases.push((
            format!("HashJoinExec Partitioned {join_type}"),
            HashJoin::new(Partitioned, join_type).build()?,
        ));
    }
    cases.push((
        "HashJoinExec Partitioned Full with filter".to_string(),
        HashJoin::new(Partitioned, Full).with_filter().build()?,
    ));

    for join_type in [Inner, Left, Full, LeftSemi, RightAnti, LeftMark] {
        cases.push((
            format!("SortMergeJoinExec {join_type}"),
            sort_merge_join(join_type, None)?,
        ));
    }
    for join_type in [Inner, Full, LeftAnti] {
        cases.push((
            format!("SortMergeJoinExec {join_type} with filter"),
            sort_merge_join(join_type, Some(value_filter()?))?,
        ));
    }

    for join_type in [Inner, Left, Full, RightSemi, LeftAnti, RightMark] {
        cases.push((
            format!("NestedLoopJoinExec {join_type}"),
            nested_loop_join(join_type)?,
        ));
    }

    cases.push((
        "CrossJoinExec".to_string(),
        Arc::new(CrossJoinExec::new(
            left_spec().with_partition_rows(&[LEFT_ROWS]).build_arc()?,
            right_spec().with_partition_rows(&[30, 0, 60]).build_arc()?,
        )),
    ));

    cases.push((
        "SymmetricHashJoinExec SinglePartition Inner".to_string(),
        symmetric_hash_join(StreamJoinPartitionMode::SinglePartition, Inner, None)?,
    ));
    cases.push((
        "SymmetricHashJoinExec Partitioned Full with filter".to_string(),
        symmetric_hash_join(
            StreamJoinPartitionMode::Partitioned,
            Full,
            Some(value_filter()?),
        )?,
    ));

    for join_type in [Inner, Full, RightSemi] {
        cases.push((
            format!("PiecewiseMergeJoinExec {join_type}"),
            piecewise_merge_join(join_type)?,
        ));
    }

    cases.push(("AsOfJoinExec".to_string(), asof_join()?));

    Ok(cases
        .into_iter()
        .map(|(name, plan)| Case::new(name, plan))
        .collect())
}

#[tokio::test]
async fn builtin_plan_findings() -> Result<()> {
    let output = audit(builtin_plans()?).await?;
    insta::assert_snapshot!(output);
    Ok(())
}

#[tokio::test]
async fn builtin_aggregate_findings() -> Result<()> {
    let output = audit(aggregate_plans()?).await?;
    insta::assert_snapshot!(output);
    Ok(())
}

#[tokio::test]
async fn builtin_join_findings() -> Result<()> {
    let output = audit(join_plans()?).await?;
    insta::assert_snapshot!(output);
    Ok(())
}
