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
//! records the findings in a snapshot.
//!
//! The snapshot is the list of known violations in the built-in plans. When
//! a plan is fixed, or a new check finds a new problem, the snapshot changes
//! and must be reviewed and updated with `cargo insta review` (or by running
//! the test with `INSTA_UPDATE=always`).
//!
//! Known findings are also listed, with their causes, in
//! `IMPLEMENTATION_STATUS.md`.

use std::fmt::Write;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_expr::{LexOrdering, Partitioning, PhysicalSortExpr};
use datafusion_physical_plan::buffer::BufferExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter::FilterExec;
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

/// The spec every source starts from: random batches of up to 16 rows,
/// including empty batches, with exact statistics
fn spec() -> SourceSpec {
    SourceSpec::new(schema()).with_batch_layout(BatchLayout::Random {
        max_rows: 16,
        empty_batches: true,
    })
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

#[expect(deprecated)]
fn coalesce_batches(
    input: Arc<dyn ExecutionPlan>,
    fetch: Option<usize>,
) -> Arc<dyn ExecutionPlan> {
    use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
    Arc::new(CoalesceBatchesExec::new(input, 8192).with_fetch(fetch))
}

/// The plans under test, each with a descriptive name
fn builtin_plans() -> Result<Vec<(&'static str, Arc<dyn ExecutionPlan>)>> {
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
    Ok(plans)
}

/// Runs on a current-thread runtime so that execution, and therefore the
/// snapshot, is deterministic. Some plans, such as a partitioned TopK
/// `SortExec`, produce output that depends on how partitions interleave.
#[tokio::test]
async fn builtin_plan_findings() -> Result<()> {
    let checker = PlanChecker::new();
    let mut output = String::new();
    for (name, plan) in builtin_plans()? {
        let report = checker.check_async(&plan).await?;
        writeln!(output, "## {name}").unwrap();
        let display = displayable(plan.as_ref()).indent(true).to_string();
        for line in display.lines() {
            writeln!(output, "    {line}").unwrap();
        }
        writeln!(output, "{report}\n").unwrap();
    }
    insta::assert_snapshot!(output);
    Ok(())
}
