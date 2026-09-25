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

use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err, not_impl_err, plan_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};

use crate::oracle;

/// How precise the statistics reported by a [`MockSourceExec`] are.
///
/// The values are always computed from the data. Only their precision
/// changes, which lets tests check how a plan handles exact, estimated and
/// missing input statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatisticsPrecision {
    /// Report the true values as `Precision::Exact`
    #[default]
    Exact,
    /// Report the true values as `Precision::Inexact`
    Inexact,
    /// Report `Statistics::new_unknown`
    Absent,
}

impl StatisticsPrecision {
    fn apply(self, schema: &Schema, statistics: &Statistics) -> Arc<Statistics> {
        Arc::new(match self {
            StatisticsPrecision::Exact => statistics.clone(),
            StatisticsPrecision::Inexact => statistics.clone().to_inexact(),
            StatisticsPrecision::Absent => Statistics::new_unknown(schema),
        })
    }
}

/// A leaf [`ExecutionPlan`] that serves fixed batches and reports properties
/// that are verified against them.
///
/// - Statistics are computed from the batches, with the precision set by
///   [`Self::with_statistics_precision`].
/// - An output ordering set with [`Self::try_with_output_ordering`] is
///   rejected unless every partition is sorted by it.
/// - Hash partitioning set with [`Self::try_with_partitioning`] is rejected
///   unless every row is in the partition `RepartitionExec` would send it to.
///
/// Because of this, checks that compare a plan with its input can treat what
/// a `MockSourceExec` reports as correct. Use [`SourceSpec`] to generate the
/// batches.
///
/// [`SourceSpec`]: crate::fixtures::SourceSpec
#[derive(Debug, Clone)]
pub struct MockSourceExec {
    schema: SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
    partitioning: Partitioning,
    output_ordering: Option<LexOrdering>,
    precision: StatisticsPrecision,
    /// Exact statistics of all partitions, before `precision` is applied
    exact_statistics: Statistics,
    /// Exact statistics of each partition, before `precision` is applied
    exact_partition_statistics: Vec<Statistics>,
    statistics: Arc<Statistics>,
    partition_statistics: Vec<Arc<Statistics>>,
    cache: Arc<PlanProperties>,
}

impl MockSourceExec {
    /// Create a source that serves `partitions`, one inner `Vec` of batches per
    /// output partition, with `UnknownPartitioning` and exact statistics.
    ///
    /// Returns an error if any batch does not have exactly `schema`.
    pub fn try_new(schema: SchemaRef, partitions: Vec<Vec<RecordBatch>>) -> Result<Self> {
        for (p, batches) in partitions.iter().enumerate() {
            for batch in batches {
                if batch.schema().as_ref() != schema.as_ref() {
                    return plan_err!(
                        "MockSourceExec partition {p} has a batch with schema {:?}, \
                         expected {schema:?}",
                        batch.schema()
                    );
                }
            }
        }
        let all_batches: Vec<RecordBatch> =
            partitions.iter().flatten().cloned().collect();
        let exact_statistics = oracle::exact_statistics(&schema, &all_batches)?;
        let exact_partition_statistics = partitions
            .iter()
            .map(|batches| oracle::exact_statistics(&schema, batches))
            .collect::<Result<Vec<_>>>()?;
        let partitioning = Partitioning::UnknownPartitioning(partitions.len());
        let precision = StatisticsPrecision::Exact;
        let cache = Self::compute_properties(&schema, &partitioning, None);

        Ok(Self {
            statistics: precision.apply(&schema, &exact_statistics),
            partition_statistics: exact_partition_statistics
                .iter()
                .map(|stats| precision.apply(&schema, stats))
                .collect(),
            schema,
            partitions,
            partitioning,
            output_ordering: None,
            precision,
            exact_statistics,
            exact_partition_statistics,
            cache,
        })
    }

    /// Set the precision of the reported statistics
    pub fn with_statistics_precision(mut self, precision: StatisticsPrecision) -> Self {
        self.precision = precision;
        self.statistics = precision.apply(&self.schema, &self.exact_statistics);
        self.partition_statistics = self
            .exact_partition_statistics
            .iter()
            .map(|stats| precision.apply(&self.schema, stats))
            .collect();
        self
    }

    /// Declare that every partition is sorted by `ordering`.
    ///
    /// Returns an error if a partition is not sorted by `ordering`.
    pub fn try_with_output_ordering(mut self, ordering: LexOrdering) -> Result<Self> {
        for (p, batches) in self.partitions.iter().enumerate() {
            if let Some(row) = oracle::first_unsorted_row(batches, &ordering)? {
                return plan_err!(
                    "MockSourceExec partition {p} is not sorted by [{ordering}] at row {row}"
                );
            }
        }
        self.output_ordering = Some(ordering);
        self.cache = Self::compute_properties(
            &self.schema,
            &self.partitioning,
            self.output_ordering.as_ref(),
        );
        Ok(self)
    }

    /// Declare the output partitioning.
    ///
    /// The partition count must match the number of partitions. For
    /// `Partitioning::Hash`, every row must be in the partition that
    /// `RepartitionExec` would send it to. `Partitioning::Range` is not
    /// supported yet.
    pub fn try_with_partitioning(mut self, partitioning: Partitioning) -> Result<Self> {
        let count = partitioning.partition_count();
        if count != self.partitions.len() {
            return plan_err!(
                "MockSourceExec has {} partitions, but the partitioning {partitioning} \
                 has {count}",
                self.partitions.len()
            );
        }
        match &partitioning {
            Partitioning::Hash(exprs, n) => {
                for (p, batches) in self.partitions.iter().enumerate() {
                    let misplaced =
                        oracle::rows_outside_hash_partition(batches, exprs, *n, p)?;
                    if misplaced > 0 {
                        return plan_err!(
                            "MockSourceExec partition {p} has {misplaced} rows that do \
                             not belong to it under {partitioning}"
                        );
                    }
                }
            }
            Partitioning::Range(_) => {
                return not_impl_err!(
                    "MockSourceExec does not support range partitioning"
                );
            }
            Partitioning::RoundRobinBatch(_) | Partitioning::UnknownPartitioning(_) => {}
        }
        self.cache = Self::compute_properties(
            &self.schema,
            &partitioning,
            self.output_ordering.as_ref(),
        );
        self.partitioning = partitioning;
        Ok(self)
    }

    /// The batches served by each partition
    pub fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.partitions
    }

    fn compute_properties(
        schema: &SchemaRef,
        partitioning: &Partitioning,
        output_ordering: Option<&LexOrdering>,
    ) -> Arc<PlanProperties> {
        let eq_properties = match output_ordering {
            Some(ordering) => EquivalenceProperties::new_with_orderings(
                Arc::clone(schema),
                [ordering.clone()],
            ),
            None => EquivalenceProperties::new(Arc::clone(schema)),
        };
        Arc::new(PlanProperties::new(
            eq_properties,
            partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl DisplayAs for MockSourceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let rows: Vec<usize> = self
            .partitions
            .iter()
            .map(|batches| batches.iter().map(RecordBatch::num_rows).sum())
            .collect();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "MockSourceExec: partitioning={}, partition_rows={rows:?}, \
                     statistics={:?}",
                    self.partitioning, self.precision
                )?;
                if let Some(ordering) = &self.output_ordering {
                    write!(f, ", output_ordering=[{ordering}]")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(f, "partition_rows={rows:?}")
            }
        }
    }
}

impl ExecutionPlan for MockSourceExec {
    fn name(&self) -> &'static str {
        "MockSourceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("MockSourceExec does not have children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(batches) = self.partitions.get(partition) else {
            return internal_err!(
                "Invalid partition index: {partition}, the partition count is {}",
                self.partitions.len()
            );
        };
        let batches = batches.clone().into_iter().map(Ok);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            futures::stream::iter(batches),
        )))
    }

    fn statistics_from_inputs(
        &self,
        _input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        match args.partition() {
            None => Ok(Arc::clone(&self.statistics)),
            Some(partition) => match self.partition_statistics.get(partition) {
                Some(stats) => Ok(Arc::clone(stats)),
                None => internal_err!(
                    "Invalid partition index: {partition}, the partition count is {}",
                    self.partition_statistics.len()
                ),
            },
        }
    }
}
