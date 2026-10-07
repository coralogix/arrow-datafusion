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
use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err, not_impl_err, plan_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{
    ConstExpr, EquivalenceProperties, LexOrdering, PhysicalExpr,
};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};
use futures::stream;

use crate::physical_plan::oracle;

/// Two expressions that a [`MockSourceExec`] declares equal on every row
pub type Equality = (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>);

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

/// A leaf [`ExecutionPlan`] that serves fixed batches and reports properties
/// that are verified against them.
///
/// - Statistics are computed from the batches, with the precision set by
///   [`Self::with_statistics_precision`].
/// - An output ordering set with [`Self::try_with_output_ordering`] is
///   rejected unless every partition is sorted by it.
/// - Constants set with [`Self::try_with_constants`] are rejected unless the
///   data has a single value for each in every partition, the same value in
///   every partition for a uniform one.
/// - Equalities set with [`Self::try_with_equalities`] are rejected unless
///   the two expressions of each are equal on every row.
/// - Hash partitioning set with [`Self::try_with_partitioning`] is rejected
///   unless every row is in the partition `RepartitionExec` would send it to.
///
/// Because of this, checks that compare a plan with its input can treat what
/// a `MockSourceExec` reports as correct. Use [`SourceSpec`] to generate the
/// batches.
///
/// [`SourceSpec`]: crate::physical_plan::fixtures::SourceSpec
#[derive(Debug, Clone)]
pub struct MockSourceExec {
    schema: SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
    partitioning: Partitioning,
    output_ordering: Option<LexOrdering>,
    constants: Vec<ConstExpr>,
    equalities: Vec<Equality>,
    precision: StatisticsPrecision,
    /// Exact statistics of all partitions, before `precision` is applied
    statistics: Statistics,
    /// Exact statistics of each partition, before `precision` is applied
    partition_statistics: Vec<Statistics>,
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
        let statistics = oracle::exact_statistics(&schema, &partitions.concat())?;
        let partition_statistics = partitions
            .iter()
            .map(|batches| oracle::exact_statistics(&schema, batches))
            .collect::<Result<Vec<_>>>()?;
        let partitioning = Partitioning::UnknownPartitioning(partitions.len());
        let cache = Self::compute_properties(&schema, &partitioning, None, &[], &[])?;
        Ok(Self {
            schema,
            partitions,
            partitioning,
            output_ordering: None,
            constants: vec![],
            equalities: vec![],
            precision: StatisticsPrecision::Exact,
            statistics,
            partition_statistics,
            cache,
        })
    }

    /// Set the precision of the reported statistics
    pub fn with_statistics_precision(mut self, precision: StatisticsPrecision) -> Self {
        self.precision = precision;
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
        self.cache = self.recompute_properties()?;
        Ok(self)
    }

    /// Declare that each of `constants` is constant: within every partition,
    /// and across partitions as its `across_partitions` says. Replaces the
    /// constants declared before.
    ///
    /// Returns an error if the data contradicts a constant (see
    /// [`oracle::constant_violations`]).
    pub fn try_with_constants(mut self, constants: Vec<ConstExpr>) -> Result<Self> {
        for constant in &constants {
            if let Some(violation) =
                oracle::constant_violations(&self.partitions, constant)?.first()
            {
                return plan_err!("MockSourceExec constant does not hold: {violation}");
            }
        }
        self.constants = constants;
        self.cache = self.recompute_properties()?;
        Ok(self)
    }

    /// Declare that the two expressions of each of `equalities` are equal on
    /// every row, which puts them in the same equivalence class. Replaces the
    /// equalities declared before.
    ///
    /// Returns an error if a row has different values for the two expressions
    /// (see [`oracle::first_unequal_row`]).
    pub fn try_with_equalities(mut self, equalities: Vec<Equality>) -> Result<Self> {
        for (left, right) in &equalities {
            for (p, batches) in self.partitions.iter().enumerate() {
                if let Some((row, left_value, right_value)) =
                    oracle::first_unequal_row(batches, left, right)?
                {
                    return plan_err!(
                        "MockSourceExec equality does not hold: {left} and {right} are \
                         {left_value} and {right_value} in row {row} of partition {p}"
                    );
                }
            }
        }
        self.equalities = equalities;
        self.cache = self.recompute_properties()?;
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
        self.partitioning = partitioning;
        self.cache = self.recompute_properties()?;
        Ok(self)
    }

    /// The batches served by each partition
    pub fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.partitions
    }

    fn recompute_properties(&self) -> Result<Arc<PlanProperties>> {
        Self::compute_properties(
            &self.schema,
            &self.partitioning,
            self.output_ordering.as_ref(),
            &self.constants,
            &self.equalities,
        )
    }

    fn compute_properties(
        schema: &SchemaRef,
        partitioning: &Partitioning,
        output_ordering: Option<&LexOrdering>,
        constants: &[ConstExpr],
        equalities: &[Equality],
    ) -> Result<Arc<PlanProperties>> {
        let mut eq_properties = match output_ordering {
            Some(ordering) => EquivalenceProperties::new_with_orderings(
                Arc::clone(schema),
                [ordering.clone()],
            ),
            None => EquivalenceProperties::new(Arc::clone(schema)),
        };
        eq_properties.add_constants(constants.iter().cloned())?;
        for (left, right) in equalities {
            eq_properties.add_equal_conditions(Arc::clone(left), Arc::clone(right))?;
        }
        Ok(Arc::new(PlanProperties::new(
            eq_properties,
            partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )))
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
                if !self.constants.is_empty() {
                    write!(
                        f,
                        ", constants=[{}]",
                        ConstExpr::format_list(&self.constants)
                    )?;
                }
                if !self.equalities.is_empty() {
                    let equalities: Vec<String> = self
                        .equalities
                        .iter()
                        .map(|(left, right)| format!("{left} = {right}"))
                        .collect();
                    write!(f, ", equalities=[{}]", equalities.join(", "))?;
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
        let batches = batches.to_vec();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream::iter(batches.into_iter().map(Ok)),
        )))
    }

    fn statistics_from_inputs(
        &self,
        _input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        let statistics = match args.partition() {
            None => &self.statistics,
            Some(partition) => match self.partition_statistics.get(partition) {
                Some(statistics) => statistics,
                None => {
                    return internal_err!(
                        "Invalid partition index: {partition}, the partition count is {}",
                        self.partition_statistics.len()
                    );
                }
            },
        };
        Ok(Arc::new(match self.precision {
            StatisticsPrecision::Exact => statistics.clone(),
            StatisticsPrecision::Inexact => statistics.clone().to_inexact(),
            StatisticsPrecision::Absent => Statistics::new_unknown(&self.schema),
        }))
    }
}
