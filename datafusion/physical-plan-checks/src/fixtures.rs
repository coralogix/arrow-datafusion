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

//! Leaf plans with controllable properties, used as inputs for the plans under
//! test.

use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, Statistics, internal_err, not_impl_err};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{EquivalenceProperties, LexOrdering, PhysicalExpr};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning,
    PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream, StatisticsArgs,
};

/// A leaf [`ExecutionPlan`] whose statistics, partition count and output
/// ordering are set by the test.
///
/// The statistics it reports are taken as the truth: checks that compare a node
/// with its input treat them as correct.
///
/// It does not produce data yet; [`ExecutionPlan::execute`] returns an error.
#[derive(Debug, Clone)]
pub struct MockSourceExec {
    schema: SchemaRef,
    statistics: Arc<Statistics>,
    partition_statistics: Vec<Arc<Statistics>>,
    output_ordering: Option<LexOrdering>,
    cache: Arc<PlanProperties>,
}

impl MockSourceExec {
    /// Create a source with a single partition and unknown statistics
    pub fn new(schema: SchemaRef) -> Self {
        let unknown = Arc::new(Statistics::new_unknown(&schema));
        let cache = Self::compute_properties(&schema, 1, None);
        Self {
            schema,
            statistics: Arc::clone(&unknown),
            partition_statistics: vec![unknown],
            output_ordering: None,
            cache,
        }
    }

    /// Set the number of partitions. The statistics of every partition, and the
    /// overall statistics, are reset to unknown.
    pub fn with_partition_count(self, partition_count: usize) -> Self {
        let unknown = Arc::new(Statistics::new_unknown(&self.schema));
        let partition_statistics = vec![unknown; partition_count];
        self.with_partition_statistics(partition_statistics)
    }

    /// Set the statistics of each partition. The number of partitions is
    /// `partition_statistics.len()`, and the overall statistics are set to the
    /// merge of the partition statistics.
    ///
    /// # Panics
    /// If any statistics do not have one column statistics entry per field of
    /// the schema.
    pub fn with_partition_statistics(
        mut self,
        partition_statistics: Vec<Arc<Statistics>>,
    ) -> Self {
        for stats in &partition_statistics {
            self.assert_valid_statistics(stats);
        }
        let statistics = if partition_statistics.is_empty() {
            Statistics::new_unknown(&self.schema)
        } else {
            Statistics::try_merge_iter(
                partition_statistics.iter().map(AsRef::as_ref),
                &self.schema,
            )
            .expect("partition statistics can be merged")
        };
        self.statistics = Arc::new(statistics);
        self.partition_statistics = partition_statistics;
        self.cache = Self::compute_properties(
            &self.schema,
            self.partition_statistics.len(),
            self.output_ordering.clone(),
        );
        self
    }

    /// Set exact row counts for each partition, leaving the other statistics
    /// unknown. The number of partitions is `num_rows.len()`.
    pub fn with_exact_partition_num_rows(self, num_rows: &[usize]) -> Self {
        let stats = num_rows
            .iter()
            .map(|n| {
                Arc::new(
                    Statistics::new_unknown(&self.schema)
                        .with_num_rows(Precision::Exact(*n)),
                )
            })
            .collect();
        self.with_partition_statistics(stats)
    }

    /// Set inexact row counts for each partition, leaving the other statistics
    /// unknown. The number of partitions is `num_rows.len()`.
    pub fn with_inexact_partition_num_rows(self, num_rows: &[usize]) -> Self {
        let stats = num_rows
            .iter()
            .map(|n| {
                Arc::new(
                    Statistics::new_unknown(&self.schema)
                        .with_num_rows(Precision::Inexact(*n)),
                )
            })
            .collect();
        self.with_partition_statistics(stats)
    }

    /// Override the overall statistics, which otherwise are the merge of the
    /// partition statistics.
    ///
    /// # Panics
    /// If `statistics` does not have one column statistics entry per field of
    /// the schema.
    pub fn with_statistics(mut self, statistics: Statistics) -> Self {
        self.assert_valid_statistics(&statistics);
        self.statistics = Arc::new(statistics);
        self
    }

    /// Declare that every partition is sorted by `ordering`
    pub fn with_output_ordering(mut self, ordering: LexOrdering) -> Self {
        self.output_ordering = Some(ordering);
        self.cache = Self::compute_properties(
            &self.schema,
            self.partition_statistics.len(),
            self.output_ordering.clone(),
        );
        self
    }

    fn assert_valid_statistics(&self, statistics: &Statistics) {
        assert_eq!(
            statistics.column_statistics.len(),
            self.schema.fields().len(),
            "MockSourceExec statistics must have one column statistics entry per field"
        );
    }

    fn compute_properties(
        schema: &SchemaRef,
        partition_count: usize,
        output_ordering: Option<LexOrdering>,
    ) -> Arc<PlanProperties> {
        let eq_properties = match output_ordering {
            Some(ordering) => {
                EquivalenceProperties::new_with_orderings(Arc::clone(schema), [ordering])
            }
            None => EquivalenceProperties::new(Arc::clone(schema)),
        };
        Arc::new(PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }
}

impl DisplayAs for MockSourceExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "MockSourceExec: partitions={}, num_rows={}",
                self.partition_statistics.len(),
                self.statistics.num_rows
            ),
            DisplayFormatType::TreeRender => {
                write!(f, "partitions={}", self.partition_statistics.len())
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
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        not_impl_err!("MockSourceExec does not produce data yet")
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
