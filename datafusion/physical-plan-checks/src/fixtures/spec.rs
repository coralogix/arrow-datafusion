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

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{Result, plan_err};
use datafusion_physical_expr::{LexOrdering, Partitioning, PhysicalExpr};
use datafusion_physical_plan::ExecutionPlan;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::values::{ValueOptions, random_array};
use super::{MockSourceExec, StatisticsPrecision, StreamBehavior};
use crate::oracle;

/// How the rows of each partition are split into batches
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchLayout {
    /// Batches of `rows` rows, except that the last batch of a partition can be
    /// smaller
    Fixed(usize),
    /// Batches of a random size between 1 and `max_rows` rows. When
    /// `empty_batches` is true, batches with no rows are added at random
    /// positions, including at the start and end of a partition.
    Random {
        max_rows: usize,
        empty_batches: bool,
    },
    /// All rows of a partition in one batch. A partition without rows has no
    /// batches.
    Single,
}

impl fmt::Display for BatchLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchLayout::Fixed(1) => write!(f, "batches of 1 row"),
            BatchLayout::Fixed(rows) => write!(f, "batches of {rows} rows"),
            BatchLayout::Random {
                max_rows,
                empty_batches,
            } => {
                write!(f, "random batches of up to {max_rows} rows")?;
                if *empty_batches {
                    write!(f, " and empty batches")?;
                }
                Ok(())
            }
            BatchLayout::Single => write!(f, "one batch per partition"),
        }
    }
}

/// Split the rows of `batch` into batches according to `layout`
pub(crate) fn split_rows(
    batch: &RecordBatch,
    layout: BatchLayout,
    rng: &mut StdRng,
) -> Vec<RecordBatch> {
    let rows = batch.num_rows();
    let mut batches = vec![];
    let mut offset = 0;
    match layout {
        BatchLayout::Fixed(size) => {
            let size = size.max(1);
            while offset < rows {
                let len = size.min(rows - offset);
                batches.push(batch.slice(offset, len));
                offset += len;
            }
        }
        BatchLayout::Random {
            max_rows,
            empty_batches,
        } => {
            let maybe_empty = |batches: &mut Vec<RecordBatch>, rng: &mut StdRng| {
                if empty_batches && rng.random_bool(0.2) {
                    batches.push(batch.slice(0, 0));
                }
            };
            while offset < rows {
                maybe_empty(&mut batches, rng);
                let len = rng.random_range(1..=max_rows.max(1)).min(rows - offset);
                batches.push(batch.slice(offset, len));
                offset += len;
            }
            maybe_empty(&mut batches, rng);
        }
        BatchLayout::Single => {
            if rows > 0 {
                batches.push(batch.clone());
            }
        }
    }
    batches
}

/// How rows are assigned to partitions
#[derive(Debug, Clone)]
enum PartitionLayout {
    /// One partition per entry, with that many rows, and `UnknownPartitioning`
    Rows(Vec<usize>),
    /// `num_rows` rows split into `partitions` partitions by the hash of
    /// `exprs`, declared as `Partitioning::Hash`
    Hash {
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        partitions: usize,
        num_rows: usize,
    },
}

/// A column of unique, increasing row ids added to a generated source
#[derive(Debug, Clone)]
struct RowIdColumn {
    name: String,
    first_id: u64,
}

/// Describes a [`MockSourceExec`] to generate.
///
/// Data is random but deterministic for a given seed. Values are drawn from a
/// small domain (see [`Self::with_distinct_values`]) so that generated data has
/// duplicate keys, ties in sort order and matches between inputs.
///
/// Expressions passed to [`Self::with_ordering`] and
/// [`Self::with_hash_partitioning`] refer to the schema passed to
/// [`Self::new`]. The optional row id column is appended after all other
/// columns, so those expressions stay valid for the generated source.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_physical_expr::expressions::col;
/// # use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
/// use datafusion_physical_plan_checks::fixtures::{BatchLayout, SourceSpec};
///
/// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
/// let ordering =
///     LexOrdering::new(vec![PhysicalSortExpr::new_default(col("a", &schema)?)]).unwrap();
/// let source = SourceSpec::new(schema)
///     .with_partition_rows(&[10, 0, 25])
///     .with_batch_layout(BatchLayout::Random { max_rows: 4, empty_batches: true })
///     .with_ordering(ordering)
///     .with_seed(42)
///     .build()?;
/// assert_eq!(source.partitions().len(), 3);
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
#[derive(Debug, Clone)]
pub struct SourceSpec {
    schema: SchemaRef,
    layout: PartitionLayout,
    batch_layout: BatchLayout,
    values: ValueOptions,
    ordering: Option<LexOrdering>,
    precision: StatisticsPrecision,
    behavior: StreamBehavior,
    row_id: Option<RowIdColumn>,
    seed: u64,
}

impl SourceSpec {
    /// Create a spec with one partition of 100 rows, random batches of up to 32
    /// rows, 10% nulls in nullable fields, 16 distinct values per column, exact
    /// statistics, finite streams and seed 0.
    pub fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            layout: PartitionLayout::Rows(vec![100]),
            batch_layout: BatchLayout::Random {
                max_rows: 32,
                empty_batches: false,
            },
            values: ValueOptions {
                null_fraction: 0.1,
                distinct_values: 16,
            },
            ordering: None,
            precision: StatisticsPrecision::Exact,
            behavior: StreamBehavior::Finite,
            row_id: None,
            seed: 0,
        }
    }

    /// Generate one partition per entry of `rows`, with that many rows, and
    /// declare `UnknownPartitioning`
    pub fn with_partition_rows(mut self, rows: &[usize]) -> Self {
        self.layout = PartitionLayout::Rows(rows.to_vec());
        self
    }

    /// Generate `num_rows` rows, split them into `partitions` partitions by the
    /// hash of `exprs` the same way `RepartitionExec` does, and declare
    /// `Partitioning::Hash(exprs, partitions)`
    pub fn with_hash_partitioning(
        mut self,
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        partitions: usize,
        num_rows: usize,
    ) -> Self {
        self.layout = PartitionLayout::Hash {
            exprs,
            partitions,
            num_rows,
        };
        self
    }

    /// Set how the rows of each partition are split into batches
    pub fn with_batch_layout(mut self, batch_layout: BatchLayout) -> Self {
        self.batch_layout = batch_layout;
        self
    }

    /// Set the probability that a value of a nullable field is null
    pub fn with_null_fraction(mut self, null_fraction: f64) -> Self {
        self.values.null_fraction = null_fraction;
        self
    }

    /// Set the number of distinct non-null values each column draws from
    pub fn with_distinct_values(mut self, distinct_values: usize) -> Self {
        self.values.distinct_values = distinct_values;
        self
    }

    /// Sort every partition by `ordering`, and declare it as the output
    /// ordering
    pub fn with_ordering(mut self, ordering: LexOrdering) -> Self {
        self.ordering = Some(ordering);
        self
    }

    /// Set the precision of the statistics the source reports
    pub fn with_statistics_precision(mut self, precision: StatisticsPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set how the streams of the source behave after serving their batches,
    /// for example to generate an unbounded input. See [`StreamBehavior`].
    pub fn with_stream_behavior(mut self, behavior: StreamBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// Append a non-nullable `UInt64` column named `name` with a unique id for
    /// every row. Ids start at `first_id` and increase by one in the order rows
    /// are produced: through partition 0, then partition 1, and so on.
    ///
    /// Row ids let checks track where an output row came from. Use a different
    /// `first_id` for each input of a plan so that ids do not overlap. The
    /// source does not declare an ordering on the row id column. An unbounded
    /// source ([`StreamBehavior::Unbounded`]) repeats its rows, and with them
    /// their ids.
    pub fn with_row_id_column(mut self, name: impl Into<String>, first_id: u64) -> Self {
        self.row_id = Some(RowIdColumn {
            name: name.into(),
            first_id,
        });
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The schema of the generated source, including the row id column if one
    /// was requested
    pub fn schema(&self) -> SchemaRef {
        match &self.row_id {
            None => Arc::clone(&self.schema),
            Some(row_id) => {
                let mut fields: Vec<Arc<Field>> =
                    self.schema.fields().iter().cloned().collect();
                fields.push(Arc::new(Field::new(&row_id.name, DataType::UInt64, false)));
                Arc::new(Schema::new_with_metadata(
                    fields,
                    self.schema.metadata().clone(),
                ))
            }
        }
    }

    /// Generate the data and build the source
    pub fn build(&self) -> Result<MockSourceExec> {
        let mut rng = StdRng::seed_from_u64(self.seed);

        // One batch per partition, holding all of the partition's rows
        let mut partitions = match &self.layout {
            PartitionLayout::Rows(rows) => rows
                .iter()
                .map(|n| self.random_batch(*n, &mut rng))
                .collect::<Result<Vec<_>>>()?,
            PartitionLayout::Hash {
                exprs,
                partitions,
                num_rows,
            } => {
                if *partitions == 0 {
                    return plan_err!("SourceSpec hash partitioning needs a partition");
                }
                let batch = self.random_batch(*num_rows, &mut rng)?;
                oracle::hash_partition(&[batch], exprs, *partitions)?
                    .iter()
                    .map(|batches| concat_batches(&self.schema, batches))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        if let Some(ordering) = &self.ordering {
            partitions = partitions
                .iter()
                .map(|batch| sort_batch(batch, ordering))
                .collect::<Result<_>>()?;
        }

        let schema = self.schema();
        if let Some(row_id) = &self.row_id {
            let mut next_id = row_id.first_id;
            partitions = partitions
                .iter()
                .map(|batch| {
                    let rows = batch.num_rows() as u64;
                    let ids: ArrayRef =
                        Arc::new(UInt64Array::from_iter_values(next_id..next_id + rows));
                    next_id += rows;
                    let mut columns = batch.columns().to_vec();
                    columns.push(ids);
                    Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
                })
                .collect::<Result<_>>()?;
        }

        let partitions = partitions
            .iter()
            .map(|batch| split_rows(batch, self.batch_layout, &mut rng))
            .collect();

        let mut source = MockSourceExec::try_new(schema, partitions)?
            .with_statistics_precision(self.precision);
        if let Some(ordering) = &self.ordering {
            source = source.try_with_output_ordering(ordering.clone())?;
        }
        if let PartitionLayout::Hash {
            exprs, partitions, ..
        } = &self.layout
        {
            source = source
                .try_with_partitioning(Partitioning::Hash(exprs.clone(), *partitions))?;
        }
        source.try_with_stream_behavior(self.behavior)
    }

    /// Generate the data and build the source as an `Arc<dyn ExecutionPlan>`
    pub fn build_arc(&self) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(self.build()?))
    }

    /// A batch of `rows` random rows with the base schema
    fn random_batch(&self, rows: usize, rng: &mut StdRng) -> Result<RecordBatch> {
        let columns = self
            .schema
            .fields()
            .iter()
            .map(|field| random_array(field, rows, self.values, rng))
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(rows));
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?)
    }
}

/// Sort the rows of `batch` by `ordering`
fn sort_batch(batch: &RecordBatch, ordering: &LexOrdering) -> Result<RecordBatch> {
    let mut sorted = oracle::sort_rows(std::slice::from_ref(batch), ordering)?;
    Ok(sorted.remove(0))
}
