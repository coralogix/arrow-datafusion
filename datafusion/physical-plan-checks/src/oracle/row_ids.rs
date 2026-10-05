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

//! Tracking the rows of an input through an output by a column of unique row
//! ids, such as the one `SourceSpec::with_row_ids` adds.

use std::collections::{HashMap, HashSet};

use arrow::array::{AsArray, RecordBatch, UInt64Array};
use arrow::datatypes::UInt64Type;
use datafusion_common::{Result, plan_datafusion_err};

/// Where the rows of an input appear in an output, see [`input_order`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputOrder {
    /// The rows cannot be tracked: the input ids are not unique, or no output
    /// row has an id of the input
    Untracked,
    /// Every row of the output is a row of the input, each output partition
    /// has rows of one input partition only, and they appear in the order of
    /// that partition. A row can repeat, as long as its copies are next to
    /// each other.
    Kept {
        /// The number of input partitions whose rows are in the output
        input_partitions: usize,
        /// The largest number of distinct input rows in one output partition
        max_partition_rows: usize,
    },
    /// Row `row` of output partition `partition` breaks the order: it is not
    /// a row of the input, it is a row of another input partition than the
    /// rows before it, or it comes before the row before it in its input
    /// partition
    Broken { partition: usize, row: usize },
}

/// Returns where the rows of `input` appear in `output`, both given as the
/// batches of each partition. Each input row is identified by the id in
/// column `input_ids` of the input batches, which the output copies into
/// column `output_ids` of its batches. Both columns must be `UInt64`. Positions
/// count rows across the batches of a partition.
///
/// Returns an error if a batch does not have such a column.
pub fn input_order(
    input: &[Vec<RecordBatch>],
    input_ids: usize,
    output: &[Vec<RecordBatch>],
    output_ids: usize,
) -> Result<InputOrder> {
    // The partition and position of every input row, by id
    let mut positions: HashMap<u64, (usize, usize)> = HashMap::new();
    for (p, batches) in input.iter().enumerate() {
        let mut position = 0;
        for batch in batches {
            for id in row_ids(batch, input_ids)? {
                let Some(id) = id else {
                    return Ok(InputOrder::Untracked);
                };
                if positions.insert(id, (p, position)).is_some() {
                    return Ok(InputOrder::Untracked);
                }
                position += 1;
            }
        }
    }

    // The input partition and position of each row of each output partition,
    // if it is a row of the input
    let output = output
        .iter()
        .map(|batches| {
            let mut rows = vec![];
            for batch in batches {
                let ids = row_ids(batch, output_ids)?;
                rows.extend(ids.map(|id| id.and_then(|id| positions.get(&id).copied())));
            }
            Ok(rows)
        })
        .collect::<Result<Vec<_>>>()?;
    if output.iter().flatten().all(Option::is_none) {
        return Ok(InputOrder::Untracked);
    }

    let mut input_partitions = HashSet::new();
    let mut max_partition_rows = 0;
    for (partition, rows) in output.iter().enumerate() {
        let mut previous: Option<(usize, usize)> = None;
        let mut distinct = 0;
        for (row, current) in rows.iter().enumerate() {
            let Some(current) = *current else {
                return Ok(InputOrder::Broken { partition, row });
            };
            match previous {
                Some(previous) if current == previous => continue,
                Some((input_partition, position))
                    if current.0 != input_partition || current.1 < position =>
                {
                    return Ok(InputOrder::Broken { partition, row });
                }
                _ => {}
            }
            previous = Some(current);
            distinct += 1;
            input_partitions.insert(current.0);
        }
        max_partition_rows = max_partition_rows.max(distinct);
    }
    Ok(InputOrder::Kept {
        input_partitions: input_partitions.len(),
        max_partition_rows,
    })
}

/// The values of the `UInt64` column `column` of `batch`
fn row_ids(
    batch: &RecordBatch,
    column: usize,
) -> Result<impl Iterator<Item = Option<u64>> + '_> {
    let ids: &UInt64Array = batch
        .columns()
        .get(column)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            plan_datafusion_err!("the batch has no UInt64 row id column at {column}")
        })?;
    Ok(ids.iter())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    /// A batch with an `Int32` column and a row id column
    fn batch(ids: &[Option<u64>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int32, false),
            Field::new("id", DataType::UInt64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![0; ids.len()])),
                Arc::new(UInt64Array::from(ids.to_vec())),
            ],
        )
        .unwrap()
    }

    /// The input: ids 0 to 4 in partition 0, and 10 to 12 in partition 1
    fn input() -> Vec<Vec<RecordBatch>> {
        let ids = |ids: std::ops::Range<u64>| ids.map(Some).collect::<Vec<_>>();
        vec![
            vec![batch(&ids(0..2)), batch(&[]), batch(&ids(2..5))],
            vec![batch(&ids(10..13))],
        ]
    }

    fn order(output: &[&[Option<u64>]]) -> InputOrder {
        let output: Vec<Vec<RecordBatch>> =
            output.iter().map(|ids| vec![batch(ids)]).collect();
        input_order(&input(), 1, &output, 1).unwrap()
    }

    #[test]
    fn kept_order() {
        // Subsequences of the input partitions, with repeated rows
        assert_eq!(
            order(&[&[Some(0), Some(2), Some(2), Some(4)], &[], &[Some(11)]]),
            InputOrder::Kept {
                input_partitions: 2,
                max_partition_rows: 3,
            }
        );
    }

    #[test]
    fn broken_order() {
        // Out of order
        assert_eq!(
            order(&[&[Some(1), Some(0)]]),
            InputOrder::Broken {
                partition: 0,
                row: 1
            }
        );
        // A repeated row that is not next to its copy
        assert_eq!(
            order(&[&[Some(0), Some(1), Some(0)]]),
            InputOrder::Broken {
                partition: 0,
                row: 2
            }
        );
        // Rows of two input partitions in one output partition
        assert_eq!(
            order(&[&[Some(12)], &[Some(3), Some(10)]]),
            InputOrder::Broken {
                partition: 1,
                row: 1
            }
        );
        // A row that is not a row of the input, or has no id
        assert_eq!(
            order(&[&[Some(0), Some(99)]]),
            InputOrder::Broken {
                partition: 0,
                row: 1
            }
        );
        assert_eq!(
            order(&[&[Some(0)], &[None]]),
            InputOrder::Broken {
                partition: 1,
                row: 0
            }
        );
    }

    #[test]
    fn untracked_rows() {
        // No output row is a row of the input
        assert_eq!(order(&[&[Some(99)], &[None]]), InputOrder::Untracked);
        assert_eq!(order(&[&[]]), InputOrder::Untracked);
        // Input ids that are not unique
        let input = vec![vec![batch(&[Some(0), Some(0)])]];
        let output = vec![vec![batch(&[Some(0)])]];
        assert_eq!(
            input_order(&input, 1, &output, 1).unwrap(),
            InputOrder::Untracked
        );
        // A column that is not a row id column is an error
        assert!(input_order(&input, 0, &output, 1).is_err());
    }
}
