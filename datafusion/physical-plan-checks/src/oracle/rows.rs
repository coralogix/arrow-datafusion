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

//! Comparisons of the rows of two sets of batches: as multisets, in order,
//! and as a prefix of a sorted sequence.
//!
//! Values are compared exactly, except floating point values, which are equal
//! when their relative difference is at most [`FLOAT_RELATIVE_TOLERANCE`].
//! Changing batch sizes or batch boundaries can change the order in which an
//! operator adds floating point values, and with it the last bits of the
//! result.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use arrow::row::{RowConverter, SortField};
use datafusion_common::{Result, plan_err};
use datafusion_physical_expr::LexOrdering;

/// Largest relative difference between two floating point values that are
/// considered equal
pub const FLOAT_RELATIVE_TOLERANCE: f64 = 1e-6;

/// One row, split into the values that are compared exactly and the floating
/// point values
#[derive(Debug, Clone)]
struct Row {
    /// The values of every column that is not floating point, in the arrow row
    /// format. Equal values have equal bytes.
    exact: Vec<u8>,
    /// The values of the floating point columns, in column order, or `None`
    /// for null
    floats: Vec<Option<f64>>,
}

/// Converts the rows of batches with the same column types to [`Row`]s
struct RowEncoder {
    types: Vec<DataType>,
    exact_columns: Vec<usize>,
    float_columns: Vec<usize>,
    /// `None` when every column is floating point
    converter: Option<RowConverter>,
}

impl RowEncoder {
    /// An encoder for the column types of the first batch in `sets`, or
    /// `None` if there are no batches
    fn try_new(sets: &[&[RecordBatch]]) -> Result<Option<Self>> {
        let Some(first) = sets.iter().flat_map(|batches| batches.iter()).next() else {
            return Ok(None);
        };
        let types: Vec<DataType> = first
            .columns()
            .iter()
            .map(|column| column.data_type().clone())
            .collect();
        let (float_columns, exact_columns): (Vec<usize>, Vec<usize>) =
            (0..types.len()).partition(|i| types[*i].is_floating());
        let converter = if exact_columns.is_empty() {
            None
        } else {
            let fields = exact_columns
                .iter()
                .map(|i| SortField::new(types[*i].clone()))
                .collect();
            Some(RowConverter::new(fields)?)
        };
        Ok(Some(Self {
            types,
            exact_columns,
            float_columns,
            converter,
        }))
    }

    fn encode(&self, batches: &[RecordBatch]) -> Result<Vec<Row>> {
        let mut rows = vec![];
        for batch in batches {
            let types: Vec<&DataType> =
                batch.columns().iter().map(|c| c.data_type()).collect();
            if types.len() != self.types.len()
                || types.iter().zip(&self.types).any(|(a, b)| *a != b)
            {
                return plan_err!(
                    "cannot compare rows of batches with different column types: \
                     {types:?} and {:?}",
                    self.types
                );
            }
            let exact: Vec<Vec<u8>> = match &self.converter {
                Some(converter) => {
                    let columns: Vec<ArrayRef> = self
                        .exact_columns
                        .iter()
                        .map(|i| Arc::clone(batch.column(*i)))
                        .collect();
                    let encoded = converter.convert_columns(&columns)?;
                    encoded.iter().map(|row| row.as_ref().to_vec()).collect()
                }
                None => vec![vec![]; batch.num_rows()],
            };
            let floats = self
                .float_columns
                .iter()
                .map(|i| {
                    let column = cast(batch.column(*i), &DataType::Float64)?;
                    Ok(column.as_primitive::<Float64Type>().iter().collect())
                })
                .collect::<Result<Vec<Vec<Option<f64>>>>>()?;
            for (row, exact) in exact.into_iter().enumerate() {
                rows.push(Row {
                    exact,
                    floats: floats.iter().map(|column| column[row]).collect(),
                });
            }
        }
        Ok(rows)
    }
}

/// Encode two sets of batches with the same encoder
fn encode_pair(
    left: &[RecordBatch],
    right: &[RecordBatch],
) -> Result<(Vec<Row>, Vec<Row>)> {
    let Some(encoder) = RowEncoder::try_new(&[left, right])? else {
        return Ok((vec![], vec![]));
    };
    Ok((encoder.encode(left)?, encoder.encode(right)?))
}

fn floats_equal(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a == b
                || (a.is_nan() && b.is_nan())
                || (a - b).abs() <= FLOAT_RELATIVE_TOLERANCE * a.abs().max(b.abs())
        }
        _ => false,
    }
}

fn rows_equal(a: &Row, b: &Row) -> bool {
    a.exact == b.exact
        && a.floats
            .iter()
            .zip(&b.floats)
            .all(|(a, b)| floats_equal(*a, *b))
}

/// A total order on rows: by the exact values, then by the floating point
/// values, with nulls first
fn compare_rows(a: &Row, b: &Row) -> Ordering {
    a.exact.cmp(&b.exact).then_with(|| {
        for (a, b) in a.floats.iter().zip(&b.floats) {
            let ordering = match (a, b) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(a), Some(b)) => a.total_cmp(b),
            };
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        Ordering::Equal
    })
}

/// Match every candidate row to a different, equal reference row, and return
/// the number of candidate rows left without a match.
///
/// Both sides are sorted and matched in order. Rows with different exact
/// values never match, so only floating point values can make the order of
/// the two sides differ. With at most one floating point column the greedy
/// matching finds a match for every candidate row whenever one exists. With
/// several, rows are matched in lexicographic order, which can miss a
/// matching when values within the tolerance sort in a different order.
fn count_unmatched(candidate: &[Row], reference: &[Row]) -> usize {
    let mut candidate: Vec<&Row> = candidate.iter().collect();
    let mut reference: Vec<&Row> = reference.iter().collect();
    candidate.sort_by(|a, b| compare_rows(a, b));
    reference.sort_by(|a, b| compare_rows(a, b));
    let mut unmatched = 0;
    let mut next = 0;
    for row in candidate {
        // Reference rows that sort before this row without being equal to it
        // cannot be equal to any later candidate row either
        while next < reference.len()
            && compare_rows(reference[next], row) == Ordering::Less
            && !rows_equal(reference[next], row)
        {
            next += 1;
        }
        if next < reference.len() && rows_equal(reference[next], row) {
            next += 1;
        } else {
            unmatched += 1;
        }
    }
    unmatched
}

/// Returns the number of rows of `candidate` that do not appear in
/// `reference`, counting rows as a multiset: a row that appears twice in
/// `candidate` must appear twice in `reference`. Zero means `candidate` is
/// contained in `reference`.
///
/// Returns an error if the batches do not all have the same column types.
pub fn unmatched_rows(
    candidate: &[RecordBatch],
    reference: &[RecordBatch],
) -> Result<usize> {
    let (candidate, reference) = encode_pair(candidate, reference)?;
    Ok(count_unmatched(&candidate, &reference))
}

/// Returns true if `left` and `right` have the same rows, as multisets, in
/// any order and split into batches in any way.
///
/// Returns an error if the batches do not all have the same column types.
pub fn same_rows(left: &[RecordBatch], right: &[RecordBatch]) -> Result<bool> {
    let (left, right) = encode_pair(left, right)?;
    Ok(left.len() == right.len() && count_unmatched(&left, &right) == 0)
}

/// Returns true if `left` and `right` have the same rows in the same order,
/// split into batches in any way.
///
/// Returns an error if the batches do not all have the same column types.
pub fn same_rows_in_order(left: &[RecordBatch], right: &[RecordBatch]) -> Result<bool> {
    let (left, right) = encode_pair(left, right)?;
    Ok(left.len() == right.len()
        && left.iter().zip(&right).all(|(a, b)| rows_equal(a, b)))
}

/// Values of each row in the arrow row format, one entry per row
type EncodedRows = Vec<Vec<u8>>;

/// The value of the sort key of every row of `left` and `right`, in the
/// arrow row format, so that equal keys have equal bytes
fn sort_keys(
    left: &[RecordBatch],
    right: &[RecordBatch],
    ordering: &LexOrdering,
) -> Result<(EncodedRows, EncodedRows)> {
    let Some(first) = left.iter().chain(right).next() else {
        return Ok((vec![], vec![]));
    };
    let fields = ordering
        .iter()
        .map(|sort_expr| {
            let data_type = sort_expr.expr.data_type(&first.schema())?;
            Ok(SortField::new_with_options(data_type, sort_expr.options))
        })
        .collect::<Result<Vec<_>>>()?;
    let converter = RowConverter::new(fields)?;
    let keys = |batches: &[RecordBatch]| -> Result<EncodedRows> {
        let mut keys = vec![];
        for batch in batches {
            let columns = ordering
                .iter()
                .map(|sort_expr| {
                    sort_expr.expr.evaluate(batch)?.into_array(batch.num_rows())
                })
                .collect::<Result<Vec<ArrayRef>>>()?;
            let rows = converter.convert_columns(&columns)?;
            keys.extend(rows.iter().map(|row| row.as_ref().to_vec()));
        }
        Ok(keys)
    };
    Ok((keys(left)?, keys(right)?))
}

/// Checks that `candidate` is a prefix of `reference` when rows that tie on
/// `ordering` may appear in any order and may be exchanged for each other.
/// This is what limiting a sorted sequence to its first rows may produce when
/// ties are broken differently. Returns the position of the first row of
/// `candidate` where this fails, or `None` if it holds.
///
/// Precisely, for every position `i` of `candidate`:
///
/// - `reference` has a row at `i`, and it has the same sort key as the row of
///   `candidate` at `i`.
/// - The rows of `candidate` in each run of equal sort keys appear, as a
///   multiset, among the rows of `reference` with that sort key in the same
///   run. The run in `reference` can continue past the end of `candidate`,
///   so which of the tied rows at the end are kept does not matter.
///
/// `candidate` can be shorter than `reference`, including empty. Sort keys
/// are compared exactly, while the other values follow the tolerance for
/// floating point values. Neither side has to be sorted: if `reference` is
/// not sorted, `candidate` must be a prefix of it with the same exceptions.
///
/// Returns an error if the ordering cannot be evaluated on the batches, or
/// the batches do not all have the same column types.
pub fn first_non_prefix_row(
    reference: &[RecordBatch],
    candidate: &[RecordBatch],
    ordering: &LexOrdering,
) -> Result<Option<usize>> {
    let (reference_keys, candidate_keys) = sort_keys(reference, candidate, ordering)?;
    let (reference_rows, candidate_rows) = encode_pair(reference, candidate)?;

    let mut start = 0;
    while start < candidate_keys.len() {
        let key = &candidate_keys[start];
        // The run of rows with this key in `candidate`, whose keys must match
        // `reference` position by position
        let mut end = start;
        while end < candidate_keys.len() && candidate_keys[end] == *key {
            if reference_keys.get(end) != Some(key) {
                return Ok(Some(end));
            }
            end += 1;
        }
        // The same run in `reference`, which can be longer
        let mut reference_end = end;
        while reference_keys.get(reference_end) == Some(key) {
            reference_end += 1;
        }
        let unmatched = count_unmatched(
            &candidate_rows[start..end],
            &reference_rows[start..reference_end],
        );
        if unmatched > 0 {
            return Ok(Some(start));
        }
        start = end;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{Field, Schema, SchemaRef};
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;

    use super::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("x", DataType::Float64, true),
        ]))
    }

    /// A batch with one row per entry of `rows`
    fn batch(rows: &[(Option<i32>, &str, Option<f64>)]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(rows.iter().map(|r| r.0).collect::<Int32Array>()),
                Arc::new(rows.iter().map(|r| Some(r.1)).collect::<StringArray>()),
                Arc::new(rows.iter().map(|r| r.2).collect::<Float64Array>()),
            ],
        )
        .unwrap()
    }

    fn by_k() -> LexOrdering {
        LexOrdering::new(vec![PhysicalSortExpr::new_default(
            col("k", &schema()).unwrap(),
        )])
        .unwrap()
    }

    #[test]
    fn same_rows_ignores_order_and_batch_boundaries() {
        let a = [
            batch(&[(Some(1), "a", Some(1.0)), (Some(2), "b", None)]),
            batch(&[(None, "c", Some(3.0))]),
        ];
        let b = [
            batch(&[]),
            batch(&[(None, "c", Some(3.0))]),
            batch(&[(Some(2), "b", None), (Some(1), "a", Some(1.0))]),
        ];
        assert!(same_rows(&a, &b).unwrap());
        assert!(!same_rows_in_order(&a, &b).unwrap());
        assert!(same_rows_in_order(&a, &a).unwrap());
    }

    #[test]
    fn same_rows_counts_duplicates() {
        let once = [batch(&[(Some(1), "a", None), (Some(2), "b", None)])];
        let twice = [batch(&[
            (Some(1), "a", None),
            (Some(1), "a", None),
            (Some(2), "b", None),
        ])];
        assert!(!same_rows(&once, &twice).unwrap());
        assert_eq!(unmatched_rows(&once, &twice).unwrap(), 0);
        assert_eq!(unmatched_rows(&twice, &once).unwrap(), 1);
    }

    #[test]
    fn nulls_differ_from_values() {
        let null = [batch(&[(None, "a", None)])];
        let value = [batch(&[(Some(0), "a", Some(0.0))])];
        assert!(!same_rows(&null, &value).unwrap());
        assert_eq!(unmatched_rows(&null, &value).unwrap(), 1);
    }

    #[test]
    fn floats_within_the_tolerance_are_equal() {
        let x = 0.1 + 0.2;
        let a = [batch(&[
            (Some(1), "a", Some(x)),
            (Some(1), "a", Some(10.0)),
        ])];
        let b = [batch(&[
            (Some(1), "a", Some(10.0 * (1.0 + 1e-9))),
            (Some(1), "a", Some(0.3)),
        ])];
        assert!(same_rows(&a, &b).unwrap());
        assert!(!same_rows_in_order(&a, &b).unwrap());

        let far = [batch(&[
            (Some(1), "a", Some(x)),
            (Some(1), "a", Some(10.001)),
        ])];
        assert!(!same_rows(&a, &far).unwrap());
        assert_eq!(unmatched_rows(&far, &a).unwrap(), 1);
    }

    #[test]
    fn floats_close_to_each_other_are_matched_in_order() {
        // Every value is within the tolerance of its neighbours, so a greedy
        // matching that did not sort both sides could leave a row unmatched
        let a = [batch(&[
            (Some(1), "a", Some(1.0)),
            (Some(1), "a", Some(1.0 + 8e-7)),
            (Some(1), "a", Some(1.0 + 1.6e-6)),
        ])];
        let b = [batch(&[
            (Some(1), "a", Some(1.0 + 1.6e-6 + 5e-7)),
            (Some(1), "a", Some(1.0 + 5e-7)),
            (Some(1), "a", Some(1.0 + 8e-7 + 5e-7)),
        ])];
        assert!(same_rows(&a, &b).unwrap());
    }

    #[test]
    fn nan_equals_nan() {
        let a = [batch(&[(Some(1), "a", Some(f64::NAN))])];
        assert!(same_rows(&a, &a).unwrap());
        assert!(same_rows_in_order(&a, &a).unwrap());
    }

    #[test]
    fn empty_inputs() {
        let rows = [batch(&[(Some(1), "a", None)])];
        assert!(same_rows(&[], &[]).unwrap());
        assert!(same_rows(&[], &[batch(&[])]).unwrap());
        assert!(!same_rows(&rows, &[]).unwrap());
        assert_eq!(unmatched_rows(&[], &rows).unwrap(), 0);
        assert_eq!(unmatched_rows(&rows, &[]).unwrap(), 1);
        assert_eq!(first_non_prefix_row(&rows, &[], &by_k()).unwrap(), None);
        assert_eq!(first_non_prefix_row(&[], &rows, &by_k()).unwrap(), Some(0));
    }

    #[test]
    fn different_column_types_are_an_error() {
        let other = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, true)])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1]))],
        )
        .unwrap();
        let rows = [batch(&[(Some(1), "a", None)])];
        assert!(same_rows(&rows, &[other]).is_err());
    }

    #[test]
    fn prefix_modulo_ties() {
        // Sorted by k; rows with k = 2 tie
        let reference = [
            batch(&[(Some(1), "a", None), (Some(2), "b", None)]),
            batch(&[
                (Some(2), "c", None),
                (Some(2), "d", None),
                (Some(3), "e", None),
            ]),
        ];
        let prefix = |rows: &[(Option<i32>, &str, Option<f64>)]| {
            first_non_prefix_row(&reference, &[batch(rows)], &by_k()).unwrap()
        };
        // An exact prefix, of any length
        assert_eq!(prefix(&[]), None);
        assert_eq!(prefix(&[(Some(1), "a", None)]), None);
        assert_eq!(
            prefix(&[
                (Some(1), "a", None),
                (Some(2), "b", None),
                (Some(2), "c", None),
                (Some(2), "d", None),
                (Some(3), "e", None),
            ]),
            None
        );
        // Tied rows in another order, or other tied rows at the end
        assert_eq!(
            prefix(&[
                (Some(1), "a", None),
                (Some(2), "d", None),
                (Some(2), "b", None)
            ]),
            None
        );
        assert_eq!(prefix(&[(Some(1), "a", None), (Some(2), "c", None)]), None);
        // A row that is not first in sort order
        assert_eq!(prefix(&[(Some(2), "b", None)]), Some(0));
        // A row skipped before a later sort key
        assert_eq!(
            prefix(&[(Some(1), "a", None), (Some(3), "e", None)]),
            Some(1)
        );
        // A tied row that is not in the reference
        assert_eq!(
            prefix(&[(Some(1), "a", None), (Some(2), "z", None)]),
            Some(1)
        );
        // A tied row repeated more often than in the reference
        assert_eq!(
            prefix(&[
                (Some(1), "a", None),
                (Some(2), "b", None),
                (Some(2), "b", None)
            ]),
            Some(1)
        );
        // Longer than the reference
        let mut longer: Vec<_> = reference.to_vec();
        longer.push(batch(&[(Some(4), "f", None)]));
        assert_eq!(
            first_non_prefix_row(&reference, &longer, &by_k()).unwrap(),
            Some(5)
        );
    }

    #[test]
    fn prefix_respects_sort_options_and_nulls() {
        let descending = LexOrdering::new(vec![PhysicalSortExpr::new(
            col("k", &schema()).unwrap(),
            arrow::compute::SortOptions {
                descending: true,
                nulls_first: true,
            },
        )])
        .unwrap();
        let reference = [batch(&[
            (None, "a", None),
            (None, "b", None),
            (Some(2), "c", None),
            (Some(1), "d", None),
        ])];
        let candidate = [batch(&[
            (None, "b", None),
            (None, "a", None),
            (Some(2), "c", None),
        ])];
        assert_eq!(
            first_non_prefix_row(&reference, &candidate, &descending).unwrap(),
            None
        );
        let candidate = [batch(&[(None, "a", None), (Some(1), "d", None)])];
        assert_eq!(
            first_non_prefix_row(&reference, &candidate, &descending).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn every_generated_type_can_be_compared() {
        use crate::fixtures::SourceSpec;
        use arrow::datatypes::TimeUnit;

        // The types `SourceSpec` generates
        let types = [
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("+01:00".into())),
            DataType::Decimal128(10, 2),
            DataType::Null,
        ];
        let fields: Vec<Field> = types
            .iter()
            .enumerate()
            .map(|(i, t)| Field::new(format!("c{i}"), t.clone(), true))
            .collect();
        let generated = SourceSpec::new(Arc::new(Schema::new(fields[..19].to_vec())))
            .with_partition_rows(&[50])
            .build()
            .unwrap();
        let batches = &generated.partitions()[0];
        assert!(same_rows(batches, batches).unwrap());
        assert!(same_rows_in_order(batches, batches).unwrap());
        assert_eq!(unmatched_rows(batches, batches).unwrap(), 0);
        let reversed: Vec<RecordBatch> = batches.iter().rev().cloned().collect();
        assert!(same_rows(batches, &reversed).unwrap());

        // `PlaceholderRowExec` produces `Null` columns
        let nulls = RecordBatch::try_new(
            Arc::new(Schema::new(vec![fields[19].clone()])),
            vec![arrow::array::new_null_array(&DataType::Null, 3)],
        )
        .unwrap();
        let nulls = std::slice::from_ref(&nulls);
        assert!(same_rows(nulls, nulls).unwrap());
    }
}
