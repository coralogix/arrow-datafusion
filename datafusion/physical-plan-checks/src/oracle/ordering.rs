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

use arrow::array::{ArrayRef, RecordBatch};
use arrow::compute::{SortColumn, concat_batches, lexsort_to_indices, take_record_batch};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion_common::Result;
use datafusion_physical_expr::LexOrdering;

/// Returns the rows of `batches` in one batch, sorted by `ordering`, or no
/// batch if there are no batches. Rows that compare equal on every sort
/// expression can be in any order.
pub fn sort_rows(
    batches: &[RecordBatch],
    ordering: &LexOrdering,
) -> Result<Vec<RecordBatch>> {
    let Some(first) = batches.first() else {
        return Ok(vec![]);
    };
    let batch = concat_batches(&first.schema(), batches)?;
    let columns = ordering
        .iter()
        .map(|sort_expr| {
            Ok(SortColumn {
                values: sort_expr
                    .expr
                    .evaluate(&batch)?
                    .into_array(batch.num_rows())?,
                options: Some(sort_expr.options),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let indices = lexsort_to_indices(&columns, None)?;
    Ok(vec![take_record_batch(&batch, &indices)?])
}

/// Returns the position of the first row of `batches` that sorts before the
/// row just before it according to `ordering`, or `None` if the rows are
/// sorted. Positions count rows across all batches, in order.
///
/// Rows that compare equal on every sort expression are considered sorted in
/// either order.
pub fn first_unsorted_row(
    batches: &[RecordBatch],
    ordering: &LexOrdering,
) -> Result<Option<usize>> {
    let fields = batches
        .first()
        .map(|batch| {
            ordering
                .iter()
                .map(|sort_expr| {
                    let data_type = sort_expr.expr.data_type(&batch.schema())?;
                    Ok(SortField::new_with_options(data_type, sort_expr.options))
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let Some(fields) = fields else {
        return Ok(None);
    };
    let converter = RowConverter::new(fields)?;

    let mut previous: Option<OwnedRow> = None;
    let mut position = 0;
    for batch in batches {
        let columns = ordering
            .iter()
            .map(|sort_expr| sort_expr.expr.evaluate(batch)?.into_array(batch.num_rows()))
            .collect::<Result<Vec<ArrayRef>>>()?;
        let rows = converter.convert_columns(&columns)?;
        for row in rows.iter() {
            if previous.as_ref().is_some_and(|prev| row < prev.row()) {
                return Ok(Some(position));
            }
            previous = Some(row.owned());
            position += 1;
        }
    }
    Ok(None)
}
