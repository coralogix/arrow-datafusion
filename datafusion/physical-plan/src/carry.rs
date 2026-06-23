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

//! Parallel prefix-scan carry above a `BoundedWindowAggExec` running
//! per-partition over `RangeRepartitionExec`-routed input.
//!
//! Each input partition produces a cumulative aggregate starting at zero.
//! `CarryExec` is pipeline-breaking: the first output partition to poll
//! drains every input partition fully into per-partition `Vec<RecordBatch>`,
//! derives each partition's final cumulative value from the last row of
//! the last batch (no separate finals state — the buffered batches ARE
//! the state), and computes the prefix sum across partition finals.
//! Concurrent and subsequent output-partition polls await the same
//! memoized result via `OnceCell`. Each output stream re-emits its
//! buffered batches with `prefix` added to the aggregate column.
//!
//! Output partitioning equals input partitioning (N → N).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::kernels::numeric::add;
use datafusion_common::{Result, ScalarValue, internal_datafusion_err};
use datafusion_execution::TaskContext;
use futures::StreamExt;
use tokio::sync::OnceCell;

use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};

#[derive(Debug)]
pub struct CarryExec {
    input: Arc<dyn ExecutionPlan>,
    /// Column index in the input schema whose values are offset by the
    /// prefix sum of prior partitions' finals.
    agg_col: usize,
    cache: Arc<PlanProperties>,
    /// First output-partition poll runs the gather; concurrent polls await
    /// its completion; later polls read the cached result. The error path
    /// stores a stringified message because `DataFusionError` isn't
    /// `Clone` and the same error must surface on every output partition.
    gathered: Arc<OnceCell<GatherResult>>,
}

type GatherResult = std::result::Result<Arc<Vec<PartitionPayload>>, Arc<String>>;

#[derive(Debug)]
struct PartitionPayload {
    batches: Vec<RecordBatch>,
    /// Already-prefix-summed offset to add to the agg column on every row
    /// of every batch. `prefix[0]` is the additive identity for the agg
    /// column's data type.
    prefix: ScalarValue,
}

impl CarryExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, agg_col: usize) -> Self {
        let cache = Arc::clone(input.properties());
        Self {
            input,
            agg_col,
            cache,
            gathered: Arc::new(OnceCell::new()),
        }
    }
}

impl DisplayAs for CarryExec {
    fn fmt_as(
        &self,
        _t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "CarryExec")
    }
}

impl ExecutionPlan for CarryExec {
    fn name(&self) -> &'static str {
        "CarryExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(children.swap_remove(0), self.agg_col)))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let gathered = Arc::clone(&self.gathered);
        let input = Arc::clone(&self.input);
        let agg_col = self.agg_col;
        let schema = self.schema();

        let body = async move {
            let payloads = gathered
                .get_or_init(|| async {
                    gather(input, context, agg_col)
                        .await
                        .map(Arc::new)
                        .map_err(|e| Arc::new(e.to_string()))
                })
                .await;
            let payloads = match payloads {
                Ok(p) => p,
                Err(msg) => return Err(internal_datafusion_err!("{}", msg)),
            };
            let payload = &payloads[partition];
            let prefix = payload.prefix.clone();
            // RecordBatch::clone is cheap (Arc'd columns).
            let batches: Vec<RecordBatch> = payload.batches.clone();
            Ok(futures::stream::iter(
                batches
                    .into_iter()
                    .map(move |batch| offset_batch(&batch, agg_col, &prefix)),
            ))
        };

        use futures::stream::{TryStreamExt, once};
        let stream = once(body).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// Drain every input partition fully, derive each partition's final from
/// the last row of its last batch, and compute the prefix sum across
/// finals. Empty partitions contribute the additive identity; their
/// prefix equals the running total at that point.
async fn gather(
    input: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    agg_col: usize,
) -> Result<Vec<PartitionPayload>> {
    let n = input.output_partitioning().partition_count();
    let mut buffers: Vec<Vec<RecordBatch>> = Vec::with_capacity(n);
    for k in 0..n {
        let mut stream = input.execute(k, Arc::clone(&ctx))?;
        let mut buf = Vec::new();
        while let Some(item) = stream.next().await {
            buf.push(item?);
        }
        buffers.push(buf);
    }

    // Derive each partition's final + cumulative prefix. `running` starts
    // as the zero scalar in the agg column's data type, taken from the
    // first non-empty batch we find.
    let agg_type = buffers
        .iter()
        .flat_map(|b| b.iter())
        .next()
        .map(|b| b.column(agg_col).data_type().clone());
    let Some(agg_type) = agg_type else {
        // No data anywhere — every partition gets an empty payload with a
        // null prefix (offset_batch passes through unchanged on null).
        return Ok((0..n)
            .map(|_| PartitionPayload {
                batches: Vec::new(),
                prefix: ScalarValue::Null,
            })
            .collect());
    };
    let mut running = ScalarValue::new_zero(&agg_type)?;
    let mut payloads = Vec::with_capacity(n);
    for batches in buffers {
        let prefix = running.clone();
        if let Some(last) = batches.last() {
            let final_i =
                ScalarValue::try_from_array(last.column(agg_col), last.num_rows() - 1)?;
            running = running.add(&final_i)?;
        }
        payloads.push(PartitionPayload { batches, prefix });
    }
    Ok(payloads)
}

/// Replace the agg column with `agg + prefix` (broadcast scalar add).
fn offset_batch(
    batch: &RecordBatch,
    agg_col: usize,
    prefix: &ScalarValue,
) -> Result<RecordBatch> {
    if prefix.is_null() {
        // Only happens when there's no data anywhere; pass through.
        return Ok(batch.clone());
    }
    let agg = batch.column(agg_col);
    // Replicate the prefix to match batch length. `arrow::array::Scalar`
    // would let us broadcast a single-element array as a Datum, but the
    // replicate cost is negligible (one scalar per batch).
    let prefix_array = prefix.to_array_of_size(batch.num_rows())?;
    let new_agg: ArrayRef = add(&agg.as_ref(), &prefix_array.as_ref())?;
    let mut cols = batch.columns().to_vec();
    cols[agg_col] = new_agg;
    Ok(RecordBatch::try_new(batch.schema(), cols)?)
}
