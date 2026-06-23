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
//! `CarryExec` is pipeline-breaking: it buffers all batches per input
//! partition, derives each partition's final cumulative value from the
//! last row of the last batch (no separate state — the buffered batches
//! ARE the state), computes the prefix sum across partition finals, and
//! re-emits each partition's batches with that prefix added to the
//! aggregate column. Output partitioning matches input.
//!
//! Stub: currently a passthrough (no buffering, no offset). Exists to
//! anchor the plan shape so `ParallelWindow`'s cumulative branch has
//! somewhere to land. Real prefix-scan body is the next commit.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_execution::TaskContext;

use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};

#[derive(Debug)]
pub struct CarryExec {
    input: Arc<dyn ExecutionPlan>,
    /// Column index of the window aggregate output in the input schema.
    /// Real implementation adds the prefix sum to this column; stub
    /// ignores it.
    agg_col: usize,
    cache: Arc<PlanProperties>,
}

impl CarryExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, agg_col: usize) -> Self {
        let cache = Arc::clone(input.properties());
        Self {
            input,
            agg_col,
            cache,
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
        let input = children.swap_remove(0);
        let cache = Arc::clone(input.properties());
        Ok(Arc::new(Self {
            input,
            agg_col: self.agg_col,
            cache,
        }))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Passthrough stub. Real implementation buffers and offsets.
        self.input.execute(partition, context)
    }
}
