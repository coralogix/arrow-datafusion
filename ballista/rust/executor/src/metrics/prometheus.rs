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

use crate::metrics::ExecutorMetricsCollector;
use ballista_core::error::{BallistaError, Result};
use ballista_core::execution_plans::ShuffleWriterExec;
use datafusion::physical_plan::ExecutionPlan;
use prometheus::{register_histogram, register_int_counter, Histogram, IntCounter};

/// ShuffleWriteExec metrics
pub struct StageMetrics {
    /// Histogram of total stage execution times
    pub write_time: Histogram,
    /// Counter for total input rows
    pub input_rows: IntCounter,
    /// Counter for total output rows
    pub output_rows: IntCounter,
}

impl StageMetrics {
    pub fn new() -> Result<Self> {
        let write_time = register_histogram!(
            "shuffle_write_time",
            "Histogram of stage execution time",
            vec![50_f64, 500_f64, 1000_f64, 5000_f64,]
        )
        .map_err(|e| {
            BallistaError::General(format!("Error registering metric: {:?}", e))
        })?;
        let output_rows =
            register_int_counter!("shuffle_output_rows", "Total output rows").map_err(
                |e| BallistaError::General(format!("Error registering metric: {:?}", e)),
            )?;
        let input_rows = register_int_counter!("shuffle_input_rows", "Total disk spills")
            .map_err(|e| {
                BallistaError::General(format!("Error registering metric: {:?}", e))
            })?;
        Ok(Self {
            write_time,
            input_rows,
            output_rows,
        })
    }
}

pub struct PrometheusMetricsCollector {
    metrics: StageMetrics,
}

impl PrometheusMetricsCollector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            metrics: StageMetrics::new()?,
        })
    }
}

impl ExecutorMetricsCollector for PrometheusMetricsCollector {
    fn record_stage(&self, _partition: usize, plan: ShuffleWriterExec) {
        if let Some(metrics) = plan.metrics().map(|m| m.aggregate_by_partition()) {
            if let Some(write_time) = metrics.time("write_time") {
                let millis = (write_time / 1_000_000) as f64;
                self.metrics.write_time.observe(millis);
            }

            if let Some(output_rows) = metrics.output_rows() {
                self.metrics.output_rows.inc_by(output_rows as u64);
            }

            if let Some(input_rows) = metrics.count("input_rows") {
                self.metrics.input_rows.inc_by(input_rows as u64);
            }
        }
    }
}
