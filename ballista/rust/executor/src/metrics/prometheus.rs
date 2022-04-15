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
use prometheus::{
    register, register_counter, register_histogram, register_int_counter, Counter,
    Encoder, Histogram, IntCounter, Opts, TextEncoder,
};

pub struct StageMetrics {
    pub write_time: Histogram,
    pub input_rows: IntCounter,
    pub output_rows: IntCounter
}

impl StageMetrics {
    pub fn new() -> Result<Self> {
        let write_time = register_histogram!(
            "shuffle_write_time",
            "Histogram of stage execution time",
            vec![50.0, 500.0, 1000.0, 1500.0, 2000.0, 5000.0]
        )
            .map_err(|e| {
                BallistaError::General(format!("Error registering metric: {:?}", e))
            })?;
        let output_rows =
            register_int_counter!("shuffle_output_rows", "Total output rows")
                .map_err(|e| {
                    BallistaError::General(format!("Error registering metric: {:?}", e))
                })?;
        let input_rows =
            register_int_counter!("shuffle_input_rows", "Total disk spills")
                .map_err(|e| {
                    BallistaError::General(format!("Error registering metric: {:?}", e))
                })?;
        Ok(Self {
            write_time, input_rows, output_rows
        })
    }
}

pub struct ExecutionPlanMetrics {
    pub total_elapsed_compute: IntCounter,
    pub elapsed_compute: Histogram,
    pub total_output_rows: IntCounter,
    pub total_spill_count: IntCounter,
    pub total_spilled_bytes: IntCounter,
}

impl ExecutionPlanMetrics {
    pub fn new(name: &str) -> Result<Self> {
        let total_elapsed_compute = register_int_counter!(
            format!("{}_elapsed_compute", name),
            "Total elapsed compute time"
        )
        .map_err(|e| {
            BallistaError::General(format!("Error registering metric: {:?}", e))
        })?;
        let elapsed_compute = register_histogram!(
            format!("{}_elapsed_compute_histogram", name),
            "Histogram of elapsed compute times",
            vec![0.0, 50.0, 500.0, 1000.0, 1500.0, 2000.0, 5000.0]
        )
        .map_err(|e| {
            BallistaError::General(format!("Error registering metric: {:?}", e))
        })?;
        let total_output_rows =
            register_int_counter!(format!("{}_output_rows", name), "Total output rows")
                .map_err(|e| {
                BallistaError::General(format!("Error registering metric: {:?}", e))
            })?;
        let total_spill_count =
            register_int_counter!(format!("{}_spill_count", name), "Total disk spills")
                .map_err(|e| {
                BallistaError::General(format!("Error registering metric: {:?}", e))
            })?;
        let total_spilled_bytes = register_int_counter!(
            format!("{}_spilled_bytes", name),
            "Total bytes spilled to disk"
        )
        .map_err(|e| {
            BallistaError::General(format!("Error registering metric: {:?}", e))
        })?;
        Ok(Self {
            total_elapsed_compute,
            elapsed_compute,
            total_output_rows,
            total_spill_count,
            total_spilled_bytes,
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
    fn record_stage(
        &self,
        _job_id: &str,
        _stage_id: usize,
        _partition: usize,
        plan: ShuffleWriterExec,
    ) {
        if let Some(metrics) = plan.metrics().map(|m| m.aggregate_by_partition()) {
            if let Some(write_time) = metrics.time("write_time") {
                self.metrics.write_time.observe(write_time as f64);
            }

            if let Some(output_rows) = metrics.output_rows() {
                self.metrics.output_rows.inc_by(output_rows as u64);
            }

            if let Some(input_rows) = metrics.count("input_rows") {
                self.metrics.input_rows.inc_by(input_rows as u64);
            }
        }

        let mut buffer = vec![];
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        encoder.encode(&metric_families, &mut buffer).unwrap();

        // Output to the standard output.
        println!("{}", String::from_utf8(buffer).unwrap());
    }
}
