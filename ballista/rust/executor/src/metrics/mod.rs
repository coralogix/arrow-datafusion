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

#[cfg(feature = "prometheus-metrics")]
pub mod prometheus;

use ballista_core::error::Result;
use ballista_core::execution_plans::ShuffleWriterExec;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use std::sync::Arc;

/// `ExecutorMetricsCollector` records metrics for `ShuffleWriteExec`
/// after they are executed.
///
/// After each stage completes, `ShuffleWriteExec::record_stage` will be
/// called.
pub trait ExecutorMetricsCollector: Send + Sync {
    /// Record metrics for stage after it is executed
    fn record_stage(&self, partition: usize, plan: ShuffleWriterExec);
}

/// Implementation of `ExecutorMetricsCollector` which logs the completed
/// plan to stdout.
#[derive(Default)]
pub struct LoggingMetricsCollector {}

impl ExecutorMetricsCollector for LoggingMetricsCollector {
    fn record_stage(&self, partition: usize, plan: ShuffleWriterExec) {
        println!(
            "=== [{}/{}/{}] Physical plan with metrics ===\n{}\n",
            plan.job_id(),
            plan.stage_id(),
            partition,
            DisplayableExecutionPlan::with_metrics(&plan).indent()
        );
    }
}

#[cfg(feature = "prometheus-metrics")]
pub fn default_metrics_collector() -> Result<Arc<dyn ExecutorMetricsCollector>> {
    Ok(Arc::new(prometheus::PrometheusMetricsCollector::new()?))
}

#[cfg(not(feature = "prometheus-metrics"))]
pub fn default_metrics_collector() -> Result<Arc<dyn ExecutorMetricsCollector>> {
    Ok(Arc::new(LoggingMetricsCollector::default()))
}
