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

//! Running the plans under test: on which runtime and task context, with a
//! timeout, and with panics caught.

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use datafusion_common::instant::Instant;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::memory_pool::MemoryPool;
use datafusion_execution::runtime_env::RuntimeEnvBuilder;
use datafusion_physical_plan::{ExecutionPlan, collect_partitioned};
use futures::FutureExt;

use crate::NodeOutput;

/// A current-thread runtime for executing plans from synchronous code. With
/// one thread, execution is deterministic, although some operators produce
/// output that depends on how tasks interleave.
pub(crate) fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

/// A default task context with the session batch size set to `batch_size`
pub(crate) fn task_context(batch_size: usize) -> Arc<TaskContext> {
    let config = SessionConfig::new().with_batch_size(batch_size);
    Arc::new(TaskContext::default().with_session_config(config))
}

/// A default task context whose memory pool is `pool`
pub(crate) fn task_context_with_pool(
    pool: Arc<dyn MemoryPool>,
) -> Result<Arc<TaskContext>> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool)
        .build_arc()?;
    Ok(Arc::new(TaskContext::default().with_runtime(runtime)))
}

/// Execute every partition of `plan` concurrently and collect the output.
/// Errors, panics and running longer than `timeout` are returned as a
/// description.
pub(crate) async fn collect_output(
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    timeout: Duration,
) -> Result<NodeOutput, String> {
    let future = AssertUnwindSafe(collect_partitioned(plan, task_context)).catch_unwind();
    match tokio::time::timeout(timeout, future).await {
        Ok(Ok(Ok(partitions))) => Ok(NodeOutput::new(partitions)),
        Ok(Ok(Err(e))) => Err(e.strip_backtrace()),
        Ok(Err(panic)) => Err(format!("panicked: {}", panic_message(&panic))),
        Err(_) => Err(format!("did not finish within {timeout:?}")),
    }
}

pub(crate) fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Wait until `released` returns true, for example once streams or memory a
/// node no longer needs are released, or until `timeout` elapses.
///
/// Such resources are released either synchronously, when what owns them is
/// dropped, or later by code running in a task, such as a spawned task that
/// notices that its receiver was dropped, or an aborted task that the runtime
/// drops. Once no task is alive on the current Tokio runtime, nothing is left
/// that could release them, so the wait ends early instead of taking the
/// whole timeout. Tokio drops a task's future before it stops counting the
/// task as alive. Work on threads outside the runtime, including
/// `spawn_blocking` tasks, which Tokio does not count, is not waited for once
/// no task is alive.
pub(crate) async fn wait_for_release(released: impl Fn() -> bool, timeout: Duration) {
    let runtime = tokio::runtime::Handle::current();
    let start = Instant::now();
    while !released()
        && start.elapsed() < timeout
        && runtime.metrics().num_alive_tasks() > 0
    {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
