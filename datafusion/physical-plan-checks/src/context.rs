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

//! [`CheckContext`]: information about a plan gathered before checks run.

use std::any::Any;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::{ExecutionPlan, collect_partitioned};
use futures::FutureExt;

/// The output of executing one node: the batches of each output partition
#[derive(Debug, Clone)]
pub struct NodeOutput {
    partitions: Vec<Vec<RecordBatch>>,
}

impl NodeOutput {
    /// The batches of each output partition
    pub fn partitions(&self) -> &[Vec<RecordBatch>] {
        &self.partitions
    }

    /// All batches of all partitions, in partition order
    pub fn batches(&self) -> impl Iterator<Item = &RecordBatch> {
        self.partitions.iter().flatten()
    }

    /// The total number of rows in all partitions
    pub fn num_rows(&self) -> usize {
        self.batches().map(RecordBatch::num_rows).sum()
    }

    /// The number of rows in each partition
    pub fn partition_num_rows(&self) -> Vec<usize> {
        self.partitions
            .iter()
            .map(|batches| batches.iter().map(RecordBatch::num_rows).sum())
            .collect()
    }
}

#[derive(Debug)]
enum Execution {
    Output(NodeOutput),
    Failed(String),
}

/// Information about a plan that checks can use, gathered by the
/// [`PlanChecker`] before any check runs.
///
/// When at least one enabled check requires execution, every node in the plan
/// is executed on its own (all of its output partitions, concurrently) and its
/// output is recorded here. Nodes are identified by the address of the node, so
/// a context is only valid for the plan it was created for.
///
/// Each node is executed on a fresh copy of its subtree made with
/// [`reset_plan_states`], so runtime state such as dynamic filters from one
/// execution does not affect the next. The plan passed to the checker is never
/// executed itself.
///
/// [`PlanChecker`]: crate::PlanChecker
#[derive(Debug, Default)]
pub struct CheckContext {
    executions: HashMap<usize, Execution>,
}

fn node_key(node: &Arc<dyn ExecutionPlan>) -> usize {
    Arc::as_ptr(node).cast::<()>() as usize
}

impl CheckContext {
    /// Execute every node of `plan` and record the outputs
    pub(crate) async fn execute(
        plan: &Arc<dyn ExecutionPlan>,
        task_context: &Arc<TaskContext>,
        timeout: Duration,
    ) -> Self {
        let mut context = Self::default();
        let mut stack = vec![Arc::clone(plan)];
        while let Some(node) = stack.pop() {
            stack.extend(node.children().into_iter().cloned());
            let key = node_key(&node);
            if context.executions.contains_key(&key) {
                continue;
            }
            let execution = execute_node(node, Arc::clone(task_context), timeout).await;
            context.executions.insert(key, execution);
        }
        context
    }

    /// The output of executing `node`, or `None` if it was not executed or
    /// failed
    pub fn output(&self, node: &Arc<dyn ExecutionPlan>) -> Option<&NodeOutput> {
        match self.executions.get(&node_key(node)) {
            Some(Execution::Output(output)) => Some(output),
            _ => None,
        }
    }

    /// The error from executing `node`, if executing it failed, panicked or
    /// timed out
    pub fn execution_error(&self, node: &Arc<dyn ExecutionPlan>) -> Option<&str> {
        match self.executions.get(&node_key(node)) {
            Some(Execution::Failed(error)) => Some(error),
            _ => None,
        }
    }

    /// The outputs of all children of `node`, or `None` if any child was not
    /// executed or failed
    pub fn child_outputs(
        &self,
        node: &Arc<dyn ExecutionPlan>,
    ) -> Option<Vec<&NodeOutput>> {
        node.children()
            .into_iter()
            .map(|child| self.output(child))
            .collect()
    }
}

async fn execute_node(
    node: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    timeout: Duration,
) -> Execution {
    let node = match reset_plan_states(node) {
        Ok(node) => node,
        Err(e) => {
            return Execution::Failed(format!(
                "resetting the plan state before execution failed: {}",
                e.strip_backtrace()
            ));
        }
    };
    let future = AssertUnwindSafe(collect_partitioned(node, task_context)).catch_unwind();
    match tokio::time::timeout(timeout, future).await {
        Ok(Ok(Ok(partitions))) => Execution::Output(NodeOutput { partitions }),
        Ok(Ok(Err(e))) => Execution::Failed(e.strip_backtrace()),
        Ok(Err(panic)) => {
            Execution::Failed(format!("panicked: {}", panic_message(&panic)))
        }
        Err(_) => Execution::Failed(format!("did not finish within {timeout:?}")),
    }
}

fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(message) = panic.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = panic.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}
