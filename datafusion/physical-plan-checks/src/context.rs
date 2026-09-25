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
use std::collections::{BTreeSet, HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::RecordBatch;
use datafusion_execution::TaskContext;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::{ExecutionPlan, collect_partitioned};
use futures::FutureExt;

use crate::experiments::{
    self, Experiment, ExperimentOptions, StreamRun, TrackingPool,
    experiment_task_context, wait_until, with_memory_pool,
};

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

/// What the [`PlanChecker`] asks a [`CheckContext`] to gather
///
/// [`PlanChecker`]: crate::PlanChecker
#[derive(Debug)]
pub(crate) struct ContextOptions {
    /// Execute every node to completion and record its output
    pub execute: bool,
    /// Stream experiments to run on every node
    pub experiments: BTreeSet<Experiment>,
    pub task_context: Arc<TaskContext>,
    /// Time allowed for executing a node to completion
    pub timeout: Duration,
    /// Time allowed for a stream experiment on inputs that never end, and for
    /// streams and memory to be released
    pub stream_timeout: Duration,
}

/// Information about a plan that checks can use, gathered by the
/// [`PlanChecker`] before any check runs.
///
/// When at least one enabled check requires execution, every node in the plan
/// is executed on its own (all of its output partitions, concurrently) and its
/// output is recorded here, along with the memory the execution left reserved.
/// Nodes are identified by the address of the node, so a context is only valid
/// for the plan it was created for.
///
/// When an enabled check requests a stream [`Experiment`], the experiment is
/// run on every node with children and its [`StreamRun`]s are recorded here
/// too. Experiments rebuild the node on inputs whose streams behave
/// differently and observe how the node drives them; see [`Experiment`].
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
    /// Bytes still reserved after executing each node to completion
    memory: HashMap<usize, usize>,
    runs: HashMap<(usize, Experiment), Vec<StreamRun>>,
}

fn node_key(node: &Arc<dyn ExecutionPlan>) -> usize {
    Arc::as_ptr(node).cast::<()>() as usize
}

impl CheckContext {
    /// Execute every node of `plan`, run the requested experiments on it, and
    /// record the results
    pub(crate) async fn gather(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ContextOptions,
    ) -> Self {
        let experiment_options = ExperimentOptions {
            task_context: Arc::new(experiment_task_context(&options.task_context)),
            execution_timeout: options.timeout,
            stream_timeout: options.stream_timeout,
        };
        let mut context = Self::default();
        let mut visited = HashSet::new();
        let mut stack = vec![Arc::clone(plan)];
        while let Some(node) = stack.pop() {
            stack.extend(node.children().into_iter().cloned());
            let key = node_key(&node);
            if !visited.insert(key) {
                continue;
            }
            if options.execute {
                let (execution, memory) = execute_node(&node, options).await;
                context.executions.insert(key, execution);
                if let Some(memory) = memory {
                    context.memory.insert(key, memory);
                }
            }
            for experiment in &options.experiments {
                let runs =
                    experiments::run(&node, *experiment, &experiment_options).await;
                context.runs.insert((key, *experiment), runs);
            }
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

    /// The number of bytes that executing `node` to completion left reserved
    /// in the memory pool, after its streams and its copy of the plan were
    /// dropped. `None` if the node was not executed, or executing it failed.
    pub fn memory_reserved_after_execution(
        &self,
        node: &Arc<dyn ExecutionPlan>,
    ) -> Option<usize> {
        self.memory.get(&node_key(node)).copied()
    }

    /// The runs of `experiment` on `node`. Empty if the experiment was not run,
    /// or does not apply to the node, for example because it has no children.
    pub fn stream_runs(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        experiment: Experiment,
    ) -> &[StreamRun] {
        self.runs
            .get(&(node_key(node), experiment))
            .map(Vec::as_slice)
            .unwrap_or_default()
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

/// Execute every partition of a fresh copy of `node` to completion. Also
/// returns the memory left reserved afterwards, if the execution succeeded.
async fn execute_node(
    node: &Arc<dyn ExecutionPlan>,
    options: &ContextOptions,
) -> (Execution, Option<usize>) {
    let node = match reset_plan_states(Arc::clone(node)) {
        Ok(node) => node,
        Err(e) => {
            let error = format!(
                "resetting the plan state before execution failed: {}",
                e.strip_backtrace()
            );
            return (Execution::Failed(error), None);
        }
    };
    let pool = Arc::new(TrackingPool::new(Arc::clone(
        options.task_context.memory_pool(),
    )));
    let task_context =
        match with_memory_pool(&options.task_context, Arc::clone(&pool) as _) {
            Ok(task_context) => Arc::new(task_context),
            Err(e) => {
                let error =
                    format!("creating the task context failed: {}", e.strip_backtrace());
                return (Execution::Failed(error), None);
            }
        };
    let future = AssertUnwindSafe(collect_partitioned(node, task_context)).catch_unwind();
    let timeout = options.timeout;
    let execution = match tokio::time::timeout(timeout, future).await {
        Ok(Ok(Ok(partitions))) => Execution::Output(NodeOutput { partitions }),
        Ok(Ok(Err(e))) => Execution::Failed(e.strip_backtrace()),
        Ok(Err(panic)) => {
            Execution::Failed(format!("panicked: {}", panic_message(&panic)))
        }
        Err(_) => Execution::Failed(format!("did not finish within {timeout:?}")),
    };
    let memory = match execution {
        Execution::Output(_) => {
            // Spawned tasks can release their reservations shortly after the
            // streams that own them are dropped
            wait_until(|| pool.tracked() == 0, options.stream_timeout).await;
            Some(pool.tracked())
        }
        Execution::Failed(_) => None,
    };
    (execution, memory)
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
