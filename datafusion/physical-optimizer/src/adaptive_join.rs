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

use crate::PhysicalOptimizerRule;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::adaptive_join::{AdaptiveJoinExec, ThresholdBoundaryExec};
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::scalar_subquery::ScalarSubqueryExec;
use std::sync::Arc;

/// When `optimizer.adaptive_join_build_side` is enabled, wraps both inputs of
/// every [`HashJoinExec`] in a [`ThresholdBoundaryExec`] and the plan in an
/// [`AdaptiveJoinExec`], which picks build sides at runtime.
#[derive(Debug, Default)]
pub struct AdaptiveJoinBuildSide {}

impl AdaptiveJoinBuildSide {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysicalOptimizerRule for AdaptiveJoinBuildSide {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.adaptive_join_build_side {
            return Ok(plan);
        }
        let max_bytes = config.optimizer.adaptive_join_max_bytes;
        let mut found = false;
        let plan = plan
            .transform_up(|node| {
                if node.downcast_ref::<HashJoinExec>().is_none() {
                    return Ok(Transformed::no(node));
                }
                found = true;
                let children = node
                    .children()
                    .into_iter()
                    .map(|child| {
                        Arc::new(ThresholdBoundaryExec::new(Arc::clone(child), max_bytes))
                            as Arc<dyn ExecutionPlan>
                    })
                    .collect();
                #[expect(deprecated)]
                let node = node.with_new_children(children)?;
                Ok(Transformed::yes(node))
            })
            .data()?;
        if !found {
            return Ok(plan);
        }
        // Scalar subqueries must run before the main input is executed, and
        // each subquery runs on its own, so give every child of a
        // `ScalarSubqueryExec` its own driver.
        let plan = plan
            .transform_up(|node| {
                if node.downcast_ref::<ScalarSubqueryExec>().is_none() {
                    return Ok(Transformed::no(node));
                }
                let children = node
                    .children()
                    .into_iter()
                    .map(|child| {
                        Arc::new(AdaptiveJoinExec::new(Arc::clone(child)))
                            as Arc<dyn ExecutionPlan>
                    })
                    .collect();
                #[expect(deprecated)]
                let node = node.with_new_children(children)?;
                Ok(Transformed::yes(node))
            })
            .data()?;
        Ok(Arc::new(AdaptiveJoinExec::new(plan)))
    }

    fn name(&self) -> &str {
        "AdaptiveJoinBuildSide"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
