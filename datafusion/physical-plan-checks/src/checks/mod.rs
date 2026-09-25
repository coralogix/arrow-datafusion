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

//! Built-in [`PlanCheck`]s. See `CHECKS.md` for what each check verifies.

use std::sync::Arc;

use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::{ExecutionPlan, StatisticsArgs, StatisticsContext};

use crate::PlanCheck;

mod cardinality;
mod statistics;
mod structure;

pub use cardinality::{
    CardinalityEffectBoundsNumRows, EqualCardinalityNumRows, FetchBoundsNumRows,
    FetchNotEqualCardinality,
};
pub use statistics::{PartitionStatisticsSum, StatisticsIgnoreInputs, StatisticsShape};
pub use structure::{CheckInvariants, PerChildLengths};

/// All built-in checks, in catalog order
pub fn all_checks() -> Vec<Arc<dyn PlanCheck>> {
    vec![
        Arc::new(EqualCardinalityNumRows),
        Arc::new(FetchNotEqualCardinality),
        Arc::new(FetchBoundsNumRows),
        Arc::new(CardinalityEffectBoundsNumRows),
        Arc::new(PerChildLengths),
        Arc::new(CheckInvariants),
        Arc::new(StatisticsShape),
        Arc::new(PartitionStatisticsSum),
        Arc::new(StatisticsIgnoreInputs),
    ]
}

/// Statistics of `plan` for all partitions, computed with a fresh
/// [`StatisticsContext`] and no statistics providers
fn overall_statistics(plan: &dyn ExecutionPlan) -> Result<Arc<Statistics>> {
    StatisticsContext::new().compute(plan, &StatisticsArgs::new())
}

/// Statistics of `plan` for a single partition, computed with a fresh
/// [`StatisticsContext`] and no statistics providers
fn partition_statistics(
    plan: &dyn ExecutionPlan,
    partition: usize,
) -> Result<Arc<Statistics>> {
    StatisticsContext::new()
        .compute(plan, &StatisticsArgs::new().with_partition(Some(partition)))
}

/// Number of output partitions of `plan`
fn partition_count(plan: &dyn ExecutionPlan) -> usize {
    plan.properties().output_partitioning().partition_count()
}
