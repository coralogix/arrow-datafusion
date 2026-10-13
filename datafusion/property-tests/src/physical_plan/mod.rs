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

//! Property checks for [`ExecutionPlan`] implementations.
//!
//! [`ExecutionPlan`] has many methods that describe a plan rather than run it,
//! such as [`ExecutionPlan::cardinality_effect`],
//! [`ExecutionPlan::supports_limit_pushdown`] and the statistics it reports.
//! Optimizer rules trust these descriptions, so an implementation that gets one
//! wrong can make DataFusion return wrong results or miss optimizations. This
//! module checks that the descriptions are consistent with each other and with
//! what the plan's inputs report.
//!
//! The easiest way to test a plan is to describe how to build it from its
//! inputs with a [`harness::PlanFactory`], and check it with a
//! [`harness::PlanHarness`]. The harness generates the inputs, meets the
//! plan's input requirements, and runs every check on several cases.
//!
//! Use [`PlanChecker`] to run every built-in check against every node of a
//! plan built by hand. Build the plan under test on inputs generated with
//! [`fixtures::SourceSpec`], which produces a [`fixtures::MockSourceExec`]
//! whose data, statistics, partitioning and ordering are known to be
//! consistent. The [`oracle`] module computes the true properties of a set
//! of batches, which the sources use to verify what they report.
//!
//! A [`PlanCheck`] looks at one node without executing it, comparing what the
//! node reports with what its children report.
//!
//! See `docs/physical_plan/CHECKS.md` in this crate for the catalog of
//! checks, why each one matters and how to fix a violation.
//!
//! [`ExecutionPlan`]: datafusion_physical_plan::ExecutionPlan
//! [`ExecutionPlan::cardinality_effect`]: datafusion_physical_plan::ExecutionPlan::cardinality_effect
//! [`ExecutionPlan::supports_limit_pushdown`]: datafusion_physical_plan::ExecutionPlan::supports_limit_pushdown

mod checker;
pub mod checks;
pub mod fixtures;
pub mod harness;
pub mod oracle;

pub use checker::{PlanCheck, PlanChecker};
