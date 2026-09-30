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

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg",
    html_favicon_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Make sure fast / cheap clones on Arc are explicit:
// https://github.com/apache/datafusion/issues/11143
#![deny(clippy::clone_on_ref_ptr)]
#![cfg_attr(test, allow(clippy::needless_pass_by_value))]

//! Property checks for [`ExecutionPlan`] implementations.
//!
//! [`ExecutionPlan`] has many methods that describe a plan rather than run it,
//! such as [`ExecutionPlan::cardinality_effect`],
//! [`ExecutionPlan::maintains_input_order`] and the statistics it reports.
//! Optimizer rules trust these descriptions, so an implementation that gets one
//! wrong can make DataFusion return wrong results or miss optimizations. This
//! crate checks that the descriptions are consistent with each other and with
//! what the plan does when it runs.
//!
//! The easiest way to test a plan is to describe how to build it from its
//! inputs with a [`harness::PlanFactory`], and check it with a
//! [`harness::PlanHarness`]. The harness generates the inputs, derives the
//! cases worth testing from the plan's input requirements, runs every check
//! on every case, and groups the findings of all cases in one report.
//!
//! Use [`PlanChecker`] to run every built-in check against every node of a
//! plan built by hand. Build the plan under test on inputs generated with
//! [`fixtures::SourceSpec`], which produces a [`fixtures::MockSourceExec`]
//! whose data, statistics, partitioning and ordering are known to be
//! consistent.
//!
//! Checks that need to see how a node drives its input streams, rather than
//! what it outputs, read the results of stream [`Experiment`]s: runs of each
//! node on copies of its inputs that stall, fail or never end, observed with
//! [`fixtures::StreamProbe`]s.
//!
//! Checks that compare a node's output with the output of a rewritten copy of
//! the node, such as the plan returned by `with_fetch`, or of the node run
//! under other settings, such as another batch size, read the results of
//! [`Variant`] runs. The [`oracle`] module compares their rows.
//!
//! See `CHECKS.md` in this crate for the catalog of checks, why each one
//! matters and how to fix a violation, and `IMPLEMENTATION_STATUS.md` for which
//! checks are implemented.
//!
//! [`ExecutionPlan`]: datafusion_physical_plan::ExecutionPlan
//! [`ExecutionPlan::cardinality_effect`]: datafusion_physical_plan::ExecutionPlan::cardinality_effect
//! [`ExecutionPlan::maintains_input_order`]: datafusion_physical_plan::ExecutionPlan::maintains_input_order

mod checker;
pub mod checks;
mod context;
mod experiments;
pub mod fixtures;
pub mod harness;
pub mod oracle;
mod report;
mod variants;

pub use checker::{PlanCheck, PlanChecker};
pub use context::{CheckContext, NodeOutput};
pub use experiments::{Experiment, RunOutcome, StreamRun};
pub use report::{Finding, Report, Severity, Violation};
pub use variants::{Variant, VariantKind, VariantRun};
