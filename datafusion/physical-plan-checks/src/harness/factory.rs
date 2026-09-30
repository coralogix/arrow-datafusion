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

//! [`PlanFactory`]: how to build the plan under test from its inputs.

use std::fmt;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;

use super::Profile;
use crate::fixtures::SourceSpec;

/// The function a [`PlanFactory`] builds its plan with
type CreateFn =
    dyn Fn(Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> + Send + Sync;

/// A check skipped for every case of a factory, with the reason
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedCheck {
    /// The [`PlanCheck::name`] of the check
    ///
    /// [`PlanCheck::name`]: crate::PlanCheck::name
    pub check: String,
    /// Why the check does not apply to the plan
    pub reason: String,
}

/// Describes a plan under test by how to build it from its inputs.
///
/// A factory has a name, one base [`SourceSpec`] per input, and a function
/// that builds the plan from input plans. The [`PlanHarness`] generates the
/// inputs for each case from the base specs, calls the function, and checks
/// the plan it returns.
///
/// # The base specs
///
/// A base spec carries what only the author of the plan knows: the schema,
/// the value distribution (for example few distinct values on join keys, so
/// that the inputs of a join match), the number of rows, and optionally an
/// ordering the plan should see, for example to exercise a sorted code path.
/// The harness sets everything else for each case, replacing what the base
/// spec says: the partition layout, hash partitioning, the batch layout, the
/// statistics precision, the seed and a row id column (see [`Profile`]). It
/// keeps the base ordering unless the plan requires another one.
///
/// # The function
///
/// `create` is called many times: to probe the plan's input requirements,
/// and once for every case. It must be a pure function of its inputs, and
/// must use the input plans it receives as they are (not copies), since the
/// harness recognizes them by identity.
///
/// **Bind expressions to the schemas of the inputs it receives, by name**,
/// for example `col("a", &inputs[0].schema())`, never by a hard-coded column
/// index or a schema captured outside `create`. The harness appends a row id
/// column to every input (`__row_id_0` for input 0, and so on), so the input
/// schemas have more columns than the base specs.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_physical_expr::expressions::col;
/// # use datafusion_physical_plan::ExecutionPlan;
/// # use datafusion_physical_plan::filter::FilterExec;
/// use datafusion_physical_plan_checks::fixtures::SourceSpec;
/// use datafusion_physical_plan_checks::harness::{PlanFactory, PlanHarness};
///
/// let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, false)]));
/// let factory = PlanFactory::new("FilterExec", vec![SourceSpec::new(schema)], |inputs| {
///     let input = Arc::clone(&inputs[0]);
///     let predicate = col("b", &input.schema())?;
///     Ok(Arc::new(FilterExec::try_new(predicate, input)?) as Arc<dyn ExecutionPlan>)
/// });
/// let report = PlanHarness::new().check(&factory)?;
/// report.assert_no_invariant_violations();
/// # Ok::<(), datafusion_common::DataFusionError>(())
/// ```
///
/// [`PlanHarness`]: crate::harness::PlanHarness
#[derive(Clone)]
pub struct PlanFactory {
    name: String,
    inputs: Vec<SourceSpec>,
    create: Arc<CreateFn>,
    allowed: Vec<AllowedCheck>,
    profiles: Option<Vec<Profile>>,
    shared_row_id_name: bool,
}

impl fmt::Debug for PlanFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanFactory")
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("allowed", &self.allowed)
            .field("profiles", &self.profiles)
            .field("shared_row_id_name", &self.shared_row_id_name)
            .finish_non_exhaustive()
    }
}

impl PlanFactory {
    /// A factory named `name` that builds its plan from one input per entry
    /// of `inputs` with `create`. A plan without inputs, such as `EmptyExec`,
    /// has no entries.
    pub fn new<F>(name: impl Into<String>, inputs: Vec<SourceSpec>, create: F) -> Self
    where
        F: Fn(Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            name: name.into(),
            inputs,
            create: Arc::new(create),
            allowed: vec![],
            profiles: None,
            shared_row_id_name: false,
        }
    }

    /// Skip the check named `check` in every case, because of `reason`,
    /// which the report shows. Use it for a known limitation of the check,
    /// not to hide a problem of the plan.
    ///
    /// # Panics
    ///
    /// If `reason` is empty.
    pub fn allow(mut self, check: impl Into<String>, reason: impl Into<String>) -> Self {
        let reason = reason.into();
        assert!(!reason.trim().is_empty(), "allowing a check needs a reason");
        self.allowed.push(AllowedCheck {
            check: check.into(),
            reason,
        });
        self
    }

    /// Test this factory with `profiles` instead of the profiles of the
    /// harness, for example to add a profile that matters for this plan or to
    /// leave out one that does not apply
    pub fn with_profiles(mut self, profiles: Vec<Profile>) -> Self {
        self.profiles = Some(profiles);
        self
    }

    /// Name the row id column of every input `__row_id` instead of
    /// `__row_id_<input>`, for plans that require all inputs to have the same
    /// schema, such as `UnionExec`. The ids still come from a separate range
    /// for each input.
    pub fn with_shared_row_id_name(mut self) -> Self {
        self.shared_row_id_name = true;
        self
    }

    /// The name, used in reports
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The base spec of each input
    pub fn inputs(&self) -> &[SourceSpec] {
        &self.inputs
    }

    /// The checks skipped for every case
    pub fn allowed(&self) -> &[AllowedCheck] {
        &self.allowed
    }

    /// The profiles set with [`Self::with_profiles`], if any
    pub fn profiles(&self) -> Option<&[Profile]> {
        self.profiles.as_deref()
    }

    /// The name of the row id column of input `input`
    pub fn row_id_name(&self, input: usize) -> String {
        if self.shared_row_id_name {
            "__row_id".to_string()
        } else {
            format!("__row_id_{input}")
        }
    }

    /// Build the plan under test on `inputs`
    pub fn create(
        &self,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        (self.create)(inputs)
    }
}
