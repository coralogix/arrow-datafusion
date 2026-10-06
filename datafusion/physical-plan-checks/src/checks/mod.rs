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
//! Each check is a function named after the check, documented with its code
//! in the catalog, and lives in the module for its [`CheckKind`].

use std::sync::Arc;

use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::{ExecutionPlan, StatisticsArgs, StatisticsContext};

use crate::{CheckContext, CheckKind, Finding, PlanCheck};

mod execution_checks;
mod static_checks;
mod stream_checks;
mod variant_checks;

use execution_checks::{
    batch_schema, cardinality_effect_holds, constants_hold, exact_statistics_hold,
    execution_succeeds, maintains_input_order_missed, memory_released, orderings_hold,
};
use static_checks::{
    cardinality_effect_bounds_num_rows, check_invariants, dynamic_expressions_reset,
    dynamic_expressions_visited, equal_cardinality_num_rows, expression_column_refs,
    fetch_bounds_num_rows, fetch_not_equal_cardinality, partition_statistics_sum,
    per_child_lengths, schema_consistency, statistics_ignore_inputs, statistics_shape,
};
use stream_checks::{
    boundedness_holds, emission_type_holds, errors_propagate, lazy_evaluation_holds,
    streams_released,
};
use variant_checks::{
    batch_boundary_invariance, batch_size_invariance, limit_pushdown_equivalent,
    limit_pushdown_missed, with_fetch_equivalent,
};

type CheckFn = fn(&Arc<dyn ExecutionPlan>, &CheckContext) -> Result<Vec<Finding>>;

const fn check(name: &'static str, kind: CheckKind, check: CheckFn) -> PlanCheck {
    PlanCheck { name, kind, check }
}

/// All built-in checks, in catalog order
pub fn all_checks() -> Vec<PlanCheck> {
    use CheckKind::*;
    vec![
        check(
            "equal_cardinality_num_rows",
            Static,
            equal_cardinality_num_rows,
        ),
        check(
            "fetch_not_equal_cardinality",
            Static,
            fetch_not_equal_cardinality,
        ),
        check("fetch_bounds_num_rows", Static, fetch_bounds_num_rows),
        check(
            "cardinality_effect_bounds_num_rows",
            Static,
            cardinality_effect_bounds_num_rows,
        ),
        check("limit_pushdown_missed", Variant, limit_pushdown_missed),
        check(
            "maintains_input_order_missed",
            Execution,
            maintains_input_order_missed,
        ),
        check("per_child_lengths", Static, per_child_lengths),
        check("check_invariants", Static, check_invariants),
        check("statistics_shape", Static, statistics_shape),
        check("partition_statistics_sum", Static, partition_statistics_sum),
        check("statistics_ignore_inputs", Static, statistics_ignore_inputs),
        check("schema_consistency", Static, schema_consistency),
        check("expression_column_refs", Static, expression_column_refs),
        check(
            "dynamic_expressions_visited",
            Static,
            dynamic_expressions_visited,
        ),
        check(
            "dynamic_expressions_reset",
            Static,
            dynamic_expressions_reset,
        ),
        check("execution_succeeds", Execution, execution_succeeds),
        check("batch_schema", Execution, batch_schema),
        check("exact_statistics_hold", Execution, exact_statistics_hold),
        check("orderings_hold", Execution, orderings_hold),
        check("constants_hold", Execution, constants_hold),
        check(
            "cardinality_effect_holds",
            Execution,
            cardinality_effect_holds,
        ),
        check("boundedness_holds", Stream, boundedness_holds),
        check("emission_type_holds", Stream, emission_type_holds),
        check("lazy_evaluation_holds", Stream, lazy_evaluation_holds),
        check("with_fetch_equivalent", Variant, with_fetch_equivalent),
        check(
            "limit_pushdown_equivalent",
            Variant,
            limit_pushdown_equivalent,
        ),
        check("batch_size_invariance", Variant, batch_size_invariance),
        check(
            "batch_boundary_invariance",
            Variant,
            batch_boundary_invariance,
        ),
        check("memory_released", Execution, memory_released),
        check("streams_released", Stream, streams_released),
        check("errors_propagate", Stream, errors_propagate),
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

/// Whether `plan` reports that its partitions can already be in the order a
/// node sorts them into: it reports an ordering, or a constant. Sorting on a
/// constant keeps rows in their order, and the equivalence properties of a
/// sort on a constant report no ordering.
fn reports_sorted_output(plan: &dyn ExecutionPlan) -> bool {
    let properties = plan.properties();
    properties.output_ordering().is_some()
        || !properties.equivalence_properties().constants().is_empty()
}

/// Number of output partitions of `plan`
fn partition_count(plan: &dyn ExecutionPlan) -> usize {
    plan.properties().output_partitioning().partition_count()
}
