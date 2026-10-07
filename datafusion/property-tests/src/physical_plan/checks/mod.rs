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

//! Built-in [`PlanCheck`]s. See `docs/physical_plan/CHECKS.md` for what each
//! check verifies. Each check is a function named after the check, and
//! documented with its code in the catalog.

use std::sync::Arc;

use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::{
    ChildStats, ExecutionPlan, StatisticsArgs, StatisticsContext,
};

use super::PlanCheck;
use crate::Finding;

mod static_checks;

use static_checks::{
    cardinality_effect_bounds_num_rows, check_invariants, display_no_panic,
    dynamic_expressions_reset, dynamic_expressions_visited, equal_cardinality_num_rows,
    expression_column_refs, fetch_bounds_num_rows, fetch_not_equal_cardinality,
    limit_pushdown_merges_partitions, limit_pushdown_missed, partition_statistics_sum,
    per_child_lengths, schema_consistency, statistics_ignore_inputs, statistics_shape,
};

type CheckFn = fn(&Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>>;

const fn check(name: &'static str, check: CheckFn) -> PlanCheck {
    PlanCheck { name, check }
}

/// All built-in checks, in catalog order
pub fn all_checks() -> Vec<PlanCheck> {
    vec![
        check("equal_cardinality_num_rows", equal_cardinality_num_rows),
        check("fetch_not_equal_cardinality", fetch_not_equal_cardinality),
        check("fetch_bounds_num_rows", fetch_bounds_num_rows),
        check(
            "cardinality_effect_bounds_num_rows",
            cardinality_effect_bounds_num_rows,
        ),
        check("limit_pushdown_missed", limit_pushdown_missed),
        check(
            "limit_pushdown_merges_partitions",
            limit_pushdown_merges_partitions,
        ),
        check("per_child_lengths", per_child_lengths),
        check("check_invariants", check_invariants),
        check("statistics_shape", statistics_shape),
        check("partition_statistics_sum", partition_statistics_sum),
        check("statistics_ignore_inputs", statistics_ignore_inputs),
        check("schema_consistency", schema_consistency),
        check("expression_column_refs", expression_column_refs),
        check("dynamic_expressions_visited", dynamic_expressions_visited),
        check("dynamic_expressions_reset", dynamic_expressions_reset),
        check("display_no_panic", display_no_panic),
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

/// Whether `node` can move the rows of its child `child`, which has several
/// partitions, between partitions: it computes the statistics of an output
/// partition from the overall statistics of the child, as `RepartitionExec`
/// does. Which output partition a row goes to then depends on the data, and
/// on generated data, such as input already hash partitioned the same way or
/// with one constant per partition, every output partition can happen to get
/// the rows of one child partition only, in order, so that the node looks as
/// if it passed its input through, which it does not do for other input.
fn moves_rows_between_partitions(node: &dyn ExecutionPlan, child: usize) -> bool {
    let children = node.children();
    let Some(child_plan) = children.get(child) else {
        return false;
    };
    partition_count(child_plan.as_ref()) > 1
        && (0..partition_count(node)).any(|p| {
            matches!(
                node.child_stats_requests(Some(p)).get(child),
                Some(ChildStats::At(None))
            )
        })
}

/// Whether the properties of `node` show that it passes the rows of its only
/// child through, which is what limit pushdown needs: its cardinality effect
/// without its fetch (that of the plan returned by `with_fetch(None)`) is
/// `Equal`, it maintains the order of the child, and each of its output
/// partitions is made of the rows of at most one partition of the child. A
/// child with one partition qualifies however the node spreads its rows, and
/// a child with several only if the node has as many partitions and does not
/// move rows between them.
fn passes_rows_through(node: &dyn ExecutionPlan) -> bool {
    let children = node.children();
    let [child] = children.as_slice() else {
        return false;
    };
    let effect = node.with_fetch(None).map_or_else(
        || node.cardinality_effect(),
        |plan| plan.cardinality_effect(),
    );
    let child_partitions = partition_count(child.as_ref());
    matches!(effect, CardinalityEffect::Equal)
        && node.maintains_input_order().first() == Some(&true)
        && (child_partitions <= 1
            || (child_partitions == partition_count(node)
                && !moves_rows_between_partitions(node, 0)))
}
