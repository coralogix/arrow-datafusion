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

//! Checks that `cardinality_effect()` and `fetch()` agree with the reported
//! statistics.

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::stats::Precision;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::CardinalityEffect;

use super::{overall_statistics, partition_count, partition_statistics};
use crate::{Finding, PlanCheck};

/// A1: a single-child node with `CardinalityEffect::Equal` and no fetch reports
/// the same `num_rows` as its input, with the same precision.
#[derive(Debug, Default, Clone, Copy)]
pub struct EqualCardinalityNumRows;

impl PlanCheck for EqualCardinalityNumRows {
    fn code(&self) -> &'static str {
        "A1"
    }

    fn name(&self) -> &'static str {
        "equal_cardinality_num_rows"
    }

    fn check_node(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>> {
        if !matches!(node.cardinality_effect(), CardinalityEffect::Equal)
            || node.fetch().is_some()
        {
            return Ok(vec![]);
        }
        let children = node.children();
        let [child] = children.as_slice() else {
            return Ok(vec![]);
        };
        // Errors are reported by `statistics_shape`
        let (Ok(input), Ok(output)) = (
            overall_statistics(child.as_ref()),
            overall_statistics(node.as_ref()),
        ) else {
            return Ok(vec![]);
        };

        let input_rows = input.num_rows;
        let output_rows = output.num_rows;
        let finding = match (input_rows, output_rows) {
            (Precision::Exact(n), Precision::Exact(m)) if n != m => {
                Some(Finding::invariant(format!(
                    "cardinality_effect() is Equal, but num_rows is {output_rows} \
                     for an input with num_rows {input_rows}"
                )))
            }
            (Precision::Inexact(_) | Precision::Absent, Precision::Exact(_)) => {
                Some(Finding::invariant(format!(
                    "cardinality_effect() is Equal, but num_rows is {output_rows} \
                     for an input with num_rows {input_rows}; an Equal node cannot \
                     know its row count more precisely than its input does"
                )))
            }
            (Precision::Exact(_), Precision::Inexact(_)) => Some(Finding::lint(format!(
                "cardinality_effect() is Equal and the input has num_rows \
                     {input_rows}, but the node reports {output_rows}; it could \
                     report {input_rows}"
            ))),
            (Precision::Inexact(n), Precision::Inexact(m)) if n != m => {
                Some(Finding::lint(format!(
                    "cardinality_effect() is Equal, but the num_rows estimate \
                     {output_rows} differs from the input estimate {input_rows}"
                )))
            }
            // Absent output for a known input is reported by
            // `statistics_ignore_inputs`
            _ => None,
        };
        Ok(finding.into_iter().collect())
    }
}

/// A2: a node with a fetch does not claim `CardinalityEffect::Equal`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FetchNotEqualCardinality;

impl PlanCheck for FetchNotEqualCardinality {
    fn code(&self) -> &'static str {
        "A2"
    }

    fn name(&self) -> &'static str {
        "fetch_not_equal_cardinality"
    }

    fn check_node(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>> {
        match (node.fetch(), node.cardinality_effect()) {
            (Some(fetch), CardinalityEffect::Equal) => {
                Ok(vec![Finding::invariant(format!(
                    "fetch() is Some({fetch}), but cardinality_effect() is Equal; \
                     a fetch can drop input rows, so the effect should be LowerEqual \
                     while a fetch is set"
                ))])
            }
            _ => Ok(vec![]),
        }
    }
}

/// A3: a node with `fetch() == Some(n)` reports at most `n` rows per partition,
/// and at most `n * partition_count` rows overall.
#[derive(Debug, Default, Clone, Copy)]
pub struct FetchBoundsNumRows;

impl FetchBoundsNumRows {
    fn check_bound(
        target: &str,
        num_rows: Precision<usize>,
        bound: usize,
        fetch: usize,
    ) -> Option<Finding> {
        match num_rows {
            Precision::Exact(n) if n > bound => Some(Finding::invariant(format!(
                "fetch() is Some({fetch}), but {target} num_rows is {num_rows}, \
                 which is more than the fetch allows ({bound})"
            ))),
            Precision::Inexact(n) if n > bound => Some(Finding::lint(format!(
                "fetch() is Some({fetch}), but the {target} num_rows estimate \
                 {num_rows} is more than the fetch allows ({bound})"
            ))),
            _ => None,
        }
    }
}

impl PlanCheck for FetchBoundsNumRows {
    fn code(&self) -> &'static str {
        "A3"
    }

    fn name(&self) -> &'static str {
        "fetch_bounds_num_rows"
    }

    fn check_node(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>> {
        let Some(fetch) = node.fetch() else {
            return Ok(vec![]);
        };
        let partitions = partition_count(node.as_ref());
        let mut findings = vec![];

        // A fetch limits each partition, so the overall bound grows with the
        // partition count. Nodes with a single output partition are bounded
        // by `fetch` exactly.
        if let Ok(stats) = overall_statistics(node.as_ref()) {
            let bound = fetch.saturating_mul(partitions.max(1));
            findings.extend(Self::check_bound("overall", stats.num_rows, bound, fetch));
        }
        if partitions > 1 {
            for partition in 0..partitions {
                if let Ok(stats) = partition_statistics(node.as_ref(), partition) {
                    let target = format!("partition {partition}");
                    findings.extend(Self::check_bound(
                        &target,
                        stats.num_rows,
                        fetch,
                        fetch,
                    ));
                }
            }
        }
        Ok(findings)
    }
}

/// A4: `num_rows` is consistent with `CardinalityEffect::LowerEqual` and
/// `CardinalityEffect::GreaterEqual`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CardinalityEffectBoundsNumRows;

impl CardinalityEffectBoundsNumRows {
    fn check_lower_equal(
        input: Precision<usize>,
        output: Precision<usize>,
    ) -> Option<Finding> {
        match (input, output) {
            (Precision::Exact(n), Precision::Exact(m)) if m > n => {
                Some(Finding::invariant(format!(
                    "cardinality_effect() is LowerEqual, but num_rows is {output} \
                     for an input with num_rows {input}"
                )))
            }
            (Precision::Exact(n) | Precision::Inexact(n), Precision::Inexact(m))
                if m > n =>
            {
                Some(Finding::lint(format!(
                    "cardinality_effect() is LowerEqual, but the num_rows estimate \
                     {output} is larger than the input num_rows {input}"
                )))
            }
            _ => None,
        }
    }

    fn check_greater_equal(
        child: usize,
        input: Precision<usize>,
        output: Precision<usize>,
    ) -> Option<Finding> {
        match (input, output) {
            (Precision::Exact(n), Precision::Exact(m)) if m < n => {
                Some(Finding::invariant(format!(
                    "cardinality_effect() is GreaterEqual, but num_rows is {output} \
                     while input {child} has num_rows {input}"
                )))
            }
            (Precision::Exact(n), Precision::Inexact(m)) if m < n => {
                Some(Finding::lint(format!(
                    "cardinality_effect() is GreaterEqual, but the num_rows estimate \
                     {output} is smaller than the num_rows {input} of input {child}"
                )))
            }
            _ => None,
        }
    }
}

impl PlanCheck for CardinalityEffectBoundsNumRows {
    fn code(&self) -> &'static str {
        "A4"
    }

    fn name(&self) -> &'static str {
        "cardinality_effect_bounds_num_rows"
    }

    fn check_node(&self, node: &Arc<dyn ExecutionPlan>) -> Result<Vec<Finding>> {
        let effect = node.cardinality_effect();
        if !matches!(
            effect,
            CardinalityEffect::LowerEqual | CardinalityEffect::GreaterEqual
        ) {
            return Ok(vec![]);
        }
        let Ok(output) = overall_statistics(node.as_ref()) else {
            return Ok(vec![]);
        };
        let children = node.children();
        let mut findings = vec![];
        match effect {
            // Only single-child nodes: which input a multi-child node's
            // LowerEqual refers to is not defined
            CardinalityEffect::LowerEqual => {
                if let [child] = children.as_slice()
                    && let Ok(input) = overall_statistics(child.as_ref())
                {
                    findings
                        .extend(Self::check_lower_equal(input.num_rows, output.num_rows));
                }
            }
            // Producing at least as many rows as all inputs together implies
            // producing at least as many rows as each input
            CardinalityEffect::GreaterEqual => {
                for (i, child) in children.iter().enumerate() {
                    if let Ok(input) = overall_statistics(child.as_ref()) {
                        findings.extend(Self::check_greater_equal(
                            i,
                            input.num_rows,
                            output.num_rows,
                        ));
                    }
                }
            }
            CardinalityEffect::Equal | CardinalityEffect::Unknown => {}
        }
        Ok(findings)
    }
}
