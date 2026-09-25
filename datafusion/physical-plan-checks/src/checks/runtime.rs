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

//! Checks that compare what a node reports with the output it produces.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Result, Statistics};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::CardinalityEffect;

use super::{overall_statistics, partition_statistics};
use crate::context::NodeOutput;
use crate::{CheckContext, Finding, PlanCheck, oracle};

/// B0: a node executes without errors, panics or timeouts.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExecutionSucceeds;

impl PlanCheck for ExecutionSucceeds {
    fn code(&self) -> &'static str {
        "B0"
    }

    fn name(&self) -> &'static str {
        "execution_succeeds"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let Some(error) = context.execution_error(node) else {
            return Ok(vec![]);
        };
        // A failing child makes its parents fail too; report it on the child
        if node
            .children()
            .into_iter()
            .any(|child| context.execution_error(child).is_some())
        {
            return Ok(vec![]);
        }
        Ok(vec![Finding::invariant(format!(
            "executing the node failed: {error}"
        ))])
    }
}

/// B1: every output batch has the node's schema, and non-nullable fields
/// contain no nulls.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchSchema;

impl PlanCheck for BatchSchema {
    fn code(&self) -> &'static str {
        "B1"
    }

    fn name(&self) -> &'static str {
        "batch_schema"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let Some(output) = context.output(node) else {
            return Ok(vec![]);
        };
        let expected = node.schema();
        let mut findings = vec![];
        // Report each kind of problem once per column, for the first batch it
        // occurs in
        let mut reported = HashSet::new();
        let mut report = |kind: &'static str, column: usize, finding: Finding| {
            if reported.insert((kind, column)) {
                findings.push(finding);
            }
        };

        for (p, batches) in output.partitions().iter().enumerate() {
            for (b, batch) in batches.iter().enumerate() {
                let actual = batch.schema();
                let at = format!("batch {b} of partition {p}");
                if actual.fields().len() != expected.fields().len() {
                    report(
                        "column count",
                        0,
                        Finding::invariant(format!(
                            "{at} has {} columns, but schema() has {} fields",
                            actual.fields().len(),
                            expected.fields().len()
                        )),
                    );
                    continue;
                }
                for (i, (actual_field, expected_field)) in
                    actual.fields().iter().zip(expected.fields()).enumerate()
                {
                    if actual_field.name() != expected_field.name()
                        || actual_field.data_type() != expected_field.data_type()
                    {
                        report(
                            "field",
                            i,
                            Finding::invariant(format!(
                                "column {i} of {at} is '{}' {}, but schema() declares \
                                 '{}' {}",
                                actual_field.name(),
                                actual_field.data_type(),
                                expected_field.name(),
                                expected_field.data_type()
                            )),
                        );
                        continue;
                    }
                    let nulls = batch.column(i).logical_null_count();
                    if !expected_field.is_nullable() && nulls > 0 {
                        report(
                            "nulls",
                            i,
                            Finding::invariant(format!(
                                "column '{}' of {at} has {nulls} nulls, but schema() \
                                 declares it non-nullable",
                                expected_field.name()
                            )),
                        );
                    } else if actual_field.is_nullable() != expected_field.is_nullable() {
                        report(
                            "nullability",
                            i,
                            Finding::lint(format!(
                                "column '{}' of {at} has nullable={}, but schema() \
                                 declares nullable={}",
                                expected_field.name(),
                                actual_field.is_nullable(),
                                expected_field.is_nullable()
                            )),
                        );
                    }
                    if actual_field.metadata() != expected_field.metadata() {
                        report(
                            "field metadata",
                            i,
                            Finding::lint(format!(
                                "column '{}' of {at} has field metadata {:?}, but \
                                 schema() declares {:?}",
                                expected_field.name(),
                                actual_field.metadata(),
                                expected_field.metadata()
                            )),
                        );
                    }
                }
                if actual.metadata() != expected.metadata() {
                    report(
                        "schema metadata",
                        0,
                        Finding::lint(format!(
                            "{at} has schema metadata {:?}, but schema() declares {:?}",
                            actual.metadata(),
                            expected.metadata()
                        )),
                    );
                }
            }
        }
        Ok(findings)
    }
}

/// B2: every exact statistic the node reports is true of its output.
#[derive(Debug, Default, Clone, Copy)]
pub struct ExactStatisticsHold;

impl ExactStatisticsHold {
    fn compare(
        target: &str,
        node: &Arc<dyn ExecutionPlan>,
        claimed: &Statistics,
        batches: &[RecordBatch],
        findings: &mut Vec<Finding>,
    ) -> Result<()> {
        let schema = node.schema();
        // Output that does not match the schema is reported by `batch_schema`
        if batches
            .iter()
            .any(|batch| batch.num_columns() != schema.fields().len())
        {
            return Ok(());
        }
        let actual = oracle::exact_statistics(&schema, batches)?;

        if let Precision::Exact(n) = claimed.num_rows
            && Precision::Exact(n) != actual.num_rows
        {
            findings.push(Finding::invariant(format!(
                "{target} statistics report num_rows Exact({n}), but the output has {} \
                 rows",
                actual.num_rows
            )));
        }
        for ((field, claimed), actual) in schema
            .fields()
            .iter()
            .zip(&claimed.column_statistics)
            .zip(&actual.column_statistics)
        {
            Self::compare_column(target, field.name(), claimed, actual, findings);
        }
        Ok(())
    }

    fn compare_column(
        target: &str,
        column: &str,
        claimed: &ColumnStatistics,
        actual: &ColumnStatistics,
        findings: &mut Vec<Finding>,
    ) {
        let mut mismatch = |statistic: &str, claimed: String, actual: String| {
            findings.push(Finding::invariant(format!(
                "{target} statistics report {statistic} {claimed} for column '{column}', \
                 but the output has {actual}"
            )));
        };
        if claimed.null_count.is_exact() == Some(true)
            && claimed.null_count != actual.null_count
        {
            mismatch(
                "null_count",
                format!("{}", claimed.null_count),
                format!("{}", actual.null_count),
            );
        }
        if claimed.distinct_count.is_exact() == Some(true)
            && claimed.distinct_count != actual.distinct_count
        {
            mismatch(
                "distinct_count",
                format!("{}", claimed.distinct_count),
                format!("{}", actual.distinct_count),
            );
        }
        // The output has no minimum or maximum when it has no non-null values,
        // so an exact claim cannot be contradicted
        if claimed.min_value.is_exact() == Some(true)
            && actual.min_value.is_exact() == Some(true)
            && claimed.min_value != actual.min_value
        {
            mismatch(
                "min_value",
                format!("{}", claimed.min_value),
                format!("{}", actual.min_value),
            );
        }
        if claimed.max_value.is_exact() == Some(true)
            && actual.max_value.is_exact() == Some(true)
            && claimed.max_value != actual.max_value
        {
            mismatch(
                "max_value",
                format!("{}", claimed.max_value),
                format!("{}", actual.max_value),
            );
        }
    }
}

impl PlanCheck for ExactStatisticsHold {
    fn code(&self) -> &'static str {
        "B2"
    }

    fn name(&self) -> &'static str {
        "exact_statistics_hold"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let Some(output) = context.output(node) else {
            return Ok(vec![]);
        };
        let mut findings = vec![];
        // Errors computing statistics are reported by `statistics_shape`
        if let Ok(claimed) = overall_statistics(node.as_ref()) {
            let batches: Vec<RecordBatch> = output.batches().cloned().collect();
            Self::compare("overall", node, &claimed, &batches, &mut findings)?;
        }
        for (p, batches) in output.partitions().iter().enumerate() {
            if let Ok(claimed) = partition_statistics(node.as_ref(), p) {
                let target = format!("partition {p}");
                Self::compare(&target, node, &claimed, batches, &mut findings)?;
            }
        }
        Ok(findings)
    }
}

/// B3: every output partition is sorted by every ordering in the node's
/// equivalence properties.
#[derive(Debug, Default, Clone, Copy)]
pub struct OrderingsHold;

impl PlanCheck for OrderingsHold {
    fn code(&self) -> &'static str {
        "B3"
    }

    fn name(&self) -> &'static str {
        "orderings_hold"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let Some(output) = context.output(node) else {
            return Ok(vec![]);
        };
        let mut findings = vec![];
        let eq_properties = node.properties().equivalence_properties();
        for ordering in eq_properties.oeq_class().iter() {
            for (p, batches) in output.partitions().iter().enumerate() {
                // An ordering that cannot be evaluated on the output points to a
                // schema mismatch, which `batch_schema` reports
                if let Ok(Some(row)) = oracle::first_unsorted_row(batches, ordering) {
                    findings.push(Finding::invariant(format!(
                        "the node reports the ordering [{ordering}], but partition {p} \
                         is not sorted by it at row {row}"
                    )));
                }
            }
        }
        Ok(findings)
    }
}

/// B8: the number of output rows agrees with `cardinality_effect()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CardinalityEffectHolds;

impl CardinalityEffectHolds {
    fn check(
        effect: &CardinalityEffect,
        output: &NodeOutput,
        inputs: &[&NodeOutput],
    ) -> Vec<Finding> {
        let out = output.num_rows();
        match (effect, inputs) {
            (CardinalityEffect::Equal, [input]) if out != input.num_rows() => {
                vec![Finding::invariant(format!(
                    "cardinality_effect() is Equal, but the node produced {out} rows \
                     from {} input rows",
                    input.num_rows()
                ))]
            }
            (CardinalityEffect::LowerEqual, [input]) if out > input.num_rows() => {
                vec![Finding::invariant(format!(
                    "cardinality_effect() is LowerEqual, but the node produced {out} \
                     rows from {} input rows",
                    input.num_rows()
                ))]
            }
            (CardinalityEffect::GreaterEqual, inputs) => inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| out < input.num_rows())
                .map(|(i, input)| {
                    Finding::invariant(format!(
                        "cardinality_effect() is GreaterEqual, but the node produced \
                         {out} rows while input {i} has {} rows",
                        input.num_rows()
                    ))
                })
                .collect(),
            _ => vec![],
        }
    }
}

impl PlanCheck for CardinalityEffectHolds {
    fn code(&self) -> &'static str {
        "B8"
    }

    fn name(&self) -> &'static str {
        "cardinality_effect_holds"
    }

    fn requires_execution(&self) -> bool {
        true
    }

    fn check_node(
        &self,
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Vec<Finding>> {
        let (Some(output), Some(inputs)) =
            (context.output(node), context.child_outputs(node))
        else {
            return Ok(vec![]);
        };
        Ok(Self::check(&node.cardinality_effect(), output, &inputs))
    }
}
