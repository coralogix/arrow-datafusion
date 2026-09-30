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

//! `CheckKind::Execution` checks: they compare what a node reports with the
//! output it produces, and check that it releases the memory it reserves.

use std::collections::HashSet;
use std::fmt::Display;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_common::stats::Precision;
use datafusion_common::{Result, Statistics};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::CardinalityEffect;

use super::{overall_statistics, partition_statistics};
use crate::{CheckContext, Finding, oracle};

/// B0: a node executes without errors, panics or timeouts.
pub(super) fn execution_succeeds(
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

/// B1: every output batch has the node's schema, and non-nullable fields
/// contain no nulls.
pub(super) fn batch_schema(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    let Some(output) = context.output(node) else {
        return Ok(vec![]);
    };
    let expected = node.schema();
    let mut findings = vec![];
    // Each problem is reported for the first batch that has it
    let mut reported = HashSet::new();
    for (p, batches) in output.partitions().iter().enumerate() {
        for (b, batch) in batches.iter().enumerate() {
            for finding in batch_problems(&expected, batch) {
                if reported.insert(finding.message.clone()) {
                    findings.push(Finding {
                        message: format!(
                            "batch {b} of partition {p}: {}",
                            finding.message
                        ),
                        ..finding
                    });
                }
            }
        }
    }
    Ok(findings)
}

/// How `batch` differs from `expected`
fn batch_problems(expected: &Schema, batch: &RecordBatch) -> Vec<Finding> {
    let actual = batch.schema();
    if actual.fields().len() != expected.fields().len() {
        return vec![Finding::invariant(format!(
            "the batch has {} columns, but schema() has {} fields",
            actual.fields().len(),
            expected.fields().len()
        ))];
    }
    let mut findings = vec![];
    for (i, (actual_field, field)) in
        actual.fields().iter().zip(expected.fields()).enumerate()
    {
        let name = field.name();
        if actual_field.name() != name || actual_field.data_type() != field.data_type() {
            findings.push(Finding::invariant(format!(
                "column {i} is '{}' {}, but schema() declares '{name}' {}",
                actual_field.name(),
                actual_field.data_type(),
                field.data_type()
            )));
            continue;
        }
        if !field.is_nullable() && batch.column(i).logical_null_count() > 0 {
            findings.push(Finding::invariant(format!(
                "column '{name}' has nulls, but schema() declares it non-nullable"
            )));
        } else if actual_field.is_nullable() != field.is_nullable() {
            findings.push(Finding::lint(format!(
                "column '{name}' has nullable={}, but schema() declares nullable={}",
                actual_field.is_nullable(),
                field.is_nullable()
            )));
        }
        if actual_field.metadata() != field.metadata() {
            findings.push(Finding::lint(format!(
                "column '{name}' has field metadata {:?}, but schema() declares {:?}",
                actual_field.metadata(),
                field.metadata()
            )));
        }
    }
    if actual.metadata() != expected.metadata() {
        findings.push(Finding::lint(format!(
            "the batch has schema metadata {:?}, but schema() declares {:?}",
            actual.metadata(),
            expected.metadata()
        )));
    }
    findings
}

/// B2: every exact statistic the node reports is true of its output.
pub(super) fn exact_statistics_hold(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    let claims = false_claims(node, context)?;
    if claims.is_empty() {
        return Ok(vec![]);
    }
    // A node computes its statistics from its children's, so a child with a
    // false exact statistic can make the node's false too, even for a node
    // that passes statistics through unchanged, such as a repartition. Report
    // it on the child. Which of the node's statistics depend on which of the
    // child's is not known, so the node is not reported at all.
    for child in node.children() {
        if !false_claims(child, context)?.is_empty() {
            return Ok(vec![]);
        }
    }
    Ok(claims.into_iter().map(Finding::invariant).collect())
}

/// For the overall statistics of `node` and then for those of each partition,
/// the exact statistics its output contradicts, if any. Empty if the node was
/// not executed or failed.
fn false_claims(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<String>> {
    let Some(output) = context.output(node) else {
        return Ok(vec![]);
    };
    let schema = node.schema();
    // Output that does not match the schema is reported by `batch_schema`
    let batches = output.batches();
    if batches
        .iter()
        .any(|batch| batch.num_columns() != schema.fields().len())
    {
        return Ok(vec![]);
    }
    let targets = std::iter::once((None, batches)).chain(
        output
            .partitions()
            .iter()
            .enumerate()
            .map(|(p, batches)| (Some(p), batches.clone())),
    );
    let mut claims = vec![];
    for (partition, batches) in targets {
        let (target, claimed) = match partition {
            None => ("overall".to_string(), overall_statistics(node.as_ref())),
            Some(p) => (
                format!("partition {p}"),
                partition_statistics(node.as_ref(), p),
            ),
        };
        // Errors computing statistics are reported by `statistics_shape`
        let Ok(claimed) = claimed else {
            continue;
        };
        let actual = oracle::exact_statistics(&schema, &batches)?;
        let mismatches = mismatches(&schema, &claimed, &actual);
        if !mismatches.is_empty() {
            claims.push(format!(
                "the {target} statistics are false: {}",
                mismatches.join("; ")
            ));
        }
    }
    Ok(claims)
}

/// Each exact statistic in `claimed` that `actual` contradicts, such as
/// `null_count of l_k@0 is Exact(9), but the output has Exact(0)`. An exact
/// minimum or maximum cannot be contradicted by a column without non-null
/// values, which has neither.
fn mismatches(schema: &Schema, claimed: &Statistics, actual: &Statistics) -> Vec<String> {
    let mut mismatches = vec![];
    if let (Precision::Exact(claimed), Precision::Exact(actual)) =
        (claimed.num_rows, actual.num_rows)
        && claimed != actual
    {
        mismatches.push(format!(
            "num_rows is Exact({claimed}), but the output has {actual} rows"
        ));
    }
    let columns = claimed
        .column_statistics
        .iter()
        .zip(&actual.column_statistics);
    for (i, (claimed, actual)) in columns.enumerate() {
        let column = format!("{}@{i}", schema.field(i).name());
        let mut compare = |statistic, claimed: &dyn Display, actual: &dyn Display| {
            mismatches.push(format!(
                "{statistic} of {column} is {claimed}, but the output has {actual}"
            ));
        };
        if claimed.null_count.is_exact() == Some(true)
            && claimed.null_count != actual.null_count
        {
            compare("null_count", &claimed.null_count, &actual.null_count);
        }
        if claimed.distinct_count.is_exact() == Some(true)
            && claimed.distinct_count != actual.distinct_count
        {
            compare(
                "distinct_count",
                &claimed.distinct_count,
                &actual.distinct_count,
            );
        }
        if claimed.min_value.is_exact() == Some(true)
            && actual.min_value.is_exact() == Some(true)
            && claimed.min_value != actual.min_value
        {
            compare("min_value", &claimed.min_value, &actual.min_value);
        }
        if claimed.max_value.is_exact() == Some(true)
            && actual.max_value.is_exact() == Some(true)
            && claimed.max_value != actual.max_value
        {
            compare("max_value", &claimed.max_value, &actual.max_value);
        }
    }
    mismatches
}

/// B3: every output partition is sorted by every ordering in the node's
/// equivalence properties.
pub(super) fn orderings_hold(
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
                    "the node reports the ordering [{ordering}], but partition {p} is \
                     not sorted by it at row {row}"
                )));
            }
        }
    }
    Ok(findings)
}

/// B8: the number of output rows agrees with `cardinality_effect()`.
pub(super) fn cardinality_effect_holds(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    let (Some(output), Some(inputs)) =
        (context.output(node), context.child_outputs(node))
    else {
        return Ok(vec![]);
    };
    let out = output.num_rows();
    let findings = match (node.cardinality_effect(), inputs.as_slice()) {
        (CardinalityEffect::Equal, [input]) if out != input.num_rows() => {
            vec![Finding::invariant(format!(
                "cardinality_effect() is Equal, but the node produced {out} rows from \
                 {} input rows",
                input.num_rows()
            ))]
        }
        (CardinalityEffect::LowerEqual, [input]) if out > input.num_rows() => {
            vec![Finding::invariant(format!(
                "cardinality_effect() is LowerEqual, but the node produced {out} rows \
                 from {} input rows",
                input.num_rows()
            ))]
        }
        (CardinalityEffect::GreaterEqual, inputs) => inputs
            .iter()
            .enumerate()
            .filter(|(_, input)| out < input.num_rows())
            .map(|(i, input)| {
                Finding::invariant(format!(
                    "cardinality_effect() is GreaterEqual, but the node produced {out} \
                     rows while input {i} has {} rows",
                    input.num_rows()
                ))
            })
            .collect(),
        _ => vec![],
    };
    Ok(findings)
}

/// F1: memory reserved while executing a node to completion is released once
/// its streams and the plan are dropped.
pub(super) fn memory_released(
    node: &Arc<dyn ExecutionPlan>,
    context: &CheckContext,
) -> Result<Vec<Finding>> {
    let mut findings = vec![];
    // A child that leaks makes its parents leak too; report it on the child
    let child_leaks = node.children().into_iter().any(|child| {
        context
            .memory_reserved_after_execution(child)
            .is_some_and(|bytes| bytes > 0)
    });
    if let Some(reserved) = context.memory_reserved_after_execution(node)
        && reserved > 0
        && !child_leaks
    {
        findings.push(Finding::invariant(format!(
            "after executing the node to completion and dropping its streams and the \
             plan, {reserved} bytes are still reserved in the memory pool"
        )));
    }
    Ok(findings)
}
