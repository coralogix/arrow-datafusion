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
use std::fmt::Display;
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion_common::stats::Precision;
use datafusion_common::{ColumnStatistics, Result, Statistics};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::execution_plan::CardinalityEffect;

use super::{Problems, overall_statistics, partition_statistics};
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

/// The statistics B2 checks, in the order they are reported
const STATISTICS: [&str; 5] = [
    "num_rows",
    "null_count",
    "distinct_count",
    "min_value",
    "max_value",
];

/// Most columns named in the summary of a B2 finding
const MAX_LISTED_COLUMNS: usize = 4;

/// Most partitions, or ranges of consecutive partitions, named for one
/// column in the summary of a B2 finding
const MAX_LISTED_PARTITION_RANGES: usize = 4;

/// Where a statistic is reported: for a column, or for the whole row
/// (`num_rows`), and in the statistics of the whole output or of a partition.
/// Ordered by column and then partition, with the overall statistics first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StatisticPlace {
    column: Option<usize>,
    partition: Option<usize>,
}

/// An exact statistic that the output contradicts
#[derive(Debug)]
struct FalseClaim {
    statistic: &'static str,
    place: StatisticPlace,
    message: String,
}

impl ExactStatisticsHold {
    fn compare(
        partition: Option<usize>,
        node: &Arc<dyn ExecutionPlan>,
        claimed: &Statistics,
        batches: &[RecordBatch],
        claims: &mut Vec<FalseClaim>,
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
        let target = match partition {
            None => "overall".to_string(),
            Some(p) => format!("partition {p}"),
        };

        if let Precision::Exact(n) = claimed.num_rows
            && Precision::Exact(n) != actual.num_rows
        {
            claims.push(FalseClaim {
                statistic: "num_rows",
                place: StatisticPlace {
                    column: None,
                    partition,
                },
                message: format!(
                    "{target} statistics report num_rows Exact({n}), but the output \
                     has {} rows",
                    actual.num_rows
                ),
            });
        }
        for (i, ((field, claimed), actual)) in schema
            .fields()
            .iter()
            .zip(&claimed.column_statistics)
            .zip(&actual.column_statistics)
            .enumerate()
        {
            let place = StatisticPlace {
                column: Some(i),
                partition,
            };
            let column = field.name();
            for (statistic, claimed, actual) in Self::compare_column(claimed, actual) {
                claims.push(FalseClaim {
                    statistic,
                    place,
                    message: format!(
                        "{target} statistics report {statistic} {claimed} for column \
                         '{column}', but the output has {actual}"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Every exact statistic of `node`, overall and for each partition, that
    /// its output contradicts, in the order: overall and then each
    /// partition, `num_rows` and then each column. `None` if the node was not
    /// executed or failed.
    fn false_claims(
        node: &Arc<dyn ExecutionPlan>,
        context: &CheckContext,
    ) -> Result<Option<Vec<FalseClaim>>> {
        let Some(output) = context.output(node) else {
            return Ok(None);
        };
        let mut claims = vec![];
        // Errors computing statistics are reported by `statistics_shape`
        if let Ok(claimed) = overall_statistics(node.as_ref()) {
            let batches: Vec<RecordBatch> = output.batches().cloned().collect();
            Self::compare(None, node, &claimed, &batches, &mut claims)?;
        }
        for (p, batches) in output.partitions().iter().enumerate() {
            if let Ok(claimed) = partition_statistics(node.as_ref(), p) {
                Self::compare(Some(p), node, &claimed, batches, &mut claims)?;
            }
        }
        Ok(Some(claims))
    }

    /// The name, claimed value and actual value of each exact column
    /// statistic in `claimed` that `actual` contradicts
    fn compare_column(
        claimed: &ColumnStatistics,
        actual: &ColumnStatistics,
    ) -> Vec<(&'static str, String, String)> {
        let mut mismatches = vec![];
        let mut mismatch = |statistic, claimed: &dyn Display, actual: &dyn Display| {
            mismatches.push((statistic, claimed.to_string(), actual.to_string()));
        };
        if claimed.null_count.is_exact() == Some(true)
            && claimed.null_count != actual.null_count
        {
            mismatch("null_count", &claimed.null_count, &actual.null_count);
        }
        if claimed.distinct_count.is_exact() == Some(true)
            && claimed.distinct_count != actual.distinct_count
        {
            mismatch(
                "distinct_count",
                &claimed.distinct_count,
                &actual.distinct_count,
            );
        }
        // The output has no minimum or maximum when it has no non-null values,
        // so an exact claim cannot be contradicted
        if claimed.min_value.is_exact() == Some(true)
            && actual.min_value.is_exact() == Some(true)
            && claimed.min_value != actual.min_value
        {
            mismatch("min_value", &claimed.min_value, &actual.min_value);
        }
        if claimed.max_value.is_exact() == Some(true)
            && actual.max_value.is_exact() == Some(true)
            && claimed.max_value != actual.max_value
        {
            mismatch("max_value", &claimed.max_value, &actual.max_value);
        }
        mismatches
    }

    /// One finding per statistic, in the order of [`STATISTICS`]. Each keeps
    /// the message of the first false claim, overall before the partitions
    /// and in column order, and summarizes the others.
    fn group(node: &Arc<dyn ExecutionPlan>, mut claims: Vec<FalseClaim>) -> Vec<Finding> {
        // A stable sort keeps the order of places within each statistic
        claims.sort_by_key(|claim| {
            STATISTICS
                .iter()
                .position(|statistic| *statistic == claim.statistic)
        });
        let mut problems = Problems::<StatisticPlace>::default();
        for claim in claims {
            problems.add(
                claim.statistic,
                claim.place,
                Finding::invariant(claim.message),
            );
        }
        let schema = node.schema();
        problems.into_findings_with(|statistic, mut finding, _, others| {
            if !others.is_empty() {
                let column_name = |i: usize| schema.field(i).name().as_str();
                finding.message = format!(
                    "{} (also {} more false {statistic} {}: {})",
                    finding.message,
                    others.len(),
                    plural(others.len(), "statistic", "statistics"),
                    summarize_places(others, column_name)
                );
            }
            finding
        })
    }
}

/// Summarize `places` by column, in column order, and then by partition, as
/// in `'a' in partitions 0-2; 'b' overall and in partition 1`. At most
/// [`MAX_LISTED_COLUMNS`] columns are named, followed by how many places are
/// in the other columns.
fn summarize_places<'a>(
    mut places: Vec<StatisticPlace>,
    column_name: impl Fn(usize) -> &'a str,
) -> String {
    places.sort();
    let mut by_column: Vec<(Option<usize>, Vec<Option<usize>>)> = vec![];
    for place in places {
        match by_column.last_mut() {
            Some((column, partitions)) if *column == place.column => {
                partitions.push(place.partition)
            }
            _ => by_column.push((place.column, vec![place.partition])),
        }
    }
    let mut parts: Vec<String> = by_column
        .iter()
        .take(MAX_LISTED_COLUMNS)
        .map(|(column, partitions)| {
            let targets = summarize_targets(partitions);
            match column {
                Some(i) => format!("'{}' {targets}", column_name(*i)),
                None => targets,
            }
        })
        .collect();
    let unlisted = &by_column[by_column.len().min(MAX_LISTED_COLUMNS)..];
    if !unlisted.is_empty() {
        let count: usize = unlisted
            .iter()
            .map(|(_, partitions)| partitions.len())
            .sum();
        parts.push(format!(
            "and {count} more in {} other {}",
            unlisted.len(),
            plural(unlisted.len(), "column", "columns")
        ));
    }
    parts.join("; ")
}

/// Describe sorted targets, `None` for the overall statistics, as in
/// `overall and in partitions 0-2, 5`. At most
/// [`MAX_LISTED_PARTITION_RANGES`] partitions or ranges of consecutive
/// partitions are named, followed by how many other partitions there are.
fn summarize_targets(targets: &[Option<usize>]) -> String {
    let overall = targets.contains(&None);
    let partitions: Vec<usize> = targets.iter().flatten().copied().collect();
    if partitions.is_empty() {
        return "overall".to_string();
    }
    let mut ranges: Vec<(usize, usize)> = vec![];
    for &p in &partitions {
        match ranges.last_mut() {
            Some((_, end)) if *end + 1 == p => *end = p,
            _ => ranges.push((p, p)),
        }
    }
    let listed: Vec<String> = ranges
        .iter()
        .take(MAX_LISTED_PARTITION_RANGES)
        .map(|&(start, end)| match end - start {
            0 => format!("{start}"),
            1 => format!("{start}, {end}"),
            _ => format!("{start}-{end}"),
        })
        .collect();
    let unlisted: usize = ranges
        .iter()
        .skip(MAX_LISTED_PARTITION_RANGES)
        .map(|(start, end)| end - start + 1)
        .sum();
    let overall = if overall { "overall and " } else { "" };
    let more = if unlisted > 0 {
        format!(" and {unlisted} more")
    } else {
        String::new()
    };
    format!(
        "{overall}in {} {}{more}",
        plural(partitions.len(), "partition", "partitions"),
        listed.join(", ")
    )
}

/// `one` if `n` is 1, `many` otherwise
fn plural(n: usize, one: &'static str, many: &'static str) -> &'static str {
    if n == 1 { one } else { many }
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
        let Some(claims) = Self::false_claims(node, context)? else {
            return Ok(vec![]);
        };
        if claims.is_empty() {
            return Ok(vec![]);
        }
        // A node computes its statistics from its children's, so a child
        // with a false exact statistic can make the node's false too, even
        // for a node that passes statistics through unchanged, such as a
        // repartition. Report it on the child. Which of the node's
        // statistics depend on which of the child's is not known, so the
        // node is not reported at all.
        for child in node.children() {
            if Self::false_claims(child, context)?.is_some_and(|child| !child.is_empty())
            {
                return Ok(vec![]);
            }
        }
        Ok(Self::group(node, claims))
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
