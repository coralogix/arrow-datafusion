<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# ExecutionPlan Check Catalog

This document lists every check that the `physical_plan` module of
`datafusion-property-tests` runs against an `ExecutionPlan`.

### Cases

Every check runs once per **case**, one per `Profile` in
`Profile::defaults()`:

- `default`: three partitions (the second empty), exact statistics
- `single partition`
- `inexact statistics`, `absent statistics`
- `empty input`: every partition empty
- `uniform constants`, `constants per partition`: every column is a declared
  constant, the same in every partition or one per partition
- `copied columns`: every column has a declared-equal `__copy` column
- `sorted by row id`: inputs are sorted by `__row_id` and declare it
- `hash partitioned`: inputs are hash partitioned on their first column and
  declare it

Every input has a `__row_id` column (excluded from the above), with a
separate id range per input. Inputs are then sorted or repartitioned as the
plan requires.

## A. Static checks

These checks call the methods that describe the plan, and never call `execute`.

### A1 `equal_cardinality_num_rows`

- **What:** a single-child node with `cardinality_effect() == Equal` and no
  fetch must report the same `num_rows` as its input, overall and per
  partition:
  - `Exact(n)` in: `Exact(m)` out with `m != n` is a violation, `Inexact` out
    is a lint.
  - `Inexact` or `Absent` in: `Exact` out is a violation.
  - `Inexact(n)` in: `Inexact(m)` out with `m != n` is a lint.
  - `Absent` out is left to A9.
- **Partitions:** a single output partition is compared with the whole input.
  Otherwise each partition is compared with the input partition it requests
  (`ChildStats::At(Some(q))`). Partitions that request the overall input
  statistics, as in `RepartitionExec`, are skipped.
- **Why:** `Equal` promises one output row per input row.
  `PassthroughStatisticsProvider`, `sort_pushdown` and
  `limit_pushdown_past_window` rely on this.
- **Severity:** Invariant. Lint when only precision is lost or inexact counts differ.
- **Fix:** pass the input `num_rows` through in `statistics_from_inputs`, or
  return a different `cardinality_effect` if the node can drop or add rows.

### A2 `fetch_not_equal_cardinality`

- **What:** a node whose `fetch()` is `Some(_)` must not return
  `CardinalityEffect::Equal`.
- **Why:** a fetch can drop rows. Rules that treat `Equal` as row-preserving,
  such as `sort_pushdown::can_push_fetch_through` and
  `PassthroughStatisticsProvider`, would ignore the fetch.
- **Severity:** Invariant.
- **Fix:** return `LowerEqual` when a fetch is set, as `SortExec` does.

### A3 `fetch_bounds_num_rows`

- **What:** a node with `fetch() == Some(n)` must report at most `n` rows for
  each partition, and at most `n * partition_count` rows overall.
- **Why:** a fetch is a hard upper bound on rows.
- **Severity:** Invariant for exact row counts. Lint for inexact ones.
- **Fix:** apply the fetch to the statistics, e.g. with
  `Statistics::with_fetch`.

### A4 `cardinality_effect_bounds_num_rows`

- **What:** overall and per partition (matched to input partitions as in A1):
  - a single-child node with `cardinality_effect() == LowerEqual` must not
    report more rows than its input;
  - a node with `cardinality_effect() == GreaterEqual` must not report fewer
    rows than any of its inputs.
- **Why:** the cardinality effect and the statistics must agree. Rules such
  as `topk_aggregation` trust the cardinality effect.
- **Severity:** Invariant for exact row counts. Lint for inexact ones.
- **Fix:** correct whichever of `cardinality_effect` and
  `statistics_from_inputs` is wrong.

### A5 `limit_pushdown_missed`

- **What:** a single-child node with `supports_limit_pushdown() == false`
  that passes its child's rows through:
  - `with_fetch(None)` has `cardinality_effect() == Equal`;
  - `maintains_input_order()[0]` is true;
  - each output partition takes rows from at most one child partition: the
    child has one partition, or each output partition requests the
    statistics of one child partition (unlike `RepartitionExec`).
- **Why:** `LimitPushdown` stops at a node that does not support limit
  pushdown, so its child produces more rows than needed.
- **Severity:** Lint.
- **False positives:** nodes whose output rows depend on later input rows,
  such as window functions like `LEAD` (`WindowAggExec`,
  `BoundedWindowAggExec`) or custom look-ahead operators, and nodes with side
  effects on the rows they read. Allow this check for such nodes.
- **Fix:** return true from `supports_limit_pushdown`.

### A6 `limit_pushdown_merges_partitions`

- **What:** a node with one output partition and
  `supports_limit_pushdown() == true` must not have a child with several
  partitions. One finding per such child. `CoalescePartitionsExec`,
  `SortPreservingMergeExec`, `GlobalLimitExec` and `LocalLimitExec` are not
  checked.
- **Why:** `LimitPushdown` passes the limit through such a node as a
  per-partition `LocalLimitExec` on the child, so a limit of `n` over `k`
  child partitions returns up to `k * n` rows. Only `CoalescePartitionsExec`
  and `SortPreservingMergeExec` are treated as combining partitions.
- **Severity:** Invariant.
- **Not checked:** several children with one partition each, which can
  return up to `n` rows per child.
- **Fix:** return false from `supports_limit_pushdown` when a child has
  several partitions.

### A7 `per_child_lengths` and `check_invariants`

- **What:**
  - `per_child_lengths`: `maintains_input_order`, `required_input_ordering`,
    `benefits_from_input_partitioning`, `input_distribution_requirements` and
    `child_stats_requests` (overall and per partition) each have one entry
    per child. Each `ChildStats::At(Some(p))` names an existing partition of
    that child.
  - `check_invariants`: `check_invariants(Always)` succeeds, and so does
    `check_invariants(Executable)` when the children meet the node's input
    requirements as `SanityCheckPlan` checks them.
- **Why:** optimizer rules index these vectors by child position, so a wrong
  length can panic, skip a child, or apply one child's requirement to
  another. `SanityCheckPlan` rejects a node that fails
  `check_invariants(Executable)`.
- **Severity:** Invariant.
- **Fix:** return one entry per child, in `children()` order. The default
  `check_invariants` checks some lengths too, so one mistake can be reported
  by both checks.

### A8 `statistics_shape` and `partition_statistics_sum`

- **What:**
  - `statistics_shape`: statistics can be computed overall and per
    partition, with one column statistics entry per field in `schema()`. A
    child's error is reported only on the child.
  - `partition_statistics_sum`: if every partition has an exact `num_rows`,
    the overall `num_rows` must be their exact sum. If the overall count is
    exact, no partition may have a larger exact count.
- **Why:** `statistics_from_inputs` should return `Statistics::new_unknown`
  rather than an error, and `column_statistics` is indexed by field
  position. Row counts feed cost-based decisions, and exact ones can answer
  queries such as `COUNT(*)` directly.
- **Severity:** Invariant. Lint when the overall row count could be exact.
- **Fix:** return `Statistics::new_unknown` instead of an error, and keep
  `column_statistics` in sync with the schema, e.g. with
  `Statistics::project`. For a per-partition fetch, sum `min(rows, fetch)`
  over partitions instead of calling `Statistics::with_fetch` on the merged
  statistics.

### A9 `statistics_ignore_inputs`

- **What:** a single-child node with `cardinality_effect() == Equal` and no
  fetch reports `num_rows` as `Absent` although its input's is known.
- **Why:** an `Equal` node can pass its input's row count through unchanged.
  Reporting `Absent` discards a known count, so every node above it loses it
  too, and cost-based decisions fall back to guesses.
- **Severity:** Lint.
- **Fix:** the usual cause is reading `input_stats` in
  `statistics_from_inputs` without overriding `child_stats_requests`, whose
  default supplies unknown statistics (the report says when). Override it to
  return `ChildStats::At(partition)`, and pass the input row count through.

### A10 `schema_consistency` and `expression_column_refs`

- **What:**
  - `schema_consistency`: `schema()` equals the schema of
    `properties().eq_properties`, including nullability and metadata.
  - `expression_column_refs`: every `Column` in the output orderings,
    equivalence classes, constants and output partitioning has an index
    inside `schema()` and the name of the field at that index. For
    single-child nodes, `Column`s in `apply_expressions` are checked the same
    way against the input schema. The exception is a `Final` `AggregateExec`,
    whose aggregate expressions refer to the input of the `Partial` aggregate
    below it, so they are checked against `AggregateExec::input_schema`.
- **Why:** stale column indexes are a common result of projection pushdown
  and child replacement. They make ordering and partitioning checks refer to
  the wrong column, which can remove a needed sort or repartition.
- **Severity:** Invariant. Lint when `schema_consistency` differs only in
  metadata.
- **Fix:** rebuild expressions and equivalence properties against the current
  schema after changing children or projections.

### A11 `dynamic_expressions_visited` and `dynamic_expressions_reset`

- **What:**
  - `dynamic_expressions_visited`: every expression in
    `dynamic_expressions_produced()` has an expression id that appears in
    the expressions visited by `apply_expressions`.
  - `dynamic_expressions_reset`: after `reset_state`, the node produces none
    of the dynamic expressions it produced before, compared by expression
    id. A filter that keeps its id keeps the values set by execution.
- **Why:** rules that find or rewrite dynamic filters rely on
  `apply_expressions` visiting them. A dynamic filter that survives
  `reset_state` makes re-execution, such as in recursive queries, use stale
  filter values.
- **Severity:** Invariant.
- **Fix:** include produced dynamic filters in `apply_expressions`, and
  recreate them in `reset_state`.
