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
`datafusion-property-tests` runs against an `ExecutionPlan`. For each check
it explains what is verified, why DataFusion depends on it, and what to
change in the plan when the check reports a violation.

## Reading this catalog

Each check has a **name** (such as `equal_cardinality_num_rows`) that is
stable and appears in reports, and a **code** (such as `A1`) that places it
in this catalog. A catalog entry can contain more than one named check when
they verify closely related properties. Use the name with
`PlanFactory::allow` or `PlanChecker::allow` to skip a check.

Every violation has a severity:

- **Invariant**: the plan reports a property that is not true. Optimizer rules
  and parent operators trust these properties, so an invariant violation can
  lead to wrong query results. These must be fixed.
- **Lint**: the plan is correct, but reports a weaker property than it could,
  so DataFusion misses an optimization. These are worth fixing, but can be
  allowed when the weaker property is intentional.

Every check in this catalog is static: it compares what a plan reports about
itself with what its children report, without running it.

Checks run on one node at a time, but the checker visits every node of a
plan. Tests should build the plan under test on inputs generated with
`fixtures::SourceSpec`. It produces a `fixtures::MockSourceExec` whose
statistics are computed from its data and whose declared ordering and
partitioning are verified against it, so checks treat what it reports as the
truth. The usual way is to describe the plan with a `harness::PlanFactory`
and check it with a `harness::PlanHarness`, which generates the inputs and
runs the checks on several cases (see [Cases](#cases)).

### Cases

The `PlanHarness` checks a plan on several **cases**, one per `Profile`,
each a fully specified set of inputs. `Profile::defaults()` gives:

- `default`: three partitions, the second of them empty, and exact
  statistics;
- `single partition`;
- `inexact statistics` and `absent statistics`, with the default layout;
- `empty input`: every partition empty;
- `uniform constants` and `constants per partition`: every column of every
  input other than the row id is constant, with one value in every
  partition, or one value per partition, and the inputs declare the
  constants;
- `copied columns`: every column of every input other than the row id has a
  copy, named with the suffix `__copy`, which the input declares equal to it;
- `sorted by row id`: every input that does not need another ordering
  declares that its partitions are sorted by the row id column, which they
  are, so every node sees sorted input. Nodes take other code paths on input
  that reports an ordering, such as `HashJoinExec`, which keeps the order of
  its probe side only then, and report orderings of their own;
- `hash partitioned`: every input that does not need another partitioning is
  hash partitioned on its first column into three partitions, and declares
  it, so every node sees hash partitioned input.

Every case splits the rows of each partition into random batches of up to 16
rows, including empty batches.

Inputs are then adapted to the plan's requirements: an input that must be
sorted is sorted, one that must be hash partitioned is hash partitioned into
the profile's number of partitions (the same for every input, so
co-partitioned inputs match), and one that must be a single partition gets
one. Every input has a `__row_id` column, with ids in a separate range per
input.

Every check runs in every case. The report lists every case, with the plan
and the findings of each case that has any.

### Terms

- A **single-child** node has exactly one entry in `children()`.
- An **exact** statistic is `Precision::Exact`. An **estimate** is
  `Precision::Inexact`.
- A fetch on a node with several output partitions limits **each
  partition**. Only nodes that combine partitions, such as
  `CoalescePartitionsExec` and `SortPreservingMergeExec`, or nodes with one
  output partition, apply a fetch to their whole output. This matches how the
  `LimitPushdown` optimizer rule uses `with_fetch`.

## A. Static checks

These checks call the methods that describe the plan, and never call
`execute`.

### A1 `equal_cardinality_num_rows`

- **Severity:** Invariant. Lint when only precision or an estimate is lost.
- **What:** a single-child node with `cardinality_effect() == Equal` and no
  fetch must report the same `num_rows` as the input rows it is made of, with
  the same precision, overall and for each partition whose input rows are
  known:
  - Input `Exact(n)` requires output `Exact(n)`. Output `Exact(m)` with
    `m != n` is an invariant violation. Output `Inexact(_)` is a lint.
  - Input `Inexact(_)` or `Absent` with output `Exact(_)` is an invariant
    violation: the node cannot know its row count better than its input
    does.
  - Input `Inexact(n)` with output `Inexact(m)` and `m != n` is a lint.
  - Output `Absent` for a known input is reported by A9.
- **Partitions:** the only partition of a node with one output partition has
  every input row. Each partition of a node with several is compared with the
  input partition that `child_stats_requests` requests for it
  (`ChildStats::At(Some(q))`), as a node that keeps rows in their partitions
  requests. A partition for which the node requests the overall input
  statistics, as `RepartitionExec` does, can have rows of every input
  partition, so it is not compared.
- **Why:** `Equal` promises exactly one output row per input row.
  `PassthroughStatisticsProvider`, `sort_pushdown` and
  `limit_pushdown_past_window` all rely on this. A node whose statistics
  disagree with its cardinality effect has one of the two wrong.
- **Fix:** if the node really produces one row per input row, pass the input
  `num_rows` through unchanged in `statistics_from_inputs` (and request the
  input statistics in `child_stats_requests`). If it can drop or add rows,
  return `LowerEqual`, `GreaterEqual` or `Unknown` from `cardinality_effect`.
- **Not checked:** multi-child nodes, because which input `Equal` refers to is
  not defined for them.

### A2 `fetch_not_equal_cardinality`

- **Severity:** Invariant.
- **What:** a node whose `fetch()` is `Some(_)` must not return
  `CardinalityEffect::Equal`.
- **Why:** a fetch can drop input rows, so `Equal` is false while a fetch is
  set. Rules that treat `Equal` nodes as row-preserving, such as
  `sort_pushdown::can_push_fetch_through` and `PassthroughStatisticsProvider`,
  will then ignore the fetch.
- **Fix:** return `LowerEqual` from `cardinality_effect` when a fetch is set,
  as `SortExec` does:

  ```rust
  fn cardinality_effect(&self) -> CardinalityEffect {
      if self.fetch.is_none() {
          CardinalityEffect::Equal
      } else {
          CardinalityEffect::LowerEqual
      }
  }
  ```

### A3 `fetch_bounds_num_rows`

- **Severity:** Invariant for exact row counts. Lint for estimates.
- **What:** a node with `fetch() == Some(n)` must report at most `n` rows for
  each partition, and at most `n * partition_count` rows overall.
- **Why:** a fetch is a hard upper bound, so an exact row count above it is
  false. An estimate above it is a bad estimate that the node could easily
  cap.
- **Fix:** apply the fetch to the statistics, for example with
  `Statistics::with_fetch`.

### A4 `cardinality_effect_bounds_num_rows`

- **Severity:** Invariant for exact row counts. Lint for estimates.
- **What:** overall and for each partition:
  - A single-child `LowerEqual` node must not report more rows than its input.
    A partition has at most the rows of the input partition it is made of
    (see A1), and in any case at most every input row.
  - A `GreaterEqual` node must not report fewer rows than any of its inputs.
    A partition has at least the rows of each input partition it is made of,
    as a partition of `UnionExec` has the rows of one partition of one input.
- **Why:** the cardinality effect and the statistics are two descriptions of
  the same thing. If they disagree, one of them is wrong, and rules such as
  `topk_aggregation` that trust the cardinality effect may make the wrong
  decision.
- **Fix:** correct whichever of `cardinality_effect` and
  `statistics_from_inputs` is wrong.

### A5 `limit_pushdown_missed`

- **Severity:** Lint.
- **What:** a single-child node for which `supports_limit_pushdown()` is
  false, but whose properties show that it passes the rows of its child
  through:
  - its cardinality effect without its fetch (that of the plan returned by
    `with_fetch(None)`) is `Equal`;
  - `maintains_input_order()[0]` is true;
  - each output partition is made of the rows of at most one partition of
    the child: the child has one partition, however the node spreads its
    rows, or the node has as many partitions as the child and does not move
    rows between them. A node moves rows between partitions when it
    computes the statistics of an output partition from the overall
    statistics of a child with several partitions (`child_stats_requests`),
    as `RepartitionExec` does.
- **Why:** `LimitPushdown` stops at a node that does not support limit
  pushdown: it gives the node the limit as a fetch, or adds a limit above it
  (`limit_pushdown.rs:268-309`), so the child produces more rows than
  needed. A node with these properties can let the limit move below it: with
  every partition of the child limited to its first `n` rows, each output
  partition has at most `n` rows, and the whole output at least
  `min(n, rows)` rows, which is all `LimitPushdown` needs.
- **False positives:** `Equal` counts rows, it does not say that an output
  row depends on its input row only. A node whose output rows depend on
  later input rows reports both properties and still needs those rows: a
  window function such as `LEAD`, `NTILE` or `CUME_DIST`, an aggregate over
  a whole window partition or over a `RANGE` frame, which includes the rows
  that tie with the current row, or a custom operator that looks ahead, such
  as a backward fill. So does a node with side effects on the rows it reads,
  such as one that writes them somewhere. `WindowAggExec` and
  `BoundedWindowAggExec` are such nodes; `limit_pushdown_past_window` pushes
  limits past the windows that only read earlier rows. Allow this check for
  such nodes.
- **Not checked:**
  - A node with several children. Which child `Equal` refers to is not
    defined (see A1), and `LimitPushdown` limits every child, including
    children such as the subqueries of `ScalarSubqueryExec`, whose rows the
    node does not pass through.
  - A node that reports `GreaterEqual`, which does not say that every input
    row produces an output row.
  - A node whose output partitions each merge rows of several partitions of
    the child, such as an order preserving `RepartitionExec` over several
    partitions. With every partition of the child limited to `n` rows, an
    output partition can get up to `n` rows from each of them.
- **Fix:** return true from `supports_limit_pushdown`.
- **Found in:** `BufferExec` (its `supports_limit_pushdown` returns the
  child's value), `CoalesceBatchesExec`, and `RepartitionExec` over a child
  with one partition, with every partitioning and with or without preserving
  order. `RepartitionExec` must not support limit pushdown over a child with
  several partitions (see A6), so its answer would depend on the
  number of input partitions.

### A6 `limit_pushdown_merges_partitions`

- **Severity:** Invariant.
- **What:** a node for which `supports_limit_pushdown()` is true and that has
  one output partition has no child with several partitions.
  `CoalescePartitionsExec`, `SortPreservingMergeExec`, `GlobalLimitExec` and
  `LocalLimitExec` are not checked.
- **Reporting:** one finding for each child with several partitions.
- **Why:** `LimitPushdown` removes `GlobalLimitExec` and `LocalLimitExec`
  nodes and keeps their limit (`extract_limit`, `limit_pushdown.rs:414-431`).
  It only adds a limit back where the limit stops moving down, as a
  `GlobalLimitExec` above a plan with one partition and a `LocalLimitExec`,
  which limits each partition, above any other (`add_limit`,
  `limit_pushdown.rs:441-450`). It only treats `CoalescePartitionsExec` and
  `SortPreservingMergeExec` as nodes that combine partitions
  (`combines_input_partitions`, `limit_pushdown.rs:435-437`), and gives them
  the limit as a fetch. Any other node that supports limit pushdown passes
  the limit to its children, so a `GlobalLimitExec` of `n` rows above a node
  with one output partition over a child with `k` partitions becomes a
  `LocalLimitExec` of `n` rows on the child, and the query returns up to
  `k * n` rows. The rule never asks limit nodes whether they support limit
  pushdown.
- **Not checked:** a node with one output partition over several children
  with one partition each. Its output partition can also have up to `n` rows
  from each child, unless the node ignores some children, as
  `ScalarSubqueryExec` ignores its subqueries.
- **Fix:** return false from `supports_limit_pushdown` when a child has
  several partitions.
- **Found in:** none of the built-in plans. A `RepartitionExec` into one
  partition over a child with several would be reported if it supported
  limit pushdown.

### A7 `per_child_lengths` and `check_invariants`

- **Severity:** Invariant.
- **What:**
  - `per_child_lengths`: `maintains_input_order`, `required_input_ordering`,
    `benefits_from_input_partitioning`, the per-child entries of
    `input_distribution_requirements`, and `child_stats_requests` (overall and
    for every partition) each have exactly `children().len()` entries. Each
    `ChildStats::At(Some(p))` request names a partition that exists on that
    child.
  - `check_invariants`: `check_invariants(InvariantLevel::Always)` succeeds.
    When the children meet the node's input requirements as
    `SanityCheckPlan` checks them (the first alternative of each ordering
    requirement, hard or soft, and each distribution requirement, allowing a
    subset of the keys), and are co-partitioned where required,
    `check_invariants(InvariantLevel::Executable)` succeeds too. A plan whose
    requirements are not met cannot be executed, so it may fail the
    executable invariants for a reason that is not the node's.
- **Why:** optimizer rules index these vectors by child position. A wrong
  length makes them panic, skip a child, or apply one child's requirement to
  another. `check_invariants` can be overridden, so a plan that replaces the
  default implementation may skip the default length checks.
  `SanityCheckPlan` calls `check_invariants(Executable)` once the requirements
  are met, so a node that fails it there is rejected although its inputs are
  valid.
- **Fix:** return one entry per child, in the same order as `children()`.
  Default `check_invariants` already checks some lengths, so a single mistake
  can be reported by both checks.

### A8 `statistics_shape` and `partition_statistics_sum`

- **Severity:** Invariant. Lint when the overall row count could be exact.
- **What:**
  - `statistics_shape`: statistics can be computed overall and for every
    partition, and each has one column statistics entry per field in
    `schema()`. An error from a child is only reported on the child.
  - `partition_statistics_sum`: if every partition has an exact `num_rows`,
    the overall `num_rows` must be exactly their sum. An inexact or absent
    overall row count in that case is a lint. If the overall row count is
    exact, no single partition may have an exact row count above it.
- **Why:** `statistics_from_inputs` is documented to return
  `Statistics::new_unknown` rather than an error. Code that uses statistics
  indexes `column_statistics` by field position. A wrong overall row count
  feeds cost-based decisions, and an exact one can be used to answer queries
  such as `COUNT(*)` directly.
- **Fix:**
  - Return `Statistics::new_unknown` instead of an error when statistics are
    not available.
  - Keep `column_statistics` in sync with the output schema, for example with
    `Statistics::project`.
  - For a per-partition fetch on a node with several partitions, the overall
    row count is the sum over partitions of `min(rows, fetch)`. Calling
    `Statistics::with_fetch(fetch, skip, 1)` on the merged input statistics
    treats the fetch as a global limit and undercounts. Compute the overall
    count from per-partition counts, or report it as `Inexact` when it cannot
    be proven.

### A9 `statistics_ignore_inputs`

- **Severity:** Lint.
- **What:** a single-child `Equal` node with no fetch, whose input has a known
  `num_rows`, reports `Absent`.
- **Why:** the most common cause is reading `input_stats` in
  `statistics_from_inputs` without overriding `child_stats_requests`, whose
  default skips every child and supplies unknown placeholder statistics. The
  report says when that is the case.
- **Fix:** override `child_stats_requests` to return
  `ChildStats::At(partition)` for the inputs the node uses, and pass the input
  row count through.

### A10 `schema_consistency` and `expression_column_refs`

- **Severity:** Invariant. Lint for a difference in metadata only
  (`schema_consistency`).
- **What:**
  - `schema_consistency`: `schema()` equals the schema of
    `properties().eq_properties`: the same fields, with the same names, data
    types and nullability, and the same field and schema metadata. A node
    whose child's two schemas differ is not reported, since it can inherit
    the difference.
  - `expression_column_refs`: every `Column` in the output orderings,
    equivalence classes, constants (equivalence classes with a constant
    value) and output partitioning (the hash expressions, and the ordering of
    a range partitioning) has an index inside `schema()` and the name of the
    field at that index. For single-child nodes, every `Column` in the
    expressions visited by `apply_expressions` refers to a field of the input
    schema in the same way. An `AggregateExec` that merges partial states,
    such as a `Final` aggregate, binds its aggregate expressions to the input
    of the partial aggregate (`AggregateExec::input_schema`), so a column that
    refers to a field of that schema is accepted.
- **Why:** stale column indexes are a common result of projection pushdown
  and child replacement. They make ordering and partitioning checks refer to
  the wrong column, which can remove a needed sort or repartition.
- **Fix:** rebuild expressions and equivalence properties against the current
  schema after changing children or projections.

### A11 `dynamic_expressions_visited` and `dynamic_expressions_reset`

- **Severity:** Invariant.
- **What:**
  - `dynamic_expressions_visited`: every expression in
    `dynamic_expressions_produced()` has an expression id, and that id appears
    somewhere in the expressions visited by `apply_expressions`.
  - `dynamic_expressions_reset`: after `reset_state`, the node's dynamic
    filters are back in their initial state: the node produces none of the
    dynamic expressions it produced before. They are compared by expression
    id, which identifies the state of a dynamic filter: copies made with
    `with_new_children` share the state and keep the id, and a new filter
    gets a new id. An expression that keeps its id after the reset still has
    the values that executing the node set.
- **Why:** `apply_expressions` is documented to visit expressions a node
  updates dynamically. Rules that find or rewrite dynamic filters depend on
  it. A dynamic filter that survives `reset_state` makes re-execution (such as
  recursive queries) use stale filter values.
- **Fix:** include produced dynamic filters in `apply_expressions`, and
  recreate them in `reset_state`.

### A12 `display_no_panic`

- **Severity:** Invariant.
- **What:** every `DisplayFormatType` and the tree renderer work without
  panicking, and `name()` is not empty. The node alone is formatted with
  `fmt_as` in `Default`, `Verbose` and `TreeRender`, and the node and its
  descendants with the tree renderer (`displayable(plan).tree_render()`),
  each into a `String`, with panics caught. Returning `fmt::Error` is
  reported too: writing to a `String` cannot fail, and `to_string()`, which
  `EXPLAIN` uses, panics on it. A panic in `name()` is reported as well.
- **Attribution:** the tree renderer formats every node of the subtree, so a
  node whose `TreeRender` output panics makes the tree renderer panic for
  every ancestor. The tree renderer is only reported for a node whose
  children all render.
- **Reporting:** one finding for an empty or panicking `name()`, and one for
  each way of displaying the node that fails, with the panic message.
- **Not checked:** `displayable(plan).indent()`, `one_line()` and
  `graphviz()`, which format each node with `fmt_as` in `Default` or
  `Verbose`, and the statistics and schema that `EXPLAIN VERBOSE` can add.
- **Why:** `EXPLAIN`, logging and error messages display plans. A display
  that panics takes down the query, or the process, that wanted to show it.
- **Fix:** do not index, unwrap or slice in `fmt_as`, and return a non-empty
  constant from `name()`.
