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

This document lists every check that `datafusion-physical-plan-checks` runs,
or plans to run, against an `ExecutionPlan`. For each check it explains what
is verified, why DataFusion depends on it, and what to change in the plan
when the check reports a violation.

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for which checks are
implemented.

## Reading this catalog

Each check has a **code** (such as `A1`) that places it in this catalog, and
a **name** (such as `equal_cardinality_num_rows`) that is stable and appears
in reports. A catalog entry can contain more than one named check when they
verify closely related properties. Use the name with `PlanChecker::allow` to
skip a check.

Every violation has a severity:

- **Invariant**: the plan reports a property that is not true. Optimizer rules
  and parent operators trust these properties, so an invariant violation can
  lead to wrong query results. These must be fixed.
- **Lint**: the plan is correct, but reports a weaker property than it could,
  so DataFusion misses an optimization. These are worth fixing, but can be
  allowed when the weaker property is intentional.

Checks come in four kinds:

- **Static** checks compare what a plan reports about itself, without running
  it (section A).
- **Execution** checks run the plan and compare its output with what it
  reports (sections B and C).
- **Differential** checks compare a plan with a rewritten version of itself,
  such as the plan returned by `with_fetch`, and require both to produce
  equivalent results (section D).
- **Metamorphic** checks run the same plan under different configurations or
  input layouts and require the same results (section E).

Section F covers the lifecycle of execution: cleanup, errors and edge cases.

Checks run on one node at a time, but the checker visits every node of a
plan. Tests should build the plan under test on top of
`fixtures::MockSourceExec`, whose statistics, partitioning and ordering are
treated as the truth.

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

These checks only call methods that describe the plan. They never call
`execute`.

### A1 `equal_cardinality_num_rows`

- **Severity:** Invariant. Lint when only precision or an estimate is lost.
- **What:** a single-child node with `cardinality_effect() == Equal` and no
  fetch must report the same `num_rows` as its input, with the same
  precision:
  - Input `Exact(n)` requires output `Exact(n)`. Output `Exact(m)` with
    `m != n` is an invariant violation. Output `Inexact(_)` is a lint.
  - Input `Inexact(_)` or `Absent` with output `Exact(_)` is an invariant
    violation: the node cannot know its row count better than its input
    does.
  - Input `Inexact(n)` with output `Inexact(m)` and `m != n` is a lint.
  - Output `Absent` for a known input is reported by A9.
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
- **What:**
  - A single-child `LowerEqual` node must not report more rows than its input.
  - A `GreaterEqual` node must not report fewer rows than any of its inputs.
- **Why:** the cardinality effect and the statistics are two descriptions of
  the same thing. If they disagree, one of them is wrong, and rules such as
  `topk_aggregation` that trust the cardinality effect may make the wrong
  decision.
- **Fix:** correct whichever of `cardinality_effect` and
  `statistics_from_inputs` is wrong.

### A5 `limit_pushdown_missed`

- **Severity:** Lint.
- **Requires execution.**
- **What:** a node for which `supports_limit_pushdown()` is false, but that
  is **prefix-closed**: for random inputs and every `n`, running the node on
  the first `n` rows of each input partition gives the first `n` rows of its
  normal output.
- **Why:** a prefix-closed node can let a limit move below it, so its inputs
  produce less data. Being `Equal` and order-preserving is not enough on its
  own: `WindowAggExec` is both, but a window over a whole partition needs rows
  after the first `n`, so it correctly does not support limit pushdown. This
  check only reports nodes whose outputs have been shown to be prefix-closed.
- **Fix:** return true from `supports_limit_pushdown`. Allow this check if the
  node is only prefix-closed for the inputs the test generated.
- **Candidates:** `BufferExec` (its `supports_limit_pushdown` returns the
  child's value) and `RepartitionExec`.

### A6 `maintains_input_order_missed`

- **Severity:** Lint.
- **Requires execution.**
- **What:** `maintains_input_order()[i]` is false, but the rows from child `i`
  keep their relative order in the output for many random inputs (see C1 for
  how order is tracked).
- **Why:** `maintains_input_order` lets the optimizer avoid re-sorting. A
  false value that should be true costs an unnecessary sort.
- **Fix:** return true for that child and make sure the output equivalence
  properties keep the child's ordering.
- **Candidates:** `CoalescePartitionsExec` with a single input partition.

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
- **Why:** optimizer rules index these vectors by child position. A wrong
  length makes them panic, skip a child, or apply one child's requirement to
  another. `check_invariants` can be overridden, so a plan that replaces the
  default implementation may skip the default length checks.
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
- **Planned addition:** `statistics_invalid_partition`: calling
  `statistics_from_inputs` directly with a partition index at or past the
  partition count returns an error rather than panicking.

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

- **Severity:** Invariant.
- **What:**
  - `schema_consistency`: `schema()` equals the schema of
    `properties().eq_properties`.
  - `expression_column_refs`: every `Column` in the output orderings,
    equivalence classes, constants and output partitioning has an index
    inside `schema()` and the name of the field at that index. For
    single-child nodes, every `Column` in `apply_expressions` refers to a
    valid field of the input schema in the same way.
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
    filters are back in their initial state.
- **Why:** `apply_expressions` is documented to visit expressions a node
  updates dynamically. Rules that find or rewrite dynamic filters depend on
  it. A dynamic filter that survives `reset_state` makes re-execution (such as
  recursive queries) use stale filter values.
- **Fix:** include produced dynamic filters in `apply_expressions`, and
  recreate them in `reset_state`.

## B. Reported properties hold at runtime

These checks run the plan on generated input and compare the output with
what the plan reports.

### B1 `batch_schema`

- **Severity:** Invariant. Lint for field metadata differences.
- **What:** every batch has exactly `schema()`, including nullability, and
  columns of non-nullable fields contain no nulls.
- **Why:** parent operators build their own schema from `schema()`. A
  mismatched batch causes errors or wrong results downstream.
- **Fix:** make the produced batches match `schema()`, or correct `schema()`.

### B2 `exact_statistics_hold`

- **Severity:** Invariant.
- **What:** every exact statistic is true for the actual output, overall and
  for each partition: `num_rows`, and for each column `min_value`,
  `max_value`, `null_count` and `distinct_count`. The check runs once with
  input statistics marked exact and once with them marked inexact, to catch
  nodes that turn estimates into exact values.
- **Why:** exact statistics are used to prove things, for example to remove a
  limit or to answer an aggregate without reading data.
- **Fix:** report `Inexact` for anything that cannot be proven.

### B3 `orderings_hold`

- **Severity:** Invariant.
- **What:** each output partition is sorted by every ordering in the output
  equivalence properties, not just `output_ordering()`, respecting descending
  and nulls-first options.
- **Why:** the optimizer removes sorts that an ordering claims are already
  satisfied. A false ordering gives unsorted query results.
- **Fix:** remove the ordering from the equivalence properties, or make the
  node preserve it.

### B4 `constants_hold`

- **Severity:** Invariant.
- **What:** each constant expression has a single value within each
  partition. If the constant is marked uniform across partitions, it has the
  same given value in every partition.
- **Why:** constants let the optimizer drop sort keys and grouping columns.
- **Fix:** only mark an expression constant when the node guarantees it.

### B5 `equivalence_classes_hold`

- **Severity:** Invariant.
- **What:** expressions in the same equivalence class are equal on every
  output row.
- **Why:** equivalence classes let an ordering or partitioning on one column
  satisfy a requirement on another.
- **Fix:** only add equivalences the node guarantees, for example from an
  equality filter or join key.

### B6 `constraints_hold`

- **Severity:** Invariant.
- **What:** primary key and unique constraints hold on the output: no
  duplicate key tuples, and no nulls in primary key columns.
- **Why:** constraints allow removing redundant grouping and distinct
  operations.
- **Fix:** drop constraints the node cannot guarantee, for example after a
  join that can duplicate rows.

### B7 `hash_partitioning_holds` and `invalid_partition_errors`

- **Severity:** Invariant.
- **What:**
  - `hash_partitioning_holds`: with `Partitioning::Hash(exprs, n)`, a key
    tuple never appears in two partitions, and the partition index equals
    `hash(key) % n` using the hash that `RepartitionExec` uses.
  - `invalid_partition_errors`: `execute(i)` with `i` at or past the
    partition count returns an error, not a panic.
- **Why:** partitioned joins and aggregates assume that two inputs hash
  partitioned on equal keys are co-partitioned. A node that claims hash
  partitioning but distributes rows differently silently drops join matches
  or splits groups.
- **Fix:** report `UnknownPartitioning` unless the node's partitioning really
  matches the repartition hash.

### B8 `cardinality_effect_holds`

- **Severity:** Invariant.
- **What:** the actual output row count agrees with `cardinality_effect()`:
  equal to, at most, or at least the input row count. As in A1 and A4,
  `Equal` and `LowerEqual` are only checked for single-child nodes.
- **Why:** see A1 and A4.
- **Fix:** correct `cardinality_effect`.

### B9 `metrics_output_rows`

- **Severity:** Invariant.
- **What:** if the node reports an `output_rows` metric, it equals the number
  of rows produced. After execution completes, the set of metric names does
  not change between calls to `metrics()`.
- **Why:** `EXPLAIN ANALYZE` and adaptive features read metrics.
- **Fix:** record output rows on every path that emits a batch, including
  early termination.

### B10 `boundedness_holds`

- **Severity:** Invariant.
- **What:** a node that reports `Boundedness::Bounded` finishes within a
  timeout on bounded input. On an unbounded mock input, a node may only
  report `Bounded` if it has a fetch.
- **Why:** the optimizer rejects plans that need to see all of an unbounded
  input, and picks streaming operators based on boundedness.
- **Fix:** derive boundedness from the inputs, for example with
  `boundedness_from_children`.

### B11 `emission_type_holds`

- **Severity:** Invariant.
- **What:** a node that reports `EmissionType::Incremental` or `Both` produces
  output before its input ends. The check uses a small batch size and a
  source that produces some batches and then stays pending forever.
- **Why:** streaming queries on unbounded input depend on incremental
  operators actually emitting output.
- **Fix:** report `Final` if the node waits for all input.

### B12 `lazy_evaluation_holds`

- **Severity:** Invariant.
- **What:** with `EvaluationType::Lazy`, calling `execute()` without polling
  the stream polls no input, and each output poll causes a bounded number of
  input polls.
- **Why:** optimizer rules use the evaluation type to reason about which
  operators drive work ahead of demand.
- **Fix:** report `EvaluationType::Eager` for nodes that spawn tasks or
  buffer ahead.

### B13 `cooperative_scheduling_holds`

- **Severity:** Invariant.
- **What:** with `SchedulingType::Cooperative`, the stream returns `Pending`
  when the Tokio task budget runs out, even over an input that is always
  ready. Any plan over such an input can be cancelled within a timeout.
- **Why:** queries must be cancellable, and a stream that never yields blocks
  the Tokio worker thread.
- **Fix:** consume budget with the helpers in the `coop` module, or report
  `NonCooperative`.

## C. Order and input requirements

Generated inputs carry a hidden `__row_id` column that increases within each
input and uses a separate range for each child. Many nodes pass it through,
which lets the checks track where each output row came from.

### C1 `maintains_input_order_holds`

- **Severity:** Invariant.
- **What:** if `maintains_input_order()[i]` is true, rows from child `i` keep
  their relative order within each output partition.
- **Why:** the optimizer keeps orderings through nodes that claim to maintain
  them and removes the sorts above.
- **Fix:** return false for that child.

### C2 `required_input_ordering_honest`

- **Severity:** Invariant.
- **What:** give the node inputs that meet its required orderings but are
  otherwise shuffled. For a deterministic node (no fetch, no volatile
  expressions), if `required_input_ordering()[i]` is `None`, permuting the
  rows of child `i` does not change the output multiset.
- **Why:** a node that needs sorted input but does not say so gives wrong
  results whenever the optimizer does not happen to sort for it.
- **Fix:** declare the ordering in `required_input_ordering`.

### C3 `input_distribution_honest`

- **Severity:** Invariant.
- **What:** if the required distribution of a child is
  `UnspecifiedDistribution`, the combined output of all partitions does not
  depend on how the child's rows are split among partitions.
- **Why:** a node that needs hash partitioned or single partition input but
  does not say so, such as a final aggregate, gives wrong results when the
  optimizer does not repartition for it.
- **Fix:** declare the distribution in `input_distribution_requirements`.

## D. Rewrite hooks are equivalent

Every hook that returns a new plan must produce the same results as the
original: the same multiset of rows, and the same sequence wherever an
ordering is claimed. Rows that tie on the sort key may appear in any order.

### D1 `with_fetch_equivalent`

- **Severity:** Invariant.
- **What:** for `with_fetch(Some(n))`:
  - `fetch()` returns `Some(n)` on the new plan.
  - Each partition has at most `n` rows, and the whole output has at most `n`
    rows for nodes that combine partitions or have one output partition.
  - The output is a subset of the unfetched output, and a prefix of it when an
    ordering is claimed.
  - The schema, partitioning and orderings do not change.
  - `with_fetch(None)` gives the full output again, and `with_fetch(Some(0))`
    works.
- **Why:** `LimitPushdown` replaces limit nodes with fetches and trusts the
  result.
- **Fix:** apply the fetch after all other work the node does, per output
  partition.

### D2 `limit_pushdown_equivalent`

- **Severity:** Invariant.
- **What:** if `supports_limit_pushdown()` is true, running the node with the
  first `n` rows of every child gives at least `min(n, total)` rows, and those
  rows are a valid limit of the normal output: a prefix when an ordering is
  required, and a subset otherwise.
- **Why:** `LimitPushdown` removes the limit above such a node and places it
  on every child. If the node drops or reorders rows, the pushed-down limit
  gives too few or the wrong rows.
- **Fix:** return false from `supports_limit_pushdown`.

### D3 `projection_swap_equivalent`

- **Severity:** Invariant.
- **What:** when `try_swapping_with_projection` returns a plan, it gives the
  same results as the projection on top of the original node.
- **Fix:** remap every expression the node uses through the projection.

### D4 `sort_pushdown_equivalent`

- **Severity:** Invariant.
- **What:** when `try_pushdown_sort` returns `Exact`, the new plan's output
  is sorted by the requested order and has the same rows. When it returns
  `Inexact`, the new plan has the same rows.
- **Fix:** only return `Exact` when the order is guaranteed.

### D5 `filter_pushdown_equivalent`

- **Severity:** Invariant.
- **What:**
  - `gather_filters_for_pushdown` returns one result per parent filter, in
    the same order.
  - A filter sent to child `i` only refers to child `i`'s columns.
  - A filter on top of the node gives the same results as the plan after the
    `FilterPushdown` rule has run.
  - A filter reported as `PushedDown::Yes` is fully applied by the child.
    `FilterExec` relies on this when it removes itself and moves its fetch to
    the child.
- **Fix:** only mark a filter as pushed down when the child applies it
  exactly.

### D6 `repartitioned_equivalent`

- **Severity:** Invariant.
- **What:** `repartitioned(n)` gives the same rows, at most `n` partitions,
  and a plan that passes the section B checks.
- **Fix:** keep the reported properties in sync with the new partitioning.

### D7 `preserve_order_holds`

- **Severity:** Invariant.
- **What:** the plan returned by `with_preserve_order(true)` produces output
  sorted by its reported orderings.
- **Fix:** keep the order when asked to, or return `None`.

### D8 `replace_children_consistent`

- **Severity:** Invariant.
- **What:**
  - With children whose properties are unchanged,
    `ChildrenPropertiesMode::Keep` and `ChildrenPropertiesMode::Recompute`
    give identical plan properties.
  - Replacing children does not depend on the previous children:
    `P(A).replace(B)` has the same properties as `P(C).replace(B)`, and
    `P(A).replace(B).replace(A)` has the same properties as `P(A)`.
  - Passing the wrong number of children returns an error.
- **Why:** these catch properties cached when the node was first built, such
  as an aggregate's input order mode, that are not recomputed for new
  children.
- **Fix:** recompute every derived field in `replace_children` with
  `Recompute`.

### D9 `reset_state_reexecution`

- **Severity:** Invariant.
- **What:** after `reset_plan_states`, executing the plan again gives the same
  results. Executing the same partition twice either gives the same results
  or returns an error, never different data.
- **Why:** recursive queries re-execute plans.
- **Fix:** clear all execution state, such as shared build sides and dynamic
  filters, in `reset_state`.

### D10 `proto_roundtrip`

- **Severity:** Invariant.
- **What:** if `try_to_proto` returns `Some`, decoding it gives a plan with
  the same display, the same properties and the same results.
- **Fix:** serialize every field that affects properties or execution.

## E. Results do not depend on configuration or input layout

Each check runs the plan under several settings and requires the same
results (as a multiset, and in order where an ordering is claimed).

### E1 `batch_size_invariance`

- **Severity:** Invariant.
- **What:** results are the same with `batch_size` set to 1, 2, 7 and 8192.

### E2 `batch_boundary_invariance`

- **Severity:** Invariant.
- **What:** results are the same when the same input rows are split into
  batches differently, including empty batches.

### E3 `partitioning_invariance`

- **Severity:** Invariant.
- **What:** results are the same across `target_partitions` values and across
  splits of input rows among partitions that still meet the node's
  distribution requirements.

### E4 `encoding_invariance`

- **Severity:** Invariant.
- **What:** results are the same with dictionary encoded and plain arrays,
  and with `StringView` and `Utf8` strings.

### E5 `memory_limit_invariance`

- **Severity:** Invariant.
- **What:** with a small memory limit, the plan either gives the same results
  as without a limit or fails with `ResourcesExhausted`. It never panics and
  never reserves more than the limit, which a tracking memory pool verifies.

**Why (all of E):** batch boundaries, partition layouts, encodings and
spilling are where operators such as sort merge join, window functions and
sort preserving merge most often break.

**Fix (all of E):** handle state that spans batches or partitions, and test
the spill path.

## F. Lifecycle and robustness

### F1 `resources_released`

- **Severity:** Invariant.
- **What:** after a complete run, the memory pool has no reservations left.
  Dropping the output stream partway through drops the input streams within
  a timeout. Spill files are removed.
- **Fix:** tie spawned tasks and reservations to the stream, for example with
  `SpawnedTask`.

### F2 `errors_propagate`

- **Severity:** Invariant.
- **What:** if an input returns an error at batch `k`, the output stream
  returns an error. It does not end early without one, hang, or panic.

### F3 `empty_input`

- **Severity:** Invariant. Lint for emitting empty batches.
- **What:** inputs with zero partitions, zero batches, or only empty batches
  work.

### F4 `partition_isolation`

- **Severity:** Lint.
- **What:** executing a single partition while the other partitions are never
  polled finishes without deadlock.

### F5 `display_no_panic`

- **Severity:** Invariant.
- **What:** every `DisplayFormatType` and the tree renderer work without
  panicking, and `name()` is not empty.
