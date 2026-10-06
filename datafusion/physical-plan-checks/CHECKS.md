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

Checks come in four kinds:

- **Static** checks compare what a plan reports about itself, without running
  it (section A). A5 and A6, which look for properties a plan could report
  but does not, need its output too.
- **Execution** checks run the plan and compare its output with what it
  reports (sections B and C).
- **Differential** checks compare a plan with a rewritten version of itself,
  such as the plan returned by `with_fetch`, and require both to produce
  equivalent results (section D).
- **Metamorphic** checks run the same plan under different configurations or
  input layouts and require the same results (section E).

Section F covers the lifecycle of execution: cleanup, errors and edge cases.

In code, each check has a `CheckKind` that says what the checker gathers for
it before checks run: nothing (`Static`, section A apart from A5 and A6), the
output of each node and the memory its execution left reserved (`Execution`,
A6, B0 to B8 and `memory_released` in F1), variant runs (`Variant`, A5,
sections D and E) or stream experiments (`Stream`, B10 to B12,
`streams_released` in F1, and F2).

Checks run on one node at a time, but the checker visits every node of a
plan. Tests should build the plan under test on inputs generated with
`fixtures::SourceSpec`. It produces a `fixtures::MockSourceExec` whose
statistics are computed from its data and whose declared ordering and
partitioning are verified against it, so checks treat what it reports as the
truth. The usual way is to describe the plan with a `harness::PlanFactory`
and check it with a `harness::PlanHarness`, which generates the inputs and
runs the checks on several cases (see [Cases](#cases)).

When any enabled check needs execution, the checker executes every node of
the plan on its own before running checks. Each execution uses a fresh copy of
the node's subtree made with `reset_plan_states`, so state left behind by one
execution, such as a dynamic filter, does not affect another.

### Stream experiments

Some checks need to see how a node drives its input streams, which its output
alone does not show. When any of them is enabled, the checker runs every
**stream experiment** (`Experiment`) on every node with children before
running checks:

- The node is rebuilt from a `reset_plan_states` copy of its subtree. The
  `MockSourceExec` leaves below some or all children are replaced by copies
  with the same data and a different `StreamBehavior`: they stall after their
  first batch, return an error after their first batch, or never end. Leaves
  that are not `MockSourceExec` are left as they are.
- Every child is wrapped in a pass-through probe that reports the child's own
  properties, so the rebuilt node computes the same properties as the
  original (unless the leaves report different properties, as unbounded
  leaves do). The probe records what the node does with the child's streams:
  when they are created, polled, finish and are dropped, and which polls
  happen outside a poll of the node's own output. Because the probes sit
  directly below the node, what they record is caused by the node, not by its
  descendants.
- Experiments use a batch size of 8, and every run has a timeout.
  Runs on finite inputs use the execution timeout (`PlanChecker::with_timeout`,
  30 seconds by default). Runs on inputs that never end, and waits for streams
  and memory to be released, use the stream timeout
  (`PlanChecker::with_stream_timeout`, 2 seconds by default). A violation that
  shows up as a node never ending is reported after that long. A wait for
  streams or memory to be released also ends as soon as no task is alive on
  the Tokio runtime, since nothing is left that could release them; cleanup
  on threads outside the runtime, such as `spawn_blocking` tasks, is not
  waited for then.

Build plans under test on finite inputs. The checker derives the stalling,
failing and unbounded variants itself. The harness runs the checks that need
stream experiments only in the `default` case (see [Cases](#cases)).

### Variant runs

The checks in sections D and E, and A5, compare a node's output with the
output of a **variant run** (`Variant`): a rewritten copy of the node, or the
node run under other settings. When any of them is enabled, the checker runs
every variant that applies to each node, after it has executed every node
normally, so that their sizes can depend on the normal outputs:

- Each run executes a fresh `reset_plan_states` copy of the node's subtree to
  completion, like the normal execution, within the execution timeout.
- `WithoutFetch` runs the plan returned by `with_fetch(None)`. For a node with
  a fetch, its output is the **unfetched output** that checks use to judge
  what the fetch kept. For a node without a fetch, the unfetched output is the
  normal output.
- `WithFetch(n)` runs the plan returned by `with_fetch(Some(n))`, and
  `LimitedInputs(n)` runs a node with children with every child limited to
  its first `n` rows per partition, for `n` of 1, 7 and one more than the
  rows of the node's output (with and without its fetch) and of each child's
  output. A limit of 0 is not tried, since `LIMIT 0` is replaced by an empty
  relation during logical optimization and never reaches a physical plan.
- `BatchSize(n)` runs the node with the session batch size set to 1, 2, 7 and 8192.
- `BatchLayout(layout)` rebuilds the node with the rows of each partition of
  its `MockSourceExec` leaves split into one row per batch, into random
  batches of up to 3 rows with empty batches, and into one batch per
  partition. The leaves keep the same rows in the same order and partitions,
  and the same `PlanProperties`, so the rebuilt node keeps its properties.

Rows are compared with the functions in `oracle`: as multisets, in order, and
as a prefix of a sorted sequence in which rows that tie on the sort key may
appear in any order and may be exchanged for each other. Values are compared
exactly. The floating point values `SourceSpec` generates are small
multiples of 0.5, so sums of them do not depend on the order in which an
operator adds them, which batch sizes and batch boundaries can change.

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
  copy, named with the suffix `__copy`, which the input declares equal to it.

Every case splits the rows of each partition into random batches of up to 16
rows, including empty batches.

Inputs are then adapted to the plan's requirements: an input that must be
sorted is sorted, one that must be hash partitioned is hash partitioned into
the profile's number of partitions (the same for every input, so
co-partitioned inputs match), and one that must be a single partition gets
one. Every input has a `__row_id` column, with ids in a separate range per
input.

Checks that need stream experiments (`boundedness_holds`,
`emission_type_holds`, `lazy_evaluation_holds`, `streams_released` and
`errors_propagate`) only run in the `default` case: they depend on how the
plan drives its streams rather than on the shape of its data, and several
of them wait for the stream timeout. Every other check runs in every case.
The report lists every case, with the plan and the findings of each case
that has any.

### Terms

- A **single-child** node has exactly one entry in `children()`.
- An **exact** statistic is `Precision::Exact`. An **estimate** is
  `Precision::Inexact`.
- A fetch on a node with several output partitions limits **each
  partition**. Only nodes that combine partitions, such as
  `CoalescePartitionsExec` and `SortPreservingMergeExec`, or nodes with one
  output partition, apply a fetch to their whole output. This matches how the
  `LimitPushdown` optimizer rule uses `with_fetch`. A fetch does not have to
  keep `n` rows in every partition that has them: the partitions of a TopK
  `SortExec` share a threshold and keep only rows that can be among the first
  `n` of the whole output. Checks require at most `n` rows per partition and
  at least `min(n, rows)` overall, which is what `LimitPushdown` relies on.

## A. Static checks

These checks call the methods that describe the plan. A5 and A6 look for
properties that a plan could report but does not, so they need its output
too. The other checks never call `execute`.

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
- **Requires execution** (the `LimitedInputs` variant, and the normal
  output).
- **What:** a node with children for which `supports_limit_pushdown()` is
  false, but that is **prefix-closed**: running the node with every child
  limited to its first `n` rows per partition gives the first `n` rows of
  each partition of its normal output, for every `n` that the
  `LimitedInputs` variant tries. Each output partition must have exactly
  `min(n, rows)` rows, which are the first rows of the same partition of the
  normal output, in the same order apart from rows that tie on the ordering
  the node reports. Only reported if the limits removed output rows in at
  least one run. Not checked:
  - A node whose cardinality effect without its fetch (that of the plan
    returned by `with_fetch(None)`) is not `Equal` or `GreaterEqual`. A node
    that can drop or merge rows, or a join, which combines rows of its inputs,
    needs later rows in general, and only looks prefix-closed when the
    generated data does not show it: a final aggregate whose input has each
    group once, or a mark join whose first input rows already have every key.
  - A node over a child that reports an ordering or a constant, and whose
    order the node does not maintain. A node that sorts its input, such as
    `SortExec`, passes input that is already sorted through, and input
    sorted on a constant keeps its order. The equivalence properties of a
    sort on a constant report no ordering, so the constant is what shows it.
- **Why:** a prefix-closed node can let a limit move below it, so its inputs
  produce less data. `LimitPushdown` stops at a node that does not support
  limit pushdown: it gives the node the limit as a fetch, or adds a limit
  above it (`limit_pushdown.rs:268-309`). Being `Equal` and order-preserving
  is not enough on its own: `WindowAggExec` is both, but a window over a whole
  partition needs rows after the first `n`, so it correctly does not support
  limit pushdown. This check only reports nodes whose outputs have been shown
  to be prefix-closed.
- **Fix:** return true from `supports_limit_pushdown`. Allow this check if the
  node is only prefix-closed for the inputs the test generated.
- **Found in:** `BufferExec` (its `supports_limit_pushdown` returns the
  child's value) and `CoalesceBatchesExec`. `RepartitionExec`, also a
  candidate when this entry was written, is not reported: its output
  partitions have rows of several input partitions, in an order that depends
  on timing.

### A6 `maintains_input_order_missed`

- **Severity:** Lint.
- **Requires execution** (the normal output of the node and of its
  children).
- **What:** `maintains_input_order()[i]` is false, but the rows of child `i`
  keep their relative order in the output. Rows are tracked by their row ids
  (see C1): the child must have exactly one `__row_id` column, with unique
  ids, and exactly one `__row_id` column of the output must have ids of the
  child. The order is kept when every row of every output partition is a row
  of the child, each output partition has rows of one partition of the child
  only, and they appear in the order of that partition. A row can repeat, as
  long as its copies are next to each other. Reported when some output
  partition has at least two rows of the child, and, for a child with several
  partitions, the output has rows of at least two of them, which shows that
  the node keeps them apart. Not checked:
  - A node that reports an output ordering. It orders its output itself, for
    example by sorting it, which keeps rows that tie on its sort key in their
    input order.
  - A child that reports an ordering or a constant. Sorted input is not in a
    random order, and can be in the order that the node sorts it into, as
    can input sorted on a constant, for which a sort reports no ordering.
- **Why:** `maintains_input_order` lets the optimizer avoid re-sorting. A
  false value that should be true costs an unnecessary sort.
- **Fix:** return true for that child and make sure the output equivalence
  properties keep the child's ordering.
- **Found in:** `CoalescePartitionsExec` with a single input partition, the
  build side of `HashJoinExec` `LeftSemi`, `LeftAnti` and `LeftMark` and of
  `NestedLoopJoinExec` `LeftAnti`, which emit the rows of the build side in
  its order once the probe side ends, and the probe side of
  `NestedLoopJoinExec` `RightSemi` and `RightMark` and of
  `PiecewiseMergeJoinExec` `RightSemi`.

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

## B. Reported properties hold at runtime

These checks run the plan on generated input and compare the output with
what the plan reports.

### B0 `execution_succeeds`

- **Severity:** Invariant.
- **What:** executing every output partition of the node on valid input
  finishes without an error, a panic, or exceeding the checker's timeout (30
  seconds by default). A failure caused by a failing child is only reported
  on the child.
- **Why:** every other execution check needs output to compare against. A
  node that fails on input meeting its requirements is broken for real
  queries too.
- **Fix:** depends on the error. If the node fails because its input does not
  meet a requirement the node does not declare, declare it (see C2 and C3).

### B1 `batch_schema`

- **Severity:** Invariant for the column count, field names, data types, and
  nulls in non-nullable fields. Lint for differences in nullability flags and
  metadata.
- **What:** every batch has the same fields as `schema()`: the same number,
  names and data types, and no nulls in fields that `schema()` declares
  non-nullable. Nulls are counted logically, so every value of a `NullArray`
  counts as null. The nullability flags and metadata of the batch should also
  match.
- **Why:** parent operators build their own schema from `schema()` and look
  up columns by position. A mismatched batch causes errors or wrong results
  downstream. A non-nullable field that contains nulls breaks operators that
  skip null handling for non-nullable input.
- **Fix:** make the produced batches match `schema()`, or correct `schema()`.

### B2 `exact_statistics_hold`

- **Severity:** Invariant.
- **What:** every exact statistic is true for the actual output, overall and
  for each partition: `num_rows`, and for each column `min_value`,
  `max_value`, `null_count`, `distinct_count` and `sum_value`. An exact
  minimum, maximum or sum is not checked when the output has no non-null
  values in that column, since there is nothing to contradict it. A sum is
  compared by value, whatever its type: integer sums may be kept in the
  column's type or in the wider type of SQL `SUM`, and adding decimals
  increases their precision. Only the sums of integer and decimal columns
  are checked, since the exact sum of floating point values depends on the
  order in which they are added, and sums that overflow the type of SQL
  `SUM` are not checked. Byte sizes and `total_byte_size` are not checked.
  The harness runs it with input statistics marked exact, marked inexact and
  absent, to catch nodes that turn estimates into exact values.
- **Attribution:** a node computes its statistics from its children's, so a
  child with a false exact statistic can make the node's false too, even
  when the node passes statistics through unchanged, as `RepartitionExec`
  does. The child is reported for those. The statistics of a node with such
  a child are computed again with `statistics_from_inputs`, from the child
  statistics that `child_stats_requests` asks for with every false exact
  statistic replaced by the true value of the child's output, and the node
  is reported for the exact statistics that are still false: with true
  inputs, they are the node's own.
- **Reporting:** one finding for the overall statistics and one for each
  partition that has a false exact statistic, listing every false statistic
  with its claimed and actual values, by column in schema order. A finding
  for statistics computed from corrected child statistics says so.
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
- **What:** each constant expression (every expression of an equivalence
  class with a constant value) has a single value within each partition. If
  the constant is marked uniform across partitions, it has the same value in
  every partition, and that value is the given one if there is one. Null is
  a value, so a column of nulls is constant. A given value is compared by
  value, after casting it to the type of the expression. Partitions without
  rows have no value to check. A constant that refers to a column by a wrong
  index or name is not checked, since `expression_column_refs` reports it.
- **Reporting:** one finding for each partition with several values, giving
  two of them, and one for each partition whose value differs from the given
  value, or for a uniform constant without a given value, from the value of
  the first partition with rows.
- **Why:** constants let the optimizer drop sort keys and grouping columns.
- **Fix:** only mark an expression constant when the node guarantees it.

### B5 `equivalence_classes_hold`

- **Severity:** Invariant.
- **What:** expressions in the same equivalence class are equal on every
  output row. Two nulls are equal, as in `IS NOT DISTINCT FROM`, since a
  sort or a hash puts them together. Each expression of a class is compared
  with the first, after casting its values to the type of the first if the
  types differ. Literals in a class are not compared: they make the class
  constant, which B4 checks. An expression that refers to a column by a
  wrong index or name is not compared, since `expression_column_refs`
  reports it.
- **Reporting:** one finding for each expression and partition with a row on
  which the expression differs from the first of its class, giving the first
  such row and both values.
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

- **Severity:** Invariant. Lint when a fetch bounds the output but the node
  reports `Unbounded`.
- **Requires execution** (the `UnboundedInput` experiment).
- **What:** the node is rebuilt with the leaves below one child unbounded, for
  each child in turn, with the other children unchanged. The unbounded leaves
  repeat their data without end.
  - A rebuilt node that reports `Boundedness::Bounded` must end its output
    within the stream timeout, while the unbounded child reports `Unbounded`
    and keeps delivering rows. A child that reports `Bounded` is responsible
    for its own claim, and a child that delivers no rows gives the node no
    chance to end, so neither case is reported on the node. To bound memory,
    each partition of an unbounded leaf stalls after 65536 rows. For a node
    with a fetch, a run in which a leaf reached that limit is inconclusive and
    not reported, since the node may need more rows to end, for example a
    limit with a large fetch or offset. Within a finite number of rows, such
    a node behaves the same as one with no limit at all. A node without a
    fetch has no reason to need a particular number of rows before it ends,
    so it is reported either way.
  - Lint: a rebuilt node that reports `Unbounded`, but has a fetch, otherwise
    passes every row through (`with_fetch(None)` returns a plan whose
    cardinality effect is `Equal`), and whose output ended. Such a node ends
    as soon as `fetch` rows reach it, which is the guarantee `GlobalLimitExec`
    reports as `Bounded`. A node that can drop rows, such as `FilterExec` with
    a fetch, is not reported: its output only ends if enough rows survive, so
    `Unbounded` is the correct report.
- **Why:** `SanityCheckPlan` rejects operators that need all of an unbounded
  input, and `EnforceSorting` and `EnforceDistribution` pick streaming
  operators, based on boundedness. A false `Bounded` lets a plan through that
  never finishes. A missed `Bounded` rejects plans that would finish, and
  `LimitPushdown`, which replaces a `GlobalLimitExec` with a fetch on the node
  below it, can turn a bounded plan into an unbounded one.
- **Fix:** derive boundedness from the inputs, for example with
  `boundedness_from_children`, and report `Bounded` when a fetch bounds the
  output of a node that does not drop rows.
- **Not checked:** that a node finishes within a timeout on bounded input,
  which `execution_succeeds` (B0) covers. An earlier version of this entry
  said a node may only report `Bounded` on unbounded input if it has a fetch;
  that is a heuristic, since a node can also be bounded because it ignores an
  input, so the check tests the claim directly instead.

### B11 `emission_type_holds`

- **Severity:** Invariant.
- **Requires execution** (the `UnboundedInput` experiment, and the normal
  output).
- **What:** the node is rebuilt with the leaves below one child unbounded, for
  each child in turn, with the other children unchanged. A rebuilt node that
  reports `EmissionType::Incremental` must produce at least one row within the
  stream timeout, while that child keeps delivering rows. Skipped when:
  - the child reports `Final` or delivered no rows, since the node then has
    nothing to emit;
  - the node produced no rows from the normal input, for example a filter that
    removes every generated row;
  - the rebuilt node reports `Final` or `Both`.
- **Why:** streaming queries on unbounded input depend on incremental
  operators actually emitting output, and optimizer rules such as
  `SanityCheckPlan` and `JoinSelection` only accept or choose operators on
  unbounded input when they are not `Final`.
- **Fix:** report `Final` if the node waits for all input.
- **Buffering:** an incremental node may buffer up to a batch before it emits
  one. Experiments use a batch size of at most 8, and each unbounded partition
  produces up to 65536 rows before it stalls, so a node that buffers up to the
  session batch size, or up to its own target such as the `target_batch_size`
  of `CoalesceBatchesExec`, has enough input to emit. A source that serves its
  data once and then stalls is not enough: a node can legitimately buffer more
  rows than the generated input has. A node that legitimately needs more than
  65536 rows per input partition before it emits anything is reported; allow
  this check for such a node. Runs where the unbounded leaves reached that
  limit are not skipped, since a node that buffers all of its input always
  reaches it.
- **Ordering:** an unbounded source that declares an ordering repeats the last
  row of each partition instead of its whole data, so the ordering still
  holds and operators that require it, such as `SortPreservingMergeExec`, see
  valid input. Partitions without rows end at once, so an operator that
  combines partitions is not blocked by a partition that would never produce
  a row.
- **Several children:** each child is made unbounded on its own, because a
  node can be incremental in one input only. A hash join emits incrementally
  on its probe side once its build side has ended, and reports `Final` when
  the build side is unbounded.
- **Not checked:** `Both`, since it allows all of the output to come at the
  end for some inputs (for example a left anti join).

### B12 `lazy_evaluation_holds`

- **Severity:** Invariant.
- **Requires execution** (the `Laziness` experiment).
- **What:** a node that reports `EvaluationType::Lazy` only polls its input
  streams from within a poll of its own output streams, on the same thread.
  Polls from `execute`, from spawned tasks, or from any other task are
  reported. The experiment polls every output partition to the end on the
  normal inputs.
- **Why:** `EnsureCooperative` treats an `Eager` node as the start of a new
  task that polls its input independently of the consumer, and wraps the
  leaves below it with `CooperativeExec` unless a cooperative ancestor already
  covers them. A node that spawns tasks but reports `Lazy` can leave those
  tasks polling non-cooperative inputs that never yield.
- **Fix:** report `EvaluationType::Eager` for nodes that spawn tasks or buffer
  ahead of demand.
- **Not checked:**
  - That each output poll causes a bounded number of input polls, as an
    earlier version of this entry said. A lazy node may legitimately consume
    all of its input in one poll, as `SortExec` does, so there is no bound to
    check.
  - A node that reports `Eager` but is lazy. The only cost is an extra
    `CooperativeExec`, and some nodes report `Eager` because an input is
    eager.

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

The harness gives every input a `__row_id` column whose ids increase within
the input and use a separate range for each input. Many nodes pass it
through, which lets the checks track where each output row came from.

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

- **Severity:** Invariant. Lint when the new plan does not report an ordering
  the node reports.
- **Requires execution** (the `WithoutFetch` and `WithFetch` variants).
- **What:** for a node whose `with_fetch` returns a plan, and fetches `n` of
  1, 7 and one more than the rows of the node's output:
  - The plan returned by `with_fetch(Some(n))` reports `fetch() == Some(n)`,
    the node's schema and the node's number of output partitions. Lint: it
    does not report an ordering that the node reports.
  - Each output partition has at most `n` rows, and the whole output has at
    least `min(n, rows of the unfetched output)` rows. A partition does not
    have to have `n` rows (see the terms above).
  - Every row appears in the unfetched output, as a multiset.
  - Where the node reports an ordering, the output keeps the first rows of
    the unfetched output, apart from rows that tie on the ordering. With one
    output partition, the output is a prefix of the unfetched output in this
    sense. With several, each partition is sorted, and the partitions
    together contain the first `min(n, rows)` rows of the whole unfetched
    output in this sense, which is what a `SortPreservingMergeExec` with the
    same fetch above them needs. Each partition does not have to be a prefix
    of its own partition: the partitions of a TopK `SortExec` share a
    threshold and reject rows that tie with it (`topk/mod.rs:606-615`), so a
    partition can keep rows it saw before another partition set the threshold
    instead of its own first rows.
  - The plan returned by `with_fetch(None)` reports `fetch() == None`, the
    node's schema and number of partitions. For a node without a fetch, it
    produces the same rows as the node, as a multiset.
  - For a node with a fetch, the node's own output satisfies the same as the
    output of `with_fetch(Some(fetch))`.
- **Unfetched output:** the output of `with_fetch(None)`. For a node with a
  fetch whose `with_fetch(None)` returns `None`, which rows the unfetched
  output has is unknown, so only the row counts are checked, with the
  node's normal output as the lower bound for the number of rows available.
- **Why:** `LimitPushdown` replaces limit nodes with fetches on the nodes below
  them and trusts the result. `PushdownSort` moves the fetch of a sort it
  removes to the input with `with_fetch`, as a per-partition fetch in place
  of a `LocalLimitExec` (`pushdown_sort.rs:110-117`).
- **Fix:** apply the fetch after all other work the node does, per output
  partition.

### D2 `limit_pushdown_equivalent`

- **Severity:** Invariant.
- **Requires execution** (the `WithoutFetch` and `LimitedInputs` variants).
- **What:** for a node with children for which `supports_limit_pushdown()` is
  true, the node is rebuilt with every child limited to its first `n` rows
  per partition, a `GlobalLimitExec` above a child with one partition and a
  `LocalLimitExec` above any other child, for `n` of 1, 7 and one more
  than the rows of the node's output and of each child's output. Then:

  - Each output partition has at most `n` rows.
  - The whole output has at least `min(n, rows of the normal output)` rows.
  - Every row appears in the normal output, or the unfetched output for a node
    with a fetch, as a multiset.
  - Where the node reports an ordering, the output keeps the first rows of
    the normal output in sort order, as in D1.

  For `CoalescePartitionsExec` and `SortPreservingMergeExec`, only the first
  `n` rows of each output partition are checked. `GlobalLimitExec` and
  `LocalLimitExec` are not checked.

- **Why:** this is what `LimitPushdown` (`limit_pushdown.rs`) assumes:

  - Above a node that supports limit pushdown, it removes the limit and
    limits each child instead, with `with_fetch` on the child where
    available, and otherwise with a `LocalLimitExec`, or a `GlobalLimitExec`
    for a child with one partition (`add_limit`). The limit applies to every
    child at once, so the node must produce a valid limit of its output from
    limited inputs on all of them. Since the limit above the node is gone, the
    node must also not produce more than `n` rows per partition: a node with
    one output partition over several input partitions would produce up to
    `n` rows from each of them.
  - `CoalescePartitionsExec` and `SortPreservingMergeExec`
    (`combines_input_partitions`) keep the limit: the rule gives them a fetch
    with `with_fetch`, or keeps a limit above them, so only their first `n`
    rows count. The rule does not limit their children, except for children
    that support `with_fetch`, so the check limits the children as for any
    other node.
  - The rule merges `GlobalLimitExec` and `LocalLimitExec` into the limit it
    pushes down (`extract_limit`) before it asks whether a node supports limit
    pushdown, so it never uses their answer. They report true, but a
    `GlobalLimitExec` with a skip would give wrong rows from limited inputs.

  `sort_pushdown` and `limit_pushdown_past_window` also push a fetch through
  nodes that support limit pushdown, when their cardinality effect is
  `Equal`.

- **Not checked:** `LimitPushdown` also pushes an offset: when the limit has a
  skip, it places `GlobalLimitExec(skip, fetch)` on the children
  (`add_limit`), so the node must produce rows `skip..skip + fetch` of its
  output from the same rows of each child.
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

Each check runs the node under several settings and requires the same
results: the same multiset of rows overall, and every ordering the node
reports holding in each partition. Rows may move between partitions and
change order where no ordering is reported. An ordering that does not hold
in the normal output is reported by `orderings_hold` (B3) instead. Executing
the node must not fail under any of the settings.

Some differences are legitimate:

- **Fetch:** which rows a node with a fetch keeps can depend on timing and on
  batch boundaries, such as how the partitions below a
  `CoalescePartitionsExec` interleave. For such a node, the output under each
  setting must be a valid result of the fetch, as in D1: at most `fetch` rows
  per partition, at least `min(fetch, rows)` overall (which fixes the number
  of rows of a node with one output partition), rows from the unfetched
  output, and the first rows in sort order where the node reports an
  ordering. Without the unfetched output, only the row counts are compared.
- **Attribution:** batch sizes and leaf layouts apply to the whole subtree. A
  node is only compared under a setting in which every child produced exactly
  the same rows, in the same order and partitions, as it does normally.
  Otherwise the node's input changed too: either the child is broken, which
  is reported on the child, or the child legitimately produced its rows in
  another order or partition, for example a sort that orders tied rows
  differently or a repartition that interleaves its inputs differently. A
  node whose output depends on the order of its input rows can then
  legitimately differ as well.

**Not checked:** a node above a child whose output changed under a setting is
not compared under that setting. A node whose output is only defined up to a
later step can legitimately differ, for example a partial aggregate that
starts to pass rows through after a number of input rows, or an aggregate
with a soft `DISTINCT` limit, which stops after the input batch in which it
has seen enough groups and relies on the `LIMIT` above it; allow these checks
for such a node.

### E1 `batch_size_invariance`

- **Severity:** Invariant.
- **Requires execution** (the `BatchSize` and `WithoutFetch` variants).
- **What:** results are the same with `batch_size` set to 1, 2, 7 and 8192.

### E2 `batch_boundary_invariance`

- **Severity:** Invariant.
- **Requires execution** (the `BatchLayout` and `WithoutFetch` variants).
- **What:** results are the same when the rows of each partition of the
  `MockSourceExec` leaves are split into one row per batch, into random
  batches of up to 3 rows with empty batches, and into one batch per
  partition. Leaves that are not `MockSourceExec` are left as they are.

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

### F1 `memory_released` and `streams_released`

- **Severity:** Invariant. Lint when input streams are only released when the
  plan is dropped.
- **Requires execution** (`memory_released`: the normal execution;
  `streams_released`: the `Cancellation` experiment).
- **What:**
  - `memory_released`: after executing the node to completion and dropping
    its streams and its copy of the plan, the memory reserved during the run
    is released within the stream timeout; the wait ends early once no task
    is alive on the runtime (see [Stream experiments](#stream-experiments)).
    The checker executes each node with a fresh `UnboundedMemoryPool` and
    reads how many bytes are still reserved in it. A node whose child also
    leaves memory reserved is not reported.
  - `streams_released`: the leaves stall after their first batch. The output
    is polled until each partition produced a batch or ended, or for 20
    milliseconds, and then dropped. Every input stream the node created must
    be dropped within the stream timeout; the wait ends early once no task is
    alive on the runtime. If they are only dropped once the rebuilt node is
    dropped as well, that is a lint: the plan keeps the input streams, or the
    tasks that poll them, alive, for example in shared execution state. If
    they are still alive after that, it is an invariant violation.
  - Planned: spill files are removed.
- **Why:** a query that is cancelled, or whose consumer stops early (for
  example a limit above the node), must stop reading its inputs and return
  its memory. Holding them until the plan is dropped keeps sources open and
  tasks running for as long as the plan is cached.
- **Fix:** tie spawned tasks and reservations to the stream, for example with
  `SpawnedTask`, and do not keep handles to them in the plan.

### F2 `errors_propagate`

- **Severity:** Invariant.
- **Requires execution** (the `InputError` experiment).
- **What:** every leaf returns an error after its first batch. If an error
  reached the node from one of its inputs, at least one output partition
  returns an error. The output ending without one, not ending within the
  execution timeout, or panicking are reported. Skipped when:
  - no error reached the node, for example because a child swallowed it
    (reported on the child) or the node stopped reading its input early;
  - the node has a fetch and an output partition produced `fetch` rows. An
    eager node can read the error into a buffer while its output is still
    serving earlier batches, and then correctly end once the fetch is met.
- **Why:** an operator that drops an error returns partial results as if they
  were complete.
- **Fix:** return errors from the input as they arrive, including errors
  received by spawned tasks.
- **Not checked:** that every output partition that depends on the failing
  input returns the error; only one partition has to.

### F3 `empty_input`

- **Severity:** Invariant. Lint for emitting empty batches.
- **What:** inputs with zero partitions, zero batches, or only empty batches
  work.
- **Today:** the harness's `empty input` case runs every check but those
  that need stream experiments on inputs whose partitions have no rows, and
  every case has empty batches, and all but `single partition` an empty
  partition. Inputs with zero partitions are not generated yet.

### F4 `partition_isolation`

- **Severity:** Lint.
- **What:** executing a single partition while the other partitions are never
  polled finishes without deadlock.

### F5 `display_no_panic`

- **Severity:** Invariant.
- **What:** every `DisplayFormatType` and the tree renderer work without
  panicking, and `name()` is not empty.
