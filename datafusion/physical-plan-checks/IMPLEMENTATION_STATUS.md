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

# Implementation Status

This file tracks which checks from [CHECKS.md](CHECKS.md) are implemented.
Update it in the same change that adds or extends a check.

Status values:

- **Done**: implemented as described in the catalog, with tests.
- **Partial**: implemented, but part of the catalog entry is missing. The
  notes say what is missing.
- **Pending**: not implemented.

## A. Static checks

| Code | Name                                 | Status  | Notes                                                                                    |
| ---- | ------------------------------------ | ------- | ---------------------------------------------------------------------------------------- |
| A1   | `equal_cardinality_num_rows`         | Partial | Compares overall statistics only, not per partition.                                     |
| A2   | `fetch_not_equal_cardinality`        | Done    |                                                                                          |
| A3   | `fetch_bounds_num_rows`              | Done    |                                                                                          |
| A4   | `cardinality_effect_bounds_num_rows` | Partial | Compares overall statistics only, not per partition.                                     |
| A5   | `limit_pushdown_missed`              | Pending | Needs execution and the prefix-closed test.                                              |
| A6   | `maintains_input_order_missed`       | Pending | Needs execution and `__row_id` tracking.                                                 |
| A7   | `per_child_lengths`                  | Done    |                                                                                          |
| A7   | `check_invariants`                   | Partial | Runs `InvariantLevel::Always` only. `Executable` needs plans whose requirements are met. |
| A8   | `statistics_shape`                   | Done    |                                                                                          |
| A8   | `partition_statistics_sum`           | Done    |                                                                                          |
| A8   | `statistics_invalid_partition`       | Pending |                                                                                          |
| A9   | `statistics_ignore_inputs`           | Done    |                                                                                          |
| A10  | `schema_consistency`                 | Pending |                                                                                          |
| A10  | `expression_column_refs`             | Pending |                                                                                          |
| A11  | `dynamic_expressions_visited`        | Pending |                                                                                          |
| A11  | `dynamic_expressions_reset`          | Pending |                                                                                          |

## B. Reported properties hold at runtime

| Code | Name                           | Status  | Notes                                                                                                                                                                                                                                                                                                                                                                  |
| ---- | ------------------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B0   | `execution_succeeds`           | Done    |                                                                                                                                                                                                                                                                                                                                                                        |
| B1   | `batch_schema`                 | Done    |                                                                                                                                                                                                                                                                                                                                                                        |
| B2   | `exact_statistics_hold`        | Partial | Checks the statistics of the input the test provides. Running each plan with both exact and inexact input statistics needs the harness to vary inputs (see Harness). `sum_value` is not checked.                                                                                                                                                                       |
| B3   | `orderings_hold`               | Done    |                                                                                                                                                                                                                                                                                                                                                                        |
| B4   | `constants_hold`               | Pending |                                                                                                                                                                                                                                                                                                                                                                        |
| B5   | `equivalence_classes_hold`     | Pending |                                                                                                                                                                                                                                                                                                                                                                        |
| B6   | `constraints_hold`             | Pending |                                                                                                                                                                                                                                                                                                                                                                        |
| B7   | `hash_partitioning_holds`      | Pending | `oracle::rows_outside_hash_partition` is the building block.                                                                                                                                                                                                                                                                                                           |
| B7   | `invalid_partition_errors`     | Pending |                                                                                                                                                                                                                                                                                                                                                                        |
| B8   | `cardinality_effect_holds`     | Done    |                                                                                                                                                                                                                                                                                                                                                                        |
| B9   | `metrics_output_rows`          | Pending |                                                                                                                                                                                                                                                                                                                                                                        |
| B10  | `boundedness_holds`            | Done    | Checks `Bounded` claims on unbounded input directly, rather than the "only with a fetch" heuristic of the earlier catalog entry. For a node with a fetch, runs in which the unbounded leaves reached their row limit are inconclusive and skipped (a large fetch or offset needs more rows). The lint only covers nodes whose fetch is the only thing that drops rows. |
| B11  | `emission_type_holds`          | Done    | Uses an unbounded input rather than one that stalls, so that nodes with large buffers have enough input to emit. A node that needs more than 65536 rows per input partition before emitting is a false positive. `Both` is not checked.                                                                                                                                |
| B12  | `lazy_evaluation_holds`        | Done    | Checks that input polls happen within output polls. The earlier "bounded number of input polls per output poll" is dropped from the catalog, since lazy nodes such as `SortExec` legitimately consume all input in one poll.                                                                                                                                           |
| B13  | `cooperative_scheduling_holds` | Pending |                                                                                                                                                                                                                                                                                                                                                                        |

## C. Order and input requirements

| Code | Name                             | Status  | Notes |
| ---- | -------------------------------- | ------- | ----- |
| C1   | `maintains_input_order_holds`    | Pending |       |
| C2   | `required_input_ordering_honest` | Pending |       |
| C3   | `input_distribution_honest`      | Pending |       |

## D. Rewrite hooks are equivalent

| Code | Name                          | Status  | Notes |
| ---- | ----------------------------- | ------- | ----- |
| D1   | `with_fetch_equivalent`       | Pending |       |
| D2   | `limit_pushdown_equivalent`   | Pending |       |
| D3   | `projection_swap_equivalent`  | Pending |       |
| D4   | `sort_pushdown_equivalent`    | Pending |       |
| D5   | `filter_pushdown_equivalent`  | Pending |       |
| D6   | `repartitioned_equivalent`    | Pending |       |
| D7   | `preserve_order_holds`        | Pending |       |
| D8   | `replace_children_consistent` | Pending |       |
| D9   | `reset_state_reexecution`     | Pending |       |
| D10  | `proto_roundtrip`             | Pending |       |

## E. Results do not depend on configuration or input layout

| Code | Name                        | Status  | Notes |
| ---- | --------------------------- | ------- | ----- |
| E1   | `batch_size_invariance`     | Pending |       |
| E2   | `batch_boundary_invariance` | Pending |       |
| E3   | `partitioning_invariance`   | Pending |       |
| E4   | `encoding_invariance`       | Pending |       |
| E5   | `memory_limit_invariance`   | Pending |       |

## F. Lifecycle and robustness

| Code | Name                  | Status  | Notes                                                                                                                                 |
| ---- | --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| F1   | `resources_released`  | Partial | Checks memory reservations after a complete run and input streams after the output is dropped. Spill file cleanup is not checked yet. |
| F2   | `errors_propagate`    | Done    | Only requires one output partition to return the error.                                                                               |
| F3   | `empty_input`         | Pending |                                                                                                                                       |
| F4   | `partition_isolation` | Pending |                                                                                                                                       |
| F5   | `display_no_panic`    | Pending |                                                                                                                                       |

## Harness

| Component                    | Status  | Notes                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PlanCheck`, `PlanChecker`   | Done    | Visits every node. Checks can be allowed by name. Sync `check` and async `check_async`.                                                                                                                                                                                                                                                                                                                                |
| `CheckContext`               | Done    | Holds the output of executing each node, each on a fresh copy made with `reset_plan_states`, the memory each execution left reserved, and the results of stream experiments. Execution and each experiment only happen when an enabled check needs them.                                                                                                                                                               |
| `Report`, `Violation`        | Done    |                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `MockSourceExec`             | Done    | Serves fixed batches. Statistics are computed from the data. Declared orderings and hash partitioning are verified against the data. Range partitioning is not supported.                                                                                                                                                                                                                                              |
| `SourceSpec` data generation | Partial | Seeded random data with partition row counts, hash partitioning, batch layouts with empty batches, null fraction, value domain size, ordering, statistics precision and a row id column. Missing: dictionary, nested, binary, interval and duration types; NaN and extreme values; constants and equivalences in the source properties.                                                                                |
| `oracle`                     | Partial | Exact statistics, sortedness and hash partition placement.                                                                                                                                                                                                                                                                                                                                                             |
| Controllable mock streams    | Done    | `StreamBehavior` on `SourceSpec` and `MockSourceExec`: pending forever after batch `k`, an error after batch `k`, and unbounded input that repeats its data (or its last row, when ordered) with an optional row budget. `StreamProbe` counts polls, batches, rows and errors per partition, and records when streams are created, first polled, finish and are dropped, and which polls are not driven by a consumer. |
| Stream experiments           | Done    | `Experiment::{Laziness, UnboundedInput, InputError, Cancellation}`. Rebuild each node with `replace_children_if_necessary` on a `reset_plan_states` copy, with `MockSourceExec` leaves swapped for variants and each child wrapped in a pass-through probe. Every run has a timeout (`with_stream_timeout`). Timeouts rely on the node yielding: a node that blocks its thread cannot be interrupted.                  |
| Input variation              | Pending | Deriving many `SourceSpec`s from one (seeds, partition counts, batch layouts, statistics precision) and running the checks for each.                                                                                                                                                                                                                                                                                   |
| Plan factories               | Pending | Build the plan under test from generated children, probe it (for example `with_fetch`, `required_input_ordering`, `input_distribution_requirements`) and generate the cases to test. See `DESIGN.md`.                                                                                                                                                                                                                  |

## Built-in plans covered

`tests/builtin_plans.rs` runs every implemented check against these plans,
on generated data with exact statistics, and records the findings in
`tests/snapshots/`:

- `EmptyExec`, `PlaceholderRowExec`
- `ProjectionExec`
- `FilterExec`, with and without fetch
- `CoalesceBatchesExec`, with and without fetch
- `CoalescePartitionsExec`, with and without fetch
- `SortExec`, with and without fetch, with and without
  `preserve_partitioning`, on a nullable column, and on input that is already
  sorted, with and without fetch
- `SortPreservingMergeExec`, with and without fetch
- `RepartitionExec`, round robin and hash
- `GlobalLimitExec`, with and without fetch, `LocalLimitExec`
- `UnionExec`
- `BufferExec`
- `CooperativeExec`

Not covered yet: aggregates, joins, windows, `DataSourceExec`, `UnnestExec`,
`InterleaveExec`, `AnalyzeExec`, `ScalarSubqueryExec`, recursive queries and
sinks.

## Known findings in built-in plans

Violations currently recorded in the snapshot. Remove an entry when the plan
is fixed and the snapshot is updated.

| Plan                                                                                     | Checks                                                    | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CoalesceBatchesExec` with fetch                                                         | `fetch_not_equal_cardinality`, `cardinality_effect_holds` | `cardinality_effect()` is `Equal` even when a fetch is set. At runtime it produces 30 rows from 600.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `CoalescePartitionsExec` with fetch                                                      | `fetch_not_equal_cardinality`, `cardinality_effect_holds` | `cardinality_effect()` is `Equal` even when a fetch is set. At runtime it produces 10 rows from 600.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `CoalesceBatchesExec` with fetch                                                         | `partition_statistics_sum`, `exact_statistics_hold`       | Reports overall `num_rows` `Exact(10)`, but produces 30 rows from 3 partitions. The per-partition fetch is applied to the merged input statistics as if it were a global limit (`with_fetch(fetch, 0, 1)`).                                                                                                                                                                                                                                                                                                                                                                         |
| `LocalLimitExec`                                                                         | `partition_statistics_sum`, `exact_statistics_hold`       | Same cause as above: reports `Exact(10)` and produces 30 rows.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `SortExec` with `preserve_partitioning` and fetch                                        | `partition_statistics_sum`, `exact_statistics_hold`       | Same cause as above for the overall count. In addition, partitions share one TopK threshold, so a partition only keeps rows that can be in the global top `fetch`. Partitions report `Exact(fetch)` but can produce fewer rows, and how many depends on how partitions interleave at runtime (see `topk/mod.rs`). The results are correct under a `SortPreservingMergeExec` with the same fetch, but the statistics are false, and the semantics of the fetch differ from other per-partition fetches. Needs a decision: report `Inexact` per partition, or document the semantics. |
| `PlaceholderRowExec`                                                                     | `batch_schema`                                            | With a non-empty schema, produces a batch of `Null` columns named `placeholder_N` instead of the declared fields.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `FilterExec` with fetch                                                                  | `fetch_bounds_num_rows`                                   | Lint: the row count estimate does not apply the fetch.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `GlobalLimitExec` without fetch                                                          | `boundedness_holds`                                       | `compute_properties` always reports `Bounded` ("limit operations are always bounded", `limit.rs:96-104`). With `fetch=None` (an `OFFSET` without `LIMIT`) the node passes through every row after the skip, so on unbounded input it never ends, and a `Final` operator above it is accepted by `SanityCheckPlan`.                                                                                                                                                                                                                                                                  |
| `CoalesceBatchesExec`, `CoalescePartitionsExec` and `SortPreservingMergeExec` with fetch | `boundedness_holds`                                       | Lint: boundedness is copied from the input and ignores the fetch (`coalesce_batches.rs:118`, `coalesce_partitions.rs:106`, `sorts/sort_preserving_merge.rs:183`), although the nodes otherwise pass every row through and end after `fetch` rows, the same guarantee `GlobalLimitExec` reports as `Bounded`. `LimitPushdown` replaces a `GlobalLimitExec` above `CoalescePartitionsExec` or `SortPreservingMergeExec` with a fetch on that node (`limit_pushdown.rs:235-252`), so the rewrite can turn a bounded plan into an unbounded one.                                        |
| `SortExec` with fetch on sorted input                                                    | `boundedness_holds`                                       | Lint: `SortExec::with_fetch` reports `Bounded` when the input is already sorted (`sorts/sort.rs:1122-1133`), but `replace_children` with `Recompute` recomputes the properties with `compute_properties`, which does not (`sorts/sort.rs:1402-1409`). The node reports `Bounded` or `Unbounded` depending on how it was built. `replace_children_consistent` (D8) would report the same cause.                                                                                                                                                                                      |
| `RepartitionExec`                                                                        | `resources_released`                                      | Lint: the handles of the input tasks are kept in the plan's shared state (`ConsumingInputStreamsState::abort_helper`, `repartition/mod.rs:368`, set at `:632`) as well as in the output streams, so dropping every output stream does not abort the input tasks. A task stops when it next receives a batch and finds no receivers (`repartition/mod.rs:2293-2321`), or when the plan is dropped, so a stalled input stays open until then.                                                                                                                                         |

The audit runs on a current-thread Tokio runtime so the snapshot is
deterministic despite the timing-dependent TopK output above. Messages from
stream experiments avoid counts that depend on timing, such as the number of
rows an unbounded input delivered before a timeout.
