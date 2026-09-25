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

| Code | Name                           | Status  | Notes |
| ---- | ------------------------------ | ------- | ----- |
| B1   | `batch_schema`                 | Pending |       |
| B2   | `exact_statistics_hold`        | Pending |       |
| B3   | `orderings_hold`               | Pending |       |
| B4   | `constants_hold`               | Pending |       |
| B5   | `equivalence_classes_hold`     | Pending |       |
| B6   | `constraints_hold`             | Pending |       |
| B7   | `hash_partitioning_holds`      | Pending |       |
| B7   | `invalid_partition_errors`     | Pending |       |
| B8   | `cardinality_effect_holds`     | Pending |       |
| B9   | `metrics_output_rows`          | Pending |       |
| B10  | `boundedness_holds`            | Pending |       |
| B11  | `emission_type_holds`          | Pending |       |
| B12  | `lazy_evaluation_holds`        | Pending |       |
| B13  | `cooperative_scheduling_holds` | Pending |       |

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

| Code | Name                  | Status  | Notes |
| ---- | --------------------- | ------- | ----- |
| F1   | `resources_released`  | Pending |       |
| F2   | `errors_propagate`    | Pending |       |
| F3   | `empty_input`         | Pending |       |
| F4   | `partition_isolation` | Pending |       |
| F5   | `display_no_panic`    | Pending |       |

## Harness

| Component                  | Status  | Notes                                                                                                    |
| -------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `PlanCheck`, `PlanChecker` | Done    | Visits every node. Checks can be allowed by name.                                                        |
| `Report`, `Violation`      | Done    |                                                                                                          |
| `MockSourceExec`           | Partial | Sets statistics, partition count and ordering. Does not produce data yet.                                |
| Data generation            | Pending | Needed for sections B to F: random batches that meet ordering and distribution requirements, `__row_id`. |
| Controllable mock streams  | Pending | Pending forever, errors at batch `k`, poll counting, drop detection.                                     |
| Plan factories             | Pending | Build a node from given children, needed for D8 and for users to register their own plans.               |

## Built-in plans covered

`tests/builtin_plans.rs` runs every implemented check against these plans
and records the findings in `tests/snapshots/`:

- `EmptyExec`, `PlaceholderRowExec`
- `ProjectionExec`
- `FilterExec`, with and without fetch
- `CoalesceBatchesExec`, with and without fetch
- `CoalescePartitionsExec`, with and without fetch
- `SortExec`, with and without fetch, with and without
  `preserve_partitioning`
- `SortPreservingMergeExec`, with and without fetch
- `RepartitionExec`, round robin and hash
- `GlobalLimitExec`, `LocalLimitExec`
- `UnionExec`
- `BufferExec`
- `CooperativeExec`

Not covered yet: aggregates, joins, windows, `DataSourceExec`, `UnnestExec`,
`InterleaveExec`, `AnalyzeExec`, `ScalarSubqueryExec`, recursive queries and
sinks.

## Known findings in built-in plans

Violations currently recorded in the snapshot. Remove an entry when the plan
is fixed and the snapshot is updated.

| Plan                                              | Check                         | Summary                                                                                                                                                                                                |
| ------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `CoalesceBatchesExec` with fetch                  | `fetch_not_equal_cardinality` | `cardinality_effect()` is `Equal` even when a fetch is set.                                                                                                                                            |
| `CoalescePartitionsExec` with fetch               | `fetch_not_equal_cardinality` | `cardinality_effect()` is `Equal` even when a fetch is set.                                                                                                                                            |
| `CoalesceBatchesExec` with fetch                  | `partition_statistics_sum`    | Reports overall `Exact(10)` for 3 partitions that each report `Exact(10)`. The per-partition fetch is applied to the merged input statistics as if it were a global limit (`with_fetch(fetch, 0, 1)`). |
| `LocalLimitExec`                                  | `partition_statistics_sum`    | Same cause as above.                                                                                                                                                                                   |
| `SortExec` with `preserve_partitioning` and fetch | `partition_statistics_sum`    | Same cause as above.                                                                                                                                                                                   |
| `FilterExec` with fetch                           | `fetch_bounds_num_rows`       | Lint: the row count estimate does not apply the fetch.                                                                                                                                                 |
