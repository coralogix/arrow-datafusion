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

# Design

This describes how the crate is put together and where it is heading. See
[CHECKS.md](CHECKS.md) for what is checked and
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for progress.

## Goal

Testing an `ExecutionPlan` should need as little test code as possible. The
end state is that a user describes how to build their plan from its inputs,
and the crate generates the inputs, decides which cases are worth testing by
probing the plan, runs every check on every case, and reports violations in a
way that can be reproduced. Today, tests still build each plan and its inputs
by hand (see `tests/builtin_plans.rs`), but each layer below is designed so
that it can be driven by that future harness without changes.

## Layers

**Checks** (`PlanCheck`, `checks/`). A check looks at one node and returns
findings about that node only. Checks are synchronous and never execute
anything themselves; everything they need beyond the node's own methods comes
from the `CheckContext`. This keeps checks small and pure, and lets the layer
above decide how and how often to execute a plan (for example once per
configuration for the metamorphic checks in section E).

**Context** (`CheckContext`). Facts about a plan that are gathered once,
before any check runs. Each node runs on a fresh copy of its subtree made with
`reset_plan_states`, so executions do not affect each other through runtime
state such as dynamic filters, and the plan under test is never mutated. The
context holds:

- The output of executing each node to completion, and the memory the
  execution left reserved (tracked by a pool that wraps the configured one).
- Stream experiments (`Experiment`, `StreamRun`), for checks that need to see
  how a node drives its inputs rather than what it outputs. An experiment
  rebuilds the node with `replace_children_if_necessary`, after replacing the
  `MockSourceExec` leaves below some children by copies with a different
  `StreamBehavior` and wrapping each child in a pass-through probe. The probes
  report the child's own properties, so the rebuilt node keeps the original
  node's properties unless the leaves change theirs (unbounded leaves do).
  Observing the direct inputs, rather than the leaves, attributes what is
  observed to the node itself, so most of these checks need no "skip if a
  child fails the same way" rule. Each run has a timeout, and experiments on
  inputs that never end stop as soon as they have seen what the checks need.

A check declares what it needs (`requires_execution`, `experiments`), and the
checker only gathers the facts that an enabled check needs. Later additions
belong here too: results of probing rewrite hooks, and outputs under other
configurations.

**Inputs** (`fixtures/`). `SourceSpec` is a plain, cloneable description of
an input: schema, partition layout, batch layout, value distribution,
ordering, statistics precision, stream behavior, row ids and seed.
`MockSourceExec` is the leaf it builds. The source is the ground truth for
every check that compares a node with its input, so it never reports anything
it has not verified: statistics are computed from the data, and declared
orderings and hash partitioning are checked against it when the source is
built. The stream behavior (`StreamBehavior`) controls what the streams do
after serving their batches: end, stall, fail, or never end. An unbounded
source reports `Unbounded` and unknown statistics, and repeats its data in a
way that keeps its declared ordering and partitioning true. A `StreamProbe`
attached to a source, or wrapped around any stream, records what happens to
the streams of each partition.

**Oracles** (`oracle/`). Reference computations of the true properties of a
set of batches: exact statistics, sortedness, hash partition placement. The
sources use them to validate themselves and the checks use them to judge
plans. They are written for clarity rather than speed, and avoid the code
paths of the operators under test where possible. The one exception is hash
partitioning, which must use `BatchPartitioner` because the property being
checked is agreement with the hash that DataFusion operators assume.

**Checker** (`PlanChecker`). Selects checks, gathers the context if any check
needs execution, visits every node, and attributes findings to nodes by path.

## Principles

- **Inputs are the truth.** If a check needs to trust something about an
  input, the input must have verified it.
- **Report at the source.** A problem is reported on the node that causes it,
  not on every ancestor that inherits it. Checks skip nodes whose children
  already failed in the same way.
- **Determinism.** Data is seeded. Tests that snapshot findings run on a
  current-thread runtime, because some operators produce timing-dependent
  output.
- **Severity reflects consequences.** An invariant violation means DataFusion
  can produce wrong results or errors by trusting the plan. A lint means a
  missed optimization.

## Toward plan factories

The harness this is heading toward looks roughly like this. The names are
placeholders and the API is deliberately not fixed yet.

```rust
trait PlanFactory {
    /// Schemas of the inputs the plan is built from
    fn input_schemas(&self) -> Vec<SchemaRef>;
    /// Build the plan under test on top of `inputs`
    fn create(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>>;
}
```

For each factory, the harness would:

1. **Build a probe plan** on default `SourceSpec`s for the input schemas.
2. **Probe it** to decide which cases to generate:
   - `required_input_ordering` gives orderings to pass to
     `SourceSpec::with_ordering`.
   - `input_distribution_requirements` gives hash expressions for
     `SourceSpec::with_hash_partitioning`, or a single partition.
   - `with_fetch(Some(n))` returning a plan adds fetch cases for D1.
   - `supports_limit_pushdown`, `try_swapping_with_projection`,
     `try_pushdown_sort` and `repartitioned` returning something add the
     matching section D cases.
3. **Vary the inputs**: seeds, partition counts including zero and one, batch
   layouts with empty batches, statistics precision, and row id columns with
   disjoint ranges per input. Stream behavior is part of the spec so that a
   factory can also build its plan directly on an unbounded input. That
   exercises constructors that depend on the boundedness of their inputs,
   which the checker's experiments, which only rebuild an existing node with
   `replace_children`, do not.
4. **Rebuild the plan** for each case with `create` and run the `PlanChecker`.
5. **Report** each violation with the case that produced it, which is a
   `SourceSpec` per input plus the probe choices, so it can be reproduced.

The pieces already fit this: `SourceSpec` is a value that can be cloned and
varied; the row id column is appended after all other columns, so
expressions that a plan binds against its input schema (orderings,
distribution requirements) stay valid; and checks work on any node without
knowing how it was built.

Open questions to settle when the harness is built:

- How a factory states which inputs are valid, such as value ranges or
  schemas that depend on each other (join keys with matching types).
- How users mark findings as expected for their plan, per case rather than
  per check.
- How to name cases so snapshots stay stable when case generation changes.
