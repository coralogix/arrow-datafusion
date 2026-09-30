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

Testing an `ExecutionPlan` should need as little test code as possible. A
user describes how to build their plan from its inputs (a `PlanFactory`), and
the crate generates the inputs, decides which cases are worth testing by
probing the plan, runs every check on every case, and reports violations in a
way that can be reproduced (the `PlanHarness`, see [Harness](#harness)). The
built-in audit in `tests/builtin_plans.rs` is written this way. The layers
below the harness can also be used on their own, for a plan built by hand.

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
- Variant runs (`Variant`, `VariantRun`), for checks that compare a node's
  output with the output of a rewritten copy of the node (the plan returned
  by `with_fetch`, or the node with its children limited the way
  `LimitPushdown` limits them) or of the node run under other settings (the
  session batch size, or `MockSourceExec` leaves split into other batches).
  Each run executes a `reset_plan_states` copy of the node's subtree to
  completion within the execution timeout, and records the whole output.
  Variants run after every node has been executed normally, so that their
  parameters, such as a fetch larger than the output, can depend on the
  normal outputs.

Variant runs are a separate concept from experiments because they answer a
different question with different machinery. An experiment observes how a
node drives its streams, often on inputs that never end, through probes, stop
conditions and observations that are only meaningful for a partial run. A
variant run needs none of that: it is a normal execution of a different plan
or with a different setting, whose output is compared row by row with the
normal output. Keeping them apart keeps `StreamRun` free of outputs and
`VariantRun` free of stream observations. They share the code that rebuilds a
node over modified `MockSourceExec` leaves (`map_mock_leaves`).

Settings such as the batch size and the leaf layout apply to the whole
subtree, so a child's output can change under a variant too. The checks that
compare variants only compare a node under a variant in which every child
produced the same rows in the same order and partitions as normally, which
attributes a difference to the node that causes it.

A check declares what it needs (`requires_execution`, `experiments`,
`variants`), and the checker only gathers the facts that an enabled check
needs. Later additions belong here too: results of probing the other rewrite
hooks of section D, and outputs under the other configurations of section E,
most of which fit as new kinds of variants.

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
set of batches: exact statistics, sortedness, hash partition placement, and
how the rows of two sets of batches compare (as multisets, in order, and as a
prefix of a sorted sequence apart from rows that tie on the sort key). Row
comparisons encode values with `arrow::row::RowConverter`, except floating
point values, which are compared with a relative tolerance because the order
in which an operator adds them can change with batching. The sources use the
oracles to validate themselves and the checks use them to judge plans. They
are written for clarity rather than speed, and avoid the code paths of the
operators under test where possible. The one exception is hash partitioning,
which must use `BatchPartitioner` because the property being checked is
agreement with the hash that DataFusion operators assume.

**Checker** (`PlanChecker`). Selects checks, gathers the context if any check
needs execution, visits every node, and attributes findings to nodes by path.

**Harness** (`harness/`). Builds a plan from a `PlanFactory` for every case
derived from a list of `Profile`s, runs a `PlanChecker` on each, and groups
the findings of all cases in a `FactoryReport`. See [Harness](#harness).

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

## Harness

```rust
let factory = PlanFactory::new("FilterExec", vec![SourceSpec::new(schema)], |inputs| {
    let input = Arc::clone(&inputs[0]);
    let predicate = col("b", &input.schema())?;
    Ok(Arc::new(FilterExec::try_new(predicate, input)?) as Arc<dyn ExecutionPlan>)
});
let report = PlanHarness::new().check(&factory)?;
report.assert_no_invariant_violations();
```

**Factories.** A `PlanFactory` is a name, one base `SourceSpec` per input,
and a closure that builds the plan from input plans. A plan without inputs,
such as `EmptyExec`, has no base specs. A closure is enough for every plan
in the audit, so there is no trait to implement. The base spec carries what
only the author of the plan knows: the schema, the value distribution (few
distinct values on join keys so that inputs match), the number of rows, and
optionally an ordering the plan should see, for example to exercise
`InputOrderMode::Sorted`. The harness sets everything else for each case:
partition layout, hash partitioning, batch layout, statistics precision,
seed and row ids. The number of rows stays with the base spec, rather than
with the harness, because a sensible size depends on the plan: a cross join
produces the product of its input sizes. `create` is called several times
per case, so it must be a pure function of its inputs; it must bind
expressions to the schemas of the inputs it receives, by name, since the
harness appends a row id column; and it must use the input plans as given,
since the harness recognizes them by identity. Factory configuration is
small: `allow(check, reason)` skips a check in every case and shows the
reason in the report, `with_profiles` replaces the harness's profiles for
one factory, and `with_shared_row_id_name` gives every input the same row id
column name for plans that need inputs with the same schema.

**Profiles and cases.** A `Profile` is a named, deterministic way to lay out
inputs: partition weights, a row multiplier, a batch layout, a statistics
precision, a seed, and whether the checks that need stream experiments run.
`Profile::defaults()` is a short curated list rather than a cross product:
`default` (three partitions with weights 1, 0 and 2, random batches of up to
16 rows with empty batches, exact statistics, every check), `single partition`, `inexact statistics`, `absent statistics` and `empty input`.
`Profile::extended()` adds two more seeds, 2 and 5 partitions and 8 times the
rows; the audit uses it with the `extended_tests` feature, as the rest of the
workspace gates slow tests, with its own snapshots. A case is the fully
specified input specs derived from one profile, after the plan's
requirements are applied. Its description is derived from the specs (for
example `default: input 0 rows [20, 0, 40]; input 1 hash [r_k@0] into 3 partitions, 90 rows; exact statistics; seeds 0, 1`), and it can be
rerun alone with `PlanHarness::run_case`, or rebuilt by hand from
`Case::inputs` and checked with `PlanHarness::checker_for`. A factory without
inputs has one case. A profile that yields the same specs as an earlier one,
for example `single partition` for a plan that requires one partition, adds
no case unless it runs checks the earlier one does not.

**Requirements.** For each profile, the harness lays out the inputs, builds
the plan, and reads what each node directly above an input requires of it:

- a hard `required_input_ordering` becomes `SourceSpec::with_ordering`, with
  default sort options where the requirement has none, replacing a base
  ordering that does not meet it (a base ordering that does is kept);
- `KeyPartitioned` becomes `with_hash_partitioning` into the profile's
  partition count, which is the same for every input, so children that must
  be co-partitioned get the same count;
- `SinglePartition` becomes one partition.

Requirements are bound to the probe input's schema, and the generated inputs
have the same schema, so they stay valid. They can depend on the inputs (an
aggregate chooses its input order mode, a join over sorted inputs can ask
for more), so the harness builds the plan again and repeats until every
requirement is met, at most `MAX_PROBES` (5) times. Soft ordering
requirements are not imposed, since the node works without them. A
requirement that cannot be met (the requirements keep changing, the inputs
already have what is asked for, or the child is a node the factory built
rather than an input) is reported as harness problem `H2` for the case, and
the case is not checked. `create` failing or panicking is `H1`, an input
that cannot be generated is `H3`, and allowing a check that does not exist
is `H4`.

**Which checks run.** Checks that need stream experiments (B10 to B12, F1,
F2) depend on how a plan drives its streams rather than on the shape of its
data, and several of them wait for the stream timeout, so only profiles
that enable them run them: the `default` profile. Every other case runs the
static, execution and variant checks. This is a property of the profile, and
the harness only filters the checker's checks by it.

**Reports.** `PlanHarness::check` returns a `FactoryReport` that groups
findings across cases. Two findings are the same if they have the same
severity, check, node path and node name, and the same message once numbers
are replaced, lists and ranges of numbers are collapsed and a trailing
`(also ...)` summary is removed; when a case has several such findings (one
per partition, say), they pair up in order with those of other cases. The
group keeps the first case's message and lists the cases it occurred in,
or `all cases`. Groups are ordered by node in plan traversal order and then by
first appearance, with harness problems first. `PlanChecker` is unchanged
and still checks a plan built by hand; the harness runs it once per case.

**Runtime.** Cases run one after another on a current-thread runtime, so
the reports are deterministic. The default audit takes about 10 to 13
seconds per snapshot test, 13 seconds for the three together (10 seconds
before the harness); the extended audit takes 20 to 55 seconds per test.
Running factories in parallel was not needed.

Open questions:

- How a factory states which inputs are valid beyond the schema and the
  value distribution, such as value ranges, or schemas that depend on each
  other (join keys with matching types).
- How users mark findings as expected per case rather than per check.
  `allow` is per factory.
- Grouping findings by their normalized message is a heuristic. Findings
  with a structured kind (as `Problems` has inside some checks) would make
  grouping exact.
- Inputs with zero partitions, other stream behaviors as base specs (an
  unbounded input given to `create`, which exercises constructors that
  depend on the boundedness of their inputs), and probing other hooks
  (`with_fetch`, `repartitioned`, `try_pushdown_sort`) to add cases for
  section D are not generated yet.
- D8 `replace_children_consistent` can build the same node on different
  valid inputs by calling `create` on the inputs of two cases; C1
  `maintains_input_order_holds` can follow the row id columns, which have a
  distinct name (`__row_id_<input>`) and range (`input * ROW_ID_RANGE`) per
  input.
