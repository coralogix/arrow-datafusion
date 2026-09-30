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
the crate generates the inputs, meets the plan's input requirements, runs
every check on several cases, and reports the findings of each case (the
`PlanHarness`). The built-in audit in `tests/builtin_plans.rs` is written this
way. The layers below the harness can also be used on their own, for a plan
built by hand.

## Layers

**Checks** (`PlanCheck`, `checks/`). A check is a name, a `CheckKind` and a
function that looks at one node and returns findings about that node only.
Checks never execute anything themselves. The kind says what the checker
gathers for the check before it runs, in the `CheckContext`:

- `Static`: nothing. The check only calls the node's own methods.
- `Execution`: the output of executing each node to completion, and the
  memory the execution left reserved.
- `Variant`: also the output of each `Variant` of each node: the plan
  returned by `with_fetch`, the node with its children limited the way
  `LimitPushdown` limits them, or the node run with another session batch
  size or with its `MockSourceExec` leaves split into other batches. Variants
  run after every node has been executed normally, so that their parameters,
  such as a fetch larger than the output, can depend on the normal outputs.
- `Stream`: also the results of each stream `Experiment` on each node, for
  checks that need to see how a node drives its inputs rather than what it
  outputs. An experiment rebuilds the node with
  `replace_children_if_necessary`, after replacing the `MockSourceExec` leaves
  below some children by copies with a different `StreamBehavior` and
  wrapping each child in a pass-through probe. The probes report the child's
  own properties, so the rebuilt node keeps the original node's properties
  unless the leaves change theirs (unbounded leaves do). Observing the direct
  inputs, rather than the leaves, attributes what is observed to the node
  itself. Each run has a timeout, and experiments on inputs that never end
  stop as soon as they have seen what the checks need.

Every run uses a fresh copy of the node's subtree made with
`reset_plan_states`, so runs do not affect each other through runtime state
such as dynamic filters, and the plan under test is never mutated. When any
enabled check needs variants, every variant runs; when any needs
experiments, every experiment runs. Checks pick the runs they need.

Variant runs and experiments are separate because they answer different
questions with different machinery. An experiment observes how a node drives
its streams, often on inputs that never end, through probes, stop conditions
and observations that are only meaningful for a partial run. A variant run
is a normal execution of a different plan or with a different setting, whose
output is compared row by row with the normal output. Both rebuild a node
over modified `MockSourceExec` leaves with `fixtures::map_mock_leaves`.

Settings such as the batch size and the leaf layout apply to the whole
subtree, so a child's output can change under a variant too. The checks that
compare variants only compare a node under a variant in which every child
produced the same rows in the same order and partitions as normally, which
attributes a difference to the node that causes it.

**Inputs** (`fixtures/`). `SourceSpec` is a plain, cloneable description of
an input: schema, partition layout, batch layout, value distribution,
ordering, statistics precision, row ids and seed. `MockSourceExec` is the
leaf it builds. The source is the ground truth for every check that compares
a node with its input, so it never reports anything it has not verified:
statistics are computed from the data, and declared orderings and hash
partitioning are checked against it when the source is built. Its
`StreamBehavior` controls what the streams do after serving their batches:
end, stall, fail, or never end. An unbounded source reports `Unbounded` and
unknown statistics, and repeats its data in a way that keeps its declared
ordering and partitioning true.

**Oracles** (`oracle/`). Reference computations of the true properties of a
set of batches: exact statistics, sortedness, hash partition placement, and
how the rows of two sets of batches compare (as multisets, in order, and as a
prefix of a sorted sequence apart from rows that tie on the sort key). Rows
are compared exactly in the arrow row format. The generated floating point
values are small multiples of 0.5, so sums of them do not depend on the order
an operator adds them in. The sources use the oracles to validate themselves
and the checks use them to judge plans. They are written for clarity rather
than speed, and avoid the code paths of the operators under test where
possible. The one exception is hash partitioning, which must use
`BatchPartitioner` because the property being checked is agreement with the
hash that DataFusion operators assume.

**Checker** (`PlanChecker`). Gathers the context the enabled checks need,
visits every node, and attributes findings to nodes by path. Plans run on a
current-thread Tokio runtime, one run after another, so reports are
deterministic although some operators produce timing-dependent output.

**Harness** (`harness/`). Builds a plan from a `PlanFactory` for every case
derived from a list of `Profile`s, and runs a `PlanChecker` on each.

## Principles

- **Inputs are the truth.** If a check needs to trust something about an
  input, the input must have verified it.
- **Report at the source.** A problem is reported on the node that causes it,
  not on every ancestor that inherits it. Checks skip nodes whose children
  already failed in the same way.
- **Determinism.** Data is seeded, and plans run on a current-thread runtime.
- **Severity reflects consequences.** An invariant violation means DataFusion
  can produce wrong results or errors by trusting the plan. A lint means a
  missed optimization.
- **Plain output.** Every problem is its own finding, reported where it
  occurs, in every case it occurs in. Findings are not grouped or summarized.

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
such as `EmptyExec`, has no base specs and one case. The base spec carries
what only the author of the plan knows: the schema, the value distribution
(few distinct values on join keys so that inputs match), the number of rows,
and optionally an ordering the plan should see, for example to exercise
`InputOrderMode::Sorted`. The harness sets everything else for each case:
partition layout, hash partitioning, batch layout, statistics precision,
seed and row ids. The number of rows stays with the base spec, rather than
with the harness, because a sensible size depends on the plan: a cross join
produces the product of its input sizes. `create` is called several times
per case, so it must be a pure function of its inputs; it must bind
expressions to the schemas of the inputs it receives, by name, since the
harness appends a `__row_id` column; and it must use the input plans as
given, since the harness recognizes them by identity. `allow(check, reason)`
skips a check in every case and shows the reason in the report.

**Profiles and cases.** A `Profile` is a named, deterministic way to lay out
inputs: partition weights, a row multiplier, a statistics precision, a seed,
and whether the stream checks run. `Profile::defaults()` is a short curated
list rather than a cross product: `default` (three partitions with weights 1,
0 and 2, exact statistics, every check), `single partition`,
`inexact statistics`, `absent statistics` and `empty input`.
`Profile::extended()` has two other seeds, 2 and 5 partitions and 8 times the
rows. With the `extended_tests` feature, as the rest of the workspace gates
slow tests, the audit also checks these cases and records them in separate
snapshots. Every case splits rows into random batches of up to 16 rows, with
empty batches. A case is the fully specified input specs derived from one
profile, after the plan's requirements are applied, and the plan built on
them. It can be rebuilt by hand from `Case::inputs` and `PlanFactory::create`.

**Requirements.** For each profile, the harness lays out the inputs, builds
the plan, and reads what each node directly above an input requires of it:

- a hard `required_input_ordering` becomes `SourceSpec::with_ordering`, with
  default sort options where the requirement has none, replacing a base
  ordering that does not meet it (a base ordering that does is kept);
- `KeyPartitioned` becomes `with_hash_partitioning` into the profile's
  partition count, which is the same for every input, so children that must
  be co-partitioned are;
- `SinglePartition` becomes one partition.

Requirements are bound to the probe input's schema, and the generated inputs
have the same schema, so they stay valid. They can depend on the inputs (an
aggregate chooses its input order mode, a join over sorted inputs can ask
for more), so the harness builds the plan again and repeats until every
requirement is met, at most 5 times. Soft ordering requirements are not
imposed, since the node works without them. A case that cannot be derived
(`create` fails, an input cannot be generated, the requirements keep
changing, or a requirement is on a child the factory built rather than on an
input) is an error, since the plan cannot be tested as the factory
describes it.

**Which checks run.** Stream checks depend on how a plan drives its streams
rather than on the shape of its data, and several of them wait for the
stream timeout, so only profiles that enable them run them: the `default`
profile. Every other case runs the static, execution and variant checks.

**Reports.** `PlanHarness::check` returns a `FactoryReport` with the plan
and the checker's report of every case. It displays every case, with the
plan and the violations of each case that has any. A violation that occurs
in several cases is listed in each, so the report is long when a plan has
problems, but each case can be read on its own.

**Runtime.** Cases run one after another on a current-thread runtime. The
default audit takes about 13 seconds for the three snapshot tests together,
and with the `extended_tests` feature, which adds three tests on the
extended profiles, about 39 seconds.

Open questions:

- How a factory states which inputs are valid beyond the schema and the
  value distribution, such as value ranges, or schemas that depend on each
  other (join keys with matching types).
- How users mark findings as expected per case rather than per check.
  `allow` is per factory.
- Inputs with zero partitions, other stream behaviors as base specs (an
  unbounded input given to `create`, which exercises constructors that
  depend on the boundedness of their inputs), and probing other hooks
  (`with_fetch`, `repartitioned`, `try_pushdown_sort`) to add cases for
  section D are not generated yet.
- D8 `replace_children_consistent` can build the same node on different
  valid inputs by calling `create` on the inputs of two cases; C1
  `maintains_input_order_holds` can follow the `__row_id` columns, whose ids
  come from a separate range (`input * ROW_ID_RANGE`) per input.
