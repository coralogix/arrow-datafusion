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

This describes how the `physical_plan` module is put together. See
[CHECKS.md](CHECKS.md) for what is checked.

## Goal

Testing an `ExecutionPlan` should need as little test code as possible. A user
describes how to build their plan from its inputs (a `PlanFactory`), and the
crate generates the inputs, meets the plan's input requirements, runs every
check on several cases, and reports the findings of each case (the
`PlanHarness`). The built-in audit in `tests/physical_plan/builtin_plans/` is
written this way. The layers below the harness can also be used on their own,
for a plan built by hand.

## Layers

**Checks** (`PlanCheck`, `checks/`). A check is a name and a function that
looks at one node and returns findings about that node only. Checks call the
methods that describe the node and its children, such as its statistics,
cardinality effect, fetch and equivalence properties, and never execute
anything.

**Inputs** (`fixtures/`). `SourceSpec` is a plain, cloneable description of an
input: schema, partition layout, value distribution, ordering,
constant and copied columns, statistics precision, row ids and seed. A
`SourceSpec` is built into a `MockSourceExec`, a leaf node that emits data with
all the properties specified by the `SourceSpec`.

**Oracles** (`oracle/`). Reference computations of the true properties of a set
of batches: exact statistics, sortedness, hash partition placement, whether an
expression is constant, and whether two expressions are equal.

**Checker** (`PlanChecker`). Visits every node, runs every check on it, and
attributes findings to nodes by path.

**Harness** (`harness/`). Builds a plan from a `PlanFactory` for every case
derived from a list of `Profile`s, and runs a `PlanChecker` on each.

## Principles

- **Inputs are the truth.** If a check needs to trust something about an
  input, the input must have verified it.
- **Report at the source.** A problem is reported on the node that causes it,
  not on every ancestor that inherits it. Checks skip nodes whose children
  already failed in the same way.
- **Determinism.** Data is seeded, so every case, and therefore every report,
  is the same on every run.
- **Severity reflects consequences.** An invariant violation means DataFusion
  can produce wrong results or errors by trusting the plan. A lint means a
  missed optimization.
- **One factory per configuration.** Following the goal of as little test
  code as possible, a simple plan needs one factory, and a plan with modes,
  flags or options one factory per configuration, such as `RepartitionExec`
  round robin, hash, and each preserving order.

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

**Factories.** A `PlanFactory` is a name, one base `SourceSpec` per input, and a
closure that builds the plan from input plans. A plan without inputs, such as
`EmptyExec`, has no base specs and one case. The base spec carries what only
the author of the plan knows: the schema, the value distribution (few distinct
values on join keys so that inputs match), the number of rows, and optionally an
ordering the plan should see, for example to exercise `InputOrderMode::Sorted`.
The harness sets everything else for each case: partition layout, hash
partitioning, statistics precision, seed and row ids. The number of rows stays
with the base spec, rather than with the harness, because a sensible size
depends on the plan: a cross join produces the product of its input sizes.
`create` is called several times per case, so it must be a pure function of its
inputs; it must bind expressions to the schemas of the inputs it receives, by
name, since the harness appends a `__row_id` column; and it must use the input
plans as given, since the harness recognizes them by identity. A factory for a
leaf, such as a `DataSourceExec`, can still take an input and build the leaf
from its rows (the `MockSourceExec` it receives), declaring what the input
declares, so that the leaf is checked with every layout of the profiles rather
than in a single case; the audit's sources and sinks do this. `allow(check, reason)` skips a check in every case and shows the reason in the report. Write
one factory per configuration of the node (see "One factory per configuration"
in the principles).

**Profiles and cases.** A `Profile` describes a layout of inputs, such as
partition weights, statistics precision and seed. A `PlanHarness` holds a list
of profiles and tests a factory once per profile. The factory's base specs say
what its inputs contain, and a profile says how any inputs are laid out, so the
same profiles apply to every factory. For each profile, the harness applies the
profile to every base spec, meets the plan's input requirements, and builds the
plan on the resulting inputs: that is a case. The input requirements come from
the plan: the harness builds the plan, reads the `required_input_ordering` and
`required_input_distribution` of each node directly above an input, changes that
input's spec to meet them, and builds the plan again until every requirement is
met. The checker then runs on the plan of every case, and the `FactoryReport`
holds one report per case. A factory without inputs has one case, for the first
profile, since every profile gives the same plan. `PlanHarness::new()` uses
`Profile::defaults()`, and `with_profiles` replaces them. A case can be rebuilt
by hand from `Case::inputs` and `PlanFactory::create`.

**Reports.** `PlanHarness::check` returns a `FactoryReport` with the plan
and the checker's report of every case. It displays every case, with the
plan and the violations of each case that has any. A violation that occurs
in several cases is listed in each, so the report is long when a plan has
problems, but each case can be read on its own.
