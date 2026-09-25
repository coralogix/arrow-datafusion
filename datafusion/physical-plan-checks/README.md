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

# Apache DataFusion ExecutionPlan Checks

[Apache DataFusion] is an extensible query execution framework, written in
Rust, that uses [Apache Arrow] as its in-memory format.

This crate checks that `ExecutionPlan` implementations are consistent with the
`ExecutionPlan` contract. Optimizer rules trust what a plan reports about
itself, such as its cardinality effect, statistics and orderings, so a plan
that reports something false can lead to wrong results or missed
optimizations. The checks run against the plans defined in DataFusion and can
be used to test your own plans.

```rust
use datafusion_physical_plan_checks::PlanChecker;
use datafusion_physical_plan_checks::fixtures::SourceSpec;

// Generate an input with known data, statistics, ordering and partitioning
let input = SourceSpec::new(schema)
    .with_partition_rows(&[100, 0, 250])
    .build_arc()?;
let plan = build_my_plan(input)?;

let report = PlanChecker::new().check(&plan)?;
report.assert_no_invariant_violations();
```

- [CHECKS.md](CHECKS.md) lists every check, why it matters and how to fix a
  violation.
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) tracks which checks are
  implemented, and the violations currently found in DataFusion's own plans.
- [DESIGN.md](DESIGN.md) describes how the crate is structured and where it is
  heading.

[apache arrow]: https://arrow.apache.org/
[apache datafusion]: https://datafusion.apache.org/
