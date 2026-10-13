// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Reference implementations that compute the true properties of a set of
//! batches: their statistics, whether they are sorted, which hash partition
//! each row belongs to, the distinct values of an expression and whether it is
//! constant, and whether two expressions are equal.
//!
//! [`MockSourceExec`] uses these to validate what it reports, so they are
//! written to be simple rather than fast, and do not reuse the code paths of
//! the operators under test where that can be avoided.
//!
//! [`MockSourceExec`]: crate::physical_plan::fixtures::MockSourceExec

mod ordering;
mod partitioning;
mod statistics;
mod values;

pub use ordering::{first_unsorted_row, sort_rows};
pub use partitioning::{hash_partition, rows_outside_hash_partition};
pub use statistics::exact_statistics;
pub use values::{constant_violations, distinct_values, first_unequal_row};
