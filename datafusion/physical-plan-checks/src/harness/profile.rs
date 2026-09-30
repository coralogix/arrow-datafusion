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

//! [`Profile`]: how the harness lays out the inputs of one case.

use crate::fixtures::{BatchLayout, StatisticsPrecision};

/// Batches of the default profiles: random batches of up to 16 rows,
/// including empty batches
const DEFAULT_BATCH_LAYOUT: BatchLayout = BatchLayout::Random {
    max_rows: 16,
    empty_batches: true,
};

/// How the [`PlanHarness`] lays out the inputs of a factory for one case.
///
/// A profile sets everything about an input that the plan's author does not
/// need to know: how many partitions it has and how its rows are spread over
/// them, how the rows are split into batches, the precision of its
/// statistics and the seed. The harness then adapts each input to the
/// requirements of the plan: an input that must be in one partition gets
/// one, an input that must be hash partitioned is hash partitioned into as
/// many partitions as the profile has, and an input that must be sorted is
/// sorted.
///
/// The size of each input is the number of rows of its base [`SourceSpec`]
/// times [`Self::with_row_multiplier`]. The size stays with the base spec
/// because only the author of the plan knows what size makes sense: a cross
/// join produces the product of its input sizes.
///
/// Profiles also decide which checks run: checks that need stream
/// experiments (such as `boundedness_holds` and `resources_released`) depend
/// on how a plan drives its streams rather than on the shape of its data, and
/// most of them wait for timeouts, so by default they only run in the
/// [`Self::default_profile`].
///
/// [`PlanHarness`]: crate::harness::PlanHarness
/// [`SourceSpec`]: crate::fixtures::SourceSpec
#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    name: String,
    partition_weights: Vec<usize>,
    row_multiplier: usize,
    batch_layout: BatchLayout,
    precision: StatisticsPrecision,
    seed: u64,
    stream_experiments: bool,
}

impl Profile {
    /// A profile named `name` with the settings of
    /// [`Self::default_profile`], except that stream experiments are off
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            partition_weights: vec![1, 0, 2],
            row_multiplier: 1,
            batch_layout: DEFAULT_BATCH_LAYOUT,
            precision: StatisticsPrecision::Exact,
            seed: 0,
            stream_experiments: false,
        }
    }

    /// `default`: three partitions, the second of them empty and the third
    /// with twice the rows of the first; random batches of up to 16 rows,
    /// including empty batches; exact statistics; seed 0; every check.
    pub fn default_profile() -> Self {
        Self::new("default").with_stream_experiments(true)
    }

    /// The profiles [`PlanHarness::new`] uses, in this order:
    ///
    /// - [`Self::default_profile`]
    /// - `single partition`: one partition
    /// - `inexact statistics`: the default layout, with inexact statistics
    /// - `absent statistics`: the default layout, without statistics
    /// - `empty input`: the default layout, with no rows
    ///
    /// All but the first skip the checks that need stream experiments.
    ///
    /// [`PlanHarness::new`]: crate::harness::PlanHarness::new
    pub fn defaults() -> Vec<Self> {
        vec![
            Self::default_profile(),
            Self::new("single partition").with_partition_weights(&[1]),
            Self::new("inexact statistics")
                .with_statistics_precision(StatisticsPrecision::Inexact),
            Self::new("absent statistics")
                .with_statistics_precision(StatisticsPrecision::Absent),
            Self::new("empty input").with_row_multiplier(0),
        ]
    }

    /// [`Self::defaults`], followed by:
    ///
    /// - `seed 1` and `seed 2`: the default layout with other data
    /// - `2 partitions` and `5 partitions`: other partition counts, the
    ///   latter with an empty partition
    /// - `large input`: the default layout with 8 times the rows, so that
    ///   each partition spans many batches, and operators with internal
    ///   buffers fill them several times
    ///
    /// Used by the built-in audit with the `extended_tests` feature.
    pub fn extended() -> Vec<Self> {
        let mut profiles = Self::defaults();
        profiles.extend([
            Self::new("seed 1").with_seed(1),
            Self::new("seed 2").with_seed(2),
            Self::new("2 partitions").with_partition_weights(&[1, 1]),
            Self::new("5 partitions").with_partition_weights(&[2, 1, 0, 3, 1]),
            Self::new("large input").with_row_multiplier(8),
        ]);
        profiles
    }

    /// Spread the rows of each input over one partition per entry of
    /// `weights`, in proportion to the weights. A weight of 0 gives an empty
    /// partition. For an input that must be hash partitioned, only the number
    /// of partitions is used.
    pub fn with_partition_weights(mut self, weights: &[usize]) -> Self {
        self.partition_weights = weights.to_vec();
        self
    }

    /// Give each input `multiplier` times the rows of its base spec. 0 gives
    /// inputs without rows.
    pub fn with_row_multiplier(mut self, multiplier: usize) -> Self {
        self.row_multiplier = multiplier;
        self
    }

    /// Set how the rows of each partition are split into batches
    pub fn with_batch_layout(mut self, batch_layout: BatchLayout) -> Self {
        self.batch_layout = batch_layout;
        self
    }

    /// Set the precision of the statistics the inputs report
    pub fn with_statistics_precision(mut self, precision: StatisticsPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set the seed. Input `i` is generated with seed `seed * 1000 + i`, so
    /// that inputs with the same schema get different data.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run the checks that need stream experiments in cases of this profile
    pub fn with_stream_experiments(mut self, stream_experiments: bool) -> Self {
        self.stream_experiments = stream_experiments;
        self
    }

    /// The name, used to refer to cases of this profile in reports
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The number of partitions of an input without a distribution
    /// requirement, or that must be hash partitioned
    pub fn partition_count(&self) -> usize {
        self.partition_weights.len()
    }

    /// The rows of each partition for an input of `num_rows` rows: in
    /// proportion to the weights, with the rows that do not divide evenly in
    /// the last partition with a non-zero weight
    pub fn partition_rows(&self, num_rows: usize) -> Vec<usize> {
        let total: usize = self.partition_weights.iter().sum();
        if total == 0 {
            return vec![0; self.partition_weights.len()];
        }
        let mut rows: Vec<usize> = self
            .partition_weights
            .iter()
            .map(|weight| num_rows * weight / total)
            .collect();
        let remainder = num_rows - rows.iter().sum::<usize>();
        if let Some(last) = self.partition_weights.iter().rposition(|w| *w > 0) {
            rows[last] += remainder;
        }
        rows
    }

    /// The factor applied to the rows of each base spec
    pub fn row_multiplier(&self) -> usize {
        self.row_multiplier
    }

    /// How the rows of each partition are split into batches
    pub fn batch_layout(&self) -> BatchLayout {
        self.batch_layout
    }

    /// The precision of the statistics the inputs report
    pub fn statistics_precision(&self) -> StatisticsPrecision {
        self.precision
    }

    /// The seed
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether the checks that need stream experiments run
    pub fn stream_experiments(&self) -> bool {
        self.stream_experiments
    }
}
