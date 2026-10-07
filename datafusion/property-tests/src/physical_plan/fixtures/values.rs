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

//! Random column values drawn from a small domain, so that generated data has
//! duplicates, ties and join matches.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray,
    StringArray, StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, TimeUnit};
use datafusion_common::{Result, not_impl_err};
use rand::Rng;
use rand::rngs::StdRng;

/// Options that control generated values
#[derive(Debug, Clone, Copy)]
pub(crate) struct ValueOptions {
    /// Probability that a value of a nullable field is null
    pub null_fraction: f64,
    /// Number of distinct non-null values to draw from
    pub distinct_values: usize,
}

/// Generate `rows` random values for `field`. Values of non-nullable fields
/// are never null.
pub(crate) fn random_array(
    field: &Field,
    rows: usize,
    options: ValueOptions,
    rng: &mut StdRng,
) -> Result<ArrayRef> {
    let null_fraction = if field.is_nullable() {
        options.null_fraction
    } else {
        0.0
    };
    // Index of the value to use for each row, or None for null
    let keys: Vec<Option<u64>> = (0..rows)
        .map(|_| {
            if null_fraction > 0.0 && rng.random_bool(null_fraction) {
                None
            } else {
                Some(rng.random_range(0..options.distinct_values.max(1) as u64))
            }
        })
        .collect();
    array_of_keys(field, &keys)
}

/// `rows` copies of the value with index `key` for `field`
pub(crate) fn constant_array(field: &Field, key: u64, rows: usize) -> Result<ArrayRef> {
    array_of_keys(field, &vec![Some(key); rows])
}

/// The value with each index of `keys` for `field`, or null for `None`.
/// Different indexes give different values, except for types with fewer
/// values than the index, such as `Boolean`.
fn array_of_keys(field: &Field, keys: &[Option<u64>]) -> Result<ArrayRef> {
    macro_rules! build {
        ($array:ty, $f:expr) => {
            Arc::new(keys.iter().map(|k| k.map($f)).collect::<$array>()) as ArrayRef
        };
    }

    let array = match field.data_type() {
        DataType::Boolean => build!(BooleanArray, |k| k % 2 == 1),
        DataType::Int8 => build!(Int8Array, |k| (k % 128) as i8),
        DataType::Int16 => build!(Int16Array, |k| k as i16),
        DataType::Int32 => build!(Int32Array, |k| k as i32),
        DataType::Int64 => build!(Int64Array, |k| k as i64),
        DataType::UInt8 => build!(UInt8Array, |k| (k % 256) as u8),
        DataType::UInt16 => build!(UInt16Array, |k| k as u16),
        DataType::UInt32 => build!(UInt32Array, |k| k as u32),
        DataType::UInt64 => build!(UInt64Array, |k| k),
        DataType::Float32 => build!(Float32Array, |k| k as f32 * 0.5),
        DataType::Float64 => build!(Float64Array, |k| k as f64 * 0.5),
        DataType::Utf8 => build!(StringArray, string_value),
        DataType::LargeUtf8 => build!(LargeStringArray, string_value),
        DataType::Utf8View => build!(StringViewArray, string_value),
        DataType::Date32 => build!(Date32Array, |k| k as i32),
        DataType::Date64 => build!(Date64Array, |k| k as i64 * 86_400_000),
        DataType::Timestamp(unit, tz) => {
            let tz = tz.clone();
            match unit {
                TimeUnit::Second => Arc::new(
                    keys.iter()
                        .map(|k| k.map(|k| k as i64))
                        .collect::<TimestampSecondArray>()
                        .with_timezone_opt(tz),
                ) as ArrayRef,
                TimeUnit::Millisecond => Arc::new(
                    keys.iter()
                        .map(|k| k.map(|k| k as i64 * 1_000))
                        .collect::<TimestampMillisecondArray>()
                        .with_timezone_opt(tz),
                ),
                TimeUnit::Microsecond => Arc::new(
                    keys.iter()
                        .map(|k| k.map(|k| k as i64 * 1_000_000))
                        .collect::<TimestampMicrosecondArray>()
                        .with_timezone_opt(tz),
                ),
                TimeUnit::Nanosecond => Arc::new(
                    keys.iter()
                        .map(|k| k.map(|k| k as i64 * 1_000_000_000))
                        .collect::<TimestampNanosecondArray>()
                        .with_timezone_opt(tz),
                ),
            }
        }
        DataType::Decimal128(precision, scale) => Arc::new(
            keys.iter()
                .map(|k| k.map(i128::from))
                .collect::<Decimal128Array>()
                .with_precision_and_scale(*precision, *scale)?,
        ),
        other => {
            return not_impl_err!(
                "SourceSpec cannot generate values of type {other} for field '{}'",
                field.name()
            );
        }
    };
    Ok(array)
}

fn string_value(k: u64) -> String {
    format!("s{k:03}")
}
