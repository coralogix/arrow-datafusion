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

//! Weak-scaling sweep for the cumulative-aggregate branch of the
//! `ParallelWindow` optimizer rule (prefix-scan via `CarryExec`).
//!
//! For each "cores" setting `N`, builds a fresh table with `N`
//! partitions of `ROWS_PER_CORE` rows each, sets `target_partitions = N`,
//! and runs a cumulative window aggregate
//! (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, no `PARTITION BY`)
//! twice: once with `ParallelWindow` filtered out of the physical
//! optimizer chain (single-partition baseline) and once with the rule
//! enabled (parallel prefix-scan via `CarryExec`). Emits one CSV row
//! per iteration to stdout.
//!
//! Under linear scaling the PoC's wall-clock stays roughly constant
//! across the sweep while the baseline grows linearly with cores —
//! same shape as the bounded-RANGE bench, validated against the
//! prefix-scan path. CarryExec is pipeline-breaking, so its sequential
//! gather + offset cost shows up as the slope on the PoC line at high
//! core counts.
//!
//! `CarryExec` offsets a single aggregate column (the last column BWAG
//! appends to the input schema), so the SQL uses one `SUM(v) OVER ...`;
//! multiple cumulative aggregates would silently produce wrong values
//! for every aggregate but the last.
//!
//! Run:
//!     cargo bench --bench parallel_window_cumulative_scaling \
//!         > cumulative.csv

use arrow::array::{Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::prelude::{SessionConfig, SessionContext};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Uniform};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Weak-scaling design: rows scale linearly with cores so total work
/// grows proportionally to the parallelism budget. The PoC line stays
/// flat under linear scaling; the baseline grows linearly with cores
/// because the cumulative aggregate serializes through one partition.
/// Single `SUM` per row is cheap, so the per-core row count is larger
/// than the bounded-RANGE bench's to keep the baseline measurable at 1
/// core.
const ROWS_PER_CORE: usize = 2_500_000;
const BATCH_SIZE: usize = 8 * 1024;
const ITERATIONS: usize = 3;
const CORE_SETTINGS: &[usize] = &[1, 2, 4, 8, 16, 32];

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]))
}

/// Build `num_partitions` partitions of `(ts, v)` rows, with `ts`
/// monotonically increasing within each partition AND between
/// partitions. Same shape as the bounded-RANGE bench's fixture —
/// keeps SortExec cheap so the bench measures BWAG + repartition +
/// CarryExec cost.
fn make_partitions(num_partitions: usize) -> Vec<Vec<RecordBatch>> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE_C0FFEE);
    let v_dist = Uniform::new(0.0f64, 1.0).unwrap();
    let schema = schema();
    (0..num_partitions)
        .map(|part| {
            let mut batches = Vec::new();
            let part_start = (part * ROWS_PER_CORE) as i64;
            let mut next_ts = part_start;
            let mut remaining = ROWS_PER_CORE;
            while remaining > 0 {
                let len = remaining.min(BATCH_SIZE);
                let ts: Vec<i64> = (0..len as i64).map(|i| next_ts + i).collect();
                next_ts += len as i64;
                let v: Vec<f64> = (0..len).map(|_| v_dist.sample(&mut rng)).collect();
                batches.push(
                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int64Array::from(ts)),
                            Arc::new(Float64Array::from(v)),
                        ],
                    )
                    .unwrap(),
                );
                remaining -= len;
            }
            batches
        })
        .collect()
}

fn make_ctx(
    data: &[Vec<RecordBatch>],
    target_partitions: usize,
    with_parallel_window: bool,
) -> SessionContext {
    let table = MemTable::try_new(schema(), data.to_vec()).unwrap();
    let config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(BATCH_SIZE);

    let mut builder = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config);
    if !with_parallel_window {
        let rules: Vec<_> = PhysicalOptimizer::new()
            .rules
            .into_iter()
            .filter(|r| r.name() != "ParallelWindow")
            .collect();
        builder = builder.with_physical_optimizer_rules(rules);
    }
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(table)).unwrap();
    ctx
}

fn run_once(ctx: &SessionContext, rt: &Runtime, sql: &str) -> usize {
    let df = rt.block_on(ctx.sql(sql)).unwrap();
    let batches = rt.block_on(df.collect()).unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    black_box(batches);
    rows
}

fn main() {
    let rt = Runtime::new().unwrap();
    let sql = "SELECT SUM(v) OVER \
               (ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
               FROM t";

    if std::env::var("EXPLAIN_PLAN").is_ok() {
        let cores: usize = std::env::var("EXPLAIN_CORES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(32);
        let data = make_partitions(cores);
        let ctx = make_ctx(&data, cores, true);
        let df = rt.block_on(ctx.sql(&format!("EXPLAIN {sql}"))).unwrap();
        let batches = rt.block_on(df.collect()).unwrap();
        for b in batches {
            eprintln!(
                "{}",
                arrow::util::pretty::pretty_format_batches(&[b]).unwrap()
            );
        }
        return;
    }

    println!("cores,with_poc,iter,seconds,rows");
    for &cores in CORE_SETTINGS {
        let data = make_partitions(cores);
        for &with_poc in &[false, true] {
            let ctx = make_ctx(&data, cores, with_poc);
            let warmup_rows = run_once(&ctx, &rt, sql);
            for iter in 0..ITERATIONS {
                let t = Instant::now();
                let rows = run_once(&ctx, &rt, sql);
                let secs = t.elapsed().as_secs_f64();
                assert_eq!(
                    rows, warmup_rows,
                    "row count drifted: warmup={warmup_rows} run={rows}"
                );
                println!("{cores},{with_poc},{iter},{secs:.6},{rows}");
                eprintln!(
                    "cores={cores:>2} poc={with_poc:<5} iter={iter} \
                     secs={secs:>6.3} rows={rows}"
                );
            }
        }
    }
}
