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

use std::cell::RefCell;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use futures::{Stream, StreamExt};

/// Source of probe ids and event sequence numbers, shared by all probes so
/// that events recorded by different probes can be ordered
static SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn next_sequence() -> u64 {
    SEQUENCE.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    /// Ids of the probes whose streams are being polled on this thread,
    /// innermost last
    static POLLING: RefCell<Vec<u64>> = const { RefCell::new(vec![]) };
}

/// Marks a probe's stream as being polled on this thread until dropped
struct PollingGuard;

impl PollingGuard {
    fn enter(id: u64) -> Self {
        POLLING.with(|polling| polling.borrow_mut().push(id));
        Self
    }
}

impl Drop for PollingGuard {
    fn drop(&mut self) {
        POLLING.with(|polling| polling.borrow_mut().pop());
    }
}

fn is_polling(id: u64) -> bool {
    POLLING.with(|polling| polling.borrow().contains(&id))
}

/// What happened to the streams of one partition, as recorded by a
/// [`StreamProbe`].
///
/// Counts cover every stream created for the partition. The `*_at` fields
/// hold the sequence number of the first such event. Sequence numbers come
/// from a counter shared by all probes, so events of different probes and
/// partitions can be ordered, but they say nothing about wall clock time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PartitionObservation {
    /// Number of streams created for the partition
    pub streams_created: usize,
    /// Number of calls to `poll_next`
    pub polls: usize,
    /// Number of calls to `poll_next` that did not happen while a stream
    /// observed by the consumer probe was being polled on the same thread. Only
    /// counted by a probe created with [`StreamProbe::with_consumer`].
    pub polls_without_demand: usize,
    /// Number of batches returned, including empty batches
    pub batches: usize,
    /// Number of rows in the returned batches
    pub rows: usize,
    /// Number of errors returned
    pub errors: usize,
    /// The first error returned
    pub first_error: Option<String>,
    /// Number of streams that returned `None`
    pub streams_finished: usize,
    /// Number of streams dropped
    pub streams_dropped: usize,
    /// When the first stream was created
    pub created_at: Option<u64>,
    /// When a stream was first polled
    pub first_polled_at: Option<u64>,
    /// When a stream first returned a batch with at least one row
    pub first_rows_at: Option<u64>,
    /// When a stream first returned `None`
    pub finished_at: Option<u64>,
    /// When a stream was first dropped
    pub dropped_at: Option<u64>,
}

impl PartitionObservation {
    /// Number of streams created and not yet dropped
    pub fn streams_alive(&self) -> usize {
        self.streams_created.saturating_sub(self.streams_dropped)
    }
}

#[derive(Debug, Default)]
struct ProbeState {
    partitions: Vec<PartitionObservation>,
}

impl ProbeState {
    fn partition(&mut self, partition: usize) -> &mut PartitionObservation {
        if self.partitions.len() <= partition {
            self.partitions
                .resize_with(partition + 1, PartitionObservation::default);
        }
        &mut self.partitions[partition]
    }
}

/// Records what happens to the streams it observes: when they are created,
/// polled, return batches, errors or `None`, and dropped.
///
/// A probe is a handle to shared state, so clones record into, and read from,
/// the same observations. Attach one to a [`MockSourceExec`] with
/// [`MockSourceExec::with_probe`], or wrap any stream with [`Self::observe`].
///
/// A probe created with [`Self::with_consumer`] also counts polls that are not
/// driven by the consumer: polls that happen while no stream observed by the
/// consumer probe is being polled on the same thread. An operator that only
/// polls its input from within its own `poll_next` makes no such polls. One
/// that polls its input from a spawned task, or from `execute`, does.
///
/// [`MockSourceExec`]: crate::fixtures::MockSourceExec
/// [`MockSourceExec::with_probe`]: crate::fixtures::MockSourceExec::with_probe
#[derive(Debug, Clone)]
pub struct StreamProbe {
    id: u64,
    consumer: Option<u64>,
    state: Arc<Mutex<ProbeState>>,
}

impl Default for StreamProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamProbe {
    /// Create a probe with no observations
    pub fn new() -> Self {
        Self {
            id: next_sequence(),
            consumer: None,
            state: Arc::default(),
        }
    }

    /// Create a probe that also counts polls not driven by the streams that
    /// `consumer` observes. See [`PartitionObservation::polls_without_demand`].
    pub fn with_consumer(consumer: &StreamProbe) -> Self {
        Self {
            consumer: Some(consumer.id),
            ..Self::new()
        }
    }

    /// Wrap `stream`, the stream of `partition`, so that it records into this
    /// probe. Records the creation of the stream.
    pub fn observe(
        &self,
        partition: usize,
        stream: SendableRecordBatchStream,
    ) -> SendableRecordBatchStream {
        self.record(partition, |observation, now| {
            observation.streams_created += 1;
            observation.created_at.get_or_insert(now);
        });
        Box::pin(ObservedStream {
            inner: stream,
            probe: self.clone(),
            partition,
            finished: false,
        })
    }

    /// Observations of `partition`. A partition with no streams has default
    /// (empty) observations.
    pub fn partition(&self, partition: usize) -> PartitionObservation {
        self.lock()
            .partitions
            .get(partition)
            .cloned()
            .unwrap_or_default()
    }

    /// Observations of every partition up to the highest one with a stream
    pub fn partitions(&self) -> Vec<PartitionObservation> {
        self.lock().partitions.clone()
    }

    /// Observations of `count` partitions, padded with empty observations for
    /// partitions without streams
    pub fn partitions_up_to(&self, count: usize) -> Vec<PartitionObservation> {
        (0..count).map(|p| self.partition(p)).collect()
    }

    /// Number of streams created and not yet dropped, over all partitions
    pub fn streams_alive(&self) -> usize {
        self.lock()
            .partitions
            .iter()
            .map(PartitionObservation::streams_alive)
            .sum()
    }

    fn lock(&self) -> MutexGuard<'_, ProbeState> {
        // Observations stay usable even if a stream panicked while recording
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn record(&self, partition: usize, f: impl FnOnce(&mut PartitionObservation, u64)) {
        let now = next_sequence();
        f(self.lock().partition(partition), now);
    }
}

/// A stream that records what happens to it in a [`StreamProbe`]
struct ObservedStream {
    inner: SendableRecordBatchStream,
    probe: StreamProbe,
    partition: usize,
    finished: bool,
}

impl Stream for ObservedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let demanded = self.probe.consumer.is_none_or(is_polling);
        self.probe.record(self.partition, |observation, now| {
            observation.polls += 1;
            observation.first_polled_at.get_or_insert(now);
            if !demanded {
                observation.polls_without_demand += 1;
            }
        });

        let poll = {
            let _guard = PollingGuard::enter(self.probe.id);
            self.inner.poll_next_unpin(cx)
        };

        match &poll {
            Poll::Ready(Some(Ok(batch))) => {
                let rows = batch.num_rows();
                self.probe.record(self.partition, |observation, now| {
                    observation.batches += 1;
                    observation.rows += rows;
                    if rows > 0 {
                        observation.first_rows_at.get_or_insert(now);
                    }
                });
            }
            Poll::Ready(Some(Err(e))) => {
                let message = e.strip_backtrace();
                self.probe.record(self.partition, |observation, _| {
                    observation.errors += 1;
                    observation.first_error.get_or_insert(message);
                });
            }
            Poll::Ready(None) if !self.finished => {
                self.finished = true;
                self.probe.record(self.partition, |observation, now| {
                    observation.streams_finished += 1;
                    observation.finished_at.get_or_insert(now);
                });
            }
            Poll::Ready(None) | Poll::Pending => {}
        }
        poll
    }
}

impl RecordBatchStream for ObservedStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Drop for ObservedStream {
    fn drop(&mut self) {
        self.probe.record(self.partition, |observation, now| {
            observation.streams_dropped += 1;
            observation.dropped_at.get_or_insert(now);
        });
    }
}
