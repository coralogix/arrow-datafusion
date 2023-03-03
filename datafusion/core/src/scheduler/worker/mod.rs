use crate::scheduler::task::Task;
use crossbeam_deque::{Injector, Worker};
use log::{debug, error, trace};
use std::cell::Cell;
use std::fmt::{Debug, Formatter};
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::thread_local;
use tokio::task::LocalSet;

thread_local! {
    static WORKER_CONTEXT: Cell<*const WorkerContext> = Cell::new(std::ptr::null());
}

pub(crate) fn is_worker() -> bool {
    !WORKER_CONTEXT.with(Cell::get).is_null()
}

pub(crate) fn spawn_local(task: Task) {
    assert!(is_worker(), "must be called from a worker");
    let ctx = unsafe { &*WORKER_CONTEXT.with(Cell::get) };

    ctx.spawn(task);
}

pub(crate) fn spawn_local_fifo(task: Task) {
    assert!(is_worker(), "must be called from a worker");
    let ctx = unsafe { &*WORKER_CONTEXT.with(Cell::get) };

    ctx.spawn_local(task);
}

pub trait Driver: Debug + Clone + Send + 'static {
    fn run(
        &mut self,
        ctx: &WorkerContext,
        panic_handler: Box<dyn Fn(Box<dyn std::any::Any + Send>) + Send + Sync + 'static>,
    );
}

#[derive(Default, Clone, Debug)]
pub struct DefaultDriver {}

impl Driver for DefaultDriver {
    fn run(
        &mut self,
        ctx: &WorkerContext,
        panic_handler: Box<dyn Fn(Box<dyn std::any::Any + Send>) + Send + Sync + 'static>,
    ) {
        debug!("[default-driver] starting run loop on worker {}", ctx.name);

        while !ctx.is_terminated() {
            if let Some(task) = ctx.next_task() {
                trace!("executing task {:#?} on {}", task, ctx.name);
                match std::panic::catch_unwind(AssertUnwindSafe(|| task.do_work())) {
                    Ok(()) => {}
                    Err(e) => panic_handler(e),
                }
            }
        }

        debug!(
            "[default-driver] terminating run loop on worker {}",
            ctx.name
        );
    }
}

#[derive(Default, Clone, Debug)]
pub struct ThreadLocalDriver {}

impl Driver for ThreadLocalDriver {
    fn run(
        &mut self,
        ctx: &WorkerContext,
        panic_handler: Box<dyn Fn(Box<dyn std::any::Any + Send>) + Send + Sync + 'static>,
    ) {
        debug!(
            "[thread-local-driver] starting run loop on worker {}",
            ctx.name
        );

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            while !ctx.is_terminated() {
                if let Some(task) = ctx.next_task() {
                    trace!("executing task {:#?} on {}", task, ctx.name);
                    match std::panic::catch_unwind(AssertUnwindSafe(|| task.do_work())) {
                        Ok(()) => {}
                        Err(e) => panic_handler(e),
                    }
                }
            }
        });

        debug!(
            "[thread-local-driver] terminating run loop on worker {}",
            ctx.name
        );
    }
}

pub struct WorkerContext {
    name: String,
    injector: Arc<Injector<Task>>,
    local: Worker<Task>,
    terminate: Arc<AtomicBool>,
}

impl WorkerContext {
    fn spawn(&self, task: Task) {
        self.injector.push(task);
    }

    fn spawn_local(&self, task: Task) {
        self.local.push(task);
    }

    pub fn next_task(&self) -> Option<Task> {
        self.local.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            std::iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.injector.steal_batch_and_pop(&self.local)
            })
            // Loop while no task was stolen and any steal operation needs to be retried.
            .find(|s| !s.is_retry())
            // Extract the stolen task, if there is one.
            .and_then(|s| s.success())
        })
    }

    pub fn is_terminated(&self) -> bool {
        self.terminate.load(Ordering::Acquire)
    }

    pub(crate) fn current() -> &'static WorkerContext {
        assert!(is_worker(), "not on a worker thread");
        unsafe { &*WORKER_CONTEXT.with(Cell::get) }
    }

    fn set_current(ctx: *const WorkerContext) {
        WORKER_CONTEXT.with(|cell| {
            assert!(cell.get().is_null(), "worker context has already been set");
            cell.set(ctx);
        })
    }
}

pub(crate) struct WorkerPool {
    injector: Arc<Injector<Task>>,
    terminate: Arc<AtomicBool>,
}

impl WorkerPool {
    pub fn new<D: Driver>(mut builder: WorkerPoolBuilder<D>) -> io::Result<Self> {
        let injector = Arc::new(Injector::new());
        let terminate = Arc::new(AtomicBool::new(false));

        for thread_idx in 0..builder.num_threads {
            let thread_name = format!("{}-{thread_idx}", builder.thread_name);
            debug!("spawning worker thread {thread_name}");

            let worker_ctx = WorkerContext {
                name: thread_name.clone(),
                injector: injector.clone(),
                local: Worker::new_fifo(),
                terminate: terminate.clone(),
            };

            let mut b = thread::Builder::new().name(thread_name);

            if let Some(size) = builder.stack_size {
                b = b.stack_size(size);
            }

            let driver = builder.driver.clone();
            let panic_handler = std::mem::take(&mut builder.panic_handler)
                .unwrap_or_else(|| Box::new(|p| error!("{}", format_worker_panic(p))));

            b.spawn(move || {
                let ctx = worker_ctx;
                WorkerContext::set_current(&ctx);

                let mut driver = driver;

                driver.run(&ctx, panic_handler)
            })?;
        }

        Ok(Self {
            injector,
            terminate,
        })
    }

    pub fn spawn(&self, task: Task) {
        self.injector.push(task);
    }
}

impl Debug for WorkerPool {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("termiante", self.terminate.as_ref())
            .finish()
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.terminate.store(true, Ordering::Release);
    }
}

pub(crate) struct WorkerPoolBuilder<D = DefaultDriver> {
    thread_name: String,
    num_threads: usize,
    stack_size: Option<usize>,
    driver: D,
    panic_handler:
        Option<Box<dyn Fn(Box<dyn std::any::Any + Send>) + Send + Sync + 'static>>,
}

impl<D: Driver> Debug for WorkerPoolBuilder<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerPoolBuilder")
            .field("thread_name", &self.thread_name)
            .field("num_threads", &self.num_threads)
            .field("stack_size", &self.stack_size)
            .field("driver", &self.driver)
            .finish()
    }
}

impl<D: Driver> WorkerPoolBuilder<D> {
    pub fn driver<D1: Driver>(self, driver: D1) -> WorkerPoolBuilder<D1> {
        WorkerPoolBuilder {
            thread_name: self.thread_name,
            num_threads: self.num_threads,
            stack_size: self.stack_size,
            driver,
            panic_handler: self.panic_handler,
        }
    }

    pub fn panic_handler<F: Fn(Box<dyn std::any::Any + Send>) + Send + Sync + 'static>(
        mut self,
        handler: F,
    ) -> Self {
        self.panic_handler = Some(Box::new(handler));
        self
    }

    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    pub fn build(self) -> io::Result<WorkerPool> {
        WorkerPool::new(self)
    }
}

impl Default for WorkerPoolBuilder {
    fn default() -> Self {
        Self {
            thread_name: "df-worker".to_string(),
            num_threads: num_cpus::get(),
            stack_size: None,
            driver: DefaultDriver::default(),
            panic_handler: None,
        }
    }
}

fn format_worker_panic(panic: Box<dyn std::any::Any + Send>) -> String {
    let maybe_idx = rayon::current_thread_index();
    let worker: &dyn std::fmt::Display = match &maybe_idx {
        Some(idx) => idx,
        None => &"UNKNOWN",
    };

    let message = if let Some(msg) = panic.downcast_ref::<&str>() {
        *msg
    } else if let Some(msg) = panic.downcast_ref::<String>() {
        msg.as_str()
    } else {
        "UNKNOWN"
    };

    format!("worker {worker} panicked with: {message}")
}
