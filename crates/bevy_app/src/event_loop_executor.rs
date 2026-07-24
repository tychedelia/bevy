//! Run closures on the event-loop thread from other threads.
//!
//! Some platforms (macOS, iOS) require certain graphics operations to run on the
//! event-loop thread.

use alloc::boxed::Box;
use alloc::sync::Arc;
use bevy_ecs::resource::Resource;
use core::any::Any;

/// Reason an [`EventLoopExecutor`] call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventLoopError {
    /// The event loop executor is not available.
    NotAvailable,
    /// The event loop is no longer running.
    Disconnected,
}

impl core::fmt::Display for EventLoopError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EventLoopError::NotAvailable => write!(f, "event loop executor not available"),
            EventLoopError::Disconnected => write!(f, "event loop disconnected"),
        }
    }
}

impl std::error::Error for EventLoopError {}

/// Runs closures on the event-loop thread.
pub trait EventLoopExecutor: Send + Sync + 'static {
    /// Runs a type-erased closure on the event-loop thread, blocking until it completes.
    fn execute_boxed(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send>,
    ) -> Result<Box<dyn Any + Send>, EventLoopError>;
}

/// Resource wrapper around an [`EventLoopExecutor`].
#[derive(Resource, Clone)]
pub struct EventLoopTaskRunner {
    executor: Arc<dyn EventLoopExecutor>,
}

impl EventLoopTaskRunner {
    /// Wraps an [`EventLoopExecutor`] implementation.
    pub fn new(executor: impl EventLoopExecutor) -> Self {
        Self {
            executor: Arc::new(executor),
        }
    }

    /// Runs a closure on the event-loop thread, blocking until it completes.
    pub fn run<F, R>(&self, f: F) -> Result<R, EventLoopError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let boxed_fn: Box<dyn FnOnce() -> Box<dyn Any + Send> + Send> =
            Box::new(move || Box::new(f()) as Box<dyn Any + Send>);

        let result = self.executor.execute_boxed(boxed_fn)?;
        result
            .downcast::<R>()
            .map(|b| *b)
            .map_err(|_| EventLoopError::Disconnected)
    }
}

impl core::fmt::Debug for EventLoopTaskRunner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EventLoopTaskRunner")
            .finish_non_exhaustive()
    }
}
