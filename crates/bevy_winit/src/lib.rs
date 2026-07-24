#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

//! `bevy_winit` provides utilities to handle window creation and the eventloop through [`winit`]
//!
//! Most commonly, the [`WinitPlugin`] is used as part of
//! [`DefaultPlugins`](https://docs.rs/bevy/latest/bevy/struct.DefaultPlugins.html).
//! The app's [runner](bevy_app::App::runner) is set by `WinitPlugin` and handles the `winit` [`EventLoop`].
//! See `winit_runner` for details.

extern crate alloc;

use bevy_derive::Deref;
use core::cell::RefCell;
use winit::{
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use bevy_app::{App, Last, Plugin};
use bevy_ecs::prelude::*;
use bevy_window::Window;
use system::check_keyboard_focus_lost;
#[cfg(not(target_arch = "wasm32"))]
pub use system::create_windows;
#[cfg(target_arch = "wasm32")]
use system::{changed_cursor_options, changed_windows, despawn_windows};
#[cfg(all(target_family = "wasm", target_os = "unknown"))]
pub use winit::platform::web::CustomCursorExtWebSys;
pub use winit::{
    event_loop::EventLoopProxy,
    window::{CustomCursor as WinitCustomCursor, CustomCursorSource},
};
pub use winit_config::*;
pub use winit_monitors::*;
pub use winit_windows::*;

use crate::accessibility::AccessKitPlugin;

#[cfg(not(target_arch = "wasm32"))]
pub use crate::state::{winit_runner, winit_runner_from_tls};

pub mod accessibility;
pub mod converters;
mod cursor;
pub mod runtime;
mod state;
mod system;
mod winit_config;
mod winit_monitors;
mod winit_windows;

thread_local! {
    /// Temporary storage of WinitWindows data to replace usage of `!Send` resources. This will be replaced with proper
    /// storage of `!Send` data after issue #17667 is complete.
    pub static WINIT_WINDOWS: RefCell<WinitWindows> = const { RefCell::new(WinitWindows::new()) };

    static WINIT_EVENT_LOOP: RefCell<Option<EventLoop<WinitUserEvent>>> = const { RefCell::new(None) };

    /// Drop queue for windows. Held one frame to let render thread release Arc references.
    pub(crate) static WINDOWS_TO_DROP: RefCell<Vec<bevy_window::WindowWrapper<winit::window::Window>>> = const { RefCell::new(Vec::new()) };
}

/// Thread-safe mapping from winit `WindowId` to bevy Entity.
#[derive(Resource, Default, Clone)]
pub struct WindowEntityMap {
    map: alloc::sync::Arc<std::sync::RwLock<bevy_platform::collections::HashMap<WindowId, Entity>>>,
}

impl WindowEntityMap {
    /// Insert a mapping from `WindowId` to Entity.
    pub fn insert(&self, window_id: WindowId, entity: Entity) {
        self.map.write().unwrap().insert(window_id, entity);
    }

    /// Get the Entity for a `WindowId`.
    pub fn get(&self, window_id: WindowId) -> Option<Entity> {
        self.map.read().unwrap().get(&window_id).copied()
    }

    /// Remove a mapping.
    pub fn remove(&self, window_id: WindowId) -> Option<Entity> {
        self.map.write().unwrap().remove(&window_id)
    }
}

/// A [`Plugin`] that uses `winit` to create and manage windows, and receive window and input
/// events.
///
/// This plugin will add systems and resources that sync with the `winit` backend and also
/// replace the existing [`App`] runner with one that constructs an [event loop](EventLoop) to
/// receive window and input events from the OS.
#[derive(Default)]
pub struct WinitPlugin {
    /// Allows the window (and the event loop) to be created on any thread
    /// instead of only the main thread.
    ///
    /// See [`EventLoopBuilder::build`](winit::event_loop::EventLoopBuilder::build) for more information on this.
    ///
    /// # Supported platforms
    ///
    /// Only works on Linux (X11/Wayland) and Windows.
    /// This field is ignored on other platforms.
    pub run_on_any_thread: bool,
}

impl Plugin for WinitPlugin {
    fn name(&self) -> &str {
        "bevy_winit::WinitPlugin"
    }

    fn build(&self, app: &mut App) {
        let mut event_loop_builder = EventLoop::<WinitUserEvent>::with_user_event();

        // linux check is needed because x11 might be enabled on other platforms.
        #[cfg(all(target_os = "linux", feature = "x11"))]
        {
            use winit::platform::x11::EventLoopBuilderExtX11;

            // This allows a Bevy app to be started and ran outside the main thread.
            // A use case for this is to allow external applications to spawn a thread
            // which runs a Bevy app without requiring the Bevy app to need to reside on
            // the main thread, which can be problematic.
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        // linux check is needed because wayland might be enabled on other platforms.
        #[cfg(all(target_os = "linux", feature = "wayland"))]
        {
            use winit::platform::wayland::EventLoopBuilderExtWayland;
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        #[cfg(target_os = "macos")]
        {
            use bevy_ecs::system::SystemState;
            use winit::platform::macos::EventLoopBuilderExtMacOS;

            // Don't request app activation on startup if all its windows should
            // start unfocused. Otherwise, app activation would focus one of the
            // windows.
            let mut initial_windows_state =
                SystemState::<Query<(Entity, &Window)>>::new(app.world_mut());
            let initial_windows = initial_windows_state.get(app.world()).unwrap();
            let initially_focused = initial_windows.iter().any(|(_, window)| window.focused);
            event_loop_builder.with_activate_ignoring_other_apps(initially_focused);
        }

        #[cfg(target_os = "windows")]
        {
            use winit::platform::windows::EventLoopBuilderExtWindows;
            event_loop_builder.with_any_thread(self.run_on_any_thread);
        }

        #[cfg(target_os = "android")]
        {
            use winit::platform::android::EventLoopBuilderExtAndroid;
            let msg = "Bevy must be setup with the #[bevy_main] macro on Android";
            event_loop_builder
                .with_android_app(bevy_android::ANDROID_APP.get().expect(msg).clone());
        }

        let event_loop = event_loop_builder
            .build()
            .expect("Failed to build event loop");

        let proxy = event_loop.create_proxy();

        // Wake up the event loop when `Ctrl+C` is received so that the app can
        // exit even while idle in a reactive update mode
        #[cfg(any(all(unix, not(target_os = "horizon")), windows))]
        {
            let proxy = proxy.clone();
            bevy_app::TerminalCtrlCHandlerPlugin::register_exit_handler(move || {
                let _ = proxy.send_event(WinitUserEvent::WakeUp);
            });
        }

        WINIT_EVENT_LOOP.with(|cell| {
            *cell.borrow_mut() = Some(event_loop);
        });

        let task_sender = WinitTaskSender::new(proxy.clone());

        app.init_resource::<WinitMonitors>()
            .init_resource::<WinitSettings>()
            .init_resource::<WindowEntityMap>()
            .insert_resource(DisplayHandleWrapper(WINIT_EVENT_LOOP.with(|cell| {
                cell.borrow()
                    .as_ref()
                    .expect("EventLoop should be stored")
                    .owned_display_handle()
            })))
            .insert_resource(EventLoopProxyWrapper(proxy))
            .insert_resource(task_sender.clone())
            .add_message::<RawWinitWindowEvent>();

        #[cfg(not(target_arch = "wasm32"))]
        app.insert_resource(bevy_app::event_loop_executor::EventLoopTaskRunner::new(
            WinitEventLoopExecutor::new(task_sender),
        ));

        #[cfg(not(target_arch = "wasm32"))]
        app.set_runner(winit_runner_from_tls);

        #[cfg(target_arch = "wasm32")]
        app.add_systems(
            Last,
            (
                changed_windows,
                changed_cursor_options,
                despawn_windows
                    .after(bevy_window::ExitSystems)
                    .after(bevy_app::OnAppExitSystems),
                check_keyboard_focus_lost,
            )
                .chain(),
        );

        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(Last, check_keyboard_focus_lost);

        app.add_plugins(AccessKitPlugin);
        app.add_plugins(cursor::WinitCursorPlugin);

        app.add_observer(
            |_window: On<Add, Window>, task_sender: Res<WinitTaskSender>| -> Result {
                task_sender.window_added()?;
                Ok(())
            },
        );
    }
}

/// A task to run on the main thread inside the winit event loop.
pub type WinitTaskFn = Box<dyn FnOnce(&ActiveEventLoop) + Send + 'static>;

/// Events and tasks that can be sent to the winit event loop.
///
/// Sent via the [`EventLoopProxyWrapper`] or [`WinitTaskSender`] resources.
///
/// # Example
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_winit::{WinitTaskSender, WinitUserEvent};
/// fn wakeup_system(task_sender: Res<WinitTaskSender>) -> Result {
///     task_sender.wake_up()?;
///     Ok(())
/// }
/// ```
pub enum WinitUserEvent {
    /// Dummy event that just wakes up the winit event loop
    WakeUp,
    /// Tell winit that a window needs to be created
    WindowAdded,
    /// A task to execute on the main thread within the winit event loop
    Task(WinitTaskFn),
    /// Signal to exit the event loop
    Exit,
}

impl core::fmt::Debug for WinitUserEvent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WinitUserEvent::WakeUp => write!(f, "WakeUp"),
            WinitUserEvent::WindowAdded => write!(f, "WindowAdded"),
            WinitUserEvent::Task(_) => write!(f, "Task(...)"),
            WinitUserEvent::Exit => write!(f, "Exit"),
        }
    }
}

/// The original window event as produced by Winit. This is meant as an escape
/// hatch for power users that wish to add custom Winit integrations.
/// If you want to process events for your app or game, you should instead use
/// `bevy::window::WindowEvent`, or one of its sub-events.
///
/// When you receive this event it has already been handled by Bevy's main loop.
/// Sending these events will NOT cause them to be processed by Bevy.
#[derive(Debug, Clone, Message)]
pub struct RawWinitWindowEvent {
    /// The window for which the event was fired.
    pub window_id: WindowId,
    /// The raw winit window event.
    pub event: winit::event::WindowEvent,
}

/// A wrapper type around [`winit::event_loop::EventLoopProxy`] with the specific
/// [`winit::event::Event::UserEvent`] used in the [`WinitPlugin`].
///
/// The `EventLoopProxy` can be used to request a redraw from outside bevy.
///
/// Use `Res<EventLoopProxyWrapper>` to retrieve this resource.
#[derive(Resource, Deref)]
pub struct EventLoopProxyWrapper(EventLoopProxy<WinitUserEvent>);

/// Error type for winit task operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinitTaskError {
    /// The event loop has been closed
    EventLoopClosed,
}

impl core::fmt::Display for WinitTaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WinitTaskError::EventLoopClosed => write!(f, "winit event loop has been closed"),
        }
    }
}

impl std::error::Error for WinitTaskError {}

/// A resource for sending tasks to run on the main thread inside the winit event loop.
#[derive(Resource, Clone)]
pub struct WinitTaskSender {
    proxy: EventLoopProxy<WinitUserEvent>,
}

impl WinitTaskSender {
    /// Creates a new `WinitTaskSender` from an event loop proxy.
    pub fn new(proxy: EventLoopProxy<WinitUserEvent>) -> Self {
        Self { proxy }
    }

    /// Sends a task to run on the main thread inside the winit event loop.
    pub fn send<F>(&self, task: F) -> Result<(), WinitTaskError>
    where
        F: FnOnce(&ActiveEventLoop) + Send + 'static,
    {
        self.proxy
            .send_event(WinitUserEvent::Task(Box::new(task)))
            .map_err(|_| WinitTaskError::EventLoopClosed)
    }

    /// Wakes up the winit event loop without sending a specific task.
    pub fn wake_up(&self) -> Result<(), WinitTaskError> {
        self.proxy
            .send_event(WinitUserEvent::WakeUp)
            .map_err(|_| WinitTaskError::EventLoopClosed)
    }

    /// Signals that a window has been added and needs to be created.
    pub fn window_added(&self) -> Result<(), WinitTaskError> {
        self.proxy
            .send_event(WinitUserEvent::WindowAdded)
            .map_err(|_| WinitTaskError::EventLoopClosed)
    }

    /// Signals that the application should exit.
    pub fn exit(&self) -> Result<(), WinitTaskError> {
        self.proxy
            .send_event(WinitUserEvent::Exit)
            .map_err(|_| WinitTaskError::EventLoopClosed)
    }

    /// Sends a task to the main thread and blocks until it completes, returning the result.
    ///
    /// # Warning
    ///
    /// Do not call from the winit (main) thread, or it will deadlock.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn scoped<F, R>(&self, task: F) -> Result<R, WinitTaskError>
    where
        F: FnOnce(&ActiveEventLoop) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.scoped_dispatch(None, task)
    }

    /// Like [`scoped`](Self::scoped), but gives up after `timeout`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn scoped_timeout<F, R>(
        &self,
        timeout: core::time::Duration,
        task: F,
    ) -> Result<R, WinitTaskError>
    where
        F: FnOnce(&ActiveEventLoop) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.scoped_dispatch(Some(timeout), task)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn scoped_dispatch<F, R>(
        &self,
        timeout: Option<core::time::Duration>,
        task: F,
    ) -> Result<R, WinitTaskError>
    where
        F: FnOnce(&ActiveEventLoop) -> R + Send + 'static,
        R: Send + 'static,
    {
        use std::sync::mpsc;

        let (tx, rx) = mpsc::channel();

        self.proxy
            .send_event(WinitUserEvent::Task(Box::new(move |event_loop| {
                let result = task(event_loop);
                let _ = tx.send(result);
            })))
            .map_err(|_| WinitTaskError::EventLoopClosed)?;

        match timeout {
            Some(timeout) => rx
                .recv_timeout(timeout)
                .map_err(|_| WinitTaskError::EventLoopClosed),
            None => rx.recv().map_err(|_| WinitTaskError::EventLoopClosed),
        }
    }

    /// Like [`scoped`](Self::scoped), but without the [`ActiveEventLoop`] reference.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute<F, R>(&self, task: F) -> Result<R, WinitTaskError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.scoped(move |_| task())
    }

    /// Like [`execute`](Self::execute), but gives up after `timeout`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute_timeout<F, R>(
        &self,
        timeout: core::time::Duration,
        task: F,
    ) -> Result<R, WinitTaskError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.scoped_timeout(timeout, move |_| task())
    }
}

/// Adapts a [`WinitTaskSender`] to [`EventLoopExecutor`](bevy_app::event_loop_executor::EventLoopExecutor).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct WinitEventLoopExecutor {
    sender: WinitTaskSender,
}

#[cfg(not(target_arch = "wasm32"))]
impl WinitEventLoopExecutor {
    /// Creates a new executor from a `WinitTaskSender`.
    pub fn new(sender: WinitTaskSender) -> Self {
        Self { sender }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl bevy_app::event_loop_executor::EventLoopExecutor for WinitEventLoopExecutor {
    fn execute_boxed(
        &self,
        f: Box<dyn FnOnce() -> Box<dyn core::any::Any + Send> + Send>,
    ) -> Result<Box<dyn core::any::Any + Send>, bevy_app::event_loop_executor::EventLoopError> {
        // Bound the wait so a stalled event loop can't hang the render thread.
        const EVENT_LOOP_TASK_TIMEOUT: core::time::Duration = core::time::Duration::from_secs(5);
        self.sender
            .execute_timeout(EVENT_LOOP_TASK_TIMEOUT, f)
            .map_err(|_| bevy_app::event_loop_executor::EventLoopError::Disconnected)
    }
}

/// A wrapper around [`winit::event_loop::OwnedDisplayHandle`]
///
/// The `DisplayHandleWrapper` can be used to build integrations that rely on direct
/// access to the display handle
///
/// Use `Res<DisplayHandleWrapper>` to receive this resource.
#[derive(Resource, Deref)]
pub struct DisplayHandleWrapper(pub winit::event_loop::OwnedDisplayHandle);
