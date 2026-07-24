//! Forwards winit events to the ECS thread over a channel.

use bevy_app::{App, AppExit};
use bevy_log::trace;
use bevy_platform::time::Instant;
use core::time::Duration;
use winit::{
    event::{DeviceEvent, DeviceId, WindowEvent},
    window::WindowId,
};

/// Events sent from the winit event loop to the ECS thread.
#[derive(Debug)]
pub enum RuntimeEvent {
    /// The application was suspended.
    Suspended,
    /// The application was resumed.
    Resumed,
    /// The application is exiting.
    Exiting,
    /// A window event.
    WindowEvent {
        /// The window that received the event.
        window_id: WindowId,
        /// When the event was received.
        timestamp: Instant,
        /// The event.
        event: WindowEvent,
    },
    /// A device event.
    DeviceEvent {
        /// The device that generated the event.
        device_id: DeviceId,
        /// When the event was received.
        timestamp: Instant,
        /// The event.
        event: DeviceEvent,
    },
    /// The system is low on memory.
    MemoryWarning,
    /// Wakes the event loop.
    WakeUp,
    /// The set of monitors changed.
    #[cfg(not(target_arch = "wasm32"))]
    MonitorsChanged(Vec<MonitorData>),
}

/// A snapshot of a monitor's properties, forwarded to the ECS thread, which cannot query
/// winit monitors directly.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct MonitorData {
    /// The winit monitor handle.
    pub handle: winit::monitor::MonitorHandle,
    /// The monitor's name, if available.
    pub name: Option<String>,
    /// Physical width in pixels.
    pub physical_width: u32,
    /// Physical height in pixels.
    pub physical_height: u32,
    /// Physical position.
    pub physical_position: bevy_math::IVec2,
    /// Refresh rate in millihertz, if available.
    pub refresh_rate_millihertz: Option<u32>,
    /// Scale factor.
    pub scale_factor: f64,
    /// Available video modes.
    pub video_modes: Vec<bevy_window::VideoMode>,
    /// Whether this is the primary monitor.
    pub is_primary: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl MonitorData {
    fn from_handle(
        handle: winit::monitor::MonitorHandle,
        primary: Option<&winit::monitor::MonitorHandle>,
    ) -> Self {
        let size = handle.size();
        let position = handle.position();
        Self {
            is_primary: primary == Some(&handle),
            name: handle.name(),
            physical_width: size.width,
            physical_height: size.height,
            physical_position: bevy_math::IVec2::new(position.x, position.y),
            refresh_rate_millihertz: handle.refresh_rate_millihertz(),
            scale_factor: handle.scale_factor(),
            video_modes: handle
                .video_modes()
                .map(|mode| {
                    let size = mode.size();
                    bevy_window::VideoMode {
                        physical_size: bevy_math::UVec2::new(size.width, size.height),
                        bit_depth: mode.bit_depth(),
                        refresh_rate_millihertz: mode.refresh_rate_millihertz(),
                    }
                })
                .collect(),
            handle,
        }
    }
}

/// Channel types for runtime events.
#[cfg(not(target_arch = "wasm32"))]
pub mod channel {
    use super::RuntimeEvent;

    /// The sender half of the runtime events channel.
    pub type RuntimeEventSender = crossbeam_channel::Sender<RuntimeEvent>;
    /// The receiver half of the runtime events channel.
    pub type RuntimeEventReceiver = crossbeam_channel::Receiver<RuntimeEvent>;

    /// Creates an unbounded channel for runtime events.
    pub fn create_channel() -> (RuntimeEventSender, RuntimeEventReceiver) {
        crossbeam_channel::unbounded()
    }
}

#[cfg(target_arch = "wasm32")]
pub mod channel {
    use super::RuntimeEvent;

    pub type RuntimeEventSender = std::sync::mpsc::Sender<RuntimeEvent>;
    pub type RuntimeEventReceiver = std::sync::mpsc::Receiver<RuntimeEvent>;

    pub fn create_channel() -> (RuntimeEventSender, RuntimeEventReceiver) {
        std::sync::mpsc::channel()
    }
}

pub use channel::{create_channel, RuntimeEventReceiver, RuntimeEventSender};

use crate::WinitUserEvent;
use winit::{application::ApplicationHandler, event_loop::ActiveEventLoop};

/// Winit application handler that forwards events to the ECS thread over a channel.
///
/// Handlers must never block on the ECS or render threads; they only push onto the unbounded
/// channel and run injected [`WinitUserEvent::Task`]s. This lets those threads safely block on
/// the event loop (e.g. surface creation) without deadlocking.
pub struct WinitApp {
    /// Sends runtime events to the ECS thread.
    pub event_sender: RuntimeEventSender,
    #[cfg(not(target_arch = "wasm32"))]
    monitors: Vec<winit::monitor::MonitorHandle>,
}

impl WinitApp {
    /// Creates a `WinitApp` with the given event sender.
    pub fn new(event_sender: RuntimeEventSender) -> Self {
        Self {
            event_sender,
            #[cfg(not(target_arch = "wasm32"))]
            monitors: Vec::new(),
        }
    }

    fn send_event(&self, event: RuntimeEvent) {
        if self.event_sender.send(event).is_err() {
            bevy_log::warn!("Failed to send runtime event: channel closed");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn send_monitors_if_changed(&mut self, event_loop: &ActiveEventLoop) {
        let current: Vec<winit::monitor::MonitorHandle> = event_loop.available_monitors().collect();
        if current != self.monitors {
            self.monitors = current.clone();
            let primary = event_loop.primary_monitor();
            let data = current
                .into_iter()
                .map(|monitor| MonitorData::from_handle(monitor, primary.as_ref()))
                .collect();
            self.send_event(RuntimeEvent::MonitorsChanged(data));
        }
    }
}

impl ApplicationHandler<WinitUserEvent> for WinitApp {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        self.send_event(RuntimeEvent::Resumed);
        #[cfg(not(target_arch = "wasm32"))]
        self.send_monitors_if_changed(_event_loop);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.send_event(RuntimeEvent::Suspended);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        crate::WINIT_WINDOWS.with_borrow(|winit_windows| {
            crate::accessibility::ACCESS_KIT_ADAPTERS.with_borrow_mut(|adapters| {
                if let Some(window_entity) = winit_windows.get_window_entity(window_id)
                    && let Some(adapter) = adapters.get_mut(&window_entity)
                    && let Some(winit_window) = winit_windows.get_window(window_entity)
                {
                    adapter.process_event(winit_window, &event);
                }
            });
        });

        let timestamp = Instant::now();
        self.send_event(RuntimeEvent::WindowEvent {
            window_id,
            timestamp,
            event,
        });
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let timestamp = Instant::now();
        self.send_event(RuntimeEvent::DeviceEvent {
            device_id,
            timestamp,
            event,
        });
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.send_event(RuntimeEvent::Exiting);
    }

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        self.send_event(RuntimeEvent::MemoryWarning);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: WinitUserEvent) {
        match event {
            WinitUserEvent::WakeUp => self.send_event(RuntimeEvent::WakeUp),
            WinitUserEvent::WindowAdded => {}
            WinitUserEvent::Task(task) => task(event_loop),
            WinitUserEvent::Exit => event_loop.exit(),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        #[cfg(not(target_arch = "wasm32"))]
        self.send_monitors_if_changed(_event_loop);
    }
}

#[derive(Default)]
struct ReceivedEvents {
    window_events: bool,
    device_events: bool,
    user_events: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_monitors_changed(world: &mut bevy_ecs::world::World, data: Vec<MonitorData>) {
    use bevy_window::{Monitor, PrimaryMonitor};

    world.resource_scope::<crate::WinitMonitors, _>(|world, mut monitors| {
        let mut seen = vec![false; monitors.monitors.len()];
        for monitor in data {
            if let Some(index) = monitors
                .monitors
                .iter()
                .position(|(handle, _)| handle == &monitor.handle)
            {
                seen[index] = true;
                continue;
            }

            let entity = world
                .spawn(Monitor {
                    name: monitor.name,
                    physical_height: monitor.physical_height,
                    physical_width: monitor.physical_width,
                    physical_position: monitor.physical_position,
                    refresh_rate_millihertz: monitor.refresh_rate_millihertz,
                    scale_factor: monitor.scale_factor,
                    video_modes: monitor.video_modes,
                })
                .id();
            if monitor.is_primary {
                world.entity_mut(entity).insert(PrimaryMonitor);
            }
            monitors.monitors.push((monitor.handle, entity));
            seen.push(true);
        }

        let mut index = 0;
        monitors.monitors.retain(|(_, entity)| {
            let keep = seen.get(index).copied().unwrap_or(true);
            index += 1;
            if !keep {
                world.despawn(*entity);
            }
            keep
        });
    });
}

/// Runs the ECS app loop on a thread separate from the winit event loop, receiving events over
/// the channel and updating at intervals set by [`UpdateMode`](crate::UpdateMode).
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app(mut app: App, event_receiver: RuntimeEventReceiver) -> AppExit {
    use crate::system::WinitWindowPressedKeys;
    use bevy_app::{First, Last};
    use bevy_ecs::change_detection::DetectChanges;
    use bevy_ecs::message::Messages;
    use bevy_input::{
        gestures::{DoubleTapGesture, PanGesture, PinchGesture, RotationGesture},
        mouse::{MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel},
    };
    use bevy_math::{ivec2, DVec2, Vec2};
    use bevy_window::{
        AppLifecycle, CursorEntered, CursorLeft, CursorMoved, FileDragAndDrop, Ime,
        WindowCloseRequested, WindowDestroyed, WindowFocused, WindowMoved, WindowOccluded,
        WindowThemeChanged,
    };
    use crossbeam_channel::RecvTimeoutError;
    use winit::event::WindowEvent as WinitWindowEvent;

    use crate::{converters, UpdateMode, WinitSettings};

    trace!("ECS thread started");

    app.add_systems(First, crate::system::create_windows);
    app.add_systems(
        bevy_app::PostUpdate,
        (
            crate::system::changed_windows,
            crate::system::changed_cursor_options,
            crate::cursor::apply_pending_cursors,
        ),
    );
    app.add_systems(
        Last,
        (
            crate::system::despawn_windows,
            crate::system::forward_redraw_requests,
        ),
    );

    // Desktop has no suspend/resume lifecycle; mobile waits for the Resumed event.
    #[cfg(any(target_os = "ios", target_os = "android"))]
    let mut is_resumed = false;
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    let mut is_resumed = true;
    let mut is_focused = true;
    // Force initial updates to create windows before waiting for events.
    let mut startup_forced_updates: u32 = 5;

    loop {
        let update_mode = app
            .world()
            .get_resource::<WinitSettings>()
            .map(|s| s.update_mode(is_focused))
            .unwrap_or(UpdateMode::Continuous);

        let mut received_events = ReceivedEvents::default();

        let use_blocking = matches!(update_mode, UpdateMode::Reactive { .. });
        let wait_duration = match update_mode {
            UpdateMode::Continuous => Duration::ZERO,
            UpdateMode::Reactive { wait, .. } => wait,
        };

        let deadline = Instant::now() + wait_duration;

        let mut process_event = |event: RuntimeEvent| -> Option<AppExit> {
            trace!("Received event: {:?}", core::mem::discriminant(&event));
            match event {
                RuntimeEvent::Exiting => {
                    trace!("Received exit signal from winit");
                    return Some(app.should_exit().unwrap_or(AppExit::Success));
                }
                RuntimeEvent::Resumed => {
                    trace!("App resumed");
                    is_resumed = true;
                    received_events.user_events = true;
                    let lifecycle = AppLifecycle::Running;
                    if let Some(mut messages) =
                        app.world_mut().get_resource_mut::<Messages<AppLifecycle>>()
                    {
                        messages.write(lifecycle);
                    }
                    app.world_mut()
                        .write_message(bevy_window::WindowEvent::from(lifecycle));
                }
                RuntimeEvent::Suspended => {
                    trace!("App suspended");
                    is_resumed = false;
                    received_events.user_events = true;
                    let lifecycle = AppLifecycle::Suspended;
                    if let Some(mut messages) =
                        app.world_mut().get_resource_mut::<Messages<AppLifecycle>>()
                    {
                        messages.write(lifecycle);
                    }
                    app.world_mut()
                        .write_message(bevy_window::WindowEvent::from(lifecycle));
                }
                RuntimeEvent::WindowEvent {
                    window_id, event, ..
                } => {
                    received_events.window_events = true;

                    app.world_mut().write_message(crate::RawWinitWindowEvent {
                        window_id,
                        event: event.clone(),
                    });

                    let window_entity = app
                        .world()
                        .get_resource::<crate::WindowEntityMap>()
                        .and_then(|map| map.get(window_id));

                    let Some(entity) = window_entity else {
                        trace!("No entity found for window {:?}", window_id);
                        return None;
                    };

                    match event {
                        WinitWindowEvent::CloseRequested => {
                            trace!("Window close requested for {:?}", entity);
                            let event = WindowCloseRequested { window: entity };
                            if let Some(mut messages) = app
                                .world_mut()
                                .get_resource_mut::<Messages<WindowCloseRequested>>()
                            {
                                messages.write(event.clone());
                            }
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Resized(size) => {
                            let (width, height) = {
                                if let Some(mut window) =
                                    app.world_mut().get_mut::<bevy_window::Window>(entity)
                                {
                                    window
                                        .resolution
                                        .set_physical_resolution(size.width, size.height);
                                    (window.width(), window.height())
                                } else {
                                    return None;
                                }
                            };
                            let event = bevy_window::WindowResized {
                                window: entity,
                                width,
                                height,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            trace!("Window {:?} scale factor changed: {}", entity, scale_factor);
                            let (prior_factor, has_override) = {
                                if let Some(mut window) =
                                    app.world_mut().get_mut::<bevy_window::Window>(entity)
                                {
                                    let prior = window.resolution.scale_factor();
                                    let has_override =
                                        window.resolution.scale_factor_override().is_some();
                                    window.resolution.set_scale_factor(scale_factor as f32);
                                    (prior, has_override)
                                } else {
                                    return None;
                                }
                            };
                            let backend_event = bevy_window::WindowBackendScaleFactorChanged {
                                window: entity,
                                scale_factor,
                            };
                            app.world_mut().write_message(backend_event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(backend_event));
                            if !has_override
                                && !approx::relative_eq!(scale_factor as f32, prior_factor)
                            {
                                let event = bevy_window::WindowScaleFactorChanged {
                                    window: entity,
                                    scale_factor,
                                };
                                app.world_mut().write_message(event.clone());
                                app.world_mut()
                                    .write_message(bevy_window::WindowEvent::from(event));
                            }
                        }
                        WinitWindowEvent::Focused(focused) => {
                            trace!("Window {:?} focus changed: {}", entity, focused);
                            is_focused = focused;
                            if let Some(mut window) =
                                app.world_mut().get_mut::<bevy_window::Window>(entity)
                            {
                                window.focused = focused;
                            }
                            let event = WindowFocused {
                                window: entity,
                                focused,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::KeyboardInput {
                            ref event,
                            is_synthetic: false,
                            ..
                        } => {
                            let keyboard_input = converters::convert_keyboard_input(event, entity);
                            if let Some(mut pressed_keys) =
                                app.world_mut().get_mut::<WinitWindowPressedKeys>(entity)
                            {
                                if event.state.is_pressed() {
                                    pressed_keys.0.insert(
                                        keyboard_input.key_code,
                                        keyboard_input.logical_key.clone(),
                                    );
                                } else {
                                    pressed_keys.0.remove(&keyboard_input.key_code);
                                }
                            }
                            app.world_mut().write_message(keyboard_input.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(keyboard_input));
                        }
                        WinitWindowEvent::CursorMoved { position, .. } => {
                            let physical_position = DVec2::new(position.x, position.y);
                            let (position, delta) = {
                                if let Some(mut window) =
                                    app.world_mut().get_mut::<bevy_window::Window>(entity)
                                {
                                    let last_position = window.physical_cursor_position();
                                    let scale = window.resolution.scale_factor();
                                    let delta = last_position.map(|last_pos| {
                                        (physical_position.as_vec2() - last_pos) / scale
                                    });
                                    window.set_physical_cursor_position(Some(physical_position));
                                    let pos = (physical_position / scale as f64).as_vec2();
                                    (pos, delta)
                                } else {
                                    return None;
                                }
                            };
                            let event = CursorMoved {
                                window: entity,
                                position,
                                delta,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::CursorEntered { .. } => {
                            let event = CursorEntered { window: entity };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::CursorLeft { .. } => {
                            if let Some(mut window) =
                                app.world_mut().get_mut::<bevy_window::Window>(entity)
                            {
                                window.set_physical_cursor_position(None);
                            }
                            let event = CursorLeft { window: entity };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::MouseInput { state, button, .. } => {
                            let event = MouseButtonInput {
                                button: converters::convert_mouse_button(button),
                                state: converters::convert_element_state(state),
                                window: entity,
                            };
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::MouseWheel { delta, phase, .. } => {
                            let phase = converters::convert_touch_phase(phase);
                            let event = match delta {
                                winit::event::MouseScrollDelta::LineDelta(x, y) => MouseWheel {
                                    unit: MouseScrollUnit::Line,
                                    x,
                                    y,
                                    window: entity,
                                    phase,
                                },
                                winit::event::MouseScrollDelta::PixelDelta(p) => MouseWheel {
                                    unit: MouseScrollUnit::Pixel,
                                    x: p.x as f32,
                                    y: p.y as f32,
                                    window: entity,
                                    phase,
                                },
                            };
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::PinchGesture { delta, .. } => {
                            let event = PinchGesture(delta as f32);
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::RotationGesture { delta, .. } => {
                            let event = RotationGesture(delta);
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::DoubleTapGesture { .. } => {
                            app.world_mut().write_message(DoubleTapGesture);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(DoubleTapGesture));
                        }
                        WinitWindowEvent::PanGesture { delta, .. } => {
                            let event = PanGesture(Vec2::new(delta.x, delta.y));
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Touch(touch) => {
                            let scale_factor = app
                                .world()
                                .get::<bevy_window::Window>(entity)
                                .map(|w| w.resolution.scale_factor() as f64)
                                .unwrap_or(1.0);
                            let location = touch.location.to_logical(scale_factor);
                            let event = converters::convert_touch_input(touch, location, entity);
                            app.world_mut().write_message(event);
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Occluded(occluded) => {
                            let event = WindowOccluded {
                                window: entity,
                                occluded,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::DroppedFile(path_buf) => {
                            let event = FileDragAndDrop::DroppedFile {
                                window: entity,
                                path_buf,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::HoveredFile(path_buf) => {
                            let event = FileDragAndDrop::HoveredFile {
                                window: entity,
                                path_buf,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::HoveredFileCancelled => {
                            let event = FileDragAndDrop::HoveredFileCanceled { window: entity };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Moved(position) => {
                            let position = ivec2(position.x, position.y);
                            if let Some(mut window) =
                                app.world_mut().get_mut::<bevy_window::Window>(entity)
                            {
                                window.position.set(position);
                            }
                            let event = WindowMoved {
                                window: entity,
                                position,
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Ime(event) => {
                            let ime_event = match event {
                                winit::event::Ime::Preedit(value, cursor) => Ime::Preedit {
                                    window: entity,
                                    value,
                                    cursor,
                                },
                                winit::event::Ime::Commit(value) => Ime::Commit {
                                    window: entity,
                                    value,
                                },
                                winit::event::Ime::Enabled => Ime::Enabled { window: entity },
                                winit::event::Ime::Disabled => Ime::Disabled { window: entity },
                            };
                            app.world_mut().write_message(ime_event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(ime_event));
                        }
                        WinitWindowEvent::ThemeChanged(theme) => {
                            let event = WindowThemeChanged {
                                window: entity,
                                theme: converters::convert_winit_theme(theme),
                            };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        WinitWindowEvent::Destroyed => {
                            let event = WindowDestroyed { window: entity };
                            app.world_mut().write_message(event.clone());
                            app.world_mut()
                                .write_message(bevy_window::WindowEvent::from(event));
                        }
                        _ => {}
                    }

                    // Mirror OS-mutated fields into `CachedWindow` so `changed_windows` won't echo them back.
                    let world = app.world_mut();
                    let os_fields = world
                        .get_entity(entity)
                        .ok()
                        .and_then(|e| e.get_ref::<bevy_window::Window>())
                        .filter(|w| w.is_changed())
                        .map(|w| {
                            (
                                w.physical_cursor_position(),
                                w.focused,
                                w.resolution.clone(),
                                w.position,
                            )
                        });
                    if let Some((cursor_position, focused, resolution, position)) = os_fields {
                        if let Some(mut cache) =
                            world.get_mut::<crate::system::CachedWindow>(entity)
                        {
                            cache.set_physical_cursor_position(
                                cursor_position.map(|p| p.as_dvec2()),
                            );
                            cache.focused = focused;
                            cache.resolution = resolution;
                            cache.position = position;
                        }
                    }
                }
                RuntimeEvent::DeviceEvent { event, .. } => {
                    received_events.device_events = true;
                    if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
                        let event = MouseMotion {
                            delta: Vec2::new(x as f32, y as f32),
                        };
                        app.world_mut().write_message(event);
                        app.world_mut()
                            .write_message(bevy_window::WindowEvent::from(event));
                    }
                }
                RuntimeEvent::MemoryWarning => {
                    trace!("Memory warning received");
                    received_events.user_events = true;
                }
                RuntimeEvent::WakeUp => {
                    trace!("WakeUp received");
                    received_events.user_events = true;
                }
                RuntimeEvent::MonitorsChanged(monitors) => {
                    trace!("Monitors changed: {} monitor(s)", monitors.len());
                    received_events.user_events = true;
                    apply_monitors_changed(app.world_mut(), monitors);
                }
            }
            None
        };

        if use_blocking {
            let timeout = deadline.saturating_duration_since(Instant::now());
            if !timeout.is_zero() {
                match event_receiver.recv_timeout(timeout) {
                    Ok(event) => {
                        if let Some(exit) = process_event(event) {
                            return exit;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {
                        trace!("Event channel disconnected, exiting ECS thread");
                        return app.should_exit().unwrap_or(AppExit::error());
                    }
                }
            }
        }

        loop {
            match event_receiver.try_recv() {
                Ok(event) => {
                    if let Some(exit) = process_event(event) {
                        return exit;
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    trace!("Event channel disconnected, exiting ECS thread");
                    return app.should_exit().unwrap_or(AppExit::error());
                }
            }
        }

        let should_update = match update_mode {
            UpdateMode::Continuous => true,
            UpdateMode::Reactive {
                react_to_device_events,
                react_to_user_events,
                react_to_window_events,
                ..
            } => {
                (react_to_window_events && received_events.window_events)
                    || (react_to_device_events && received_events.device_events)
                    || (react_to_user_events && received_events.user_events)
                    || (!received_events.window_events
                        && !received_events.device_events
                        && !received_events.user_events)
            }
        };

        let force_update = startup_forced_updates > 0;
        if force_update {
            startup_forced_updates -= 1;
        }

        if force_update || (is_resumed && should_update) {
            app.update();

            if (force_update || matches!(update_mode, UpdateMode::Continuous))
                && let Some(task_sender) = app.world().get_resource::<crate::WinitTaskSender>()
            {
                let _ = task_sender.send(|_| {
                    crate::WINIT_WINDOWS.with_borrow(|winit_windows| {
                        for window in winit_windows.windows.values() {
                            window.request_redraw();
                        }
                    });
                });
            }
        }

        if let Some(exit) = app.should_exit() {
            if let Some(task_sender) = app.world().get_resource::<crate::WinitTaskSender>() {
                let _ = task_sender.exit();
            }
            return exit;
        }
    }
}
