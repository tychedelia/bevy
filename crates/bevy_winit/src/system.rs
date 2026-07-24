use std::collections::HashMap;

use bevy_derive::{Deref, DerefMut};
#[cfg(target_arch = "wasm32")]
use bevy_ecs::system::Local;
use bevy_ecs::{
    change_detection::DetectChangesMut,
    entity::Entity,
    lifecycle::RemovedComponents,
    message::MessageWriter,
    prelude::{Changed, Commands, Component},
    system::Query,
    world::EntityWorldMut,
};
use bevy_input::keyboard::{Key, KeyCode, KeyboardFocusLost, KeyboardInput};
use bevy_window::{
    ClosingWindow, CursorOptions, OnMonitor, RawHandleWrapper, RequestRedraw, Window, WindowClosed,
    WindowClosing, WindowCreated, WindowEvent, WindowFocused, WindowMode, WindowPosition,
    WindowResized,
};
#[cfg(target_arch = "wasm32")]
use bevy_window::{WindowScaleFactorChanged, WindowWrapper};
use tracing::{error, info, warn};

use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition, PhysicalSize};

use crate::{
    accessibility::ACCESS_KIT_ADAPTERS,
    converters::{
        convert_enabled_buttons, convert_resize_direction, convert_window_level,
        convert_window_theme, convert_winit_theme,
    },
    resolve_exclusive_fullscreen, select_monitor,
    winit_monitors::WinitMonitors,
    WINIT_WINDOWS,
};
use bevy_app::AppExit;
use bevy_ecs::{prelude::MessageReader, query::With, system::Res};
#[cfg(target_os = "ios")]
use winit::platform::ios::WindowExtIOS;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

/// Check whether keyboard focus was lost. This is different from window
/// focus in that swapping between Bevy windows keeps window focus.
pub(crate) fn check_keyboard_focus_lost(
    mut window_focused_reader: MessageReader<WindowFocused>,
    mut keyboard_focus_lost_writer: MessageWriter<KeyboardFocusLost>,
    mut keyboard_input_writer: MessageWriter<KeyboardInput>,
    mut window_event_writer: MessageWriter<WindowEvent>,
    mut q_windows: Query<&mut WinitWindowPressedKeys>,
) {
    let mut focus_lost = vec![];
    let mut focus_gained = false;
    for e in window_focused_reader.read() {
        if e.focused {
            focus_gained = true;
        } else {
            focus_lost.push(e.window);
        }
    }

    if !focus_gained {
        if !focus_lost.is_empty() {
            window_event_writer.write(WindowEvent::KeyboardFocusLost(KeyboardFocusLost));
            keyboard_focus_lost_writer.write(KeyboardFocusLost);
        }

        for window in focus_lost {
            let Ok(mut pressed_keys) = q_windows.get_mut(window) else {
                continue;
            };
            for (key_code, logical_key) in pressed_keys.0.drain() {
                let event = KeyboardInput {
                    key_code,
                    logical_key,
                    state: bevy_input::ButtonState::Released,
                    repeat: false,
                    window,
                    text: None,
                };
                window_event_writer.write(WindowEvent::KeyboardInput(event.clone()));
                keyboard_input_writer.write(event);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn despawn_windows(
    closing: Query<Entity, With<ClosingWindow>>,
    mut closed: RemovedComponents<Window>,
    window_entities: Query<Entity, With<Window>>,
    mut closing_event_writer: MessageWriter<WindowClosing>,
    mut closed_event_writer: MessageWriter<WindowClosed>,
    mut windows_to_drop: Local<Vec<WindowWrapper<winit::window::Window>>>,
    mut exit_event_reader: MessageReader<AppExit>,
) {
    // Drop all the windows that are waiting to be closed
    windows_to_drop.clear();
    for window in closing.iter() {
        closing_event_writer.write(WindowClosing { window });
    }
    for window in closed.read() {
        info!("Closing window {}", window);
        // Guard to verify that the window is in fact actually gone,
        // rather than having the component added
        // and removed in the same frame.
        if !window_entities.contains(window) {
            WINIT_WINDOWS.with_borrow_mut(|winit_windows| {
                if let Some((wrapper, _winit_id)) = winit_windows.remove_window(window) {
                    // Keeping WindowWrapper that are dropped for one frame
                    // Otherwise the last `Arc` of the window could be in the rendering thread, and dropped there
                    // This would hang on macOS
                    // Keeping the wrapper and dropping it next frame in this system ensure its dropped in the main thread
                    windows_to_drop.push(wrapper);
                }
            });
            closed_event_writer.write(WindowClosed { window });
        }
    }

    // On macOS, many things need to be dropped on the main thread, or the app will hang:
    // - notify the rendering thread the windows are about to close
    // - take the `WindowWrapper`s out of `WINIT_WINDOWS` and into the local `windows_to_drop`
    if !exit_event_reader.is_empty() {
        exit_event_reader.clear();
        WINIT_WINDOWS.with_borrow_mut(|winit_windows| {
            for window in window_entities.iter() {
                closing_event_writer.write(WindowClosing { window });
                if let Some(wrapper) = winit_windows.remove_window(window) {
                    windows_to_drop.push(wrapper);
                }
            }
        });
    }
}

/// Despawns closed windows.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn despawn_windows(
    closing: Query<Entity, With<ClosingWindow>>,
    mut closed: RemovedComponents<Window>,
    window_entities: Query<Entity, With<Window>>,
    mut closing_event_writer: MessageWriter<WindowClosing>,
    mut closed_event_writer: MessageWriter<WindowClosed>,
    mut exit_event_reader: MessageReader<AppExit>,
    task_sender: Res<crate::WinitTaskSender>,
    window_entity_map: Res<crate::WindowEntityMap>,
) {
    let _ = task_sender.scoped(|_| {
        crate::WINDOWS_TO_DROP.with_borrow_mut(Vec::clear);
    });

    for window in closing.iter() {
        closing_event_writer.write(WindowClosing { window });
    }

    for window in closed.read() {
        info!("Closing window {}", window);
        if !window_entities.contains(window) {
            let removed_window_id = task_sender.scoped(move |_event_loop| {
                WINIT_WINDOWS.with_borrow_mut(|winit_windows| {
                    if let Some((wrapper, winit_id)) = winit_windows.remove_window(window) {
                        crate::WINDOWS_TO_DROP.with_borrow_mut(|queue| queue.push(wrapper));
                        Some(winit_id)
                    } else {
                        None
                    }
                })
            });

            if let Ok(Some(winit_id)) = removed_window_id {
                window_entity_map.remove(winit_id);
            }

            closed_event_writer.write(WindowClosed { window });
        }
    }

    if !exit_event_reader.is_empty() {
        exit_event_reader.clear();
        for window in window_entities.iter() {
            closing_event_writer.write(WindowClosing { window });
        }
    }
}

/// The cached state of the window so we can check which properties were changed from within the app.
#[derive(Debug, Clone, Component, Deref, DerefMut)]
pub(crate) struct CachedWindow(Window);

/// The cached state of the window so we can check which properties were changed from within the app.
#[derive(Debug, Clone, Component, Deref, DerefMut)]
pub(crate) struct CachedCursorOptions(CursorOptions);

/// Propagates changes from [`Window`] entities to the [`winit`] backend.
///
/// # Notes
///
/// - [`Window::present_mode`] and [`Window::composite_alpha_mode`] changes are handled by the `bevy_render` crate.
/// - [`Window::transparent`] cannot be changed after the window is created.
/// - [`Window::canvas`] cannot be changed after the window is created.
/// - [`Window::focused`] cannot be manually changed to `false` after the window is created.
#[cfg(target_arch = "wasm32")]
pub(crate) fn changed_windows(
    mut commands: Commands,
    mut changed_windows: Query<
        (Entity, &mut Window, &mut CachedWindow, Option<&OnMonitor>),
        Changed<Window>,
    >,
    monitors: Res<WinitMonitors>,
    mut window_resized: MessageWriter<WindowResized>,
    mut window_event: MessageWriter<WindowEvent>,
    mut window_rescaled: MessageWriter<WindowScaleFactorChanged>,
) {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        for (entity, mut window, mut cache, monitor_relationship) in &mut changed_windows {
            let Some(winit_window) = winit_windows.get_window(entity) else {
                continue;
            };

            if window.title != cache.title {
                winit_window.set_title(window.title.as_str());
            }

            if window.mode != cache.mode {
                let new_mode = match window.mode {
                    WindowMode::BorderlessFullscreen(monitor_selection) => {
                        Some(Some(winit::window::Fullscreen::Borderless(select_monitor(
                            &monitors,
                            winit_window.primary_monitor(),
                            winit_window.current_monitor(),
                            &monitor_selection,
                        ))))
                    }
                    WindowMode::Fullscreen(monitor_selection, video_mode_selection) => {
                        let monitor = select_monitor(
                            &monitors,
                            winit_window.primary_monitor(),
                            winit_window.current_monitor(),
                            &monitor_selection,
                        );
                        Some(Some(resolve_exclusive_fullscreen(
                            monitor,
                            monitor_selection,
                            video_mode_selection,
                        )))
                    }
                    WindowMode::Windowed => Some(None),
                };

                if let Some(new_mode) = new_mode
                    && winit_window.fullscreen() != new_mode {
                        winit_window.set_fullscreen(new_mode);
                    }
            }

            // Set position before size so the window is on the correct monitor
            // (and thus using the correct scale factor) when size is applied.
            if window.position != cache.position
                && let Some(position) = crate::winit_window_position(
                    &window.position,
                    &window.resolution,
                    &monitors,
                    winit_window.primary_monitor(),
                    winit_window.current_monitor(),
                ) {
                    let should_set = match winit_window.outer_position() {
                        Ok(current_position) => current_position != position,
                        _ => true,
                    };

                    if should_set {
                        winit_window.set_outer_position(position);
                    }
                }

            if window.resolution != cache.resolution {
                let cache_physical_size = PhysicalSize::new(
                    cache.resolution.physical_width(),
                    cache.resolution.physical_height(),
                );
                let requested_physical_size = PhysicalSize::new(
                    window.resolution.physical_width(),
                    window.resolution.physical_height(),
                );

                if cache_physical_size != requested_physical_size {
                    // In `None` case, the request will be handled by winit::event::WindowEvent::Resized
                    if let Some(new_physical_size) = winit_window.request_inner_size(requested_physical_size) {
                        let event = react_to_resize(entity, &mut window, new_physical_size);
                        // Need to send two very similar events because different systems rely on those.
                        window_resized.write(event.clone());
                        window_event.write(event.into());
                    }
                }

                let cache_scale_factor = cache.scale_factor();
                let requested_scale_factor = window.scale_factor();

                if cache_scale_factor != requested_scale_factor {
                    // If the scale factor has changed we don't query anything from winit, but send events for camera system to handle.
                    let event = WindowScaleFactorChanged { scale_factor: requested_scale_factor as f64, window: entity};
                    // Need to send two very similar events because different systems rely on those.
                    window_rescaled.write(event.clone());
                    window_event.write(event.into());
                }
            }

            if window.physical_cursor_position() != cache.physical_cursor_position()
                && let Some(physical_position) = window.physical_cursor_position() {
                    let position = PhysicalPosition::new(physical_position.x, physical_position.y);

                    if let Err(err) = winit_window.set_cursor_position(position) {
                        error!("could not set cursor position: {}", err);
                    }
                }

            if window.decorations != cache.decorations
                && window.decorations != winit_window.is_decorated()
            {
                winit_window.set_decorations(window.decorations);
            }

            if window.resizable != cache.resizable
                && window.resizable != winit_window.is_resizable()
            {
                winit_window.set_resizable(window.resizable);
            }

            if window.enabled_buttons != cache.enabled_buttons {
                winit_window.set_enabled_buttons(convert_enabled_buttons(window.enabled_buttons));
            }

            if window.resize_constraints != cache.resize_constraints {
                let constraints = window.resize_constraints.check_constraints();
                let min_inner_size = LogicalSize {
                    width: constraints.min_width,
                    height: constraints.min_height,
                };
                let max_inner_size = LogicalSize {
                    width: constraints.max_width,
                    height: constraints.max_height,
                };

                winit_window.set_min_inner_size(Some(min_inner_size));
                winit_window.set_max_inner_size(
                    if constraints.max_width.is_finite() && constraints.max_height.is_finite() {
                        Some(max_inner_size)
                    } else {
                        None
                    },
                );
            }

            if let Some(monitor_link) = monitor_relationship {
                if let Some(winit_monitor) = winit_window.current_monitor() {
                    if let Some(linked_monitor) = monitors.find_entity(monitor_link.0) &&
                        winit_monitor != linked_monitor &&
                        let Some((_, winit_monitor_entity)) = monitors.monitors.iter().find(|(h, _)| h == &winit_monitor) {
                        commands.entity(entity).insert(OnMonitor(winit_monitor_entity.to_owned()));
                    }
                } else {
                    commands.entity(entity).remove::<OnMonitor>();
                }
            } else {
                if let Some(winit_monitor) = winit_window.current_monitor()
                    && let Some((_, winit_monitor_entity)) = monitors.monitors.iter()
                    .find(|(h, _)| h == &winit_monitor) {
                    commands.entity(entity).insert(OnMonitor(winit_monitor_entity.to_owned()));
                }
            }

            if let Some(maximized) = window.internal.take_maximize_request() {
                winit_window.set_maximized(maximized);
            }

            if let Some(minimized) = window.internal.take_minimize_request() {
                winit_window.set_minimized(minimized);
            }

            if window.internal.take_move_request()
                && let Err(e) = winit_window.drag_window() {
                    warn!("Winit returned an error while attempting to drag the window: {e}");
                }

            if let Some(resize_direction) = window.internal.take_resize_request()
                && let Err(e) =
                    winit_window.drag_resize_window(convert_resize_direction(resize_direction))
                {
                    warn!("Winit returned an error while attempting to drag resize the window: {e}");
                }

            if window.focused != cache.focused && window.focused {
                winit_window.focus_window();
            }

            if window.window_level != cache.window_level {
                winit_window.set_window_level(convert_window_level(window.window_level));
            }

            // Currently unsupported changes
            if window.transparent != cache.transparent {
                window.transparent = cache.transparent;
                warn!("Winit does not currently support updating transparency after window creation.");
            }

            #[cfg(target_arch = "wasm32")]
            if window.canvas != cache.canvas {
                window.canvas.clone_from(&cache.canvas);
                warn!(
                    "Bevy currently doesn't support modifying the window canvas after initialization."
                );
            }

            if window.ime_enabled != cache.ime_enabled {
                winit_window.set_ime_allowed(window.ime_enabled);
            }

            if window.ime_position != cache.ime_position {
                winit_window.set_ime_cursor_area(
                    LogicalPosition::new(window.ime_position.x, window.ime_position.y),
                    PhysicalSize::new(10, 10),
                );
            }

            if window.window_theme != cache.window_theme {
                winit_window.set_theme(window.window_theme.map(convert_window_theme));
            }

            if window.visible != cache.visible {
                winit_window.set_visible(window.visible);
            }

            #[cfg(target_os = "ios")]
            {
                if window.recognize_pinch_gesture != cache.recognize_pinch_gesture {
                    winit_window.recognize_pinch_gesture(window.recognize_pinch_gesture);
                }
                if window.recognize_rotation_gesture != cache.recognize_rotation_gesture {
                    winit_window.recognize_rotation_gesture(window.recognize_rotation_gesture);
                }
                if window.recognize_doubletap_gesture != cache.recognize_doubletap_gesture {
                    winit_window.recognize_doubletap_gesture(window.recognize_doubletap_gesture);
                }
                if window.recognize_pan_gesture != cache.recognize_pan_gesture {
                    match (
                        window.recognize_pan_gesture,
                        cache.recognize_pan_gesture,
                    ) {
                        (Some(_), Some(_)) => {
                            warn!("Bevy currently doesn't support modifying PanGesture number of fingers recognition. Please disable it before re-enabling it with the new number of fingers");
                        }
                        (Some((min, max)), _) => winit_window.recognize_pan_gesture(true, min, max),
                        _ => winit_window.recognize_pan_gesture(false, 0, 0),
                    }
                }

                if window.prefers_home_indicator_hidden != cache.prefers_home_indicator_hidden {
                    winit_window
                        .set_prefers_home_indicator_hidden(window.prefers_home_indicator_hidden);
                }
                if window.prefers_status_bar_hidden != cache.prefers_status_bar_hidden {
                    winit_window.set_prefers_status_bar_hidden(window.prefers_status_bar_hidden);
                }
                if window.preferred_screen_edges_deferring_system_gestures
                    != cache
                        .preferred_screen_edges_deferring_system_gestures
                {
                    use crate::converters::convert_screen_edge;
                    let preferred_edge =
                        convert_screen_edge(window.preferred_screen_edges_deferring_system_gestures);
                    winit_window.set_preferred_screen_edges_deferring_system_gestures(preferred_edge);
                }
            }
            **cache = window.clone();
        }
    });
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn changed_cursor_options(
    mut changed_windows: Query<
        (
            Entity,
            &Window,
            &mut CursorOptions,
            &mut CachedCursorOptions,
        ),
        Changed<CursorOptions>,
    >,
) {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        for (entity, window, mut cursor_options, mut cache) in &mut changed_windows {
            // This system already only runs when the cursor options change, so we need to bypass change detection or the next frame will also run this system
            let cursor_options = cursor_options.bypass_change_detection();
            let Some(winit_window) = winit_windows.get_window(entity) else {
                continue;
            };
            // Don't check the cache for the grab mode. It can change through external means, leaving the cache outdated.
            if let Err(err) =
                crate::winit_windows::attempt_grab(winit_window, cursor_options.grab_mode)
            {
                warn!(
                    "Could not set cursor grab mode for window {}: {}",
                    window.title, err
                );
                cursor_options.grab_mode = cache.grab_mode;
            } else {
                cache.grab_mode = cursor_options.grab_mode;
            }

            if cursor_options.visible != cache.visible {
                winit_window.set_cursor_visible(cursor_options.visible);
                cache.visible = cursor_options.visible;
            }

            if cursor_options.hit_test != cache.hit_test {
                if let Err(err) = winit_window.set_cursor_hittest(cursor_options.hit_test) {
                    warn!(
                        "Could not set cursor hit test for window {}: {}",
                        window.title, err
                    );
                    cursor_options.hit_test = cache.hit_test;
                } else {
                    cache.hit_test = cursor_options.hit_test;
                }
            }
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
struct CursorOptionsResult {
    window_found: bool,
    grab_success: bool,
    hit_test_success: bool,
}

/// Syncs changed `CursorOptions` to the winit backend.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn changed_cursor_options(
    mut changed_windows: Query<
        (
            Entity,
            &Window,
            &mut CursorOptions,
            &mut CachedCursorOptions,
        ),
        Changed<CursorOptions>,
    >,
    task_sender: Res<crate::WinitTaskSender>,
) {
    for (entity, window, mut cursor_options, mut cache) in &mut changed_windows {
        let cursor_options = cursor_options.bypass_change_detection();
        let grab_mode = cursor_options.grab_mode;
        let visible = cursor_options.visible;
        let hit_test = cursor_options.hit_test;
        let cache_visible = cache.visible;
        let cache_hit_test = cache.hit_test;

        let result = match task_sender.scoped(move |_event_loop| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                let Some(winit_window) = winit_windows.get_window(entity) else {
                    return CursorOptionsResult {
                        window_found: false,
                        grab_success: false,
                        hit_test_success: false,
                    };
                };

                let grab_success =
                    crate::winit_windows::attempt_grab(winit_window, grab_mode).is_ok();

                if visible != cache_visible {
                    winit_window.set_cursor_visible(visible);
                }

                let hit_test_success = if hit_test != cache_hit_test {
                    winit_window.set_cursor_hittest(hit_test).is_ok()
                } else {
                    true
                };

                CursorOptionsResult {
                    window_found: true,
                    grab_success,
                    hit_test_success,
                }
            })
        }) {
            Ok(result) => result,
            Err(err) => {
                error!("Failed to send cursor options update task: {}", err);
                continue;
            }
        };

        if !result.window_found {
            continue;
        }

        if result.grab_success {
            cache.grab_mode = grab_mode;
        } else {
            warn!("Could not set cursor grab mode for window {}", window.title);
            cursor_options.grab_mode = cache.grab_mode;
        }

        cache.visible = visible;

        if result.hit_test_success {
            cache.hit_test = hit_test;
        } else {
            warn!("Could not set cursor hit test for window {}", window.title);
            cursor_options.hit_test = cache.hit_test;
        }
    }
}

/// This keeps track of which keys are pressed on each window.
/// When a window is unfocused, this is used to send key release events for all the currently held keys.
#[derive(Default, Component)]
pub struct WinitWindowPressedKeys(pub(crate) HashMap<KeyCode, Key>);

/// Result of creating a window on the winit thread.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct WindowCreationResult {
    pub window_id: winit::window::WindowId,
    pub scale_factor: f64,
    pub theme: Option<bevy_window::WindowTheme>,
    pub raw_handle: Option<RawHandleWrapper>,
}

/// Creates new windows.
#[cfg(not(target_arch = "wasm32"))]
pub fn create_windows(
    mut commands: Commands,
    created_windows: Query<
        (
            Entity,
            &Window,
            &CursorOptions,
            Option<&bevy_window::RawHandleWrapperHolder>,
        ),
        bevy_ecs::prelude::Added<Window>,
    >,
    mut window_created_events: MessageWriter<WindowCreated>,
    handlers: Res<crate::accessibility::WinitActionRequestHandlers>,
    accessibility_requested: Res<bevy_a11y::AccessibilityRequested>,
    monitors: Res<WinitMonitors>,
    task_sender: Res<crate::WinitTaskSender>,
    window_entity_map: Res<crate::WindowEntityMap>,
) {
    for (entity, window, cursor_options, handle_holder) in &created_windows {
        let already_exists =
            WINIT_WINDOWS.with_borrow(|winit_windows| winit_windows.get_window(entity).is_some());
        if already_exists {
            continue;
        }

        info!("Creating new window {} ({})", window.title.as_str(), entity);

        let window_data = window.clone();
        let cursor_data = cursor_options.clone();
        let monitors_data = (*monitors).clone();
        let accessibility_data = (*accessibility_requested).clone();
        let mut handlers_clone =
            crate::accessibility::WinitActionRequestHandlers((*handlers).clone());

        let result = match task_sender.scoped(move |event_loop| {
            // WinitMonitors may be empty at startup, so select from the event loop's live list.
            let mut live_monitors = monitors_data.clone();
            live_monitors.monitors = event_loop
                .available_monitors()
                .map(|handle| {
                    let monitor_entity = monitors_data
                        .monitors
                        .iter()
                        .find(|(existing, _)| existing == &handle)
                        .map(|(_, mapped)| *mapped)
                        .unwrap_or(Entity::PLACEHOLDER);
                    (handle, monitor_entity)
                })
                .collect();
            WINIT_WINDOWS.with_borrow_mut(|winit_windows| {
                ACCESS_KIT_ADAPTERS.with_borrow_mut(|adapters| {
                    let winit_window = winit_windows.create_window(
                        event_loop,
                        entity,
                        &window_data,
                        &cursor_data,
                        adapters,
                        &mut handlers_clone,
                        &accessibility_data,
                        &live_monitors,
                    );

                    let window_id = winit_window.id();
                    let scale_factor = winit_window.scale_factor();
                    let theme = winit_window.theme().map(convert_winit_theme);
                    let raw_handle = RawHandleWrapper::new(winit_window).ok();

                    WindowCreationResult {
                        window_id,
                        scale_factor,
                        theme,
                        raw_handle,
                    }
                })
            })
        }) {
            Ok(result) => result,
            Err(err) => {
                error!("Failed to create window: {}", err);
                continue;
            }
        };

        window_entity_map.insert(result.window_id, entity);

        let scale_factor = result.scale_factor;
        let theme = result.theme;
        commands
            .entity(entity)
            .queue(move |mut entity: EntityWorldMut| {
                if let Some(mut window) = entity.get_mut::<Window>() {
                    if let Some(theme) = theme {
                        window.window_theme = Some(theme);
                    }
                    window
                        .resolution
                        .set_scale_factor_and_apply_to_physical_size(scale_factor as f32);
                }
            });

        let mut cached = window.clone();
        if let Some(theme) = result.theme {
            cached.window_theme = Some(theme);
        }
        cached
            .resolution
            .set_scale_factor_and_apply_to_physical_size(result.scale_factor as f32);

        commands.entity(entity).insert((
            CachedWindow(cached),
            CachedCursorOptions(cursor_options.clone()),
            WinitWindowPressedKeys::default(),
        ));

        if let Some(raw_handle) = result.raw_handle {
            commands.entity(entity).insert(raw_handle.clone());
            if let Some(holder) = handle_holder {
                *holder.0.lock().unwrap() = Some(raw_handle);
            }
        }

        window_created_events.write(WindowCreated { window: entity });
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct WindowUpdateResult {
    /// If false, the window was missing; leave the cache untouched.
    window_found: bool,
    /// Actual applied size; may differ from the request when the OS constrains it.
    actual_resize: Option<(u32, u32)>,
    current_monitor: Option<winit::monitor::MonitorHandle>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct ResolutionChangeData {
    physical_width: u32,
    physical_height: u32,
    cached_physical_width: u32,
    cached_physical_height: u32,
    base_scale_factor: f32,
    scale_factor: f32,
    cached_scale_factor: f32,
    scale_factor_override: Option<f32>,
    cached_scale_factor_override: Option<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct WindowUpdateData {
    entity: Entity,
    title: Option<String>,
    mode: Option<WindowMode>,
    resolution_change: Option<ResolutionChangeData>,
    cursor_position: Option<(f32, f32)>,
    decorations: Option<bool>,
    resizable: Option<bool>,
    enabled_buttons: Option<bevy_window::EnabledButtons>,
    resize_constraints: Option<bevy_window::WindowResizeConstraints>,
    position: Option<(WindowPosition, bevy_window::WindowResolution)>,
    maximize_request: Option<bool>,
    minimize_request: Option<bool>,
    move_request: bool,
    resize_request: Option<bevy_math::CompassOctant>,
    focused: Option<bool>,
    window_level: Option<bevy_window::WindowLevel>,
    ime_enabled: Option<bool>,
    ime_position: Option<bevy_math::Vec2>,
    window_theme: Option<Option<bevy_window::WindowTheme>>,
    visible: Option<bool>,
    #[cfg(target_os = "ios")]
    recognize_pinch_gesture: Option<bool>,
    #[cfg(target_os = "ios")]
    recognize_rotation_gesture: Option<bool>,
    #[cfg(target_os = "ios")]
    recognize_doubletap_gesture: Option<bool>,
    #[cfg(target_os = "ios")]
    recognize_pan_gesture: Option<Option<(u8, u8)>>,
    #[cfg(target_os = "ios")]
    prefers_home_indicator_hidden: Option<bool>,
    #[cfg(target_os = "ios")]
    prefers_status_bar_hidden: Option<bool>,
    #[cfg(target_os = "ios")]
    preferred_screen_edges_deferring_system_gestures: Option<bevy_window::ScreenEdge>,
}

/// Syncs changed `Window` components to the winit backend.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn changed_windows(
    mut commands: Commands,
    mut changed_windows: Query<
        (Entity, &mut Window, &mut CachedWindow, Option<&OnMonitor>),
        Changed<Window>,
    >,
    monitors: Res<WinitMonitors>,
    task_sender: Res<crate::WinitTaskSender>,
    mut window_resized: MessageWriter<WindowResized>,
    mut window_event: MessageWriter<WindowEvent>,
) {
    for (entity, mut window, mut cache, monitor_relationship) in &mut changed_windows {
        let mut update = WindowUpdateData {
            entity,
            title: None,
            mode: None,
            resolution_change: None,
            cursor_position: None,
            decorations: None,
            resizable: None,
            enabled_buttons: None,
            resize_constraints: None,
            position: None,
            maximize_request: None,
            minimize_request: None,
            move_request: false,
            resize_request: None,
            focused: None,
            window_level: None,
            ime_enabled: None,
            ime_position: None,
            window_theme: None,
            visible: None,
            #[cfg(target_os = "ios")]
            recognize_pinch_gesture: None,
            #[cfg(target_os = "ios")]
            recognize_rotation_gesture: None,
            #[cfg(target_os = "ios")]
            recognize_doubletap_gesture: None,
            #[cfg(target_os = "ios")]
            recognize_pan_gesture: None,
            #[cfg(target_os = "ios")]
            prefers_home_indicator_hidden: None,
            #[cfg(target_os = "ios")]
            prefers_status_bar_hidden: None,
            #[cfg(target_os = "ios")]
            preferred_screen_edges_deferring_system_gestures: None,
        };

        let mut has_changes = false;

        if window.title != cache.title {
            update.title = Some(window.title.clone());
            has_changes = true;
        }

        if window.mode != cache.mode {
            update.mode = Some(window.mode);
            has_changes = true;
        }

        if window.resolution != cache.resolution {
            update.resolution_change = Some(ResolutionChangeData {
                physical_width: window.resolution.physical_width(),
                physical_height: window.resolution.physical_height(),
                cached_physical_width: cache.physical_width(),
                cached_physical_height: cache.physical_height(),
                base_scale_factor: window.resolution.base_scale_factor(),
                scale_factor: window.scale_factor(),
                cached_scale_factor: cache.scale_factor(),
                scale_factor_override: window.resolution.scale_factor_override(),
                cached_scale_factor_override: cache.resolution.scale_factor_override(),
            });
            has_changes = true;
        }

        if window.physical_cursor_position() != cache.physical_cursor_position()
            && let Some(pos) = window.physical_cursor_position()
        {
            update.cursor_position = Some((pos.x, pos.y));
            has_changes = true;
        }

        if window.decorations != cache.decorations {
            update.decorations = Some(window.decorations);
            has_changes = true;
        }

        if window.resizable != cache.resizable {
            update.resizable = Some(window.resizable);
            has_changes = true;
        }

        if window.enabled_buttons != cache.enabled_buttons {
            update.enabled_buttons = Some(window.enabled_buttons);
            has_changes = true;
        }

        if window.resize_constraints != cache.resize_constraints {
            update.resize_constraints = Some(window.resize_constraints);
            has_changes = true;
        }

        if window.position != cache.position {
            update.position = Some((window.position, window.resolution.clone()));
            has_changes = true;
        }

        if let Some(maximized) = window.internal.take_maximize_request() {
            update.maximize_request = Some(maximized);
            has_changes = true;
        }

        if let Some(minimized) = window.internal.take_minimize_request() {
            update.minimize_request = Some(minimized);
            has_changes = true;
        }

        if window.internal.take_move_request() {
            update.move_request = true;
            has_changes = true;
        }

        if let Some(direction) = window.internal.take_resize_request() {
            update.resize_request = Some(direction);
            has_changes = true;
        }

        // Focus can only be set to `true`, not cleared.
        if window.focused != cache.focused && window.focused {
            update.focused = Some(true);
            has_changes = true;
        }

        if window.window_level != cache.window_level {
            update.window_level = Some(window.window_level);
            has_changes = true;
        }

        if window.ime_enabled != cache.ime_enabled {
            update.ime_enabled = Some(window.ime_enabled);
            has_changes = true;
        }

        if window.ime_position != cache.ime_position {
            update.ime_position = Some(window.ime_position);
            has_changes = true;
        }

        if window.window_theme != cache.window_theme {
            update.window_theme = Some(window.window_theme);
            has_changes = true;
        }

        if window.visible != cache.visible {
            update.visible = Some(window.visible);
            has_changes = true;
        }

        #[cfg(target_os = "ios")]
        {
            if window.recognize_pinch_gesture != cache.recognize_pinch_gesture {
                update.recognize_pinch_gesture = Some(window.recognize_pinch_gesture);
                has_changes = true;
            }
            if window.recognize_rotation_gesture != cache.recognize_rotation_gesture {
                update.recognize_rotation_gesture = Some(window.recognize_rotation_gesture);
                has_changes = true;
            }
            if window.recognize_doubletap_gesture != cache.recognize_doubletap_gesture {
                update.recognize_doubletap_gesture = Some(window.recognize_doubletap_gesture);
                has_changes = true;
            }
            if window.recognize_pan_gesture != cache.recognize_pan_gesture {
                update.recognize_pan_gesture = Some(window.recognize_pan_gesture);
                has_changes = true;
            }
            if window.prefers_home_indicator_hidden != cache.prefers_home_indicator_hidden {
                update.prefers_home_indicator_hidden = Some(window.prefers_home_indicator_hidden);
                has_changes = true;
            }
            if window.prefers_status_bar_hidden != cache.prefers_status_bar_hidden {
                update.prefers_status_bar_hidden = Some(window.prefers_status_bar_hidden);
                has_changes = true;
            }
            if window.preferred_screen_edges_deferring_system_gestures
                != cache.preferred_screen_edges_deferring_system_gestures
            {
                update.preferred_screen_edges_deferring_system_gestures =
                    Some(window.preferred_screen_edges_deferring_system_gestures);
                has_changes = true;
            }
        }

        if window.transparent != cache.transparent {
            window.transparent = cache.transparent;
            warn!("Winit does not currently support updating transparency after window creation.");
        }

        #[cfg(target_arch = "wasm32")]
        if window.canvas != cache.canvas {
            window.canvas.clone_from(&cache.canvas);
            warn!(
                "Bevy currently doesn't support modifying the window canvas after initialization."
            );
        }

        if !has_changes {
            continue;
        }

        let monitors_clone = (*monitors).clone();

        let result = task_sender.scoped(move |_event_loop| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                let Some(winit_window) = winit_windows.get_window(update.entity) else {
                    return WindowUpdateResult::default();
                };

                let mut result = WindowUpdateResult {
                    window_found: true,
                    actual_resize: None,
                    current_monitor: None,
                };

                if let Some(title) = update.title {
                    winit_window.set_title(&title);
                }

                if let Some(mode) = update.mode {
                    let new_mode = match mode {
                        WindowMode::BorderlessFullscreen(monitor_selection) => {
                            Some(Some(winit::window::Fullscreen::Borderless(select_monitor(
                                &monitors_clone,
                                winit_window.primary_monitor(),
                                winit_window.current_monitor(),
                                &monitor_selection,
                            ))))
                        }
                        WindowMode::Fullscreen(monitor_selection, video_mode_selection) => {
                            let monitor = select_monitor(
                                &monitors_clone,
                                winit_window.primary_monitor(),
                                winit_window.current_monitor(),
                                &monitor_selection,
                            );
                            Some(Some(resolve_exclusive_fullscreen(
                                monitor,
                                monitor_selection,
                                video_mode_selection,
                            )))
                        }
                        WindowMode::Windowed => Some(None),
                    };

                    if let Some(new_mode) = new_mode
                        && winit_window.fullscreen() != new_mode
                    {
                        winit_window.set_fullscreen(new_mode);
                    }
                }

                // Set position before size so the window is on the correct monitor
                // (and thus using the correct scale factor) when size is applied.
                if let Some((position, resolution)) = update.position
                    && let Some(position) = crate::winit_window_position(
                        &position,
                        &resolution,
                        &monitors_clone,
                        winit_window.primary_monitor(),
                        winit_window.current_monitor(),
                    )
                {
                    let should_set = match winit_window.outer_position() {
                        Ok(current_position) => current_position != position,
                        _ => true,
                    };
                    if should_set {
                        winit_window.set_outer_position(position);
                    }
                }

                if let Some(res) = update.resolution_change {
                    let mut physical_size =
                        PhysicalSize::new(res.physical_width, res.physical_height);
                    let cached_physical_size =
                        PhysicalSize::new(res.cached_physical_width, res.cached_physical_height);

                    if res.scale_factor != res.cached_scale_factor && !winit_window.is_maximized() {
                        let logical_size =
                            if let Some(cached_factor) = res.cached_scale_factor_override {
                                physical_size.to_logical::<f32>(cached_factor as f64)
                            } else {
                                physical_size.to_logical::<f32>(res.base_scale_factor as f64)
                            };

                        physical_size = if let Some(forced_factor) = res.scale_factor_override {
                            logical_size.to_physical::<u32>(forced_factor as f64)
                        } else {
                            logical_size.to_physical::<u32>(res.base_scale_factor as f64)
                        };
                    }

                    if physical_size != cached_physical_size {
                        if let Some(actual_size) = winit_window.request_inner_size(physical_size) {
                            result.actual_resize = Some((actual_size.width, actual_size.height));
                        }
                    }
                }

                if let Some((x, y)) = update.cursor_position {
                    let position = PhysicalPosition::new(x, y);
                    if let Err(err) = winit_window.set_cursor_position(position) {
                        error!("could not set cursor position: {}", err);
                    }
                }

                if let Some(decorations) = update.decorations {
                    winit_window.set_decorations(decorations);
                }

                if let Some(resizable) = update.resizable {
                    winit_window.set_resizable(resizable);
                }

                if let Some(enabled_buttons) = update.enabled_buttons {
                    winit_window.set_enabled_buttons(convert_enabled_buttons(enabled_buttons));
                }

                if let Some(constraints) = update.resize_constraints {
                    let constraints = constraints.check_constraints();
                    let min_inner_size = LogicalSize {
                        width: constraints.min_width,
                        height: constraints.min_height,
                    };
                    let max_inner_size = LogicalSize {
                        width: constraints.max_width,
                        height: constraints.max_height,
                    };
                    winit_window.set_min_inner_size(Some(min_inner_size));
                    if constraints.max_width.is_finite() && constraints.max_height.is_finite() {
                        winit_window.set_max_inner_size(Some(max_inner_size));
                    }
                }

                if let Some(maximized) = update.maximize_request {
                    winit_window.set_maximized(maximized);
                }

                if let Some(minimized) = update.minimize_request {
                    winit_window.set_minimized(minimized);
                }

                if update.move_request
                    && let Err(e) = winit_window.drag_window()
                {
                    warn!("Winit returned an error while attempting to drag the window: {e}");
                }

                if let Some(direction) = update.resize_request
                    && let Err(e) =
                        winit_window.drag_resize_window(convert_resize_direction(direction))
                {
                    warn!(
                        "Winit returned an error while attempting to drag resize the window: {e}"
                    );
                }

                if update.focused == Some(true) {
                    winit_window.focus_window();
                }

                if let Some(level) = update.window_level {
                    winit_window.set_window_level(convert_window_level(level));
                }

                if let Some(ime_enabled) = update.ime_enabled {
                    winit_window.set_ime_allowed(ime_enabled);
                }

                if let Some(ime_position) = update.ime_position {
                    winit_window.set_ime_cursor_area(
                        LogicalPosition::new(ime_position.x, ime_position.y),
                        PhysicalSize::new(10, 10),
                    );
                }

                if let Some(theme) = update.window_theme {
                    winit_window.set_theme(theme.map(convert_window_theme));
                }

                if let Some(visible) = update.visible {
                    winit_window.set_visible(visible);
                }

                #[cfg(target_os = "ios")]
                {
                    if let Some(recognize) = update.recognize_pinch_gesture {
                        winit_window.recognize_pinch_gesture(recognize);
                    }
                    if let Some(recognize) = update.recognize_rotation_gesture {
                        winit_window.recognize_rotation_gesture(recognize);
                    }
                    if let Some(recognize) = update.recognize_doubletap_gesture {
                        winit_window.recognize_doubletap_gesture(recognize);
                    }
                    if let Some(pan_gesture) = update.recognize_pan_gesture {
                        match pan_gesture {
                            Some((min, max)) => winit_window.recognize_pan_gesture(true, min, max),
                            None => winit_window.recognize_pan_gesture(false, 0, 0),
                        }
                    }
                    if let Some(hidden) = update.prefers_home_indicator_hidden {
                        winit_window.set_prefers_home_indicator_hidden(hidden);
                    }
                    if let Some(hidden) = update.prefers_status_bar_hidden {
                        winit_window.set_prefers_status_bar_hidden(hidden);
                    }
                    if let Some(edge) = update.preferred_screen_edges_deferring_system_gestures {
                        use crate::converters::convert_screen_edge;
                        winit_window.set_preferred_screen_edges_deferring_system_gestures(
                            convert_screen_edge(edge),
                        );
                    }
                }

                result.current_monitor = winit_window.current_monitor();

                result
            })
        });

        match result {
            Ok(update_result) => {
                if update_result.window_found {
                    **cache = window.clone();

                    if let Some((actual_width, actual_height)) = update_result.actual_resize {
                        let requested_width = window.resolution.physical_width();
                        let requested_height = window.resolution.physical_height();

                        if actual_width != requested_width || actual_height != requested_height {
                            window
                                .resolution
                                .set_physical_resolution(actual_width, actual_height);
                            cache
                                .resolution
                                .set_physical_resolution(actual_width, actual_height);
                        }

                        let event = WindowResized {
                            window: entity,
                            width: window.width(),
                            height: window.height(),
                        };
                        window_resized.write(event.clone());
                        window_event.write(WindowEvent::WindowResized(event));
                    }

                    if let Some(monitor_link) = monitor_relationship {
                        if let Some(winit_monitor) = update_result.current_monitor {
                            if let Some(linked_monitor) = monitors.find_entity(monitor_link.0)
                                && winit_monitor != linked_monitor
                                && let Some((_, winit_monitor_entity)) =
                                    monitors.monitors.iter().find(|(h, _)| h == &winit_monitor)
                            {
                                commands
                                    .entity(entity)
                                    .insert(OnMonitor(winit_monitor_entity.to_owned()));
                            }
                        } else {
                            commands.entity(entity).remove::<OnMonitor>();
                        }
                    } else if let Some(winit_monitor) = update_result.current_monitor
                        && let Some((_, winit_monitor_entity)) =
                            monitors.monitors.iter().find(|(h, _)| h == &winit_monitor)
                    {
                        commands
                            .entity(entity)
                            .insert(OnMonitor(winit_monitor_entity.to_owned()));
                    }
                }
            }
            Err(e) => {
                error!("Failed to send window update task: {}", e);
            }
        }
    }
}

/// Forwards `RequestRedraw` messages to the winit thread.
#[cfg(not(target_arch = "wasm32"))]
pub fn forward_redraw_requests(
    mut redraw_reader: MessageReader<RequestRedraw>,
    task_sender: Res<crate::WinitTaskSender>,
) {
    if redraw_reader.read().next().is_some() {
        let _ = task_sender.send(move |_event_loop| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                for window in winit_windows.windows.values() {
                    window.request_redraw();
                }
            });
        });
    }
}
