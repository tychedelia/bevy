//! Winit event-loop runners.

use bevy_app::{App, AppExit, PluginsState};
#[cfg(target_arch = "wasm32")]
use bevy_ecs::entity::Entity;
use bevy_log::trace;
#[cfg(target_arch = "wasm32")]
use bevy_window::{Window, WindowResized};
#[cfg(target_arch = "wasm32")]
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;

use crate::WinitUserEvent;

/// [`winit_runner`] with the [`EventLoop`] taken from thread-local storage.
#[cfg(not(target_arch = "wasm32"))]
pub fn winit_runner_from_tls(app: App) -> AppExit {
    let event_loop = crate::WINIT_EVENT_LOOP.with(|cell| {
        cell.borrow_mut()
            .take()
            .expect("EventLoop should be stored in thread-local storage")
    });
    winit_runner(app, event_loop)
}

/// [`App::runner`] that runs ECS on a separate thread from the winit event loop.
#[cfg(not(target_arch = "wasm32"))]
pub fn winit_runner(mut app: App, event_loop: EventLoop<WinitUserEvent>) -> AppExit {
    use crate::runtime::{create_channel, run_app, WinitApp};

    // Non-send resources must be finalized on the main thread.
    if app.plugins_state() == PluginsState::Ready {
        app.finish();
        app.cleanup();
    }

    trace!("starting threaded winit event loop");

    let (event_sender, event_receiver) = create_channel();

    let ecs_handle = std::thread::Builder::new()
        .name("ECS".to_string())
        .spawn(move || run_app(app, event_receiver))
        .expect("failed to spawn ECS thread");

    let mut winit_app = WinitApp::new(event_sender);
    if let Err(err) = event_loop.run_app(&mut winit_app) {
        bevy_log::error!("winit event loop returned an error: {err}");
    }

    match ecs_handle.join() {
        Ok(exit) => exit,
        Err(_) => {
            bevy_log::error!("ECS thread panicked");
            AppExit::error()
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn react_to_resize(
    window_entity: Entity,
    window: &mut Window,
    size: PhysicalSize<u32>,
) -> WindowResized {
    window
        .resolution
        .set_physical_resolution(size.width, size.height);

    WindowResized {
        window: window_entity,
        width: window.width(),
        height: window.height(),
    }
}
