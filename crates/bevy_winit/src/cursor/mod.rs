#[cfg(feature = "custom_cursor")]
mod custom_cursor;

#[cfg(feature = "custom_cursor")]
pub use custom_cursor::*;

use crate::{converters::convert_system_cursor_icon, WINIT_WINDOWS};
use bevy_app::{App, Last, Plugin};
#[cfg(feature = "custom_cursor")]
use bevy_asset::Assets;
use bevy_ecs::{entity::EntityHashSet, prelude::*};
#[cfg(feature = "custom_cursor")]
use bevy_image::{Image, TextureAtlasLayout};
#[cfg(feature = "custom_cursor")]
use bevy_window::CustomCursor;
use bevy_window::{CursorIcon, SystemCursorIcon, Window};

/// Adds support for custom cursors.
pub(crate) struct WinitCursorPlugin;

impl Plugin for WinitCursorPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(feature = "custom_cursor")]
        {
            if !app.is_plugin_added::<bevy_image::TextureAtlasPlugin>() {
                app.add_plugins(bevy_image::TextureAtlasPlugin);
            }

            app.init_resource::<WinitCustomCursorCache>();
        }

        app.add_systems(Last, update_cursors)
            .add_observer(on_remove_cursor_icon);
    }
}

/// A source for a cursor. Consumed by the winit event loop.
#[derive(Debug)]
pub enum CursorSource {
    #[cfg(feature = "custom_cursor")]
    /// A custom cursor was identified to be cached, no reason to recreate it.
    CustomCached(CustomCursorCacheKey),
    #[cfg(feature = "custom_cursor")]
    /// A custom cursor was not cached, so it needs to be created by the winit event loop.
    Custom((CustomCursorCacheKey, winit::window::CustomCursorSource)),
    /// A system cursor was requested.
    System(winit::window::CursorIcon),
}

/// Component that indicates what cursor should be used for a window. Inserted
/// automatically after changing `CursorIcon` and consumed by the winit event
/// loop.
#[derive(Component, Debug)]
pub struct PendingCursor(pub Option<CursorSource>);

/// Applies pending cursor changes.
///
/// TODO: cache custom cursors; the cache lives on the ECS thread but creation
/// happens on the event loop thread.
#[cfg(not(target_arch = "wasm32"))]
pub fn apply_pending_cursors(
    mut windows: Query<(Entity, &mut PendingCursor), Changed<PendingCursor>>,
    task_sender: Res<crate::WinitTaskSender>,
    #[cfg(feature = "custom_cursor")] cursor_cache: Res<WinitCustomCursorCache>,
) {
    for (entity, mut pending_cursor) in windows.iter_mut() {
        let Some(cursor_source) = pending_cursor.0.take() else {
            continue;
        };

        #[cfg(feature = "custom_cursor")]
        let cached_cursor = match &cursor_source {
            CursorSource::CustomCached(cache_key) => cursor_cache.0.get(cache_key).cloned(),
            _ => None,
        };

        if let Err(e) = task_sender.send(move |_event_loop| {
            WINIT_WINDOWS.with_borrow(|winit_windows| {
                let Some(winit_window) = winit_windows.get_window(entity) else {
                    return;
                };

                let final_cursor: winit::window::Cursor = match cursor_source {
                    #[cfg(feature = "custom_cursor")]
                    CursorSource::CustomCached(_) => {
                        if let Some(cursor) = cached_cursor {
                            cursor.into()
                        } else {
                            tracing::error!("Cursor should have been cached, but was not found");
                            return;
                        }
                    }
                    #[cfg(feature = "custom_cursor")]
                    CursorSource::Custom((_cache_key, source)) => {
                        _event_loop.create_custom_cursor(source).into()
                    }
                    CursorSource::System(system_cursor) => system_cursor.into(),
                };
                winit_window.set_cursor(final_cursor);
            });
        }) {
            tracing::error!("Failed to send cursor update task: {}", e);
        }
    }
}

fn update_cursors(
    mut commands: Commands,
    windows: Query<(Entity, Ref<CursorIcon>), With<Window>>,
    #[cfg(feature = "custom_cursor")] cursor_cache: Res<WinitCustomCursorCache>,
    #[cfg(feature = "custom_cursor")] images: Res<Assets<Image>>,
    #[cfg(feature = "custom_cursor")] texture_atlases: Res<Assets<TextureAtlasLayout>>,
    mut queue: Local<EntityHashSet>,
) {
    for (entity, cursor) in windows.iter() {
        if !(queue.remove(&entity) || cursor.is_changed()) {
            continue;
        }

        let cursor_source = match cursor.as_ref() {
            #[cfg(feature = "custom_cursor")]
            CursorIcon::Custom(CustomCursor::Image(c)) => {
                let bevy_window::CustomCursorImage {
                    handle,
                    texture_atlas,
                    flip_x,
                    flip_y,
                    rect,
                    hotspot,
                } = c;

                let cache_key = CustomCursorCacheKey::Image {
                    id: handle.id(),
                    texture_atlas_layout_id: texture_atlas.as_ref().map(|a| a.layout.id()),
                    texture_atlas_index: texture_atlas.as_ref().map(|a| a.index),
                    flip_x: *flip_x,
                    flip_y: *flip_y,
                    rect: *rect,
                };

                if cursor_cache.0.contains_key(&cache_key) {
                    CursorSource::CustomCached(cache_key)
                } else {
                    let Some(image) = images.get(handle) else {
                        tracing::warn!(
                            "Cursor image {handle:?} is not loaded yet and couldn't be used. Trying again next frame."
                        );
                        queue.insert(entity);
                        continue;
                    };

                    let (rect, needs_sub_image) =
                        calculate_effective_rect(&texture_atlases, image, texture_atlas, rect);

                    let (maybe_rgba, hotspot) = if *flip_x || *flip_y || needs_sub_image {
                        (
                            extract_and_transform_rgba_pixels(image, *flip_x, *flip_y, rect),
                            transform_hotspot(*hotspot, *flip_x, *flip_y, rect),
                        )
                    } else {
                        (extract_rgba_pixels(image), *hotspot)
                    };

                    let Some(rgba) = maybe_rgba else {
                        tracing::warn!("Cursor image {handle:?} not accepted because it's not rgba8 or rgba32float format");
                        continue;
                    };

                    let source = match winit::window::CustomCursor::from_rgba(
                        rgba,
                        rect.width() as u16,
                        rect.height() as u16,
                        hotspot.0,
                        hotspot.1,
                    ) {
                        Ok(source) => source,
                        Err(err) => {
                            tracing::warn!("Cursor image {handle:?} is invalid: {err}");
                            continue;
                        }
                    };

                    CursorSource::Custom((cache_key, source))
                }
            }
            #[cfg(feature = "custom_cursor")]
            CursorIcon::Custom(CustomCursor::Url(_c)) => {
                #[cfg(all(target_family = "wasm", target_os = "unknown"))]
                {
                    let cache_key = CustomCursorCacheKey::Url(_c.url.clone());

                    if cursor_cache.0.contains_key(&cache_key) {
                        CursorSource::CustomCached(cache_key)
                    } else {
                        use crate::CustomCursorExtWebSys;
                        let source = winit::window::CustomCursor::from_url(
                            _c.url.clone(),
                            _c.hotspot.0,
                            _c.hotspot.1,
                        );
                        CursorSource::Custom((cache_key, source))
                    }
                }
                #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
                {
                    bevy_log::error_once!("CustomCursor::Url is not supported on this platform. Falling back to CursorIcon::System(SystemCursorIcon::Default)");
                    CursorSource::System(winit::window::CursorIcon::Default)
                }
            }
            CursorIcon::System(system_cursor_icon) => {
                CursorSource::System(convert_system_cursor_icon(*system_cursor_icon))
            }
        };

        commands
            .entity(entity)
            .insert(PendingCursor(Some(cursor_source)));
    }
}

/// Resets the cursor to the default icon when `CursorIcon` is removed.
fn on_remove_cursor_icon(remove: On<Remove, CursorIcon>, mut commands: Commands) {
    // Use `try_insert` to avoid panic if the window is being destroyed.
    commands
        .entity(remove.entity)
        .try_insert(PendingCursor(Some(CursorSource::System(
            convert_system_cursor_icon(SystemCursorIcon::Default),
        ))));
}
