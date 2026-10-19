//! The rendering pipeline of a [`PostProcessCamera`].

use crate::{
    schedule::{CorePostProcess, CorePostProcessSystems},
    tonemapping::{tonemapping, DebandDither, Tonemapping},
    upscaling::upscaling,
};
use bevy_app::{App, Plugin};
use bevy_camera::PostProcessCamera;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{camera::CameraRenderGraph, view::Msaa, RenderApp};

/// Renders [`PostProcessCamera`]s.
pub struct PostProcessCameraPlugin;

impl Plugin for PostProcessCameraPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components_with::<PostProcessCamera, CameraRenderGraph>(|| {
            CameraRenderGraph::new(CorePostProcess)
        })
        .register_required_components::<PostProcessCamera, Tonemapping>()
        .register_required_components_with::<PostProcessCamera, DebandDither>(|| {
            DebandDither::Enabled
        })
        // The camera draws nothing, so it has no edges to anti-alias.
        .register_required_components_with::<PostProcessCamera, Msaa>(|| Msaa::Off);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_schedule(CorePostProcess::base_schedule())
            .add_systems(
                CorePostProcess,
                (
                    tonemapping.in_set(CorePostProcessSystems::PostProcess),
                    upscaling.after(CorePostProcessSystems::PostProcess),
                ),
            );
    }
}
