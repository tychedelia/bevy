mod node;
mod material;

use crate::core_2d::graph::{Core2d, Node2d};
use crate::core_2d::{Camera2d, MainOpaquePass2dNode, MainTransparentPass2dNode};
use crate::fullscreen::graph::{CoreFullscreen, NodeFullscreen};
use crate::tonemapping::TonemappingNode;
use crate::upscaling::UpscalingNode;
use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_ecs::component::Component;
use bevy_ecs::prelude::World;
use bevy_ecs::query::QueryItem;
use bevy_reflect::Reflect;
use bevy_render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy_render::render_graph::{
    EmptyNode, NodeRunError, RenderGraphApp, RenderGraphContext, RenderSubGraph, ViewNode,
    ViewNodeRunner,
};
use bevy_render::renderer::RenderContext;
use bevy_render::{prelude::Shader, render_resource::VertexState, RenderApp};

pub const FULLSCREEN_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("481fb759-d0b1-4175-8319-c439acde30a2");

/// uses the [`FULLSCREEN_SHADER_HANDLE`] to output a
/// ```wgsl
/// struct FullscreenVertexOutput {
///     [[builtin(position)]]
///     position: vec4<f32>;
///     [[location(0)]]
///     uv: vec2<f32>;
/// };
/// ```
/// from the vertex shader.
/// The draw call should render one triangle: `render_pass.draw(0..3, 0..1);`
pub fn fullscreen_shader_vertex_state() -> VertexState {
    VertexState {
        shader: FULLSCREEN_SHADER_HANDLE,
        shader_defs: Vec::new(),
        entry_point: "fullscreen".into(),
        buffers: Vec::new(),
    }
}

pub struct FullscreenPlugin;

impl Plugin for FullscreenPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            FULLSCREEN_SHADER_HANDLE,
            "fullscreen/fullscreen.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<FullscreenCamera>()
            .add_plugins(ExtractComponentPlugin::<FullscreenCamera>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_sub_graph(CoreFullscreen)
            .add_render_graph_node::<EmptyNode>(CoreFullscreen, NodeFullscreen::StartFullscreenPass)
            .add_render_graph_node::<ViewNodeRunner<FullscreenViewNode>>(
                CoreFullscreen,
                NodeFullscreen::Upscaling,
            )
            .add_render_graph_node::<EmptyNode>(CoreFullscreen, NodeFullscreen::Fullscreen)
            .add_render_graph_node::<ViewNodeRunner<TonemappingNode>>(
                CoreFullscreen,
                NodeFullscreen::Tonemapping,
            )
            .add_render_graph_node::<ViewNodeRunner<UpscalingNode>>(
                CoreFullscreen,
                NodeFullscreen::Upscaling,
            )
            .add_render_graph_edges(
                Core2d,
                (
                    NodeFullscreen::StartFullscreenPass,
                    NodeFullscreen::EndFullscreenPass,
                    NodeFullscreen::Tonemapping,
                    NodeFullscreen::Upscaling,
                ),
            );
    }
}

pub struct FullscreenShader(Handle<Shader>);

#[derive(Default)]
pub struct FullscreenViewNode;

impl ViewNode for FullscreenViewNode {
    type ViewQuery = ();

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        view_query: QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        todo!()
    }
}

#[derive(Component, Reflect, Clone, ExtractComponent)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Default, Clone)]
#[require(
    Camera,
    CameraRenderGraph::new(CoreFullscreen),
    FullscreenShader,
    Tonemapping,
    ColorGrading
)]
pub struct FullscreenCamera;

pub mod graph {
    use bevy_render::render_graph::{RenderLabel, RenderSubGraph};

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderSubGraph)]
    pub struct CoreFullscreen;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub enum NodeFullscreen {
        MsaaWriteback,
        StartFullscreenPass,
        Fullscreen,
        Tonemapping,
        Upscaling,
        EndFullscreenPass,
    }
}
