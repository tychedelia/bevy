use crate::{
    core_2d::graph::{Core2d, Node2d},
    core_3d::graph::{Core3d, Node3d},
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
};
use bevy_app::prelude::*;
use bevy_asset::{load_internal_asset, Handle};
use bevy_ecs::{prelude::*, query::QueryItem};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
    prelude::Camera,
    render_graph::RenderGraphApp,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    texture::BevyDefault,
    view::{ExtractedView, ViewTarget},
    Render, RenderApp, RenderSet,
};
use bevy_render::render_component::{RenderComponent, RenderComponentPlugin};

mod node;

pub use node::CasNode;

/// Applies a contrast adaptive sharpening (CAS) filter to the camera.
///
/// CAS is usually used in combination with shader based anti-aliasing methods
/// such as FXAA or TAA to regain some of the lost detail from the blurring that they introduce.
///
/// CAS is designed to adjust the amount of sharpening applied to different areas of an image
/// based on the local contrast. This can help avoid over-sharpening areas with high contrast
/// and under-sharpening areas with low contrast.
///
/// To use this, add the [`ContrastAdaptiveSharpening`] component to a 2D or 3D camera.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default)]
pub struct ContrastAdaptiveSharpening {
    /// Enable or disable sharpening.
    pub enabled: bool,
    /// Adjusts sharpening strength. Higher values increase the amount of sharpening.
    ///
    /// Clamped between 0.0 and 1.0.
    ///
    /// The default value is 0.6.
    pub sharpening_strength: f32,
    /// Whether to try and avoid sharpening areas that are already noisy.
    ///
    /// You probably shouldn't use this, and just leave it set to false.
    /// You should generally apply any sort of film grain or similar effects after CAS
    /// and upscaling to avoid artifacts.
    pub denoise: bool,
}

#[deprecated(since = "0.15.0", note = "Renamed to `ContrastAdaptiveSharpening`")]
pub type ContrastAdaptiveSharpeningSettings = ContrastAdaptiveSharpening;

impl Default for ContrastAdaptiveSharpening {
    fn default() -> Self {
        ContrastAdaptiveSharpening {
            enabled: true,
            sharpening_strength: 0.6,
            denoise: false,
        }
    }
}

#[derive(Component, RenderComponent)]
pub struct UseContrastAdaptiveSharpening;

#[derive(Component, Default, Reflect, Clone)]
#[reflect(Component, Default)]
pub struct DenoiseCas(bool);

/// The uniform struct extracted from [`ContrastAdaptiveSharpening`] attached to a [`Camera`].
/// Will be available for use in the CAS shader.
#[doc(hidden)]
#[derive(Component, ShaderType, Clone)]
pub struct CasUniform {
    sharpness: f32,
}

impl ExtractComponent for ContrastAdaptiveSharpening {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    type Out = (DenoiseCas, CasUniform, UseContrastAdaptiveSharpening);

    fn extract_component(item: QueryItem<Self::QueryData>) -> Option<Self::Out> {
        if !item.enabled || item.sharpening_strength == 0.0 {
            return None;
        }
        Some((
            DenoiseCas(item.denoise),
            CasUniform {
                // above 1.0 causes extreme artifacts and fireflies
                sharpness: item.sharpening_strength.clamp(0.0, 1.0),
            },
            UseContrastAdaptiveSharpening,
        ))
    }
}

const CONTRAST_ADAPTIVE_SHARPENING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(6925381244141981602);

/// Adds Support for Contrast Adaptive Sharpening (CAS).
pub struct CasPlugin;

impl Plugin for CasPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CONTRAST_ADAPTIVE_SHARPENING_SHADER_HANDLE,
            "robust_contrast_adaptive_sharpening.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ContrastAdaptiveSharpening>();
        app.add_plugins((
            ExtractComponentPlugin::<ContrastAdaptiveSharpening>::default(),
            UniformComponentPlugin::<CasUniform>::default(),
            RenderComponentPlugin::<UseContrastAdaptiveSharpening>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<SpecializedRenderPipelines<CasPipeline>>()
            .add_systems(Render, prepare_cas_pipelines.in_set(RenderSet::Prepare));

        {
            render_app
                .add_render_graph_node::<CasNode>(Core3d, Node3d::ContrastAdaptiveSharpening)
                .add_render_graph_edge(
                    Core3d,
                    Node3d::Tonemapping,
                    Node3d::ContrastAdaptiveSharpening,
                )
                .add_render_graph_edges(
                    Core3d,
                    (
                        Node3d::Fxaa,
                        Node3d::ContrastAdaptiveSharpening,
                        Node3d::EndMainPassPostProcessing,
                    ),
                );
        }
        {
            render_app
                .add_render_graph_node::<CasNode>(Core2d, Node2d::ContrastAdaptiveSharpening)
                .add_render_graph_edge(
                    Core2d,
                    Node2d::Tonemapping,
                    Node2d::ContrastAdaptiveSharpening,
                )
                .add_render_graph_edges(
                    Core2d,
                    (
                        Node2d::Fxaa,
                        Node2d::ContrastAdaptiveSharpening,
                        Node2d::EndMainPassPostProcessing,
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<CasPipeline>();
    }
}

#[derive(Resource)]
pub struct CasPipeline {
    texture_bind_group: BindGroupLayout,
    sampler: Sampler,
}

impl FromWorld for CasPipeline {
    fn from_world(render_world: &mut World) -> Self {
        let render_device = render_world.resource::<RenderDevice>();
        let texture_bind_group = render_device.create_bind_group_layout(
            "sharpening_texture_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                    // CAS Settings
                    uniform_buffer::<CasUniform>(true),
                ),
            ),
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        CasPipeline {
            texture_bind_group,
            sampler,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct CasPipelineKey {
    texture_format: TextureFormat,
    denoise: bool,
}

impl SpecializedRenderPipeline for CasPipeline {
    type Key = CasPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec![];
        if key.denoise {
            shader_defs.push("RCAS_DENOISE".into());
        }
        RenderPipelineDescriptor {
            label: Some("contrast_adaptive_sharpening".into()),
            layout: vec![self.texture_bind_group.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: CONTRAST_ADAPTIVE_SHARPENING_SHADER_HANDLE,
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: key.texture_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: Vec::new(),
        }
    }
}

fn prepare_cas_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<CasPipeline>>,
    sharpening_pipeline: Res<CasPipeline>,
    views: Query<(Entity, &ExtractedView, &DenoiseCas), With<CasUniform>>,
) {
    for (entity, view, cas) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &sharpening_pipeline,
            CasPipelineKey {
                denoise: cas.0,
                texture_format: if view.hdr {
                    ViewTarget::TEXTURE_FORMAT_HDR
                } else {
                    TextureFormat::bevy_default()
                },
            },
        );

        commands.entity(entity).insert(ViewCasPipeline(pipeline_id));
    }
}

#[derive(Component)]
pub struct ViewCasPipeline(CachedRenderPipelineId);
