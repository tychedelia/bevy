use crate::FullscreenShader;
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d},
        *,
    },
    renderer::RenderDevice,
    GpuResourceAppExt, RenderApp, RenderStartup,
};
use bevy_shader::Shader;
use bevy_utils::default;

/// Adds support for specialized "blit pipelines", which can be used to write one texture to another.
pub struct BlitPlugin;

impl Plugin for BlitPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "blit.wesl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .allow_ambiguous_resource::<SpecializedRenderPipelines<BlitPipeline>>()
            .init_gpu_resource::<SpecializedRenderPipelines<BlitPipeline>>()
            .add_systems(RenderStartup, init_blit_pipeline);
    }
}

#[derive(Resource)]
pub struct BlitPipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub sampler: Sampler,
    /// The layout for [`FilterMode::Linear`] blits.
    pub linear_layout: BindGroupLayoutDescriptor,
    /// The sampler for [`FilterMode::Linear`] blits.
    pub linear_sampler: Sampler,
    pub fullscreen_shader: FullscreenShader,
    pub fragment_shader: Handle<Shader>,
}

pub fn init_blit_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "blit_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                sampler(SamplerBindingType::NonFiltering),
            ),
        ),
    );

    let linear_layout = BindGroupLayoutDescriptor::new(
        "blit_linear_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        ..default()
    });

    commands.insert_resource(BlitPipeline {
        layout,
        sampler,
        linear_layout,
        linear_sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "blit.wesl"),
    });
}

impl BlitPipeline {
    pub fn create_bind_group(
        &self,
        render_device: &RenderDevice,
        src_texture: &TextureView,
        pipeline_cache: &PipelineCache,
    ) -> BindGroup {
        render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&self.layout),
            &BindGroupEntries::sequential((src_texture, &self.sampler)),
        )
    }

    /// Creates the bind group for a pipeline specialized with the given
    /// [`BlitPipelineKey::filter`].
    pub fn create_bind_group_with_filter(
        &self,
        render_device: &RenderDevice,
        src_texture: &TextureView,
        pipeline_cache: &PipelineCache,
        filter: FilterMode,
    ) -> BindGroup {
        let (layout, sampler) = match filter {
            FilterMode::Linear => (&self.linear_layout, &self.linear_sampler),
            FilterMode::Nearest => (&self.layout, &self.sampler),
        };
        render_device.create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(layout),
            &BindGroupEntries::sequential((src_texture, sampler)),
        )
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct BlitPipelineKey {
    pub target_format: TextureFormat,
    pub blend_state: Option<BlendState>,
    pub samples: u32,
    /// Color space of the source texture. When `Some(Srgb)` or `Some(Oklab)`, the blit converts
    /// to linear RGB before writing to the output target.
    pub source_space: Option<CompositingSpace>,
    /// How the source texture is sampled. Bind groups for a [`FilterMode::Linear`] pipeline come
    /// from [`BlitPipeline::create_bind_group_with_filter`].
    pub filter: FilterMode,
}

impl SpecializedRenderPipeline for BlitPipeline {
    type Key = BlitPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();
        match key.source_space {
            Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_TO_LINEAR".into()),
            Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_TO_LINEAR".into()),
            Some(CompositingSpace::Linear) | None => {}
        }

        RenderPipelineDescriptor {
            label: Some("blit pipeline".into()),
            layout: vec![match key.filter {
                FilterMode::Linear => self.linear_layout.clone(),
                FilterMode::Nearest => self.layout.clone(),
            }],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: key.blend_state,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            multisample: MultisampleState {
                count: key.samples,
                ..default()
            },
            ..default()
        }
    }
}
