//! Simple example demonstrating the use of the [`Readback`] component to read back data from the
//! GPU using both a storage buffer and texture.
//!
//! Also demonstrates multi-stage readback with [`ReadbackLabel`]: a compute shader increments a
//! buffer twice per frame, and labeled [`readback`] calls between the dispatches capture
//! independent snapshots.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{readback, Readback, ReadbackComplete, ReadbackLabel},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer, texture_storage_2d},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/gpu_readback.wgsl";

// The length of the buffer sent to the gpu
const BUFFER_LEN: usize = 16;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            GpuReadbackExamplePlugin,
            ExtractResourcePlugin::<ReadbackBuffer>::default(),
            ExtractResourcePlugin::<ReadbackImage>::default(),
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .run();
}

// We need a plugin to organize all the systems and render node required for this example
struct GpuReadbackExamplePlugin;
impl Plugin for GpuReadbackExamplePlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_compute_pipeline)
            .add_systems(
                Render,
                prepare_bind_group
                    .in_set(RenderSystems::PrepareBindGroups)
                    // We don't need to recreate the bind group every frame
                    .run_if(not(resource_exists::<GpuBufferBindGroup>)),
            )
            .add_systems(
                RenderGraph,
                (
                    compute_first,
                    readback(Some("first")),
                    compute_second,
                    readback(Some("second")),
                )
                    .chain(),
            );
    }
}

#[derive(Resource, ExtractResource, Clone)]
struct ReadbackBuffer(Handle<ShaderBuffer>);

#[derive(Resource, ExtractResource, Clone)]
struct ReadbackImage(Handle<Image>);

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    // Create a storage buffer with some data
    let buffer: Vec<u32> = (0..BUFFER_LEN as u32).collect();
    let mut buffer = ShaderBuffer::from(buffer);
    // We need to enable the COPY_SRC usage so we can copy the buffer to the cpu
    buffer.buffer_description.usage |= BufferUsages::COPY_SRC;
    let buffer = buffers.add(buffer);

    // Create a storage texture
    let size = Extent3d {
        width: BUFFER_LEN as u32,
        height: 1,
        ..default()
    };
    let mut image = Image::new_uninit(
        size,
        TextureDimension::D2,
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING;
    let image = images.add(image);

    // Labeled buffer readbacks: "first" runs after the first compute dispatch,
    // "second" after the second dispatch.
    commands
        .spawn((Readback::buffer(buffer.clone()), ReadbackLabel("first")))
        .observe(|event: On<ReadbackComplete>| {
            let data: Vec<u32> = event.to_shader_type();
            info!("[first]   frame={} {:?}", event.frame, data);
        });
    commands
        .spawn((Readback::buffer(buffer.clone()), ReadbackLabel("second")))
        .observe(|event: On<ReadbackComplete>| {
            let data: Vec<u32> = event.to_shader_type();
            info!("[second]  frame={} {:?}", event.frame, data);
        });

    // It is also possible to read only a range of the buffer.
    commands
        .spawn(Readback::buffer_range(
            buffer.clone(),
            4 * u32::SHADER_SIZE.get(), // skip the first four elements
            8 * u32::SHADER_SIZE.get(), // read eight elements
        ))
        .observe(|event: On<ReadbackComplete>| {
            let data: Vec<u32> = event.to_shader_type();
            info!("[range]   frame={} {:?}", event.frame, data);
        });

    // Textures can also be read back from the GPU. Pay careful attention to the format of the
    // texture, as it will affect how the data is interpreted.
    commands
        .spawn(Readback::texture(image.clone()))
        .observe(|event: On<ReadbackComplete>| {
            let data: Vec<u32> = event.to_shader_type();
            info!("[texture] frame={} {:?}", event.frame, data);
        });

    commands.insert_resource(ReadbackBuffer(buffer));
    commands.insert_resource(ReadbackImage(image));
}

#[derive(Resource)]
struct GpuBufferBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    buffer: Res<ReadbackBuffer>,
    image: Res<ReadbackImage>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    let buffer = buffers.get(&buffer.0).unwrap();
    let image = images.get(&image.0).unwrap();
    let bind_group = render_device.create_bind_group(
        None,
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            buffer.buffer.as_entire_buffer_binding(),
            image.texture_view.into_binding(),
        )),
    );
    commands.insert_resource(GpuBufferBindGroup(bind_group));
}

#[derive(Resource)]
struct ComputePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

fn init_compute_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer::<Vec<u32>>(false),
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("GPU readback compute shader".into()),
        layout: vec![layout.clone()],
        shader: shader.clone(),
        ..default()
    });
    commands.insert_resource(ComputePipeline { layout, pipeline });
}

/// First compute dispatch, increments every element by 1.
fn compute_first(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<ComputePipeline>,
    bind_group: Res<GpuBufferBindGroup>,
) {
    dispatch_compute(&mut render_context, &pipeline_cache, &pipeline, &bind_group);
}

/// Second compute dispatch, increments by 1 again.
fn compute_second(
    mut render_context: RenderContext,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<ComputePipeline>,
    bind_group: Res<GpuBufferBindGroup>,
) {
    dispatch_compute(&mut render_context, &pipeline_cache, &pipeline, &bind_group);
}

fn dispatch_compute(
    render_context: &mut RenderContext,
    pipeline_cache: &PipelineCache,
    pipeline: &ComputePipeline,
    bind_group: &GpuBufferBindGroup,
) {
    if let Some(init_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) {
        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("GPU readback compute pass"),
                    ..default()
                });

        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.set_pipeline(init_pipeline);
        pass.dispatch_workgroups(BUFFER_LEN as u32, 1, 1);
    }
}
