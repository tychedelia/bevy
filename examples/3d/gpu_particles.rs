//! GPU-authored instance batches: end-to-end example.
//!
//! Demonstrates the [`GpuInstanceBatch`] API — rendering many mesh
//! instances whose per-instance transforms are authored on the GPU
//! rather than extracted from ECS entities on the CPU.
//!
//! The example runs a compute shader each frame that writes
//! `world_from_local` and per-slot culling data into bevy's shared GPU
//! preprocessing input buffer, at offsets corresponding to the batch's
//! reservation. Bevy's existing preprocessing, indirect build, and
//! `multi_draw_indexed_indirect` rendering then renders the batch
//! across all four phases (opaque, shadow, prepass, deferred).
//!
//! # Requirements
//!
//! Requires GPU preprocessing (compute shaders). The plugin warns and
//! does nothing on unsupported devices (e.g. WebGL).

use std::borrow::Cow;

use bevy::{
    camera::primitives::Aabb,
    core_pipeline::Core3d,
    math::Vec3A,
    pbr::{
        early_gpu_preprocess,
        gpu_instance_batch::{
            GpuInstanceBatch, GpuInstanceBatchPlugin, GpuInstanceBatchReservations,
        },
        MeshCullingDataBuffer, MeshFlags, MeshInputUniform, MeshUniform,
    },
    prelude::*,
    render::{
        batching::gpu_preprocessing::BatchedInstanceBuffers,
        render_resource::{
            binding_types::{storage_buffer_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::MainEntityHashMap,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const SHADER_ASSET_PATH: &str = "shaders/gpu_particles_simulate.wgsl";
const WORKGROUP_SIZE: u32 = 64;
const PARTICLES_PER_EMITTER: u32 = 256;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GpuInstanceBatchPlugin)
        .add_plugins(GpuParticlesSimulationPlugin)
        .add_systems(Startup, setup)
        .run();
}

// ---------------------------------------------------------------------------
// Main world: scene setup.
// ---------------------------------------------------------------------------

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Particle batch — stripped scene, no ground/reference cube, to
    // isolate the batch rendering.
    let particle_mesh = meshes.add(Cuboid::new(0.3, 0.3, 0.3));
    let particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.4, 0.2),
        emissive: LinearRgba::rgb(1.0, 0.2, 0.0) * 3.0,
        ..default()
    });

    commands.spawn((
        GpuInstanceBatch {
            mesh: particle_mesh,
            max_capacity: PARTICLES_PER_EMITTER,
            aabb: Aabb {
                center: Vec3A::ZERO,
                half_extents: Vec3A::splat(6.0),
            },
            // Disable frustum culling while debugging — if something is
            // wrong with the per-slot MeshCullingData path, this
            // guarantees we still see the particles.
            flags: MeshFlags::NO_FRUSTUM_CULLING,
        },
        MeshMaterial3d(particle_material),
        Transform::default(),
        Visibility::default(),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

// ---------------------------------------------------------------------------
// Compute plugin: pipeline setup + per-frame bind groups + dispatch.
// ---------------------------------------------------------------------------

struct GpuParticlesSimulationPlugin;

impl Plugin for GpuParticlesSimulationPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ParticleSimBindGroups>()
            .add_systems(RenderStartup, init_particle_sim_pipeline)
            .add_systems(
                Render,
                prepare_particle_sim_bind_groups.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(Core3d, dispatch_particle_sim.before(early_gpu_preprocess));
    }
}

#[derive(Resource)]
struct ParticleSimPipeline {
    bind_group_layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

/// Per-batch parameters passed to the compute shader via a uniform buffer.
#[derive(Copy, Clone, Default, ShaderType)]
struct ParticleSimParams {
    base_input_index: u32,
    count: u32,
    time: f32,
    _pad: u32,
}

/// Bind groups for each live batch, rebuilt each frame because the
/// underlying buffers can resize and invalidate them.
#[derive(Resource, Default)]
struct ParticleSimBindGroups {
    per_batch: MainEntityHashMap<PerBatchBindGroup>,
}

struct PerBatchBindGroup {
    bind_group: BindGroup,
    dispatch_count: u32,
}

fn init_particle_sim_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let bind_group_layout = BindGroupLayoutDescriptor::new(
        "ParticleSimBindGroupLayout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                uniform_buffer::<ParticleSimParams>(false),
            ),
        ),
    );

    let shader = asset_server.load(SHADER_ASSET_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("particle_sim_pipeline".into()),
        layout: vec![bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("simulate")),
        ..default()
    });

    commands.insert_resource(ParticleSimPipeline {
        bind_group_layout,
        pipeline,
    });
}

fn prepare_particle_sim_bind_groups(
    pipeline: Res<ParticleSimPipeline>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    culling_data_buffer: Res<MeshCullingDataBuffer>,
    reservations: Res<GpuInstanceBatchReservations>,
    time: Res<Time>,
    mut sim_bind_groups: ResMut<ParticleSimBindGroups>,
    mut frame: Local<u32>,
) {
    sim_bind_groups.per_batch.clear();

    *frame += 1;
    let log_this_frame = *frame < 5 || *frame % 60 == 0;

    let pipeline_state = pipeline_cache.get_compute_pipeline_state(pipeline.pipeline);
    if log_this_frame {
        info!("[gpu_particles] frame {} pipeline_state = {:?}", *frame, match pipeline_state {
            CachedPipelineState::Queued => "Queued",
            CachedPipelineState::Creating(_) => "Creating",
            CachedPipelineState::Ok(_) => "Ok",
            CachedPipelineState::Err(e) => {
                info!("  err: {}", e);
                "Err"
            }
        });
    }

    // Wait for compute pipeline to finish compiling.
    if !matches!(pipeline_state, CachedPipelineState::Ok(_)) {
        return;
    }

    let input_buffer_opt = batched_instance_buffers.current_input_buffer.buffer().buffer();
    let culling_buffer_opt = culling_data_buffer.buffer();

    if log_this_frame {
        info!(
            "[gpu_particles] input_buffer: {}, culling_buffer: {}, reservations: {}",
            input_buffer_opt.is_some(),
            culling_buffer_opt.is_some(),
            reservations.by_entity.len(),
        );
    }

    let Some(input_buffer) = input_buffer_opt else {
        return;
    };
    let Some(culling_buffer) = culling_buffer_opt else {
        return;
    };

    for (main_entity, reservation) in reservations.by_entity.iter() {
        let params = ParticleSimParams {
            base_input_index: reservation.input_buffer_base,
            count: reservation.max_capacity,
            time: time.elapsed_secs(),
            _pad: 0,
        };

        let mut uniform = UniformBuffer::from(params);
        uniform.write_buffer(&render_device, &render_queue);

        let bind_group = render_device.create_bind_group(
            Some("particle_sim_bind_group"),
            &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                input_buffer.as_entire_binding(),
                culling_buffer.as_entire_binding(),
                uniform.binding().unwrap(),
            )),
        );

        let dispatch_count = reservation.max_capacity.div_ceil(WORKGROUP_SIZE);
        sim_bind_groups.per_batch.insert(
            *main_entity,
            PerBatchBindGroup {
                bind_group,
                dispatch_count,
            },
        );
    }
}

fn dispatch_particle_sim(
    mut render_context: RenderContext,
    sim_bind_groups: Res<ParticleSimBindGroups>,
    pipeline: Res<ParticleSimPipeline>,
    pipeline_cache: Res<PipelineCache>,
    mut frame: Local<u32>,
) {
    *frame += 1;
    let log = *frame < 5 || *frame % 60 == 0;
    if log {
        info!(
            "[gpu_particles] dispatch: {} bind groups ready",
            sim_bind_groups.per_batch.len()
        );
    }

    if sim_bind_groups.per_batch.is_empty() {
        return;
    }
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("particle_sim"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute_pipeline);

    for per_batch in sim_bind_groups.per_batch.values() {
        pass.set_bind_group(0, &per_batch.bind_group, &[]);
        pass.dispatch_workgroups(per_batch.dispatch_count, 1, 1);
    }
}
