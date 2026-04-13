//! GPU-authored instance batches: a particle swarm whose transforms are
//! written by a compute shader. Move the mouse to attract the particles.

use std::borrow::Cow;

use bevy::{
    camera::{primitives::Aabb, Hdr},
    core_pipeline::Core3d,
    math::Vec3A,
    pbr::{
        early_gpu_preprocess,
        gpu_instance_batch::{
            GpuBatchedMesh3d, GpuInstanceBatchPlugin, GpuInstanceBatchReservations,
        },
        MeshCullingDataBuffer, MeshInputUniform, MeshUniform,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        batching::gpu_preprocessing::BatchedInstanceBuffers,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{storage_buffer_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            ShaderType, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_world::MainEntityHashMap,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const SHADER_ASSET_PATH: &str = "shaders/gpu_particles_simulate.wgsl";
const WORKGROUP_SIZE: u32 = 64;
const PARTICLES_PER_EMITTER: u32 = 4096;
const PARTICLE_STATE_SIZE: u64 = 32;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GpuInstanceBatchPlugin)
        .add_plugins(GpuParticlesSimulationPlugin)
        .init_resource::<MouseWorldPos>()
        .add_systems(Startup, setup)
        .add_systems(Update, update_mouse_world_pos)
        .run();
}

#[derive(Resource, Default, Clone, Copy, ExtractResource)]
struct MouseWorldPos(Vec3);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(32.0, 32.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.1, 0.1, 0.12),
            perceptual_roughness: 0.4,
            metallic: 0.2,
            ..default()
        })),
        Transform::from_xyz(0.0, -2.0, 0.0),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.5, 4.0, 1.5))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.22, 0.28),
            perceptual_roughness: 0.35,
            metallic: 0.6,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    let particle_mesh = meshes.add(Cuboid::new(0.22, 0.22, 0.22));
    let particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.78, 0.66),
        metallic: 0.0,
        perceptual_roughness: 0.55,
        reflectance: 0.3,
        ..default()
    });

    commands.spawn((
        GpuBatchedMesh3d {
            mesh: particle_mesh,
            max_capacity: PARTICLES_PER_EMITTER,
        },
        Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(16.0),
        },
        MeshMaterial3d(particle_material),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5_000.0,
            color: Color::srgb(1.0, 0.95, 0.9),
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            color: Color::srgb(0.3, 0.6, 1.0),
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(-5.0, 3.0, -4.0),
    ));
    commands.spawn((
        PointLight {
            intensity: 2_000_000.0,
            color: Color::srgb(1.0, 0.4, 0.2),
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(5.0, 3.0, 4.0),
    ));
    commands.spawn((
        PointLight {
            intensity: 1_200_000.0,
            color: Color::srgb(0.8, 1.0, 0.6),
            range: 20.0,
            ..default()
        },
        Transform::from_xyz(0.0, 6.0, -6.0),
    ));

    commands.spawn((
        Camera3d::default(),
        Hdr,
        Bloom::default(),
        Transform::from_xyz(0.0, 3.0, 12.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));
}

#[derive(Component)]
struct MainCamera;

fn update_mouse_world_pos(
    windows: Query<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut mouse_pos: ResMut<MouseWorldPos>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    if ray.direction.y.abs() < 1e-4 {
        return;
    }
    let t = -ray.origin.y / ray.direction.y;
    if t > 0.0 {
        mouse_pos.0 = ray.origin + ray.direction * t;
    }
}

struct GpuParticlesSimulationPlugin;

impl Plugin for GpuParticlesSimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<MouseWorldPos>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ParticleSimBindGroups>()
            .init_resource::<ParticleStateBuffers>()
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

#[derive(Copy, Clone, Default, ShaderType)]
struct ParticleSimParams {
    base_input_index: u32,
    count: u32,
    time: f32,
    dt: f32,
    mouse_world_pos: Vec4,
}

#[derive(Resource, Default)]
struct ParticleStateBuffers {
    per_batch: MainEntityHashMap<Buffer>,
}

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
    mouse_world_pos: Res<MouseWorldPos>,
    mut state_buffers: ResMut<ParticleStateBuffers>,
    mut sim_bind_groups: ResMut<ParticleSimBindGroups>,
) {
    sim_bind_groups.per_batch.clear();

    if !matches!(
        pipeline_cache.get_compute_pipeline_state(pipeline.pipeline),
        CachedPipelineState::Ok(_)
    ) {
        return;
    }

    let Some(input_buffer) = batched_instance_buffers
        .current_input_buffer
        .buffer()
        .buffer()
    else {
        return;
    };
    let Some(culling_buffer) = culling_data_buffer.buffer() else {
        return;
    };

    for (main_entity, reservation) in reservations.by_entity.iter() {
        let state_buffer = state_buffers
            .per_batch
            .entry(*main_entity)
            .or_insert_with(|| {
                let size = reservation.max_capacity as u64 * PARTICLE_STATE_SIZE;
                let buffer = render_device.create_buffer(&BufferDescriptor {
                    label: Some("particle_state"),
                    size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Seed `pos.w = -1.0` to trigger the shader's init path.
                let mut seed =
                    vec![0u8; reservation.max_capacity as usize * PARTICLE_STATE_SIZE as usize];
                for i in 0..reservation.max_capacity as usize {
                    let base = i * PARTICLE_STATE_SIZE as usize;
                    let bytes = (-1.0f32).to_ne_bytes();
                    seed[base + 12..base + 16].copy_from_slice(&bytes);
                }
                render_queue.write_buffer(&buffer, 0, &seed);
                buffer
            });

        let params = ParticleSimParams {
            base_input_index: reservation.input_buffer_base,
            count: reservation.max_capacity,
            time: time.elapsed_secs(),
            dt: time.delta_secs().min(1.0 / 30.0),
            mouse_world_pos: mouse_world_pos.0.extend(0.0),
        };

        let mut uniform = UniformBuffer::from(params);
        uniform.write_buffer(&render_device, &render_queue);

        let bind_group = render_device.create_bind_group(
            Some("particle_sim_bind_group"),
            &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                input_buffer.as_entire_binding(),
                culling_buffer.as_entire_binding(),
                state_buffer.as_entire_binding(),
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
) {
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
