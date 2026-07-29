//! GPU-authored mesh instances: a particle swarm whose transforms are
//! written by a compute shader. Move the mouse to attract the particles.
//!
//! A [`GpuInstances3d`] reserves a contiguous range of slots in the GPU-only
//! instance pools. Each frame the simulation compute shader writes each
//! particle's transform into its slot in the input pool and an `alive` value
//! into the culling pool; retired particles are culled on GPU. Bevy's
//! standard PBR pipeline then draws the whole emitter as one indirect draw,
//! with no per-particle entities or CPU transforms.

use std::borrow::Cow;
use std::num::NonZeroU32;

use bytemuck::{Pod, Zeroable};

use bevy::{
    camera::{primitives::Aabb, Hdr},
    math::Vec3A,
    pbr::gpu_instances::{
        GpuInstanceAuthoringSystems, GpuInstancePools, GpuInstanceReservations, GpuInstances3d,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            binding_types::{storage_buffer_sized, uniform_buffer},
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache, ShaderStages,
            ShaderType, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        sync_world::MainEntityHashMap,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

const SHADER_ASSET_PATH: &str = "shaders/gpu_particles_simulate.wgsl";
const WORKGROUP_SIZE: u32 = 64;
const PARTICLES_PER_EMITTER: u32 = 4096;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GpuParticlesSimulationPlugin)
        .init_resource::<MouseWorldPos>()
        .add_systems(Startup, setup)
        .add_systems(Update, update_mouse_world_pos)
        .run();
}

#[derive(Resource, Default, Clone, Copy, ExtractResource)]
#[extract_app(RenderApp)]
struct MouseWorldPos(Vec3);

#[derive(Component)]
struct MainCamera;

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

    // The emitter entity: one entity for the whole swarm. The `Aabb` bounds the
    // entire volume the particles can move in; per-instance culling volumes
    // come from the simulation shader.
    commands.spawn((
        GpuInstances3d {
            mesh: particle_mesh,
            capacity: PARTICLES_PER_EMITTER,
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
            shadow_maps_enabled: true,
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

fn update_mouse_world_pos(
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut mouse_pos: ResMut<MouseWorldPos>,
) {
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
            .add_systems(
                RenderGraph,
                dispatch_particle_sim.in_set(GpuInstanceAuthoringSystems),
            );
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

/// Mirrors the `ParticleState` struct in the simulation shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleState {
    /// `xyz` = world position, `w` = age in seconds; negative = uninitialized.
    pos: Vec4,
    /// `xyz` = velocity.
    vel: Vec4,
}

#[derive(Resource, Default)]
struct ParticleStateBuffers {
    per_emitter: MainEntityHashMap<ParticleEmitterBuffers>,
}

/// The per-emitter GPU resources the simulation owns: the particle state
/// buffer and the parameters uniform, both reused across frames.
struct ParticleEmitterBuffers {
    state: Buffer,
    capacity: NonZeroU32,
    sim_params: UniformBuffer<ParticleSimParams>,
}

#[derive(Resource, Default)]
struct ParticleSimBindGroups {
    per_emitter: MainEntityHashMap<PerEmitterBindGroup>,
}

struct PerEmitterBindGroup {
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
    pools: Res<GpuInstancePools>,
    reservations: Res<GpuInstanceReservations>,
    time: Res<Time>,
    mouse_world_pos: Res<MouseWorldPos>,
    mut state_buffers: ResMut<ParticleStateBuffers>,
    mut sim_bind_groups: ResMut<ParticleSimBindGroups>,
) {
    sim_bind_groups.per_emitter.clear();

    if !matches!(
        pipeline_cache.get_compute_pipeline_state(pipeline.pipeline),
        CachedPipelineState::Ok(_)
    ) {
        return;
    }

    let (Some(input_buffer), Some(culling_buffer)) = (pools.input_buffer(), pools.culling_buffer())
    else {
        return;
    };

    // Drop state buffers for emitters that no longer exist.
    state_buffers
        .per_emitter
        .retain(|main_entity, _| reservations.by_entity.contains_key(main_entity));

    for (main_entity, reservation) in reservations.by_entity.iter() {
        // Create the state buffer, or recreate it if the emitter's capacity
        // changed.
        let emitter_buffers = state_buffers
            .per_emitter
            .entry(*main_entity)
            .and_modify(|emitter_buffers| {
                if emitter_buffers.capacity != reservation.capacity {
                    emitter_buffers.state = create_particle_state_buffer(
                        &render_device,
                        &render_queue,
                        reservation.capacity,
                    );
                    emitter_buffers.capacity = reservation.capacity;
                }
            })
            .or_insert_with(|| ParticleEmitterBuffers {
                state: create_particle_state_buffer(
                    &render_device,
                    &render_queue,
                    reservation.capacity,
                ),
                capacity: reservation.capacity,
                sim_params: UniformBuffer::default(),
            });

        emitter_buffers.sim_params.set(ParticleSimParams {
            base_input_index: reservation.base_input_index,
            count: reservation.capacity.get(),
            time: time.elapsed_secs(),
            dt: time.delta_secs().min(1.0 / 30.0),
            mouse_world_pos: mouse_world_pos.0.extend(0.0),
        });
        emitter_buffers
            .sim_params
            .write_buffer(&render_device, &render_queue);

        let bind_group = render_device.create_bind_group(
            Some("particle_sim_bind_group"),
            &pipeline_cache.get_bind_group_layout(&pipeline.bind_group_layout),
            &BindGroupEntries::sequential((
                input_buffer.as_entire_binding(),
                culling_buffer.as_entire_binding(),
                emitter_buffers.state.as_entire_binding(),
                emitter_buffers.sim_params.binding().unwrap(),
            )),
        );

        let dispatch_count = reservation.capacity.get().div_ceil(WORKGROUP_SIZE);
        sim_bind_groups.per_emitter.insert(
            *main_entity,
            PerEmitterBindGroup {
                bind_group,
                dispatch_count,
            },
        );
    }
}

fn create_particle_state_buffer(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    capacity: NonZeroU32,
) -> Buffer {
    // Seed `pos.w = -1.0` to trigger the shader's init path.
    let seed = vec![
        ParticleState {
            pos: Vec4::new(0.0, 0.0, 0.0, -1.0),
            vel: Vec4::ZERO,
        };
        capacity.get() as usize
    ];
    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("particle_state"),
        size: size_of_val(seed.as_slice()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    render_queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&seed));
    buffer
}

fn dispatch_particle_sim(
    mut render_context: RenderContext,
    sim_bind_groups: Res<ParticleSimBindGroups>,
    pipeline: Res<ParticleSimPipeline>,
    pipeline_cache: Res<PipelineCache>,
) {
    if sim_bind_groups.per_emitter.is_empty() {
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

    for per_emitter in sim_bind_groups.per_emitter.values() {
        pass.set_bind_group(0, &per_emitter.bind_group, &[]);
        pass.dispatch_workgroups(per_emitter.dispatch_count, 1, 1);
    }
}
