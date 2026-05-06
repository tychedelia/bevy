//! Demonstrates per-attribute vertex buffer bindings with compute shader
//! displacement.
//!
//! A subdivided plane is created with each vertex attribute in its own GPU
//! buffer (via `deinterleave()`). A compute shader then displaces the
//! positions every frame to create a wave effect — without touching the
//! normal or UV buffers at all.
//!
//! This is the key benefit of separate bindings: a compute shader can
//! read/write a single attribute's buffer without needing to understand
//! the layout of other attributes.

use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::allocator::{MeshAllocator, MeshAllocatorSettings},
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer, uniform_buffer},
            BufferUsages, *,
        },
        renderer::{RenderContext, RenderQueue},
        Render, RenderApp, RenderStartup,
    },
};

const SHADER_ASSET_PATH: &str = "shaders/compute_vertex_displacement.wgsl";

const SUBDIVISIONS: u32 = 64;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            VertexDisplacementPlugin,
            ExtractComponentPlugin::<DisplaceMesh>::default(),
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .run();
}

struct VertexDisplacementPlugin;

impl Plugin for VertexDisplacementPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_displacement_pipeline)
            .add_systems(Render, run_displacement.before(camera_driver));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages = BufferUsages::STORAGE;
    }
}

/// Marker component: the mesh on this entity will be displaced.
#[derive(Component, ExtractComponent, Clone)]
struct DisplaceMesh(Handle<Mesh>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut plane = Plane3d::default()
        .mesh()
        .size(10.0, 10.0)
        .subdivisions(SUBDIVISIONS)
        .build();

    // Put each attribute into its own vertex buffer binding so the compute
    // shader can write to position without touching normals or UVs.
    plane.deinterleave();

    let handle = meshes.add(plane);

    commands.spawn((
        DisplaceMesh(handle.clone()),
        Mesh3d(handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.6, 1.0),
            perceptual_roughness: 0.4,
            ..default()
        })),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(20.0, 20.0).build())),
        MeshMaterial3d(materials.add(Color::srgb(0.15, 0.15, 0.15))),
        Transform::from_xyz(0.0, -1.0, 0.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -1.0, 0.5, 0.0)),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8.0, 6.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

#[derive(Resource)]
struct DisplacementPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

#[derive(ShaderType)]
struct DisplacementParams {
    position_start: u32,
    vertex_count: u32,
    time: f32,
}

fn init_displacement_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "displacement_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DisplacementParams>(false),
                storage_buffer::<Vec<u32>>(false),
            ),
        ),
    );
    let shader = asset_server.load(SHADER_ASSET_PATH);
    let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("vertex_displacement_pipeline".into()),
        layout: vec![layout.clone()],
        shader,
        ..default()
    });
    commands.insert_resource(DisplacementPipeline { layout, pipeline });
}

fn run_displacement(
    mut render_context: RenderContext,
    meshes_to_displace: Query<&DisplaceMesh>,
    render_meshes: Res<RenderAssets<bevy::render::mesh::RenderMesh>>,
    mesh_allocator: Res<MeshAllocator>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<DisplacementPipeline>,
    render_queue: Res<RenderQueue>,
    time: Res<Time>,
) {
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    for displace_mesh in &meshes_to_displace {
        let mesh_id = displace_mesh.0.id();

        let Some(gpu_mesh) = render_meshes.get(mesh_id) else {
            continue;
        };

        let Some(pos_binding) =
            gpu_mesh.layout.0.binding_index_for_attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            continue;
        };

        let Some(pos_slice) = mesh_allocator.mesh_vertex_slice(&mesh_id, pos_binding as u8)
        else {
            continue;
        };

        let params = DisplacementParams {
            // Each vertex is 3 f32s, so multiply element offset by 3.
            position_start: pos_slice.range.start * 3,
            vertex_count: gpu_mesh.vertex_count,
            time: time.elapsed_secs(),
        };

        let mut uniforms = UniformBuffer::from(params);
        uniforms.write_buffer(render_context.render_device(), &render_queue);

        let bind_group = render_context.render_device().create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(&pipeline.layout),
            &BindGroupEntries::sequential((
                &uniforms,
                pos_slice.buffer.as_entire_buffer_binding(),
            )),
        );

        let workgroup_count = gpu_mesh.vertex_count.div_ceil(64);

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("vertex_displacement_pass"),
                    ..default()
                });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(compute_pipeline);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
