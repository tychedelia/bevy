//! Demonstrates splitting vertex attributes across multiple GPU buffers.
//!
//! This example places static geometry (positions, normals) in vertex buffer slot 0,
//! and dynamic per-vertex height offsets in slot 1.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError, VertexFormat,
    },
};

const ATTRIBUTE_HEIGHT_OFFSET: MeshVertexAttribute =
    MeshVertexAttribute::new("HeightOffset", 1, VertexFormat::Float32).with_slot_index(1);

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, MaterialPlugin::<WaveMaterial>::default()))
        .add_systems(Startup, setup)
        .add_systems(Update, animate_height_offsets)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaveMaterial>>,
) {
    let mesh = create_wave_mesh(16, 16);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(WaveMaterial {})),
        WaveMesh,
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(3.0, 3.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

#[derive(Component)]
struct WaveMesh;

fn create_wave_mesh(width: u32, height: u32) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut height_offsets = Vec::new();
    let mut indices = Vec::new();

    let half_w = width as f32 / 2.0;
    let half_h = height as f32 / 2.0;
    let scale = 0.2;

    for z in 0..=height {
        for x in 0..=width {
            let px = (x as f32 - half_w) * scale;
            let pz = (z as f32 - half_h) * scale;
            positions.push([px, 0.0, pz]);
            normals.push([0.0, 1.0, 0.0]);
            height_offsets.push(0.0_f32);
        }
    }

    for z in 0..height {
        for x in 0..width {
            let i = z * (width + 1) + x;
            indices.push(i);
            indices.push(i + width + 1);
            indices.push(i + 1);
            indices.push(i + 1);
            indices.push(i + width + 1);
            indices.push(i + width + 2);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all())
        .with_inserted_indices(Indices::U32(indices))
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(ATTRIBUTE_HEIGHT_OFFSET, height_offsets)
}

fn animate_height_offsets(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh3d, With<WaveMesh>>,
) {
    let t = time.elapsed_secs();

    for mesh_handle in &query {
        let Some(mesh) = meshes.get_mut(mesh_handle) else {
            continue;
        };

        let Some(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
            continue;
        };
        let positions: Vec<[f32; 3]> = positions.as_float3().unwrap().to_vec();

        let offsets: Vec<f32> = positions
            .iter()
            .map(|[x, _, z]| (x * 3.0 + z * 2.0 + t * 2.0).sin() * 0.15)
            .collect();

        mesh.insert_attribute(ATTRIBUTE_HEIGHT_OFFSET, offsets);
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct WaveMaterial {}

impl Material for WaveMaterial {
    fn vertex_shader() -> bevy::shader::ShaderRef {
        "shaders/split_vertex_attributes.wgsl".into()
    }

    fn fragment_shader() -> bevy::shader::ShaderRef {
        "shaders/split_vertex_attributes.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layouts = layout.0.get_layouts(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_HEIGHT_OFFSET.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vertex_layouts;
        Ok(())
    }
}
