//! Demonstrates submeshes - a single mesh divided into multiple draw regions.
//!
//! This example creates a UV sphere and divides it into three horizontal bands,
//! each represented as a separate submesh. In a full multi-material system,
//! each submesh could use a different material.

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use std::f32::consts::PI;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

/// Creates a UV sphere mesh with submeshes dividing it into horizontal bands.
fn create_banded_sphere(radius: f32, sectors: u32, stacks: u32, bands: u32) -> Mesh {
    // Generate UV sphere geometry (based on bevy_mesh sphere.rs)
    let sectors_f32 = sectors as f32;
    let stacks_f32 = stacks as f32;
    let length_inv = 1. / radius;
    let sector_step = 2. * PI / sectors_f32;
    let stack_step = PI / stacks_f32;

    let n_vertices = ((stacks + 1) * (sectors + 1)) as usize;
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n_vertices);

    // Generate vertices
    for i in 0..stacks + 1 {
        let stack_angle = PI / 2. - (i as f32) * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..sectors + 1 {
            let sector_angle = (j as f32) * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            vertices.push([x, z, y]); // Note: swapped y/z for Y-up
            normals.push([x * length_inv, z * length_inv, y * length_inv]);
            uvs.push([(j as f32) / sectors_f32, (i as f32) / stacks_f32]);
        }
    }

    // Generate indices, tracking where each band starts/ends
    let stacks_per_band = stacks / bands;
    let mut all_indices: Vec<u32> = Vec::new();
    let mut band_ranges: Vec<std::ops::Range<u32>> = Vec::new();

    for band in 0..bands {
        let band_start_stack = band * stacks_per_band;
        let band_end_stack = if band == bands - 1 {
            stacks // Last band gets remaining stacks
        } else {
            (band + 1) * stacks_per_band
        };

        let index_start = all_indices.len() as u32;

        for i in band_start_stack..band_end_stack {
            let mut k1 = i * (sectors + 1);
            let mut k2 = k1 + sectors + 1;

            for _j in 0..sectors {
                // Top cap triangles (only at i == 0)
                if i != 0 {
                    all_indices.push(k1);
                    all_indices.push(k2);
                    all_indices.push(k1 + 1);
                }
                // Bottom cap triangles (only at i == stacks - 1)
                if i != stacks - 1 {
                    all_indices.push(k1 + 1);
                    all_indices.push(k2);
                    all_indices.push(k2 + 1);
                }
                k1 += 1;
                k2 += 1;
            }
        }

        let index_end = all_indices.len() as u32;
        band_ranges.push(index_start..index_end);
    }

    // Create submeshes for each band
    let submeshes: Vec<bevy::mesh::SubMesh> = band_ranges
        .into_iter()
        .enumerate()
        .map(|(i, range)| bevy::mesh::SubMesh::new(range, i as u32))
        .collect();

    info!(
        "Created banded sphere with {} submeshes: {:?}",
        submeshes.len(),
        submeshes
    );

    Mesh::new(
        PrimitiveTopology::TriangleList,
        bevy::asset::RenderAssetUsages::default(),
    )
    .with_inserted_indices(Indices::U32(all_indices))
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_submeshes(submeshes)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create a banded sphere with 3 submeshes (top, middle, bottom bands)
    let sphere_mesh = create_banded_sphere(1.0, 32, 18, 3);
    let sphere_handle = meshes.add(sphere_mesh);

    // Spawn the sphere
    // Note: Currently all submeshes use the same material since we haven't
    // implemented MeshMaterials<M> yet. But the GPU will emit 3 separate
    // draw calls, one per submesh.
    commands.spawn((
        Mesh3d(sphere_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.2, 0.2),
            ..default()
        })),
        Transform::from_xyz(0.0, 1.0, 0.0),
    ));

    // Ground plane for reference
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.3))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // Light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            intensity: 2_000_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-3.0, 3.0, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    info!("Submesh example loaded. The sphere has 3 submeshes (horizontal bands).");
    info!("Each submesh generates a separate draw call.");
}
