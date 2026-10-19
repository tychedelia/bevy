//! Renders several cameras into one [`MainTexture`] and post-processes the result once.
//!
//! Every camera renders into a main texture before its output is written to its render target.
//! Cameras usually get that texture chosen for them. Here three cameras point at the same
//! [`MainTexture`] entity instead:
//!
//! - The world camera and the overlay camera draw the scene. They don't tonemap and they don't
//!   write to the window.
//! - A [`PostProcessCamera`] draws nothing. It runs bloom and tonemapping over what the other
//!   two left in the texture, and writes the result to the window.

use bevy::{
    camera::{visibility::RenderLayers, CameraMainTexture, CameraOutputMode, MainTexture},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::TextureFormat,
};

const OVERLAY_LAYER: RenderLayers = RenderLayers::layer(1);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate)
        .run();
}

#[derive(Component)]
struct Rotates;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // An HDR format keeps values above 1.0 for bloom and tonemapping.
    let main_texture = commands
        .spawn(MainTexture {
            format: Some(TextureFormat::Rgba16Float),
            ..default()
        })
        .id();

    commands.spawn((
        Camera3d::default(),
        Camera {
            output_mode: CameraOutputMode::Skip,
            ..default()
        },
        Tonemapping::None,
        DebandDither::Disabled,
        Transform::from_xyz(0.0, 3.0, 8.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraMainTexture(main_texture),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 1,
            // Draw over what the world camera rendered.
            clear_color: ClearColorConfig::None,
            output_mode: CameraOutputMode::Skip,
            ..default()
        },
        Tonemapping::None,
        DebandDither::Disabled,
        Transform::from_xyz(0.0, 0.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
        CameraMainTexture(main_texture),
        OVERLAY_LAYER,
    ));

    commands.spawn((
        PostProcessCamera,
        Camera {
            order: 2,
            ..default()
        },
        Bloom::NATURAL,
        CameraMainTexture(main_texture),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials.add(Color::srgb(0.8, 0.7, 0.6))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Rotates,
    ));
    commands.spawn((
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(0.3))),
        MeshMaterial3d(materials.add(StandardMaterial {
            emissive: LinearRgba::rgb(20.0, 4.0, 1.0),
            ..default()
        })),
        Transform::from_xyz(1.2, -0.8, 0.0),
        OVERLAY_LAYER,
    ));
}

fn rotate(time: Res<Time>, mut query: Query<&mut Transform, With<Rotates>>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_secs());
    }
}
