//! Demonstrates manual multi-material extraction with submesh slots.
//!
//! This example shows material properties that CANNOT be achieved with a single material,
//! even using textures - proving the value of multi-material support:
//!
//! 1. **Left sphere (Air vs Moissanite IOR)**: Different regions have different indices of
//!    refraction. There is NO `ior_texture` in StandardMaterial, so varying IOR per-region
//!    is impossible with a single material. Moissanite (IOR 2.65) bends light dramatically
//!    while air (IOR 1.0) shows almost no refraction.
//!
//! 2. **Center sphere (PBR vs Fresnel Glow)**: Combines StandardMaterial with a completely
//!    different custom shader. This is impossible with any single material - you cannot mix
//!    PBR lighting with a non-physical fresnel edge glow effect.
//!
//! 3. **Right sphere (Clear vs Amber attenuation)**: Different regions absorb different
//!    wavelengths of light. There is NO `attenuation_color_texture`, so varying the color
//!    tint of transmitted light is impossible with a single material. Amber regions absorb
//!    blue light while clear regions stay neutral.
//!
//! The multi-material model works as follows:
//! - Submeshes (geometry slices) are defined on the mesh asset via `Mesh::with_inserted_submeshes`
//! - Each `SubMesh` defines an index range and material slot
//! - `RenderMaterialInstance` uses `submesh_index` to reference the submesh slot:
//!   - Slot 0 = full mesh (default for single-material)
//!   - Slots 1+ = submeshes from the mesh asset

use bevy::{
    core_pipeline::Skybox,
    ecs::system::SystemChangeTick,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    mesh::{Indices, PrimitiveTopology, SubMesh},
    pbr::{
        check_entities_needing_specialization, EntitiesNeedingSpecialization,
        EntitySpecializationTickPair, EntitySpecializationTicks,
        MaterialExtractEntitiesNeedingSpecializationSystems, MaterialExtractionSystems,
        RenderMaterialInstance, RenderMaterialInstances, SpecializedMaterialPipelineCache,
    },
    prelude::*,
    reflect::TypePath,
    render::{
        render_resource::AsBindGroup, sync_world::MainEntity, view::ExtractedView, Extract,
        RenderApp,
    },
    shader::ShaderRef,
    utils::Parallel,
};
use std::{any::TypeId, f32::consts::PI};

/// Custom corruption material - glitchy, reality-breaking effect.
/// This is a completely different shader from StandardMaterial, demonstrating
/// that multi-material support can mix entirely different material types.
const CORRUPTION_SHADER: &str = "shaders/fresnel_glow.wgsl";

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct CorruptionMaterial {
    #[uniform(0)]
    pub color_a: LinearRgba,
    #[uniform(1)]
    pub color_b: LinearRgba,
    #[uniform(2)]
    pub glitch_speed: f32,
    #[texture(3)]
    #[sampler(4)]
    pub noise_texture: Handle<Image>,
}

impl Material for CorruptionMaterial {
    fn fragment_shader() -> ShaderRef {
        CORRUPTION_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }
}

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<CorruptionMaterial>::default(),
            MultiMaterialExtractionPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_sphere)
        .run();
}

/// Marker for center sphere (mixed material types demo)
#[derive(Component)]
struct CenterSphere;

/// Marker for rotating sphere
#[derive(Component)]
struct RotatingSphere;

fn rotate_sphere(time: Res<Time>, mut query: Query<&mut Transform, With<RotatingSphere>>) {
    for mut transform in &mut query {
        transform.rotation = Quat::from_rotation_y(time.elapsed_secs() * 0.15);
    }
}


/// Plugin that demonstrates multi-material extraction.
struct MultiMaterialExtractionPlugin;

impl Plugin for MultiMaterialExtractionPlugin {
    fn build(&self, app: &mut App) {
        // Track entities needing specialization when MultiMaterialSlots or MixedMaterialSlots changes.
        // IMPORTANT: Must run AFTER the standard material's check system, which clears
        // the EntitiesNeedingSpecialization resource each frame.
        app.init_resource::<EntitiesNeedingSpecialization<StandardMaterial>>()
            .init_resource::<EntitiesNeedingSpecialization<CorruptionMaterial>>()
            .add_systems(
                PostUpdate,
                (
                    check_multi_material_entities_needing_specialization
                        .after(check_entities_needing_specialization::<StandardMaterial>),
                    check_mixed_material_entities_needing_specialization
                        .after(check_entities_needing_specialization::<StandardMaterial>)
                        .after(check_entities_needing_specialization::<CorruptionMaterial>),
                ),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            ExtractSchedule,
            (
                extract_multi_materials
                    .in_set(MaterialExtractionSystems)
                    .ambiguous_with_all(),
                extract_mixed_materials
                    .in_set(MaterialExtractionSystems)
                    .ambiguous_with_all(),
                extract_multi_materials_needing_specialization
                    .in_set(MaterialExtractEntitiesNeedingSpecializationSystems),
                extract_mixed_materials_needing_specialization
                    .in_set(MaterialExtractEntitiesNeedingSpecializationSystems),
            ),
        );
    }
}

/// Component that holds multiple StandardMaterials with their associated submesh slots.
/// This is a manual multi-material component for demonstration purposes.
#[derive(Component)]
struct MultiMaterialSlots {
    /// Each entry is (material handle, submesh slot index).
    /// Slot indices correspond to submeshes defined on the mesh asset:
    /// - Slot 1 = first submesh from `mesh.submeshes()`
    /// - Slot 2 = second submesh, etc.
    /// (Slot 0 is reserved for the full mesh)
    materials: Vec<(Handle<StandardMaterial>, u16)>,
}

impl MultiMaterialSlots {
    fn new(materials: Vec<(Handle<StandardMaterial>, u16)>) -> Self {
        Self { materials }
    }
}

/// Component that holds a mix of StandardMaterial and CorruptionMaterial.
/// This demonstrates the ultimate multi-material capability: different shader types
/// on different regions of the same mesh.
#[derive(Component)]
struct MixedMaterialSlots {
    standard: Vec<(Handle<StandardMaterial>, u16)>,
    fresnel: Vec<(Handle<CorruptionMaterial>, u16)>,
}

impl MixedMaterialSlots {
    fn new(
        standard: Vec<(Handle<StandardMaterial>, u16)>,
        fresnel: Vec<(Handle<CorruptionMaterial>, u16)>,
    ) -> Self {
        Self { standard, fresnel }
    }
}

/// Creates a UV sphere mesh with full indices, plus a submesh for 2/3 of the sphere.
/// - Slot 0: Full sphere (for PBR base layer)
/// - Slot 1: 2/3 sphere submesh (for corruption overlay)
/// The submesh uses radial UVs centered at (0.5, 0.5) for proper noise mask falloff
fn create_partial_sphere_submesh(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let sectors_f32 = sectors as f32;
    let stacks_f32 = stacks as f32;
    let length_inv = 1. / radius;
    let sector_step = 2. * PI / sectors_f32;
    let stack_step = PI / stacks_f32;
    let partial_sectors = (sectors * 2) / 3; // 2/3 of the sphere

    // First, create the full sphere vertices (for PBR)
    let full_sphere_vertex_count = ((stacks + 1) * (sectors + 1)) as usize;
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(full_sphere_vertex_count * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(full_sphere_vertex_count * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(full_sphere_vertex_count * 2);

    // Full sphere vertices with standard UVs
    for i in 0..stacks + 1 {
        let stack_angle = PI / 2. - (i as f32) * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        for j in 0..sectors + 1 {
            let sector_angle = (j as f32) * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            vertices.push([x, z, y]);
            normals.push([x * length_inv, z * length_inv, y * length_inv]);
            // Standard sphere UVs for PBR
            uvs.push([(j as f32) / sectors_f32, (i as f32) / stacks_f32]);
        }
    }

    // Partial-sphere vertices with RADIAL UVs centered at (0.5, 0.5)
    // This maps the center of the partial-sphere to UV (0.5, 0.5) and edges to the border
    let partial_vertex_start = vertices.len() as u32;

    // Center of partial-sphere in spherical coords
    let center_sector = partial_sectors as f32 / 2.0;
    let center_stack = stacks_f32 / 2.0;

    for i in 0..stacks + 1 {
        let stack_angle = PI / 2. - (i as f32) * stack_step;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();

        // 2/3 of sectors + 1 for the seam
        for j in 0..partial_sectors + 1 {
            let sector_angle = (j as f32) * sector_step;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();

            vertices.push([x, z, y]);
            normals.push([x * length_inv, z * length_inv, y * length_inv]);

            // Radial UVs: distance from center of partial-sphere determines UV distance from (0.5, 0.5)
            // Normalize sector/stack to [-1, 1] range relative to center
            let norm_sector = ((j as f32) - center_sector) / center_sector; // -1 to 1
            let norm_stack = ((i as f32) - center_stack) / center_stack;    // -1 to 1

            // Map to [0, 1] UV space centered at 0.5
            let u = 0.5 + norm_sector * 0.5;
            let v = 0.5 + norm_stack * 0.5;
            uvs.push([u, v]);
        }
    }

    // Build full sphere indices
    let mut all_indices: Vec<u32> = Vec::new();

    for i in 0..stacks {
        let mut k1 = i * (sectors + 1);
        let mut k2 = k1 + sectors + 1;

        for _j in 0..sectors {
            if i != 0 {
                all_indices.push(k1);
                all_indices.push(k2);
                all_indices.push(k1 + 1);
            }
            if i != stacks - 1 {
                all_indices.push(k1 + 1);
                all_indices.push(k2);
                all_indices.push(k2 + 1);
            }
            k1 += 1;
            k2 += 1;
        }
    }

    // Build partial-sphere indices using the SECOND vertex set (with centered UVs)
    let partial_start = all_indices.len() as u32;

    for i in 0..stacks {
        let mut k1 = partial_vertex_start + i * (partial_sectors + 1);
        let mut k2 = k1 + partial_sectors + 1;

        for _j in 0..partial_sectors {
            if i != 0 {
                all_indices.push(k1);
                all_indices.push(k2);
                all_indices.push(k1 + 1);
            }
            if i != stacks - 1 {
                all_indices.push(k1 + 1);
                all_indices.push(k2);
                all_indices.push(k2 + 1);
            }
            k1 += 1;
            k2 += 1;
        }
    }

    let partial_end = all_indices.len() as u32;

    let submeshes = vec![SubMesh::new(partial_start..partial_end, 1)];

    info!(
        "Created sphere with 2/3 submesh: {} full triangles, {} partial triangles, {} vertices",
        (partial_start as usize) / 3,
        (partial_end - partial_start) as usize / 3,
        vertices.len()
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
    mut corruption_materials: ResMut<Assets<CorruptionMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // === CENTER SPHERE: PBR vs Fresnel Glow ===
    // IMPOSSIBLE with single material - completely different shader types!
    // Chrome PBR - physically-based reflections
    let chrome_pbr = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.9, 0.95),
        metallic: 1.0,
        perceptual_roughness: 0.1,
        reflectance: 1.0,
        ..default()
    });

    // Corruption - glitchy, reality-breaking effect using noise texture with clamping
    // Use clamp mode so edges sample the black border instead of wrapping
    let noise_texture: Handle<Image> = asset_server.load_with_settings(
        "textures/corruption_mask.png",
        |s: &mut bevy::image::ImageLoaderSettings| {
            s.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::ClampToEdge,
                address_mode_v: ImageAddressMode::ClampToEdge,
                ..default()
            });
        },
    );
    let corruption = corruption_materials.add(CorruptionMaterial {
        color_a: LinearRgba::new(1.0, 0.0, 0.5, 1.0), // Hot pink
        color_b: LinearRgba::new(0.0, 1.0, 1.0, 1.0), // Cyan
        glitch_speed: 3.0,
        noise_texture,
    });

    // Create sphere with 2/3 as a submesh for corruption overlay
    // PBR renders on full mesh, corruption only on the partial submesh
    let center_sphere_mesh = create_partial_sphere_submesh(1.0, 64, 32);
    let center_sphere_handle = meshes.add(center_sphere_mesh);

    commands.spawn((
        Mesh3d(center_sphere_handle),
        // PBR on slot 0 (full mesh), corruption on slot 1 (partial submesh)
        MixedMaterialSlots::new(
            vec![(chrome_pbr, 0)],     // PBR base - full sphere
            vec![(corruption, 1)],     // Corruption overlay - 2/3 sphere with noise dissolve
        ),
        Transform::from_xyz(0.0, 1.0, 0.0),
        CenterSphere,
        RotatingSphere,
    ));

    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.3),
            perceptual_roughness: 0.8,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // Light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Camera - positioned to see both materials as sphere rotates
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 1.5, 4.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        EnvironmentMapLight {
            intensity: 500.0,
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            ..default()
        },
        Skybox {
            brightness: 1000.0,
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            ..default()
        },
    ));

    info!("Manual multi-material example loaded.");
    info!("Chrome PBR (top) vs Corruption shader (bottom) - different shader types on same mesh!");
}

/// Checks for entities needing specialization when MultiMaterialSlots changes.
fn check_multi_material_entities_needing_specialization(
    needs_specialization: Query<
        Entity,
        (
            Or<(
                Changed<Mesh3d>,
                AssetChanged<Mesh3d>,
                Added<MultiMaterialSlots>,
                Changed<MultiMaterialSlots>,
            )>,
            With<MultiMaterialSlots>,
        ),
    >,
    mut par_local: Local<Parallel<Vec<Entity>>>,
    mut entities_needing_specialization: ResMut<EntitiesNeedingSpecialization<StandardMaterial>>,
) {
    // Note: Don't clear - let standard material extraction add its entities too
    needs_specialization
        .par_iter()
        .for_each(|entity| par_local.borrow_local_mut().push(entity));

    par_local.drain_into(&mut entities_needing_specialization);
}

/// Extracts multi-material instances into RenderMaterialInstances.
/// Each material gets its own entry with the corresponding submesh_index.
fn extract_multi_materials(
    mut material_instances: ResMut<RenderMaterialInstances>,
    changed_query: Extract<
        Query<
            (Entity, &ViewVisibility, &MultiMaterialSlots),
            Or<(Changed<ViewVisibility>, Changed<MultiMaterialSlots>)>,
        >,
    >,
) {
    let last_change_tick = material_instances.current_change_tick;
    let material_type_id = TypeId::of::<StandardMaterial>();

    for (entity, view_visibility, multi_material) in &changed_query {
        let main_entity = MainEntity::from(entity);

        if view_visibility.get() {
            // First, collect existing instance_ids from our multi-material entries
            // (those with submesh_index > 0, since slot 0 is the full mesh)
            let existing_instance_ids: Vec<_> = material_instances
                .instances
                .get(&main_entity)
                .map(|instances| {
                    instances
                        .iter()
                        .filter(|inst| {
                            inst.asset_id.type_id() == material_type_id && inst.submesh_index > 0
                        })
                        .map(|inst| inst.instance_id)
                        .collect()
                })
                .unwrap_or_default();

            // Pre-allocate new instance IDs for any materials beyond existing count
            let new_count = multi_material
                .materials
                .len()
                .saturating_sub(existing_instance_ids.len());
            let new_instance_ids: Vec<_> = (0..new_count)
                .map(|_| material_instances.allocate_instance_id())
                .collect();

            let instances = material_instances.instances.entry(main_entity).or_default();

            // Remove ALL old StandardMaterial entries (both from standard extraction and our custom ones)
            // since we're replacing them entirely with our multi-material entries
            instances.retain(|inst| inst.asset_id.type_id() != material_type_id);

            // Add new entries for each material with its submesh slot
            // Reuse existing instance_ids where possible for stability
            for (i, (handle, submesh_slot)) in multi_material.materials.iter().enumerate() {
                let instance_id = if i < existing_instance_ids.len() {
                    existing_instance_ids[i]
                } else {
                    new_instance_ids[i - existing_instance_ids.len()]
                };

                instances.push(RenderMaterialInstance {
                    asset_id: handle.id().untyped(),
                    submesh_index: *submesh_slot,
                    last_change_tick,
                    instance_id,
                });
            }

            info!(
                "Extracted {} material instances with submesh slots for entity {:?} (total: {})",
                multi_material.materials.len(),
                main_entity,
                instances.len()
            );
        } else if let Some(instances) = material_instances.instances.get_mut(&main_entity) {
            // Remove our multi-material entries when invisible
            // (those with submesh_index > 0)
            instances.retain(|inst| {
                inst.asset_id.type_id() != material_type_id || inst.submesh_index == 0
            });
        }
    }
}

/// Extracts entities needing specialization for multi-material.
fn extract_multi_materials_needing_specialization(
    entities_needing_specialization: Extract<Res<EntitiesNeedingSpecialization<StandardMaterial>>>,
    mut entity_specialization_ticks: ResMut<EntitySpecializationTicks>,
    mut removed_components: Extract<RemovedComponents<MultiMaterialSlots>>,
    mut specialized_material_pipeline_cache: ResMut<SpecializedMaterialPipelineCache>,
    render_material_instances: Res<RenderMaterialInstances>,
    views: Query<&ExtractedView>,
    ticks: SystemChangeTick,
) {
    // Clean up despawned entities
    for entity in removed_components.read() {
        let main_entity = MainEntity::from(entity);
        // Only remove if not re-added this frame
        if entity_specialization_ticks
            .get(&main_entity)
            .is_some_and(|ticks| {
                ticks.material_instances_tick == render_material_instances.current_change_tick
            })
        {
            continue;
        }

        entity_specialization_ticks.remove(&main_entity);
        for view in &views {
            if let Some(cache) =
                specialized_material_pipeline_cache.get_mut(&view.retained_view_entity)
            {
                // Use retain to filter out entries for this main_entity
                cache.retain(|phase_item_id, _| phase_item_id.main_entity != main_entity);
            }
        }
    }

    // Track specialization ticks for entities with multi-materials
    for entity in entities_needing_specialization.iter() {
        let main_entity = MainEntity::from(*entity);
        entity_specialization_ticks.insert(
            main_entity,
            EntitySpecializationTickPair {
                system_tick: ticks.this_run(),
                material_instances_tick: render_material_instances.current_change_tick,
            },
        );
    }
}

/// Checks for entities needing specialization when MixedMaterialSlots changes.
fn check_mixed_material_entities_needing_specialization(
    needs_specialization: Query<
        Entity,
        (
            Or<(
                Changed<Mesh3d>,
                AssetChanged<Mesh3d>,
                Added<MixedMaterialSlots>,
                Changed<MixedMaterialSlots>,
            )>,
            With<MixedMaterialSlots>,
        ),
    >,
    mut par_local_standard: Local<Parallel<Vec<Entity>>>,
    mut par_local_fresnel: Local<Parallel<Vec<Entity>>>,
    mut standard_entities: ResMut<EntitiesNeedingSpecialization<StandardMaterial>>,
    mut fresnel_entities: ResMut<EntitiesNeedingSpecialization<CorruptionMaterial>>,
) {
    // Add to both specialization lists since MixedMaterialSlots contains both types
    needs_specialization
        .par_iter()
        .for_each(|entity| {
            par_local_standard.borrow_local_mut().push(entity);
            par_local_fresnel.borrow_local_mut().push(entity);
        });

    par_local_standard.drain_into(&mut standard_entities);
    par_local_fresnel.drain_into(&mut fresnel_entities);
}

/// Extracts mixed material instances (StandardMaterial + CorruptionMaterial) into RenderMaterialInstances.
fn extract_mixed_materials(
    mut material_instances: ResMut<RenderMaterialInstances>,
    changed_query: Extract<
        Query<
            (Entity, &ViewVisibility, &MixedMaterialSlots),
            Or<(Changed<ViewVisibility>, Changed<MixedMaterialSlots>)>,
        >,
    >,
) {
    let last_change_tick = material_instances.current_change_tick;
    let standard_type_id = TypeId::of::<StandardMaterial>();
    let fresnel_type_id = TypeId::of::<CorruptionMaterial>();

    for (entity, view_visibility, mixed_material) in &changed_query {
        let main_entity = MainEntity::from(entity);

        if view_visibility.get() {
            // Pre-allocate instance IDs before borrowing instances
            let total_materials = mixed_material.standard.len() + mixed_material.fresnel.len();
            let instance_ids: Vec<_> = (0..total_materials)
                .map(|_| material_instances.allocate_instance_id())
                .collect();

            let instances = material_instances.instances.entry(main_entity).or_default();

            // Remove old entries for both material types
            instances.retain(|inst| {
                inst.asset_id.type_id() != standard_type_id
                    && inst.asset_id.type_id() != fresnel_type_id
            });

            let mut id_iter = instance_ids.into_iter();

            // Add StandardMaterial entries
            for (handle, submesh_slot) in &mixed_material.standard {
                instances.push(RenderMaterialInstance {
                    asset_id: handle.id().untyped(),
                    submesh_index: *submesh_slot,
                    last_change_tick,
                    instance_id: id_iter.next().unwrap(),
                });
            }

            // Add CorruptionMaterial entries
            for (handle, submesh_slot) in &mixed_material.fresnel {
                instances.push(RenderMaterialInstance {
                    asset_id: handle.id().untyped(),
                    submesh_index: *submesh_slot,
                    last_change_tick,
                    instance_id: id_iter.next().unwrap(),
                });
            }

            info!(
                "Extracted {} standard + {} fresnel material instances for entity {:?}",
                mixed_material.standard.len(),
                mixed_material.fresnel.len(),
                main_entity,
            );
        } else if let Some(instances) = material_instances.instances.get_mut(&main_entity) {
            // Remove our mixed material entries when invisible
            instances.retain(|inst| {
                inst.asset_id.type_id() != standard_type_id
                    && inst.asset_id.type_id() != fresnel_type_id
            });
        }
    }
}

/// Extracts entities needing specialization for mixed materials.
fn extract_mixed_materials_needing_specialization(
    standard_entities: Extract<Res<EntitiesNeedingSpecialization<StandardMaterial>>>,
    fresnel_entities: Extract<Res<EntitiesNeedingSpecialization<CorruptionMaterial>>>,
    mut entity_specialization_ticks: ResMut<EntitySpecializationTicks>,
    mut removed_components: Extract<RemovedComponents<MixedMaterialSlots>>,
    mut specialized_material_pipeline_cache: ResMut<SpecializedMaterialPipelineCache>,
    render_material_instances: Res<RenderMaterialInstances>,
    views: Query<&ExtractedView>,
    ticks: SystemChangeTick,
) {
    // Clean up despawned entities
    for entity in removed_components.read() {
        let main_entity = MainEntity::from(entity);
        if entity_specialization_ticks
            .get(&main_entity)
            .is_some_and(|ticks| {
                ticks.material_instances_tick == render_material_instances.current_change_tick
            })
        {
            continue;
        }

        entity_specialization_ticks.remove(&main_entity);
        for view in &views {
            if let Some(cache) =
                specialized_material_pipeline_cache.get_mut(&view.retained_view_entity)
            {
                cache.retain(|phase_item_id, _| phase_item_id.main_entity != main_entity);
            }
        }
    }

    // Track specialization ticks for entities with mixed materials
    // (entities appear in both standard and fresnel lists)
    for entity in standard_entities.iter().chain(fresnel_entities.iter()) {
        let main_entity = MainEntity::from(*entity);
        entity_specialization_ticks.insert(
            main_entity,
            EntitySpecializationTickPair {
                system_tick: ticks.this_run(),
                material_instances_tick: render_material_instances.current_change_tick,
            },
        );
    }
}
