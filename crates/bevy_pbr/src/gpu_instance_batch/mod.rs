//! GPU-authored instance batches.
//!
//! This module exposes a mechanism for rendering `N` instances of a `Mesh`
//! whose per-instance transform data is written into the GPU preprocessing
//! input buffer by the user's own compute shaders, rather than extracted from
//! ECS entities on the CPU. It is useful for GPU-driven particle systems,
//! foliage scatter, point clouds, GPU-baked animation output, and any other
//! system that produces per-instance transforms as the output of a compute
//! pipeline.
//!
//! # Usage
//!
//! A user spawns one entity per batch:
//!
//! ```ignore
//! commands.spawn((
//!     GpuInstanceBatch {
//!         mesh: mesh_handle,
//!         max_capacity: 4096,
//!         aabb: emitter_bounds,
//!         dispatch_args_buffer: my_dispatch_args_buffer,
//!         dispatch_args_offset: 0,
//!         flags: MeshFlags::empty(),
//!     },
//!     MeshMaterial3d(material_handle),
//! ));
//! ```
//!
//! No `Mesh3d` is attached: the mesh handle lives inside `GpuInstanceBatch`,
//! which prevents the entity from being picked up by normal mesh extraction.
//! The `MeshMaterial3d<M>` component is extracted by the existing
//! `MaterialPlugin<M>`, which populates the type-erased `RenderMaterialInstances`
//! the rendering code here reads from.
//!
//! # Scope (v1)
//!
//! - Forward opaque rendering only. Shadow casting, prepass, and deferred are
//!   not yet wired up for batches.
//! - No motion vectors / TAA support (`previous_input_index` is always
//!   `u32::MAX`).
//! - No late-phase occlusion culling for batches — batches participate only in
//!   the frustum-culling early pass.
//! - `max_capacity` is CPU-declared and mutable only from the main world.

use bevy_app::{App, Plugin};
use bevy_asset::AssetId;
use bevy_camera::primitives::Aabb;
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_log::warn;
use bevy_math::{UVec2, Vec3, Vec4};
use bevy_mesh::Mesh;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::batching::gpu_preprocessing::{BatchedInstanceBuffers, GpuPreprocessingSupport};
use bevy_render::mesh::allocator::MeshAllocator;
use bevy_render::render_resource::Buffer;
use bevy_render::sync_world::{MainEntity, MainEntityHashMap};
use bevy_render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};

use crate::{
    MaterialExtractionSystems, MeshCullingData, MeshCullingDataBuffer, MeshFlags, MeshInputUniform,
    MeshUniform, RenderMaterialBindings, RenderMaterialInstances,
};

/// Component declaring that an entity represents a batch of up to
/// `max_capacity` GPU-authored mesh instances.
///
/// The user must also attach a `MeshMaterial3d<M>` component for whatever
/// material type `M` they want the batch to render with; the existing
/// `MaterialPlugin<M>` handles material extraction.
///
/// Do **not** attach a `Mesh3d` component: the mesh handle is stored inside
/// this component so that normal mesh extraction ignores the entity.
#[derive(Component, Clone)]
pub struct GpuInstanceBatch {
    /// The mesh whose vertices and indices are drawn, instanced
    /// `max_capacity` times.
    pub mesh: bevy_asset::Handle<Mesh>,
    /// Upper bound on the number of instances this batch can render.
    ///
    /// Contiguous slots are reserved in the preprocessing input buffer,
    /// culling data buffer, and indirect work-item buffer at this size. The
    /// live instance count may vary 0..=`max_capacity` each frame, driven by
    /// the user's dispatch-args buffer.
    pub max_capacity: u32,
    /// Axis-aligned bounding box that is copied into every slot of the
    /// culling data buffer for this batch. A single AABB is shared by all
    /// instances for v1; per-instance AABBs are a future extension.
    pub aabb: Aabb,
    /// User-owned buffer whose contents at `dispatch_args_offset` encode the
    /// indirect dispatch arguments (`[workgroup_x, 1, 1]` as `vec3<u32>`) for
    /// the preprocessing compute pass over this batch.
    ///
    /// The user's own compute shaders are responsible for writing this buffer
    /// each frame based on their simulation's live particle count.
    pub dispatch_args_buffer: Buffer,
    /// Byte offset into `dispatch_args_buffer` where the indirect arguments
    /// begin.
    pub dispatch_args_offset: u64,
    /// [`MeshFlags`] to stamp into every slot of the input uniform buffer for
    /// this batch. Used for visibility range, no-frustum-culling, and other
    /// per-mesh rendering toggles.
    pub flags: MeshFlags,
}

/// Render-world mirror of [`GpuInstanceBatch`], extracted each frame.
#[derive(Clone)]
pub struct ExtractedGpuInstanceBatch {
    pub mesh_asset_id: AssetId<Mesh>,
    pub max_capacity: u32,
    pub aabb: Aabb,
    pub dispatch_args_buffer: Buffer,
    pub dispatch_args_offset: u64,
    pub flags: MeshFlags,
}

/// Render-world collection of extracted batches for the current frame.
///
/// Populated by [`extract_gpu_instance_batches`] and consumed by
/// [`allocate_gpu_instance_batch_reservations`] (and, in later PRs, by the
/// queue system and the preprocessing node).
#[derive(Resource, Default)]
pub struct ExtractedGpuInstanceBatches {
    pub batches: MainEntityHashMap<ExtractedGpuInstanceBatch>,
}

/// Stable per-batch allocation handles.
///
/// The input-buffer and culling-buffer ranges for a batch are allocated on its
/// first successful extraction and persist until the batch is removed from the
/// main world.
#[derive(Clone, Copy)]
pub struct GpuInstanceBatchReservation {
    pub input_buffer_base: u32,
    pub culling_buffer_base: u32,
    pub max_capacity: u32,
    pub mesh_asset_id: AssetId<Mesh>,
}

#[derive(Resource, Default)]
pub struct GpuInstanceBatchReservations {
    pub by_entity: HashMap<MainEntity, GpuInstanceBatchReservation>,
}

/// Plugin that registers GPU instance batch extraction and reservation
/// lifecycle.
///
/// Systems in this plugin are gated on [`GpuPreprocessingSupport::is_available`]
/// — on devices without GPU preprocessing, the plugin does nothing and logs a
/// warning.
pub struct GpuInstanceBatchPlugin;

impl Plugin for GpuInstanceBatchPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExtractedGpuInstanceBatches>()
            .init_resource::<GpuInstanceBatchReservations>()
            .add_systems(
                ExtractSchedule,
                extract_gpu_instance_batches.after(MaterialExtractionSystems),
            )
            .add_systems(
                Render,
                allocate_gpu_instance_batch_reservations.in_set(RenderSystems::PrepareResources),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app(RenderApp) else {
            return;
        };
        if !render_app
            .world()
            .resource::<GpuPreprocessingSupport>()
            .is_available()
        {
            warn!(
                "GpuInstanceBatchPlugin requires GPU preprocessing support; \
                 batches will not be processed on this device."
            );
        }
    }
}

/// Extracts [`GpuInstanceBatch`] components from the main world into the
/// render world's [`ExtractedGpuInstanceBatches`] resource.
///
/// Runs unconditionally (all batches every frame, not change-detected) since
/// the expected batch count is small — dozens, typically.
pub fn extract_gpu_instance_batches(
    mut extracted: ResMut<ExtractedGpuInstanceBatches>,
    query: Extract<Query<(Entity, &GpuInstanceBatch)>>,
) {
    extracted.batches.clear();
    for (entity, batch) in query.iter() {
        extracted.batches.insert(
            MainEntity::from(entity),
            ExtractedGpuInstanceBatch {
                mesh_asset_id: batch.mesh.id(),
                max_capacity: batch.max_capacity,
                aabb: batch.aabb,
                dispatch_args_buffer: batch.dispatch_args_buffer.clone(),
                dispatch_args_offset: batch.dispatch_args_offset,
                flags: batch.flags,
            },
        );
    }
}

/// Allocates input-buffer and culling-buffer ranges for newly-seen batches,
/// and frees ranges for batches no longer present.
pub fn allocate_gpu_instance_batch_reservations(
    extracted: Res<ExtractedGpuInstanceBatches>,
    mut reservations: ResMut<GpuInstanceBatchReservations>,
    mut batched_instance_buffers: ResMut<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    mut culling_data_buffer: ResMut<MeshCullingDataBuffer>,
    mesh_allocator: Res<MeshAllocator>,
    render_material_instances: Res<RenderMaterialInstances>,
    render_material_bindings: Res<RenderMaterialBindings>,
    frame_count: Res<FrameCount>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    if !gpu_preprocessing_support.is_available() {
        return;
    }

    let input_uniform_buffer = &mut batched_instance_buffers.current_input_buffer;

    // Free reservations for batches no longer present this frame.
    let alive_entities: HashSet<MainEntity> = extracted.batches.keys().copied().collect();
    reservations.by_entity.retain(|entity, reservation| {
        if alive_entities.contains(entity) {
            true
        } else {
            input_uniform_buffer
                .remove_range(reservation.input_buffer_base, reservation.max_capacity);
            // `MeshCullingDataBuffer` does not support freeing ranges yet;
            // slots persist until the app exits. Acceptable for v1 because
            // batches are expected to be long-lived.
            false
        }
    });

    // Allocate reservations for newly-seen batches.
    for (main_entity, batch) in extracted.batches.iter() {
        if reservations.by_entity.contains_key(main_entity) {
            continue;
        }

        // Look up the material binding for this batch's material. The
        // material is extracted independently by `MaterialPlugin<M>` via
        // `MeshMaterial3d<M>`.
        let Some(material_instance) = render_material_instances.instances.get(main_entity) else {
            // Material not yet extracted (e.g. first frame); retry next frame.
            continue;
        };
        let Some(material_binding) = render_material_bindings
            .get(&material_instance.asset_id)
            .copied()
        else {
            // Material not yet prepared; retry next frame.
            continue;
        };

        // Look up the mesh's location in the mesh allocator.
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&batch.mesh_asset_id) else {
            // Mesh not yet uploaded; retry next frame.
            continue;
        };
        let first_vertex_index = vertex_slice.range.start;
        let vertex_count = vertex_slice.range.end - vertex_slice.range.start;

        let (mesh_is_indexed, first_index_index, index_count) =
            match mesh_allocator.mesh_index_slice(&batch.mesh_asset_id) {
                Some(index_slice) => (
                    true,
                    index_slice.range.start,
                    index_slice.range.end - index_slice.range.start,
                ),
                None => (false, 0, 0),
            };
        let resolved_index_count = if mesh_is_indexed {
            index_count
        } else {
            vertex_count
        };

        let material_slot = u32::from(material_binding.slot);
        // Lightmap slot is set to u16::MAX (no lightmap) for v1.
        let lightmap_slot = u16::MAX as u32;
        let material_and_lightmap_bind_group_slot = material_slot | (lightmap_slot << 16);

        let template = MeshInputUniform {
            // Zeroed; the user's compute shader overwrites this every frame.
            world_from_local: [Vec4::ZERO; 3],
            lightmap_uv_rect: UVec2::ZERO,
            flags: batch.flags.bits(),
            // No TAA / motion vector support for batches in v1.
            previous_input_index: u32::MAX,
            timestamp: frame_count.0,
            first_vertex_index,
            first_index_index,
            index_count: resolved_index_count,
            current_skin_index: u32::MAX,
            material_and_lightmap_bind_group_slot,
            tag: 0,
            morph_descriptor_index: u32::MAX,
        };

        let input_buffer_base =
            input_uniform_buffer.add_many_with(batch.max_capacity, |_| template);

        let culling_data = MeshCullingData {
            aabb_center: Vec3::from(batch.aabb.center).extend(0.0),
            aabb_half_extents: Vec3::from(batch.aabb.half_extents).extend(0.0),
        };
        let culling_buffer_base =
            culling_data_buffer.push_many_identical(culling_data, batch.max_capacity);

        reservations.by_entity.insert(
            *main_entity,
            GpuInstanceBatchReservation {
                input_buffer_base,
                culling_buffer_base,
                max_capacity: batch.max_capacity,
                mesh_asset_id: batch.mesh_asset_id,
            },
        );
    }
}
