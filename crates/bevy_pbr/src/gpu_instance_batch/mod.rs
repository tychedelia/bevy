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
//! # Aliveness control
//!
//! To mark a slot as "dead", the user's simulation compute shader writes a
//! per-slot [`MeshCullingData`] with impossible AABB extents (or a center far
//! outside any frustum). Bevy's existing frustum culling in
//! `mesh_preprocess.wgsl` then rejects that slot, so the indirect draw's
//! `instance_count` only reflects surviving instances. No indirect dispatch
//! is required for the preprocess pass; the existing direct dispatch iterates
//! all work items (including ours) and culling drops the dead ones.
//!
//! # Scope (v1)
//!
//! Batches render through all standard phases (opaque, shadow, prepass,
//! deferred) once the bevy-side patch introducing
//! `RenderMeshInstanceBatches` + range-variant phase items lands. Known
//! limitations:
//!
//! - No transparent-phase support. GPU-authored depths can't be correctly
//!   interleaved with other transparent geometry via CPU sort keys; this is
//!   structurally incompatible with `SortedRenderPhase` rather than a
//!   scope cut. Proper transparent GPU-authored batches would need OIT or
//!   per-particle GPU sorting.
//! - No motion vectors / TAA (`previous_input_index = u32::MAX` always).
//!   Slot mapping isn't stable across frames when the user's simulation
//!   compacts particles.
//! - No late-phase occlusion culling for batches — batches participate only
//!   in the frustum-culling early pass.
//! - No per-instance attribute variation beyond transform (v2+ extension
//!   point).
//! - `max_capacity` is CPU-declared and mutable only from the main world.
//!
//! # Architecture note
//!
//! Batches do not go through `RenderMeshInstances` (which encodes 1:1
//! ECS-entity-to-instance semantics). Instead, a separate parallel registry
//! `RenderMeshInstanceBatches` — introduced in a companion bevy patch —
//! carries one `RenderMeshInstanceBatch` entry per batch, describing a
//! range of `count` GPU-authored instances. Queue systems (opaque, shadow,
//! prepass, deferred) iterate both registries and emit range-variant phase
//! items (`BinnedRenderPhaseType::InstanceBatch`) for the batch registry.
//! `batch_and_prepare_binned_render_phase` handles the variant by pushing
//! `count` work items instead of one. Every subsequent stage — GPU
//! preprocessing, indirect-parameter building, `DrawMesh` — is unchanged;
//! the shared infrastructure operates on work items and indirect
//! parameters, which are populated identically whether they came from a
//! single mesh entity or a batch range.

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
    /// culling data buffer, and indirect work-item buffer at this size.
    /// Dead slots (marked via [`MeshCullingData`] with impossible AABBs from
    /// the user's simulation shader) are rejected by frustum culling so the
    /// final indirect draw's `instance_count` matches the live particle
    /// count without any CPU bookkeeping.
    pub max_capacity: u32,
    /// Axis-aligned bounding box that is copied into every slot of the
    /// culling data buffer at reservation time.
    ///
    /// The user's simulation compute is free to overwrite the per-slot AABB
    /// each frame to signal alive/dead status. See the module-level docs.
    pub aabb: Aabb,
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
    pub flags: MeshFlags,
}

/// Render-world collection of extracted batches for the current frame.
///
/// Populated by [`extract_gpu_instance_batches`] and consumed by
/// [`allocate_gpu_instance_batch_reservations`]. Cleared and repopulated each
/// frame; lookups key by [`MainEntity`].
#[derive(Resource, Default)]
pub struct ExtractedGpuInstanceBatches {
    pub batches: MainEntityHashMap<ExtractedGpuInstanceBatch>,
}

/// Stable per-batch allocation handles.
///
/// The input-buffer and culling-buffer ranges for a batch are allocated on
/// its first successful extraction and persist until the batch is removed
/// from the main world.
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
/// Systems in this plugin are gated on
/// [`GpuPreprocessingSupport::is_available`] — on devices without GPU
/// preprocessing, the plugin does nothing and logs a warning.
///
/// # v1 limitations
///
/// This plugin currently wires up extraction and reservation only. Queue
/// and draw integration depend on a companion bevy-side patch that
/// introduces `RenderMeshInstanceBatches` plus a range-variant phase item
/// (see the trailing `NOTE` in this module's source). Until that lands,
/// batches allocate input and culling buffer ranges and template
/// [`MeshInputUniform`] data, but nothing renders.
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

// NOTE: rendering integration is completed by two follow-up pieces in
// the upstream bevy renderer:
//
// 1. A bevy-side patch adding `RenderMeshInstanceBatch` +
//    `RenderMeshInstanceBatches` and a `BinnedRenderPhaseType::InstanceBatch`
//    variant, with range handling in `batch_and_prepare_binned_render_phase`
//    and awareness in `queue_material_meshes` / shadow / prepass / deferred
//    queue systems. Lands independently as it's broadly useful to any
//    GPU-driven instance provider.
//
// 2. A thin consumer layer in this module that, during reservation,
//    populates `RenderMeshInstanceBatches` with an entry keyed by the
//    batch's `MainEntity`, carrying `base_input_index`, `count =
//    max_capacity`, and the resolved material binding. With that in place,
//    the registry-aware queue systems emit range phase items for us and
//    the existing GPU preprocessing + indirect build + draw path renders
//    batches across all phases with zero further integration.
//
// Until those land, this module reserves input / culling buffer ranges
// and populates `MeshInputUniform` templates, but produces no visible
// output.
