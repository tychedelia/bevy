//! GPU-authored instance batches: rendering `N` instances of a `Mesh`
//! whose per-instance transforms are written into the GPU preprocessing
//! input buffer by the user's own compute shaders, rather than extracted
//! from ECS entities on the CPU.
//!
//! Useful for GPU-driven particle systems, foliage scatter, point clouds,
//! GPU-baked animation output, and similar "N instances of one (mesh,
//! material)" workloads.
//!
//! # Usage
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
//! The user's compute shader then writes `world_from_local` (and
//! optionally per-slot [`MeshCullingData`]) into the reserved range of
//! bevy's shared input buffer each frame. To mark a slot as "dead",
//! write an impossible AABB for it — frustum culling rejects it and the
//! indirect draw's `instance_count` reflects only surviving instances.
//!
//! # Known limitations
//!
//! - **Transparent rendering**: unsupported. GPU-authored depths can't be
//!   correctly interleaved with other transparent geometry via CPU sort
//!   keys; [`SortedRenderPhase`] is structurally incompatible. Would
//!   require OIT or per-particle GPU sort.
//! - **Motion vectors / TAA**: `previous_input_index = u32::MAX` always;
//!   slot mapping isn't stable across frames when simulations compact
//!   particles.
//! - **Late-phase occlusion culling**: batches participate only in the
//!   early frustum-culling pass.
//! - **Per-instance attributes beyond transform**: not yet plumbed.
//! - **`max_capacity`**: CPU-declared, mutable only from the main world.

use core::num::NonZeroU32;

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
    MeshUniform, RenderMaterialBindings, RenderMaterialInstances, RenderMeshInstanceBatch,
    RenderMeshInstanceBatches,
};

/// An entity with this component represents a batch of up to
/// `max_capacity` GPU-authored mesh instances.
///
/// The entity must also carry a `MeshMaterial3d<M>` so the existing
/// `MaterialPlugin<M>` picks up its material. It must **not** carry a
/// `Mesh3d` — that would cause the entity to also participate in
/// normal mesh extraction.
#[derive(Component, Clone)]
pub struct GpuInstanceBatch {
    pub mesh: bevy_asset::Handle<Mesh>,
    /// Upper bound on live instances. Contiguous slots at this size are
    /// reserved in the preprocessing input buffer, culling data buffer,
    /// and work-item buffer.
    pub max_capacity: u32,
    /// Emitter-level AABB, stamped into every slot at reservation time.
    /// The simulation shader can overwrite per-slot AABBs each frame to
    /// signal alive/dead via frustum culling.
    pub aabb: Aabb,
    pub flags: MeshFlags,
}

#[derive(Clone)]
pub struct ExtractedGpuInstanceBatch {
    pub mesh_asset_id: AssetId<Mesh>,
    pub max_capacity: u32,
    pub aabb: Aabb,
    pub flags: MeshFlags,
}

#[derive(Resource, Default)]
pub struct ExtractedGpuInstanceBatches {
    pub batches: MainEntityHashMap<ExtractedGpuInstanceBatch>,
}

/// Stable per-batch allocation handles. The input-buffer and
/// culling-buffer ranges persist until the batch is removed from the
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

/// Registers [`GpuInstanceBatch`] extraction and reservation. On devices
/// without GPU preprocessing, logs a warning and the plugin's systems
/// no-op.
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
/// registers them in [`RenderMeshInstanceBatches`] so the bevy-side queue
/// systems can emit phase items, and frees ranges for batches no longer
/// present.
pub fn allocate_gpu_instance_batch_reservations(
    extracted: Res<ExtractedGpuInstanceBatches>,
    mut reservations: ResMut<GpuInstanceBatchReservations>,
    mut batched_instance_buffers: ResMut<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    mut culling_data_buffer: ResMut<MeshCullingDataBuffer>,
    mut render_mesh_instance_batches: ResMut<RenderMeshInstanceBatches>,
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

    // Free reservations for batches no longer present this frame, and
    // remove them from the render-world registry so queue systems stop
    // emitting phase items for them.
    let alive_entities: HashSet<MainEntity> = extracted.batches.keys().copied().collect();
    reservations.by_entity.retain(|entity, reservation| {
        if alive_entities.contains(entity) {
            true
        } else {
            input_uniform_buffer
                .remove_range(reservation.input_buffer_base, reservation.max_capacity);
            render_mesh_instance_batches.remove(entity);
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

        // The low 16 bits of `MeshFlags` encode the visibility-range /
        // LOD index. `u16::MAX` is the sentinel for "no LOD" and makes
        // `mesh_preprocess.wgsl` skip the visibility-range cull
        // (otherwise the default low-bit value of 0 causes it to index
        // `visibility_ranges[0]` — usually zeroed — and early-return).
        // Users spawning `GpuInstanceBatch` shouldn't need to think
        // about this encoding, so we OR the sentinel in here rather
        // than requiring them to set it on the component's `flags`.
        let lod_sentinel = u16::MAX as u32;
        let resolved_flags = batch.flags.bits() | lod_sentinel;

        let template = MeshInputUniform {
            // Zeroed; the user's compute shader overwrites this every frame.
            world_from_local: [Vec4::ZERO; 3],
            lightmap_uv_rect: UVec2::ZERO,
            flags: resolved_flags,
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

        // Register the batch in the render-world registry. Queue systems
        // (`queue_material_meshes`, shadow, prepass, deferred) iterate
        // this map and emit one `BinnedRenderPhaseType::InstanceBatch`
        // phase item per entry.
        let Some(count) = NonZeroU32::new(batch.max_capacity) else {
            // max_capacity == 0 shouldn't happen in practice; `add_many_with`
            // panics on zero counts. Skip registration defensively.
            continue;
        };
        render_mesh_instance_batches.insert(
            *main_entity,
            RenderMeshInstanceBatch {
                asset_id: batch.mesh_asset_id,
                material_binding,
                base_input_index: input_buffer_base,
                count,
                flags: batch.flags,
            },
        );
    }
}

// Rendering integration is now complete via the companion bevy patch
// that introduced `RenderMeshInstanceBatch` + `RenderMeshInstanceBatches`
// plus the `BinnedRenderPhaseType::InstanceBatch` variant. Each batch
// inserted into `RenderMeshInstanceBatches` by
// `allocate_gpu_instance_batch_reservations` is picked up by the
// bevy-side queue systems (`queue_material_meshes`, shadow, prepass,
// deferred) which emit range-variant phase items. The shared GPU
// preprocessing + indirect build + draw path handles the rest, rendering
// batches across all supported phases.
