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
//!     GpuBatchedMesh3d {
//!         mesh: mesh_handle,
//!         max_capacity: 4096,
//!     },
//!     // Optional: sibling `Aabb` for CPU-side whole-batch frustum culling
//!     // if you remove `NoFrustumCulling`. Per-slot GPU culling still uses
//!     // the per-slot `MeshCullingData` the simulation writes.
//!     MeshMaterial3d(material_handle),
//! ));
//! ```
//!
//! The user's compute shader then writes `world_from_local` (and
//! optionally per-slot [`MeshCullingData`]) into the reserved range of
//! bevy's shared input buffer each frame. To mark a slot as "dead", write
//! a nonzero value into [`MeshCullingData::dead`] — the preprocessing pass
//! skips it entirely, so it costs nothing past that point and the indirect
//! draw's `instance_count` reflects only live instances. Slots default to
//! alive (`dead == 0.0`) after reservation, so simulations only need to
//! write the flag when retiring a slot.
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
//! - **Per-instance attributes beyond transform**: not yet plumbed.
//! - **`max_capacity`**: CPU-declared, mutable only from the main world.

use core::num::NonZeroU32;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetEvent, AssetId};
use bevy_camera::primitives::Aabb;
use bevy_camera::visibility::{
    add_visibility_class, NoFrustumCulling, ViewVisibility, Visibility, VisibilityClass,
};
use bevy_diagnostic::FrameCount;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::world::DeferredWorld;
use bevy_log::warn;
use bevy_math::{UVec2, Vec4};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::batching::gpu_preprocessing::{BatchedInstanceBuffers, GpuPreprocessingSupport};
use bevy_render::mesh::allocator::MeshAllocator;
use bevy_render::sync_world::{MainEntity, MainEntityHashMap};
use bevy_render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy_transform::components::Transform;

use crate::{
    MeshCullingData, MeshCullingDataBuffer, MeshFlags, MeshInputUniform, MeshUniform,
    RenderMaterialBindings, RenderMaterialInstances, RenderMeshInstanceBatch,
    RenderMeshInstanceBatches,
};

/// An entity with this component represents a batch of up to
/// `max_capacity` GPU-authored mesh instances.
///
/// The entity must also carry a `MeshMaterial3d<M>` so the existing
/// `MaterialPlugin<M>` picks up its material. It must **not** carry a
/// [`Mesh3d`] — that would cause the entity to also participate in
/// normal mesh extraction and produce duplicate draws. A runtime warning
/// fires if both are present.
///
/// `NoFrustumCulling` is required by default because per-instance GPU
/// culling is the authoritative mechanism for batches. Removing it opts
/// into CPU whole-batch frustum culling against a sibling [`Aabb`]
/// component (if present).
#[derive(Component, Clone)]
#[component(on_add = gpu_batched_mesh_3d_on_add)]
#[require(Transform, Visibility, VisibilityClass, NoFrustumCulling)]
pub struct GpuBatchedMesh3d {
    pub mesh: bevy_asset::Handle<Mesh>,
    /// Upper bound on live instances. Contiguous slots at this size are
    /// reserved in the preprocessing input buffer, culling data buffer,
    /// and work-item buffer.
    pub max_capacity: u32,
}

/// `on_add` hook: adds `GpuBatchedMesh3d`'s `TypeId` to the entity's
/// [`VisibilityClass`] bucket, and warns if [`Mesh3d`] is also present.
fn gpu_batched_mesh_3d_on_add(mut world: DeferredWorld<'_>, ctx: HookContext) {
    add_visibility_class::<GpuBatchedMesh3d>(world.reborrow(), ctx);
    if world.get::<Mesh3d>(ctx.entity).is_some() {
        warn!(
            "entity {:?} has both `Mesh3d` and `GpuBatchedMesh3d`; \
             behavior is undefined — expect duplicate draws",
            ctx.entity
        );
    }
}

/// Extracted (diff) state of a single [`GpuBatchedMesh3d`] entity.
#[derive(Clone)]
pub struct ExtractedGpuBatchedMesh {
    pub mesh_asset_id: AssetId<Mesh>,
    pub max_capacity: u32,
    /// Emitter-level AABB, if the entity has a sibling [`Aabb`]. When
    /// `None`, the reservation uses the "infinitely-large" path of
    /// [`MeshCullingData::new`].
    pub aabb: Option<Aabb>,
}

/// Per-frame diff of [`GpuBatchedMesh3d`] extractions.
///
/// Populated by [`extract_gpu_batched_mesh_changes`] and consumed by
/// [`prepare_gpu_batched_mesh_reservations`].
#[derive(Resource, Default)]
pub struct ExtractedGpuBatchedMeshChanges {
    pub added_or_changed: MainEntityHashMap<ExtractedGpuBatchedMesh>,
    pub removed: HashSet<MainEntity>,
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

/// Registers [`GpuBatchedMesh3d`] extraction and reservation. On devices
/// without GPU preprocessing, logs a warning and the plugin's systems
/// no-op.
pub struct GpuInstanceBatchPlugin;

impl Plugin for GpuInstanceBatchPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PostUpdate,
            (
                mark_gpu_batched_meshes_as_changed_if_their_assets_changed,
                warn_on_gpu_batched_mesh_mesh3d_overlap,
            ),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExtractedGpuBatchedMeshChanges>()
            .init_resource::<GpuInstanceBatchReservations>()
            .add_systems(ExtractSchedule, extract_gpu_batched_mesh_changes)
            .add_systems(
                Render,
                prepare_gpu_batched_mesh_reservations.in_set(RenderSystems::PrepareResources),
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

/// Main-world sibling of
/// [`bevy_mesh::mark_3d_meshes_as_changed_if_their_assets_changed`] for
/// [`GpuBatchedMesh3d`]: when the mesh asset the batch points at is
/// modified, trigger a `Changed<GpuBatchedMesh3d>` signal so downstream
/// change-tracking re-specializes the batch.
pub fn mark_gpu_batched_meshes_as_changed_if_their_assets_changed(
    mut batched_meshes: Query<&mut GpuBatchedMesh3d>,
    mut mesh_asset_events: MessageReader<AssetEvent<Mesh>>,
) {
    let mut changed_meshes: HashSet<AssetId<Mesh>> = HashSet::default();
    for mesh_asset_event in mesh_asset_events.read() {
        if let AssetEvent::Modified { id } = mesh_asset_event {
            changed_meshes.insert(*id);
        }
    }

    if changed_meshes.is_empty() {
        return;
    }

    for mut batched_mesh in &mut batched_meshes {
        if changed_meshes.contains(&batched_mesh.mesh.id()) {
            batched_mesh.set_changed();
        }
    }
}

/// Warn once per entity (throttled via `Local`) if both [`Mesh3d`] and
/// [`GpuBatchedMesh3d`] are present. The on-add hook catches the case
/// where `GpuBatchedMesh3d` is added to an entity that already has
/// `Mesh3d`; this system catches the other insertion order and the case
/// where both are present from separate spawns.
pub fn warn_on_gpu_batched_mesh_mesh3d_overlap(
    offenders: Query<Entity, (With<Mesh3d>, With<GpuBatchedMesh3d>)>,
    mut already_warned: Local<HashSet<Entity>>,
) {
    for entity in &offenders {
        if already_warned.insert(entity) {
            warn!(
                "entity {:?} has both `Mesh3d` and `GpuBatchedMesh3d`; \
                 behavior is undefined — expect duplicate draws",
                entity
            );
        }
    }
}

/// Diff-based extract of [`GpuBatchedMesh3d`] entities.
///
/// Clears the previous frame's `added_or_changed` (which the prepare
/// system consumed) and `removed` (which it also consumed), then writes
/// this frame's diff. `RemovedComponents` must be read here since it
/// only fires once per reader.
pub fn extract_gpu_batched_mesh_changes(
    mut extracted: ResMut<ExtractedGpuBatchedMeshChanges>,
    query: Extract<
        Query<
            (Entity, &GpuBatchedMesh3d, Option<&Aabb>, &ViewVisibility),
            Or<(
                Changed<GpuBatchedMesh3d>,
                Changed<Aabb>,
                Changed<ViewVisibility>,
            )>,
        >,
    >,
    mut removed: Extract<RemovedComponents<GpuBatchedMesh3d>>,
) {
    // `removed` is a message reader — each event fires exactly once per
    // reader. We accumulate into `extracted.removed`; the consumer drains
    // it. `added_or_changed` is *not* cleared here — entries stay around
    // until the consumer successfully reserves them (material/mesh might
    // not be ready this frame; retry next frame).
    for entity in removed.read() {
        let main_entity = MainEntity::from(entity);
        extracted.added_or_changed.remove(&main_entity);
        extracted.removed.insert(main_entity);
    }

    for (entity, batch, aabb, _view_visibility) in query.iter() {
        // Zero-capacity batches would panic the downstream allocators
        // (`add_many_with` / `push_many_identical` both require count > 0)
        // and have no rendering meaning anyway.
        if batch.max_capacity == 0 {
            warn!(
                "GpuBatchedMesh3d on {entity} has max_capacity = 0; ignoring. \
                 Set a positive capacity to render any instances."
            );
            continue;
        }
        extracted.added_or_changed.insert(
            MainEntity::from(entity),
            ExtractedGpuBatchedMesh {
                mesh_asset_id: batch.mesh.id(),
                max_capacity: batch.max_capacity,
                aabb: aabb.copied(),
            },
        );
    }
}

pub fn prepare_gpu_batched_mesh_reservations(
    mut extracted: ResMut<ExtractedGpuBatchedMeshChanges>,
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
        extracted.added_or_changed.clear();
        extracted.removed.clear();
        return;
    }

    let input_uniform_buffer = &mut batched_instance_buffers.current_input_buffer;

    // Release buffer ranges for removed entities. The two buffers are
    // indexed in lockstep (the culling shader reads
    // `mesh_culling_data[input_index]`), so both frees must cover the
    // same range.
    for main_entity in extracted.removed.drain() {
        if let Some(reservation) = reservations.by_entity.remove(&main_entity) {
            input_uniform_buffer
                .remove_range(reservation.input_buffer_base, reservation.max_capacity);
            culling_data_buffer
                .remove_range(reservation.culling_buffer_base, reservation.max_capacity);
            render_mesh_instance_batches.remove(&main_entity);
        }
    }

    // Allocate reservations for added/changed entities not already
    // reserved. For entities already reserved, treat the diff as a
    // notification only — `max_capacity` and mesh changes that require
    // re-reservation aren't supported here. (Mesh *asset* changes
    // propagate through extract_meshes_for_gpu_building's slab metadata,
    // and the specialization re-fires via
    // mark_gpu_batched_meshes_as_changed_if_their_assets_changed which
    // routes through DirtySpecializations.)
    //
    // Entries are retained in `added_or_changed` until they successfully
    // reserve; on the spawn frame the material/mesh may not be ready yet.
    extracted.added_or_changed.retain(|main_entity, batch| {
        if reservations.by_entity.contains_key(main_entity) {
            // Already reserved — drop from the pending-diff map.
            return false;
        }

        // Material / mesh may not be ready on the first frame after spawn;
        // retain this entry and retry next frame.
        let Some(material_instance) = render_material_instances.instances.get(main_entity) else {
            return true;
        };
        let Some(material_binding) = render_material_bindings
            .get(&material_instance.asset_id)
            .copied()
        else {
            return true;
        };
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&batch.mesh_asset_id) else {
            return true;
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
        let lightmap_slot = u16::MAX as u32;
        let material_and_lightmap_bind_group_slot = material_slot | (lightmap_slot << 16);

        // The low 16 bits of `MeshFlags` are the visibility-range / LOD
        // index. `mesh_preprocess.wgsl` reads them as an index into
        // `visibility_ranges` — `u16::MAX` is the sentinel for "no LOD"
        // that tells the shader to skip the cull. Leaving this at 0
        // causes a silent early-return against `visibility_ranges[0]`.
        let lod_sentinel = u16::MAX as u32;
        let resolved_flags = MeshFlags::empty().bits() | lod_sentinel;

        let template = MeshInputUniform {
            world_from_local: [Vec4::ZERO; 3],
            lightmap_uv_rect: UVec2::ZERO,
            flags: resolved_flags,
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

        let culling_data = MeshCullingData::new(batch.aabb.as_ref());
        let culling_buffer_base =
            culling_data_buffer.push_many_identical(culling_data, batch.max_capacity);

        // The preprocessing shader uses a single `input_index` to address
        // both `current_input` and `mesh_culling_data`, so the two
        // allocators must hand out matching bases. They have independent
        // free lists but operate in lockstep here; if this ever fires, the
        // two free lists have drifted and the shader will pair each
        // instance's transform with the wrong AABB.
        debug_assert_eq!(
            input_buffer_base, culling_buffer_base,
            "input-buffer and culling-buffer reservations diverged for {main_entity:?}",
        );

        reservations.by_entity.insert(
            *main_entity,
            GpuInstanceBatchReservation {
                input_buffer_base,
                culling_buffer_base,
                max_capacity: batch.max_capacity,
                mesh_asset_id: batch.mesh_asset_id,
            },
        );

        // `extract_gpu_batched_mesh_changes` filters out zero-capacity
        // batches, so this is infallible.
        let count = NonZeroU32::new(batch.max_capacity)
            .expect("zero-capacity batches must be filtered at extract time");
        render_mesh_instance_batches.insert(
            *main_entity,
            RenderMeshInstanceBatch {
                asset_id: batch.mesh_asset_id,
                material_binding,
                base_input_index: input_buffer_base,
                count,
                flags: MeshFlags::empty(),
            },
        );

        // Successfully reserved — drop from the pending-diff map.
        false
    });
}

