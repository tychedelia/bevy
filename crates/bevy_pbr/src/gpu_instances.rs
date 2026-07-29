//! GPU-authored mesh instances: per-instance transforms are written by
//! user compute shaders rather than extracted from ECS entities.
//!
//! A [`GpuInstances3d`] reserves a contiguous range of `capacity`
//! slots in [`GpuInstancePools`], a pair of GPU-only buffers dedicated
//! to GPU-authored instances. A user compute shader, dispatched in the
//! [`GpuInstanceAuthoringSystems`] set of the `RenderGraph` schedule, writes
//! each slot's `world_from_local` transform into the input pool and its cull
//! volume and `alive` value into the culling pool, once per frame for all
//! views. Bevy's standard mesh
//! pipeline then renders the whole reservation as a single indirect draw,
//! with GPU frustum culling compacting out the instances whose `alive` is
//! zero or less.
//!
//! The pools have no CPU-side copy, and growing them copies their contents
//! forward, so slots persist once written: a simulation can initialize its
//! reservation once and then operate on it purely with compute dispatches.
//! Changing an entity's mesh, material, or [`Aabb`] rewrites its slots in
//! place; changing its `capacity` re-reserves, which may move the base.
//! In both cases the slots are reset to their initial state and the
//! reservation's [`generation`] is incremented, so simulations should
//! re-initialize their slots when they observe a new generation.
//!
//! A slot's `MeshInput` is shared with the renderer: simulations own
//! `world_from_local` and `tag` and must leave every other field as they
//! found it, since the renderer resolves mesh, material, and motion-vector
//! state through them. The `MeshCullingData` slots are entirely
//! simulation-owned.
//!
//! The WGSL definitions of the two buffers' element types, `MeshInput` and
//! `MeshCullingData`, can be imported from `bevy_pbr::mesh_preprocess_types`.
//!
//! [`generation`]: GpuInstanceReservation::generation
//!
//! Transparent rendering, motion vectors, lightmaps, and per-instance
//! attributes beyond the transform are not supported. GPU instances require
//! GPU culling; they won't render on platforms without compute shader support
//! or for cameras with `NoIndirectDrawing`.

use core::num::NonZeroU32;
use core::ops::Range;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{AssetEvent, AssetEventSystems, AssetId, Handle};
use bevy_camera::primitives::Aabb;
use bevy_camera::visibility::{
    add_visibility_class, NoFrustumCulling, Visibility, VisibilityClass,
};
use bevy_diagnostic::FrameCount;
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::ReflectComponent;
use bevy_light::{NonMeshShadowCaster, NotShadowCaster, NotShadowReceiver};
use bevy_log::{debug, warn, warn_once};
use bevy_math::{UVec2, Vec4};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_reflect::Reflect;
use bevy_render::batching::gpu_preprocessing::GpuPreprocessingSupport;
use bevy_render::camera::DirtySpecializations;
use bevy_render::material_bind_groups::RenderMaterialBindings;
use bevy_render::mesh::allocator::MeshAllocator;
use bevy_render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
};
use bevy_render::renderer::{RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue};
use bevy_render::sync_world::{MainEntity, MainEntityHashMap};
use bevy_render::view::{RenderVisibleEntities, RenderVisibleEntitiesClass};
use bevy_render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy_transform::components::Transform;
use smallvec::SmallVec;

use crate::{
    MeshCullingData, MeshFlags, MeshInputUniform, PreviousMeshInputUniform, RenderMaterialInstances,
};

pub use crate::render::{RenderGpuInstanceRange, RenderGpuInstanceRanges};

/// Up to `capacity` GPU-authored instances of a mesh, drawn as a single
/// indirect draw. Must be paired with a `MeshMaterial3d<M>` and must not
/// carry a [`Mesh3d`].
///
/// The entity's [`Aabb`], if present, seeds every slot's model-space culling
/// volume. It should bound a single instance's mesh, not all instances: an
/// undersized volume makes instances vanish once frustum or occlusion culling
/// rejects it, even though the mesh is still visible; an oversized one is
/// safe, merely culling more conservatively. Without an [`Aabb`], slots start
/// with an infinitely large volume and are never culled.
#[derive(Component, Clone, Debug, Reflect, PartialEq)]
#[reflect(Component, Clone, PartialEq)]
#[component(on_add = add_visibility_class::<GpuInstances3d>)]
#[require(
    Transform,
    Visibility,
    VisibilityClass,
    NoFrustumCulling,
    NonMeshShadowCaster
)]
pub struct GpuInstances3d {
    /// The mesh that every instance draws.
    pub mesh: Handle<Mesh>,
    /// The number of instance slots to reserve.
    pub capacity: u32,
}

/// The data extracted from a [`GpuInstances3d`] entity that changed this
/// frame.
#[derive(Clone, Debug)]
pub struct ExtractedGpuInstances {
    /// The [`AssetId`] of the mesh.
    pub mesh_asset_id: AssetId<Mesh>,
    /// The number of instance slots to reserve.
    pub capacity: NonZeroU32,
    /// The bounding volume used to initialize each slot's culling data.
    pub aabb: Option<Aabb>,
    /// Whether the instances cast shadows.
    pub shadow_caster: bool,
    /// Whether the instances receive shadows.
    pub shadow_receiver: bool,
}

/// [`GpuInstances3d`] entities that were added, changed, or removed and
/// haven't been turned into reservations yet.
///
/// Entities stay in `added_or_changed` until their material and mesh are
/// ready, so an entity whose assets are still loading is retried every
/// frame.
#[derive(Resource, Default)]
pub struct ExtractedGpuInstancesChanges {
    /// Entities whose instances must be (re)reserved.
    pub added_or_changed: MainEntityHashMap<ExtractedGpuInstances>,
    /// Entities whose reservation must be freed.
    pub removed: HashSet<MainEntity>,
}

/// A live reservation: the contiguous range of slots in
/// [`GpuInstancePools`] that a [`GpuInstances3d`] owns.
///
/// User compute shaders read this to find the slots they should write. The
/// input pool and the culling pool are addressed by the same index, so one
/// base serves both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuInstanceReservation {
    /// The index of the first of the entity's slots in the pools.
    pub base_input_index: u32,
    /// The number of slots in the reservation.
    pub capacity: NonZeroU32,
    /// The [`AssetId`] of the mesh.
    pub mesh_asset_id: AssetId<Mesh>,
    /// Incremented every time the entity's slots are rewritten to their
    /// initial state, including when the base moves. Simulations should
    /// re-initialize their slots when they observe a new generation.
    pub generation: u32,
}

/// All live [`GpuInstanceReservation`]s, keyed by the main-world entity
/// that owns them.
///
/// A reservation's base is stable for as long as its entity's `capacity`
/// is unchanged: pool growth copies the buffers forward, and changes to the
/// entity's mesh, material, or [`Aabb`] rewrite its slots in place. Changing
/// `capacity` re-reserves, which may move the base. In all of these cases
/// the slots are reset to their initial state and the reservation's
/// [`generation`] is incremented, so simulations should re-initialize their
/// slots when they observe a new generation.
///
/// [`generation`]: GpuInstanceReservation::generation
#[derive(Resource, Default)]
pub struct GpuInstanceReservations {
    /// Each [`GpuInstances3d`] entity's live reservation.
    pub by_entity: HashMap<MainEntity, GpuInstanceReservation>,
}

/// A sorted, coalesced list of disjoint free runs over some index space.
#[derive(Default, Clone, Debug)]
struct FreeRunList {
    runs: Vec<Range<u32>>,
}

impl FreeRunList {
    /// Inserts `[base, base + count)` into the list, merging with adjacent
    /// or overlapping runs. No-op if `count` is zero.
    fn free(&mut self, base: u32, count: u32) {
        if count == 0 {
            return;
        }
        let mut merged = base..base + count;

        let idx = self.runs.partition_point(|r| r.end < merged.start);

        while idx < self.runs.len() && self.runs[idx].start <= merged.end {
            debug_assert!(
                self.runs[idx].end <= merged.start || self.runs[idx].start >= merged.end,
                "FreeRunList::free: range {:?} overlaps existing run {:?} (double-free?)",
                base..base + count,
                self.runs[idx],
            );
            merged.start = merged.start.min(self.runs[idx].start);
            merged.end = merged.end.max(self.runs[idx].end);
            self.runs.remove(idx);
        }

        self.runs.insert(idx, merged);
    }

    /// First-fit allocation of `count` contiguous slots. Splits the chosen
    /// run, leaving the remainder on the list.
    fn allocate(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return None;
        }
        for i in 0..self.runs.len() {
            let run = &mut self.runs[i];
            if run.end - run.start >= count {
                let base = run.start;
                run.start += count;
                if run.start == run.end {
                    self.runs.remove(i);
                }
                return Some(base);
            }
        }
        None
    }

    /// Whether `index` falls within any free run.
    #[cfg(test)]
    fn contains(&self, index: u32) -> bool {
        let idx = self.runs.partition_point(|r| r.end <= index);
        idx < self.runs.len() && self.runs[idx].start <= index
    }

    /// Whether the list has no free runs.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }
}

/// The GPU-only buffers that back [`GpuInstances3d`] reservations.
///
/// Instance data lives in these dedicated pools rather than in the mesh
/// preprocessing input buffers. The pools have no CPU-side copy, so data that
/// GPU simulations write is never overwritten by the renderer's uploads, and
/// growing them copies the old contents forward, so reservations keep their
/// bases.
///
/// User compute shaders bind [`Self::input_buffer`] and
/// [`Self::culling_buffer`] to write per-instance transforms and culling
/// data. The element types are `MeshInput` and `MeshCullingData` from
/// `bevy_pbr::mesh_preprocess_types`. Growth replaces the [`Buffer`]
/// objects themselves, so re-fetch them (and rebuild any bind groups) every
/// frame rather than caching them.
#[derive(Resource, Default)]
pub struct GpuInstancePools {
    input: Option<Buffer>,
    culling: Option<Buffer>,
    previous_input_stub: Option<Buffer>,
    len: u32,
    capacity: u32,
    free_runs: FreeRunList,
}

impl GpuInstancePools {
    /// The pool of `MeshInput`s that preprocessing reads per-instance
    /// transforms from.
    pub fn input_buffer(&self) -> Option<&Buffer> {
        self.input.as_ref()
    }

    /// The stub buffer bound as the previous input for GPU instance
    /// preprocessing.
    pub(crate) fn previous_input_stub(&self) -> Option<&Buffer> {
        self.previous_input_stub.as_ref()
    }

    /// The pool of `MeshCullingData` that preprocessing reads culling
    /// volumes and `alive` values from.
    pub fn culling_buffer(&self) -> Option<&Buffer> {
        self.culling.as_ref()
    }

    fn allocate(
        &mut self,
        count: u32,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) -> u32 {
        if let Some(base) = self.free_runs.allocate(count) {
            return base;
        }
        let base = self.len;
        self.len += count;
        self.ensure_capacity(render_device, render_queue);
        base
    }

    fn free(&mut self, base: u32, count: u32) {
        self.free_runs.free(base, count);
    }

    /// Grows the pools to hold at least `self.len` slots, copying the old
    /// buffers' contents forward so that existing reservations keep both
    /// their bases and the data their simulations have written.
    fn ensure_capacity(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        if self.len <= self.capacity && self.input.is_some() {
            return;
        }

        // GPU instances have no previous-frame transforms, but the
        // preprocessing bind group layout requires a previous-input binding.
        // Slots pin `previous_input_index` to `u32::MAX`, so this one-element
        // stub is never read.
        if self.previous_input_stub.is_none() {
            self.previous_input_stub = Some(render_device.create_buffer(&BufferDescriptor {
                label: Some("gpu instances previous input stub"),
                size: size_of::<PreviousMeshInputUniform>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        let new_capacity = self.len.next_power_of_two().max(1024);

        let create = |label: &str, element_size: usize| {
            render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: new_capacity as u64 * element_size as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let new_input = create("gpu instances input pool", size_of::<MeshInputUniform>());
        let new_culling = create("gpu instances culling pool", size_of::<MeshCullingData>());

        if let (Some(old_input), Some(old_culling)) = (&self.input, &self.culling) {
            let mut command_encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("gpu instances pool grow copy"),
                });
            command_encoder.copy_buffer_to_buffer(old_input, 0, &new_input, 0, old_input.size());
            command_encoder.copy_buffer_to_buffer(
                old_culling,
                0,
                &new_culling,
                0,
                old_culling.size(),
            );
            render_queue.submit([command_encoder.finish()]);
        }

        self.input = Some(new_input);
        self.culling = Some(new_culling);
        self.capacity = new_capacity;

        debug!("gpu instance pools grew to {new_capacity} slots");
    }

    /// Fills the range's slots in both pools with the given values.
    fn write_range(
        &self,
        base: u32,
        count: u32,
        input: MeshInputUniform,
        culling: MeshCullingData,
        render_queue: &RenderQueue,
    ) {
        fn write_chunked<T: bytemuck::Pod>(
            render_queue: &RenderQueue,
            buffer: &Buffer,
            base: u32,
            count: u32,
            value: T,
        ) {
            const CHUNK: u32 = 1024;
            let chunk = vec![value; CHUNK.min(count) as usize];
            let mut written = 0;
            while written < count {
                let n = CHUNK.min(count - written);
                render_queue.write_buffer(
                    buffer,
                    (base + written) as u64 * size_of::<T>() as u64,
                    bytemuck::cast_slice(&chunk[..n as usize]),
                );
                written += n;
            }
        }

        let (Some(input_buffer), Some(culling_buffer)) = (&self.input, &self.culling) else {
            return;
        };
        write_chunked(render_queue, input_buffer, base, count, input);
        write_chunked(render_queue, culling_buffer, base, count, culling);
    }
}

/// Adds support for [`GpuInstances3d`].
/// The `RenderGraph` schedule set for the user compute dispatches that
/// author GPU instance data.
///
/// Systems in this set run after the renderer's buffer uploads and before
/// any view's mesh preprocessing reads the pools, which is the window in
/// which slots must be written. The set runs once per frame, not per view:
/// the pools hold world-space data that every view reads, so authoring that
/// advances state belongs here, while anything view-dependent belongs to
/// the per-view machinery (culling and compaction are already per-view on
/// the engine side, and view-dependent appearance such as billboarding
/// belongs in the material's vertex stage, where it applies to every view
/// that renders the instance).
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuInstanceAuthoringSystems;

pub struct GpuInstancePlugin;

impl Plugin for GpuInstancePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<GpuInstances3d>();
        app.add_systems(
            PostUpdate,
            (
                mark_gpu_instances_as_changed_if_their_assets_changed.after(AssetEventSystems),
                warn_on_gpu_instances_mesh3d_overlap,
            ),
        );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExtractedGpuInstancesChanges>()
            .init_resource::<GpuInstanceReservations>()
            .init_resource::<GpuInstancePools>()
            .configure_sets(
                RenderGraph,
                GpuInstanceAuthoringSystems
                    .after(RenderGraphSystems::Begin)
                    .before(RenderGraphSystems::Render),
            )
            .add_systems(ExtractSchedule, extract_gpu_instances_changes)
            .add_systems(
                Render,
                prepare_gpu_instance_reservations.in_set(RenderSystems::PrepareMeshes),
            );
    }
}

/// Marks [`GpuInstances3d`]s as changed when their mesh assets are
/// modified, so that their reservations get rebuilt with the meshes' new
/// locations in the vertex and index buffers.
pub fn mark_gpu_instances_as_changed_if_their_assets_changed(
    mut instances_query: Query<&mut GpuInstances3d>,
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

    for mut instances in &mut instances_query {
        if changed_meshes.contains(&instances.mesh.id()) {
            instances.set_changed();
        }
    }
}

/// Adds visible [`GpuInstances3d`] entities to the shadow-map visibility
/// lists of every shadow-mapping light.
///
/// Returns the visible-entity classes that mesh specialization and queuing
/// operate on: [`Mesh3d`] and [`GpuInstances3d`], whichever are present.
pub(crate) fn mesh_and_gpu_instances_visibility_classes(
    visible_entities: &RenderVisibleEntities,
) -> SmallVec<[&RenderVisibleEntitiesClass; 2]> {
    visible_entities
        .get::<Mesh3d>()
        .into_iter()
        .chain(visible_entities.get::<GpuInstances3d>())
        .collect()
}

/// Warns if an entity has both a [`Mesh3d`] and a [`GpuInstances3d`], which
/// would render the mesh both as an ordinary mesh instance and as GPU
/// instances.
pub fn warn_on_gpu_instances_mesh3d_overlap(
    offenders: Query<Entity, (With<Mesh3d>, With<GpuInstances3d>)>,
    mut already_warned: Local<HashSet<Entity>>,
) {
    for entity in &offenders {
        if already_warned.insert(entity) {
            warn!(
                "entity {entity} has both `Mesh3d` and `GpuInstances3d`; \
                 this will result in duplicate draws"
            );
        }
    }
}

/// Copies added, changed, and removed [`GpuInstances3d`]s into
/// [`ExtractedGpuInstancesChanges`].
pub fn extract_gpu_instances_changes(
    mut extracted: ResMut<ExtractedGpuInstancesChanges>,
    changed_query: Extract<
        Query<
            (
                Entity,
                &GpuInstances3d,
                Option<&Aabb>,
                Has<NotShadowCaster>,
                Has<NotShadowReceiver>,
            ),
            Or<(
                Changed<GpuInstances3d>,
                Changed<Aabb>,
                Changed<NotShadowCaster>,
                Changed<NotShadowReceiver>,
            )>,
        >,
    >,
    all_query: Extract<
        Query<(
            Entity,
            &GpuInstances3d,
            Option<&Aabb>,
            Has<NotShadowCaster>,
            Has<NotShadowReceiver>,
        )>,
    >,
    mut removed: Extract<RemovedComponents<GpuInstances3d>>,
    mut removed_aabbs: Extract<RemovedComponents<Aabb>>,
    mut removed_not_shadow_casters: Extract<RemovedComponents<NotShadowCaster>>,
    mut removed_not_shadow_receivers: Extract<RemovedComponents<NotShadowReceiver>>,
) {
    for entity in removed.read() {
        let main_entity = MainEntity::from(entity);
        extracted.added_or_changed.remove(&main_entity);
        extracted.removed.insert(main_entity);
    }

    for instances in changed_query.iter() {
        extract_instances(&mut extracted, instances);
    }

    // `Changed` filters don't fire on component removal, so entities that
    // removed an `Aabb` or a shadow marker are re-extracted here.
    for entity in removed_aabbs
        .read()
        .chain(removed_not_shadow_casters.read())
        .chain(removed_not_shadow_receivers.read())
    {
        if let Ok(instances) = all_query.get(entity) {
            extract_instances(&mut extracted, instances);
        }
    }
}

fn extract_instances(
    extracted: &mut ExtractedGpuInstancesChanges,
    (entity, instances, aabb, not_shadow_caster, not_shadow_receiver): (
        Entity,
        &GpuInstances3d,
        Option<&Aabb>,
        bool,
        bool,
    ),
) {
    let main_entity = MainEntity::from(entity);
    let Some(capacity) = NonZeroU32::new(instances.capacity) else {
        warn!(
            "GpuInstances3d on {entity} has capacity = 0; freeing its \
             reservation. Set a positive capacity to render any instances."
        );
        extracted.added_or_changed.remove(&main_entity);
        extracted.removed.insert(main_entity);
        return;
    };
    extracted.added_or_changed.insert(
        main_entity,
        ExtractedGpuInstances {
            mesh_asset_id: instances.mesh.id(),
            capacity,
            aabb: aabb.copied(),
            shadow_caster: !not_shadow_caster,
            shadow_receiver: !not_shadow_receiver,
        },
    );
}

/// Turns [`ExtractedGpuInstancesChanges`] into
/// [`GpuInstanceReservations`], allocating (or re-allocating) each
/// entity's slot ranges and writing its `MeshInputUniform` and
/// [`MeshCullingData`] templates.
pub fn prepare_gpu_instance_reservations(
    mut extracted: ResMut<ExtractedGpuInstancesChanges>,
    mut reservations: ResMut<GpuInstanceReservations>,
    mut pools: ResMut<GpuInstancePools>,
    mut render_gpu_instance_ranges: ResMut<RenderGpuInstanceRanges>,
    mesh_allocator: Res<MeshAllocator>,
    render_material_instances: Res<RenderMaterialInstances>,
    render_material_bindings: Res<RenderMaterialBindings>,
    frame_count: Res<FrameCount>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut dirty_specializations: ResMut<DirtySpecializations>,
) {
    if !gpu_preprocessing_support.is_culling_supported() {
        if !extracted.added_or_changed.is_empty() {
            warn_once!(
                "GpuInstances3d requires GPU culling support; \
                 GPU instances will not be rendered on this device."
            );
        }
        extracted.added_or_changed.clear();
        extracted.removed.clear();
        return;
    }

    let pools = &mut *pools;
    let render_gpu_instance_ranges = &mut *render_gpu_instance_ranges;

    for main_entity in extracted.removed.drain() {
        if let Some(reservation) = reservations.by_entity.remove(&main_entity) {
            pools.free(reservation.base_input_index, reservation.capacity.get());
            render_gpu_instance_ranges.remove(&main_entity);
        }
    }

    extracted.added_or_changed.retain(|main_entity, instances| {
        // If the material or mesh isn't ready yet, keep the entity around and
        // retry next frame.
        let Some(material_instance) = render_material_instances.instances.get(main_entity) else {
            return true;
        };
        let Some(material_binding) = render_material_bindings
            .get(&material_instance.asset_id)
            .copied()
        else {
            return true;
        };
        let Some(vertex_slice) = mesh_allocator.mesh_vertex_slice(&instances.mesh_asset_id) else {
            return true;
        };

        // A changed entity keeps its range if its capacity is unchanged;
        // otherwise its old range is freed and a new one allocated. Either
        // way the range's slots are rewritten below, which picks up the
        // mesh's new location in the vertex and index buffers after its
        // asset is modified, since the mesh allocator may move it.
        let (existing_base, generation) = match reservations.by_entity.remove(main_entity) {
            Some(old_reservation) if old_reservation.capacity == instances.capacity => (
                Some(old_reservation.base_input_index),
                old_reservation.generation.wrapping_add(1),
            ),
            Some(old_reservation) => {
                pools.free(
                    old_reservation.base_input_index,
                    old_reservation.capacity.get(),
                );
                render_gpu_instance_ranges.remove(main_entity);
                (None, old_reservation.generation.wrapping_add(1))
            }
            None => (None, 0),
        };

        let first_vertex_index = vertex_slice.range.start;
        let vertex_count = vertex_slice.range.end - vertex_slice.range.start;

        let (mesh_is_indexed, first_index_index, index_count) =
            match mesh_allocator.mesh_index_slice(&instances.mesh_asset_id) {
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

        let metadata_index = mesh_allocator
            .mesh_metadata_slice(&instances.mesh_asset_id)
            .map(|mesh_metadata_slice| mesh_metadata_slice.range.start)
            .unwrap_or(0);

        let material_slot = u32::from(material_binding.slot);
        let lightmap_slot = u16::MAX as u32;
        let material_and_lightmap_bind_group_slot = material_slot | (lightmap_slot << 16);

        let mut flags = MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        if instances.shadow_receiver {
            flags |= MeshFlags::SHADOW_RECEIVER;
        }
        // Without an `Aabb` the culling data's half-extents are infinite,
        // which would produce NaNs in the frustum test, so skip it.
        if instances.aabb.is_none() {
            flags |= MeshFlags::NO_FRUSTUM_CULLING;
        }
        // The low 16 bits of `MeshFlags` are an index into
        // `visibility_ranges`; `u16::MAX` is the "no LOD" sentinel.
        let lod_sentinel = (u16::MAX as u32) << MeshFlags::LOD_INDEX_SHIFT;
        let resolved_flags = flags.bits() | lod_sentinel;

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
            metadata_index,
            ..Default::default()
        };

        let base_input_index = existing_base.unwrap_or_else(|| {
            pools.allocate(instances.capacity.get(), &render_device, &render_queue)
        });

        let culling_data = MeshCullingData::new(instances.aabb.as_ref());
        pools.write_range(
            base_input_index,
            instances.capacity.get(),
            template,
            culling_data,
            &render_queue,
        );

        reservations.by_entity.insert(
            *main_entity,
            GpuInstanceReservation {
                base_input_index,
                capacity: instances.capacity,
                mesh_asset_id: instances.mesh_asset_id,
                generation,
            },
        );

        render_gpu_instance_ranges.insert(
            *main_entity,
            RenderGpuInstanceRange {
                asset_id: instances.mesh_asset_id,
                base_input_index,
                count: instances.capacity,
                shadow_caster: instances.shadow_caster,
            },
        );

        // The retained phase bins cache the instances' base and mesh from
        // queue
        // time, so a reservation that lands frames after the entity was
        // queued (say, because its mesh was still loading), or that moved
        // because the entity changed, would otherwise leave them stale. Mark
        // the entity as changed so this frame's queue systems dequeue and
        // requeue it with the reservation we just made.
        dirty_specializations
            .changed_renderables
            .insert(*main_entity);

        false
    });
}

#[cfg(test)]
mod tests {
    use super::FreeRunList;

    #[test]
    fn free_run_list_coalesces_adjacent_frees() {
        let mut list = FreeRunList::default();
        list.free(10, 5);
        list.free(20, 5);
        list.free(15, 5);
        assert_eq!(list.allocate(15), Some(10));
        assert!(list.is_empty());
    }

    #[test]
    fn free_run_list_coalesces_backward_and_forward() {
        let mut list = FreeRunList::default();
        list.free(10, 5);
        list.free(20, 5);
        list.free(15, 5);
        list.free(25, 5);
        assert_eq!(list.allocate(20), Some(10));
        assert!(list.is_empty());
    }

    #[test]
    fn free_run_list_allocate_splits_run() {
        let mut list = FreeRunList::default();
        list.free(0, 100);

        assert_eq!(list.allocate(30), Some(0));
        assert!(list.contains(30));
        assert!(!list.contains(29));
        assert_eq!(list.allocate(70), Some(30));
        assert!(list.is_empty());
    }

    #[test]
    fn free_run_list_allocate_none_when_no_run_fits() {
        let mut list = FreeRunList::default();
        list.free(0, 10);
        list.free(20, 10);
        assert_eq!(list.allocate(15), None);
        assert!(list.contains(5));
        assert!(list.contains(25));
    }

    #[test]
    fn free_run_list_contains_boundary_cases() {
        let mut list = FreeRunList::default();
        list.free(5, 3);
        assert!(!list.contains(4));
        assert!(list.contains(5));
        assert!(list.contains(7));
        assert!(!list.contains(8));
    }
}
