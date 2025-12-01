//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

use core::num::{NonZero, NonZeroU64};

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_core_pipeline::{
    core_3d::graph::{Core3d, Node3d},
    experimental::mip_generation::ViewDepthPyramid,
    prepass::{DepthPrepass, PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    prelude::resource_exists,
    query::{Has, Or, QueryState, With, Without},
    resource::Resource,
    schedule::IntoScheduleConfigs as _,
    system::{lifetimeless::Read, Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_render::{
    batching::gpu_preprocessing::{
        BatchedInstanceBuffers, GpuOcclusionCullingWorkItemBuffers, GpuPreprocessingMode,
        GpuPreprocessingSupport, IndirectBatchSet, IndirectParametersBuffers,
        IndirectParametersCpuMetadata, IndirectParametersGpuMetadata, IndirectParametersIndexed,
        IndirectParametersNonIndexed, LatePreprocessWorkItemIndirectParameters, PreprocessWorkItem,
        PreprocessWorkItemBuffers, UntypedPhaseBatchedInstanceBuffers,
        UntypedPhaseIndirectParametersBuffers,
    },
    diagnostic::RecordDiagnostics,
    experimental::occlusion_culling::OcclusionCulling,
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt},
    render_resource::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindingResource, Buffer,
        BufferBinding, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DynamicBindGroupLayoutEntries, PipelineCache, PushConstantRange, RawBufferVec,
        ShaderStages, ShaderType, SpecializedComputePipeline, SpecializedComputePipelines,
        TextureSampleType, UninitBufferVec,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    view::{ExtractedView, NoIndirectDrawing, RetainedViewEntity, ViewUniform, ViewUniformOffset, ViewUniforms},
    Render, RenderApp, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::{default, TypeIdMap};
use bitflags::bitflags;
use smallvec::{smallvec, SmallVec};
use tracing::warn;

use crate::{
    graph::NodePbr, invalidate_submesh_offsets, write_submesh_buffer, DrawData, MeshCullingData,
    MeshCullingDataBuffer, MeshInputUniform, MeshUniform, RenderMeshInstances, SubMeshBuffer,
    SubMeshDescriptor,
};

use super::{ShadowView, ViewLightEntities};

use bytemuck::{Pod, Zeroable};

/// The GPU workgroup size.
const WORKGROUP_SIZE: usize = 64;

/// Entry in flat material list for Stage 2 material expansion.
///
/// One per (mesh, material) pair. Separate buffers for indexed vs non-indexed
/// matching the preprocess work item split.
///
/// Stage 2 dispatches over these entries, checks visibility, and writes draw_data.
/// Geometry is resolved on GPU from SubMeshBuffer using submesh_slot.
#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default)]
#[repr(C)]
pub struct MaterialEntry {
    /// Index into MeshInput buffer (for visibility check and geometry lookup).
    pub mesh_input_index: u32,
    /// Material bind group slot.
    pub material_bind_group_slot: u32,
    /// Submesh slot (0 = full mesh, 1+ = submesh indices).
    pub submesh_slot: u32,
    /// Output destination:
    /// - Direct mode: per-draw output index
    /// - Indirect mode: index into IndirectParametersGpuMetadata
    pub indirect_parameters_index: u32,
}

/// Per-material geometry slice for indirect draws.
///
/// Written by Stage 2 material expansion, read by `build_indirect_params`.
/// Keyed by `indirect_parameters_index`.
#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default)]
#[repr(C)]
pub struct MaterialIndirectGeometry {
    /// First index in the index buffer (for indexed meshes).
    pub first_index: u32,
    /// Number of indices (for indexed meshes).
    pub index_count: u32,
    /// Base vertex offset (for indexed meshes).
    pub base_vertex: u32,
    /// First vertex (for non-indexed meshes).
    pub first_vertex: u32,
    /// Number of vertices (for non-indexed meshes).
    pub vertex_count: u32,
    /// Flags: bit 0 = indexed (1) or non-indexed (0).
    pub flags: u32,
}

impl MaterialIndirectGeometry {
    /// Flag indicating this geometry uses indexed drawing.
    pub const FLAG_INDEXED: u32 = 1;
}

/// Per-phase material expansion buffers.
///
/// Each render phase (e.g., Opaque3d, AlphaMask3d) has its own draw_data buffer
/// because output indices are phase-local (starting from 0 per phase).
pub struct PhaseMaterialExpansionBuffers {
    /// Per-draw output data written by Stage 2.
    ///
    /// Contains material slot and mesh index for each draw instance.
    /// Read by draw shaders via `DrawData[instance_index]`.
    /// Sized to match the phase's data_buffer (MeshUniform output).
    pub draw_data: RawBufferVec<DrawData>,

    /// Per-material geometry slices for indirect indexed draws.
    ///
    /// Written by Stage 2, read by `build_indirect_params`.
    /// Indexed by `indirect_parameters_index` (which starts at 0 for indexed batches).
    /// Separate from non-indexed to avoid index collision.
    pub material_indirect_geometry_indexed: RawBufferVec<MaterialIndirectGeometry>,

    /// Per-material geometry slices for indirect non-indexed draws.
    ///
    /// Written by Stage 2, read by `build_indirect_params`.
    /// Indexed by `indirect_parameters_index` (which starts at 0 for non-indexed batches).
    /// Separate from indexed to avoid index collision.
    pub material_indirect_geometry_non_indexed: RawBufferVec<MaterialIndirectGeometry>,

    /// Per-phase flat material entries for indexed draws.
    ///
    /// Stage 2 dispatches over these, checks visibility, writes draw_data and geometry.
    pub material_entries_indexed: RawBufferVec<MaterialEntry>,

    /// Per-phase flat material entries for non-indexed draws.
    pub material_entries_non_indexed: RawBufferVec<MaterialEntry>,
}

impl Default for PhaseMaterialExpansionBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseMaterialExpansionBuffers {
    /// Creates a new empty set of per-phase buffers.
    pub fn new() -> Self {
        Self {
            draw_data: RawBufferVec::new(bevy_render::render_resource::BufferUsages::STORAGE),
            material_indirect_geometry_indexed: RawBufferVec::new(
                bevy_render::render_resource::BufferUsages::STORAGE,
            ),
            material_indirect_geometry_non_indexed: RawBufferVec::new(
                bevy_render::render_resource::BufferUsages::STORAGE,
            ),
            material_entries_indexed: RawBufferVec::new(
                bevy_render::render_resource::BufferUsages::STORAGE,
            ),
            material_entries_non_indexed: RawBufferVec::new(
                bevy_render::render_resource::BufferUsages::STORAGE,
            ),
        }
    }

    /// Clears the buffers for a new frame.
    pub fn clear(&mut self) {
        self.draw_data.clear();
        self.material_indirect_geometry_indexed.clear();
        self.material_indirect_geometry_non_indexed.clear();
        self.material_entries_indexed.clear();
        self.material_entries_non_indexed.clear();
    }

    /// Writes the buffers to the GPU.
    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.draw_data.write_buffer(render_device, render_queue);
        self.material_indirect_geometry_indexed
            .write_buffer(render_device, render_queue);
        self.material_indirect_geometry_non_indexed
            .write_buffer(render_device, render_queue);
        self.material_entries_indexed
            .write_buffer(render_device, render_queue);
        self.material_entries_non_indexed
            .write_buffer(render_device, render_queue);
    }
}

/// Buffers for the Stage 2 material expansion pass.
///
/// This resource holds the buffers that enable decoupling visibility culling
/// (Stage 1) from material fan-out (Stage 2).
///
/// The key insight is that Stage 2 directly consumes PreprocessWorkItem buffers
/// (both early and late) and uses a material lookup table to get the material
/// slot for each mesh. This avoids CPU dependency on late work items.
/// Information about a single material draw (one per submesh).
///
/// For meshes without submeshes, there's one entry per mesh.
/// For multi-material meshes, there's one entry per submesh.
/// Multiple entries can share the same `input_index` (same mesh entity).
#[derive(Clone, Copy, Debug)]
pub struct MaterialDrawInfo {
    /// Index into MeshInputUniform buffer (identifies the mesh entity).
    pub input_index: u32,
    /// Submesh index within the mesh (0 for single-material meshes).
    /// GPU resolves geometry from SubMeshBuffer[submesh_offset + submesh_index].
    pub submesh_index: u16,
    /// Material bind group slot for this draw.
    pub material_slot: u32,
    /// Whether this draw uses indexed geometry.
    pub is_indexed: bool,
}

#[derive(Resource)]
pub struct MaterialExpansionBuffers {
    /// Per-mesh visibility flags written by Stage 1, read by Stage 2.
    ///
    /// Indexed by `mesh_input_index`. Value of 1 = visible, 0 = culled.
    /// Sized to the high-water mark of MeshInput indices.
    /// This is global (shared across phases) because visibility is mesh-specific.
    pub visibility_flags: RawBufferVec<u32>,

    /// Per-(phase, view) buffers (draw_data, material_indirect_geometry, material_entries).
    ///
    /// Keyed by (TypeId of render phase, RetainedViewEntity).
    /// Each (phase, view) pair has separate output indices starting from 0.
    /// This ensures correct multi-view handling where different views may have
    /// different output/indirect indices within the same phase.
    pub per_phase_view: bevy_platform::collections::HashMap<(core::any::TypeId, RetainedViewEntity), PhaseMaterialExpansionBuffers>,

    /// CPU-side tracking of material draws.
    ///
    /// One entry per draw (per submesh). Multiple entries can share the same
    /// `input_index` for multi-material meshes.
    /// Populated during extraction, used to build MaterialEntries.
    pub(crate) material_draws: Vec<MaterialDrawInfo>,

    /// Tracks the highest input_index seen, for sizing visibility_flags.
    pub(crate) max_input_index: u32,

    /// Legacy global draw_data for CPU path.
    pub draw_data: RawBufferVec<DrawData>,
}

impl Default for MaterialExpansionBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl MaterialExpansionBuffers {
    /// Creates a new empty set of material expansion buffers.
    pub fn new() -> Self {
        MaterialExpansionBuffers {
            visibility_flags: RawBufferVec::new(
                bevy_render::render_resource::BufferUsages::STORAGE,
            ),
            per_phase_view: bevy_platform::collections::HashMap::default(),
            material_draws: Vec::new(),
            max_input_index: 0,
            // Legacy for CPU path
            draw_data: RawBufferVec::new(bevy_render::render_resource::BufferUsages::STORAGE),
        }
    }

    /// Clears all buffers for a new frame.
    pub fn clear(&mut self) {
        self.visibility_flags.clear();
        for phase_buffers in self.per_phase_view.values_mut() {
            phase_buffers.clear();
        }
        self.material_draws.clear();
        self.max_input_index = 0;
        // Legacy
        self.draw_data.clear();
    }

    /// Gets or creates per-(phase, view) buffers for the given key.
    pub fn get_or_create_phase_view_buffers(
        &mut self,
        phase_type_id: core::any::TypeId,
        view_entity: RetainedViewEntity,
    ) -> &mut PhaseMaterialExpansionBuffers {
        self.per_phase_view
            .entry((phase_type_id, view_entity))
            .or_insert_with(PhaseMaterialExpansionBuffers::new)
    }

    /// Gets buffers for a phase by finding the first view's buffers for that phase.
    ///
    /// This is for compatibility with per-phase operations like `build_indirect_params`
    /// that don't have access to a specific view. For multi-view correctness,
    /// callers should use `get_phase_view_buffers` with a specific view entity.
    pub fn get_first_phase_buffers(
        &self,
        phase_type_id: &core::any::TypeId,
    ) -> Option<&PhaseMaterialExpansionBuffers> {
        self.per_phase_view
            .iter()
            .find(|((pt, _), _)| pt == phase_type_id)
            .map(|(_, buffers)| buffers)
    }

    /// Records a material draw for a mesh (or submesh).
    ///
    /// Called during extraction. For meshes with submeshes, this is called
    /// multiple times with the same `input_index` but different geometry/material.
    pub(crate) fn record_material_draw(&mut self, draw: MaterialDrawInfo) {
        self.max_input_index = self.max_input_index.max(draw.input_index);
        self.material_draws.push(draw);
    }

    /// Writes the buffers to the GPU.
    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.visibility_flags
            .write_buffer(render_device, render_queue);

        // Write per-(phase, view) buffers
        for phase_buffers in self.per_phase_view.values_mut() {
            phase_buffers.write_buffers(render_device, render_queue);
        }

        // Legacy buffers for CPU path
        self.draw_data.write_buffer(render_device, render_queue);
    }

    /// Builds MaterialEntry buffers from MaterialWorkItemInfo (GPU path).
    ///
    /// Uses the real output_index from batching instead of the placeholder
    /// indirect_parameters_index from extraction.
    ///
    /// Builds per-(phase, view) MaterialEntry buffers so each (phase, view) pair
    /// has its own entries with view-local output indices.
    pub fn build_material_entries_from_work_items(
        &mut self,
        batched_buffers: &BatchedInstanceBuffers<MeshUniform, MeshInputUniform>,
    ) {
        use bevy_platform::collections::HashMap;

        // Clone material draws into HashMap to avoid borrow conflict.
        // We need owned data since we'll mutate self.per_phase_view during iteration.
        let draws_by_key: HashMap<(u32, u16), MaterialDrawInfo> = self
            .material_draws
            .iter()
            .map(|draw| ((draw.input_index, draw.submesh_index), draw.clone()))
            .collect();

        // Pre-create all per-(phase, view) buffers to avoid mutable borrow during iteration
        for (&phase_type_id, phase_buffers) in &batched_buffers.phase_instance_buffers {
            for &view_entity in phase_buffers.material_work_item_info.keys() {
                self.per_phase_view
                    .entry((phase_type_id, view_entity))
                    .or_insert_with(PhaseMaterialExpansionBuffers::new);
            }
        }

        // Process ALL (phase, view) pairs
        for (&phase_type_id, phase_buffers) in &batched_buffers.phase_instance_buffers {
            // Process each view's work items
            for (&view_entity, view_work_item_info) in &phase_buffers.material_work_item_info {
                // Get per-(phase, view) buffers (guaranteed to exist from pre-creation above)
                let phase_view_expansion = self
                    .per_phase_view
                    .get_mut(&(phase_type_id, view_entity))
                    .unwrap();

                // Clear per-(phase, view) entries
                phase_view_expansion.material_entries_indexed.clear();
                phase_view_expansion.material_entries_non_indexed.clear();

                for info in view_work_item_info {
                    // Look up the corresponding MaterialDrawInfo
                    let Some(draw) = draws_by_key.get(&(info.input_index, info.submesh_index))
                    else {
                        // No matching draw info - this can happen for non-multi-material meshes
                        // which don't go through record_material_draw
                        continue;
                    };

                    // Geometry is resolved on GPU from SubMeshBuffer using submesh_slot
                    let entry = MaterialEntry {
                        mesh_input_index: draw.input_index,
                        material_bind_group_slot: draw.material_slot,
                        submesh_slot: info.submesh_index as u32,
                        indirect_parameters_index: info.output_index,
                    };

                    // Push to per-(phase, view) buffers - get a fresh mutable reference
                    let phase_view_expansion = self
                        .per_phase_view
                        .get_mut(&(phase_type_id, view_entity))
                        .unwrap();
                    if info.is_indexed {
                        phase_view_expansion.material_entries_indexed.push(entry);
                    } else {
                        phase_view_expansion.material_entries_non_indexed.push(entry);
                    }
                }
            }
        }
    }

    /// Populates draw_data buffer directly for CPU/direct rendering mode.
    ///
    /// In direct mode (no GPU culling), all meshes are visible and we can write
    /// draw_data directly from CPU without needing Stage 2 compute shader.
    pub(crate) fn populate_draw_data_for_direct_mode(&mut self) {
        self.draw_data.clear();

        for draw in &self.material_draws {
            self.draw_data.push(DrawData {
                material_bind_group_slot: draw.material_slot,
            });
        }
    }
}

/// System that prepares material expansion buffers for GPU-driven Stage 2.
///
/// This runs AFTER batching (in PrepareResourcesFlush) when PreprocessWorkItems
/// have been created with their real output indices.
///
/// Builds MaterialEntry buffers (indexed and non-indexed) that Stage 2 dispatches over.
/// Stage 2 checks visibility and writes draw_data for each visible material draw.
pub fn build_material_work_items_system(
    mut material_expansion_buffers: ResMut<MaterialExpansionBuffers>,
    gpu_batched_instance_buffers: Option<
        Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let Some(batched_buffers) = gpu_batched_instance_buffers else {
        // For non-GPU path (direct mode without batching):
        // Only draw_data is needed - no Stage 2 material expansion dispatch.
        material_expansion_buffers.populate_draw_data_for_direct_mode();
        material_expansion_buffers.write_buffers(&render_device, &render_queue);
        return;
    };

    // For GPU path: build MaterialEntry buffers using real output_index from batching
    material_expansion_buffers.build_material_entries_from_work_items(&batched_buffers);

    // Size visibility_flags to match input count (indexed by mesh_input_index)
    let input_count = material_expansion_buffers.max_input_index as usize + 1;
    material_expansion_buffers
        .visibility_flags
        .values_mut()
        .resize(input_count, 0);

    // For each (phase, view), size the draw_data and material_indirect_geometry buffers.
    // Stage 2 will populate draw_data[output_index] for each visible mesh.
    for (&phase_type_id, phase_buffers) in &batched_buffers.phase_instance_buffers {
        // Get the size of this phase's MeshUniform output buffer
        let data_buffer_len = phase_buffers.data_buffer.len();

        // Get indirect metadata batch counts for sizing material_indirect_geometry.
        // Indexed and non-indexed have separate batch counts starting from 0.
        let (indexed_batch_count, non_indexed_batch_count) = indirect_parameters_buffers
            .get(&phase_type_id)
            .map(|b| (b.indexed.batch_count(), b.non_indexed.batch_count()))
            .unwrap_or((0, 0));

        // Process each view within this phase
        for &view_entity in phase_buffers.material_work_item_info.keys() {
            // Get or create per-(phase, view) buffers
            let phase_view_buffers = material_expansion_buffers
                .per_phase_view
                .entry((phase_type_id, view_entity))
                .or_insert_with(PhaseMaterialExpansionBuffers::new);

            // Clear and resize draw_data for this (phase, view)
            phase_view_buffers.draw_data.clear();
            phase_view_buffers
                .draw_data
                .values_mut()
                .resize(data_buffer_len, DrawData::default());

            // Size material_indirect_geometry buffers to match indirect metadata batch counts.
            // Stage 2 writes geometry at indirect_parameters_index, which indexes into
            // the indirect metadata. build_indirect_params reads this geometry.
            // Indexed and non-indexed have separate buffers to avoid index collision.
            phase_view_buffers.material_indirect_geometry_indexed.clear();
            if indexed_batch_count > 0 {
                phase_view_buffers
                    .material_indirect_geometry_indexed
                    .values_mut()
                    .resize(indexed_batch_count, MaterialIndirectGeometry::default());
            }
            phase_view_buffers
                .material_indirect_geometry_non_indexed
                .clear();
            if non_indexed_batch_count > 0 {
                phase_view_buffers
                    .material_indirect_geometry_non_indexed
                    .values_mut()
                    .resize(non_indexed_batch_count, MaterialIndirectGeometry::default());
            }
        }
    }

    // Legacy: also update global draw_data for backward compatibility
    // (Will be removed once per-phase binding is complete)
    material_expansion_buffers.draw_data.clear();
    let mut max_output_index: u32 = 0;
    for phase_buffers in batched_buffers.phase_instance_buffers.values() {
        max_output_index = max_output_index.max(phase_buffers.data_buffer.len() as u32);
    }
    material_expansion_buffers
        .draw_data
        .values_mut()
        .resize(max_output_index as usize, DrawData::default());

    // Write buffers to GPU so they're available for bind groups
    material_expansion_buffers.write_buffers(&render_device, &render_queue);
}

/// System that populates draw_data for CPU path where mesh extraction
/// doesn't go through record_material_draw.
///
/// For CPU path, we iterate over all RenderMeshInstances and populate draw_data
/// with identity mapping (mesh_input_index = instance_index).
pub fn populate_draw_data_for_cpu_path(
    render_mesh_instances: Res<RenderMeshInstances>,
    mut material_expansion_buffers: ResMut<MaterialExpansionBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let RenderMeshInstances::CpuBuilding(ref instances) = *render_mesh_instances else {
        return;
    };

    material_expansion_buffers.draw_data.clear();

    // For CPU path, instance_index corresponds to the order in the batched buffer.
    // We iterate in arbitrary order here, but since we're using identity mapping
    // Each draw_data entry just needs to contain the material slot.
    // mesh[] is indexed directly by instance_index, so no mesh indirection needed.
    for (_, mesh_instance) in instances.iter() {
        material_expansion_buffers.draw_data.push(DrawData {
            material_bind_group_slot: mesh_instance.material_bindings_index.slot.into(),
        });
    }

    // Write to GPU
    material_expansion_buffers.write_buffers(&render_device, &render_queue);
}

/// A plugin that builds mesh uniforms on GPU.
///
/// This will only be added if the platform supports compute shaders (e.g. not
/// on WebGL 2).
pub struct GpuMeshPreprocessPlugin {
    /// Whether we're building [`MeshUniform`]s on GPU.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
}

/// The render node that clears out the GPU-side indirect metadata buffers.
///
/// This is only used when indirect drawing is enabled.
#[derive(Default)]
pub struct ClearIndirectParametersMetadataNode;

/// The render node for the first mesh preprocessing pass.
///
/// This pass runs a compute shader to cull meshes outside the view frustum (if
/// that wasn't done by the CPU), cull meshes that weren't visible last frame
/// (if occlusion culling is on), transform them, and, if indirect drawing is
/// on, populate indirect draw parameter metadata for the subsequent
/// [`EarlyPrepassBuildIndirectParametersNode`].
pub struct EarlyGpuPreprocessNode {
    view_query: QueryState<
        (
            Read<ExtractedView>,
            Option<Read<PreprocessBindGroups>>,
            Option<Read<ViewUniformOffset>>,
            Has<NoIndirectDrawing>,
            Has<OcclusionCulling>,
        ),
        Without<SkipGpuPreprocess>,
    >,
    main_view_query: QueryState<Read<ViewLightEntities>>,
}

/// The render node for the second mesh preprocessing pass.
///
/// This pass runs a compute shader to cull meshes outside the view frustum (if
/// that wasn't done by the CPU), cull meshes that were neither visible last
/// frame nor visible this frame (if occlusion culling is on), transform them,
/// and, if indirect drawing is on, populate the indirect draw parameter
/// metadata for the subsequent [`LatePrepassBuildIndirectParametersNode`].
pub struct LateGpuPreprocessNode {
    view_query: QueryState<
        (
            Read<ExtractedView>,
            Read<PreprocessBindGroups>,
            Read<ViewUniformOffset>,
        ),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            With<OcclusionCulling>,
            With<DepthPrepass>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws the meshes visible from the previous frame.
///
/// This node runs a compute shader on the output of the
/// [`EarlyGpuPreprocessNode`] in order to transform the
/// [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct EarlyPrepassBuildIndirectParametersNode {
    view_query: QueryState<
        (Read<ExtractedView>, Read<PreprocessBindGroups>),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            Or<(With<DepthPrepass>, With<ShadowView>)>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws the meshes that are potentially visible on this frame but weren't
/// visible on the previous frame.
///
/// This node runs a compute shader on the output of the
/// [`LateGpuPreprocessNode`] in order to transform the
/// [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct LatePrepassBuildIndirectParametersNode {
    view_query: QueryState<
        (Read<ExtractedView>, Read<PreprocessBindGroups>),
        (
            Without<SkipGpuPreprocess>,
            Without<NoIndirectDrawing>,
            Or<(With<DepthPrepass>, With<ShadowView>)>,
            With<OcclusionCulling>,
        ),
    >,
}

/// The render node for the part of the indirect parameter building pass that
/// draws all meshes, both those that are newly-visible on this frame and those
/// that were visible last frame.
///
/// This node runs a compute shader on the output of the
/// [`EarlyGpuPreprocessNode`] and [`LateGpuPreprocessNode`] in order to
/// transform the [`IndirectParametersGpuMetadata`] into properly-formatted
/// [`IndirectParametersIndexed`] and [`IndirectParametersNonIndexed`].
pub struct MainBuildIndirectParametersNode {
    view_query: QueryState<
        (Read<ExtractedView>, Read<PreprocessBindGroups>),
        (Without<SkipGpuPreprocess>, Without<NoIndirectDrawing>),
    >,
}

/// The render node for the Stage 2 material expansion pass.
///
/// This pass runs a compute shader that reads visibility_flags from Stage 1
/// and writes DrawData (with material_bind_group_slot) for each visible mesh.
/// This decouples visibility culling from material fan-out.
pub struct MaterialExpandNode {
    view_query: QueryState<
        (
            Read<ExtractedView>,
            Read<MaterialExpandBindGroups>,
        ),
        Without<SkipGpuPreprocess>,
    >,
}

/// Per-phase bind groups for Stage 2 material expansion on a single view.
///
/// This mirrors the structure of [`PhasePreprocessBindGroups`] to handle
/// direct, indirect frustum culling, and indirect occlusion culling modes.
#[derive(Clone)]
pub enum PhaseMaterialExpandBindGroups {
    /// The bind groups used when indirect drawing is *not* being used.
    Direct {
        /// The bind group for indexed meshes.
        indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes.
        non_indexed: Option<BindGroup>,
        /// Number of indexed work items.
        indexed_work_item_count: u32,
        /// Number of non-indexed work items.
        non_indexed_work_item_count: u32,
    },

    /// The bind groups used when indirect drawing is enabled but occlusion
    /// culling is not.
    IndirectFrustumCulling {
        /// The bind group for indexed meshes.
        indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes.
        non_indexed: Option<BindGroup>,
        /// Number of indexed work items.
        indexed_work_item_count: u32,
        /// Number of non-indexed work items.
        non_indexed_work_item_count: u32,
    },

    /// The bind groups used when both indirect drawing and occlusion culling
    /// are enabled.
    IndirectOcclusionCulling {
        /// The bind group for indexed meshes during the early phase.
        early_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the early phase.
        early_non_indexed: Option<BindGroup>,
        /// The bind group for indexed meshes during the late phase.
        late_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the late phase.
        late_non_indexed: Option<BindGroup>,
        /// Number of early indexed work items.
        early_indexed_work_item_count: u32,
        /// Number of early non-indexed work items.
        early_non_indexed_work_item_count: u32,
        /// Offset into the late indirect parameters buffer for indexed meshes.
        late_indirect_parameters_indexed_offset: u32,
        /// Offset into the late indirect parameters buffer for non-indexed meshes.
        late_indirect_parameters_non_indexed_offset: u32,
    },
}

/// The compute shader bind groups for Stage 2 material expansion.
///
/// This is a per-view component (like [`PreprocessBindGroups`]) that maps
/// phase TypeId to the bind groups for that phase on this view.
#[derive(Component, Clone, Deref, DerefMut)]
pub struct MaterialExpandBindGroups(pub TypeIdMap<PhaseMaterialExpandBindGroups>);

/// The compute shader pipelines for the GPU mesh preprocessing and indirect
/// parameter building passes.
#[derive(Resource)]
pub struct PreprocessPipelines {
    /// The pipeline used for CPU culling. This pipeline doesn't populate
    /// indirect parameter metadata.
    pub direct_preprocess: PreprocessPipeline,
    /// The pipeline used for mesh preprocessing when GPU frustum culling is in
    /// use, but occlusion culling isn't.
    ///
    /// This pipeline populates indirect parameter metadata.
    pub gpu_frustum_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the first phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub early_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline used for the second phase of occlusion culling.
    ///
    /// This pipeline culls, transforms meshes, and populates indirect parameter
    /// metadata.
    pub late_gpu_occlusion_culling_preprocess: PreprocessPipeline,
    /// The pipeline that builds indirect draw parameters for indexed meshes,
    /// when frustum culling is enabled but occlusion culling *isn't* enabled.
    pub gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline that builds indirect draw parameters for non-indexed
    /// meshes, when frustum culling is enabled but occlusion culling *isn't*
    /// enabled.
    pub gpu_frustum_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// Compute shader pipelines for the early prepass phase that draws meshes
    /// visible in the previous frame.
    pub early_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the late prepass phase that draws meshes
    /// that weren't visible in the previous frame, but became visible this
    /// frame.
    pub late_phase: PreprocessPhasePipelines,
    /// Compute shader pipelines for the main color phase.
    pub main_phase: PreprocessPhasePipelines,

    /// Stage 2: Material expansion pipeline for direct rendering (no indirect).
    ///
    /// Writes `DrawData` with material slot for each visible mesh.
    pub material_expand_direct: MaterialExpandPipeline,

    /// Stage 2: Material expansion pipeline for indirect indexed meshes.
    ///
    /// Resolves geometry from SubMeshBuffer for indexed draws.
    pub material_expand_indirect_indexed: MaterialExpandPipeline,

    /// Stage 2: Material expansion pipeline for indirect non-indexed meshes.
    ///
    /// Resolves geometry from SubMeshBuffer for non-indexed draws.
    pub material_expand_indirect_non_indexed: MaterialExpandPipeline,

    /// Stage 2: Material expansion pipeline for late phase indexed meshes.
    ///
    /// Processes late work items (GPU-only) for indexed meshes.
    pub material_expand_late_indexed: MaterialExpandPipeline,

    /// Stage 2: Material expansion pipeline for late phase non-indexed meshes.
    ///
    /// Processes late work items (GPU-only) for non-indexed meshes.
    pub material_expand_late_non_indexed: MaterialExpandPipeline,
}

/// Compute shader pipelines for a specific phase: early, late, or main.
///
/// The distinction between these phases is relevant for occlusion culling.
#[derive(Clone)]
pub struct PreprocessPhasePipelines {
    /// The pipeline that resets the indirect draw counts used in
    /// `multi_draw_indirect_count` to 0 in preparation for a new pass.
    pub reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline,
    /// The pipeline used for indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into indexed indirect
    /// parameters.
    pub gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline,
    /// The pipeline used for non-indexed indirect parameter building.
    ///
    /// This pipeline converts indirect parameter metadata into non-indexed
    /// indirect parameters.
    pub gpu_occlusion_culling_build_non_indexed_indirect_params: BuildIndirectParametersPipeline,
}

/// The pipeline for the GPU mesh preprocessing shader.
pub struct PreprocessPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the batch set count reset shader.
///
/// This shader resets the indirect batch set count to 0 for each view. It runs
/// in between every phase (early, late, and main).
#[derive(Clone)]
pub struct ResetIndirectBatchSetsPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the indirect parameter building shader.
#[derive(Clone)]
pub struct BuildIndirectParametersPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

/// The pipeline for the Stage 2 material expansion shader.
///
/// This shader writes per-draw `DrawData` (material slot + mesh index) for
/// visible meshes, decoupling material binding from mesh uniform data.
#[derive(Clone)]
pub struct MaterialExpandPipeline {
    /// The bind group layout for the compute shader.
    pub bind_group_layout: BindGroupLayoutDescriptor,
    /// The shader asset handle.
    pub shader: Handle<Shader>,
    /// The pipeline ID for the compute shader.
    ///
    /// This gets filled in `prepare_preprocess_pipelines`.
    pub pipeline_id: Option<CachedComputePipelineId>,
}

bitflags! {
    /// Specifies variants of the mesh preprocessing shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PreprocessPipelineKey: u8 {
        /// Whether GPU frustum culling is in use.
        ///
        /// This `#define`'s `FRUSTUM_CULLING` in the shader.
        const FRUSTUM_CULLING = 1;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 2;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 4;
    }

    /// Specifies variants of the indirect parameter building shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct BuildIndirectParametersPipelineKey: u8 {
        /// Whether the indirect parameter building shader is processing indexed
        /// meshes (those that have index buffers).
        ///
        /// This defines `INDEXED` in the shader.
        const INDEXED = 1;
        /// Whether the GPU and driver supports `multi_draw_indirect_count`.
        ///
        /// This defines `MULTI_DRAW_INDIRECT_COUNT_SUPPORTED` in the shader.
        const MULTI_DRAW_INDIRECT_COUNT_SUPPORTED = 2;
        /// Whether GPU two-phase occlusion culling is in use.
        ///
        /// This `#define`'s `OCCLUSION_CULLING` in the shader.
        const OCCLUSION_CULLING = 4;
        /// Whether this is the early phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `EARLY_PHASE` in the shader.
        const EARLY_PHASE = 8;
        /// Whether this is the late phase of GPU two-phase occlusion culling.
        ///
        /// This `#define`'s `LATE_PHASE` in the shader.
        const LATE_PHASE = 16;
        /// Whether this is the phase that runs after the early and late phases,
        /// and right before the main drawing logic, when GPU two-phase
        /// occlusion culling is in use.
        ///
        /// This `#define`'s `MAIN_PHASE` in the shader.
        const MAIN_PHASE = 32;
        /// Whether Stage 2 material expansion is in use.
        ///
        /// When enabled, instance counts come from `draw_count` instead of
        /// `early_instance_count`. This `#define`'s `MATERIAL_EXPANSION` in the shader.
        const MATERIAL_EXPANSION = 64;
    }

    /// Specifies variants of the material expansion shader.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MaterialExpandPipelineKey: u8 {
        /// Whether indirect drawing is in use (vs direct drawing).
        ///
        /// This `#define`'s `INDIRECT` in the shader.
        const INDIRECT = 1;
        /// Whether this is the late phase of GPU occlusion culling.
        ///
        /// This `#define`'s `LATE_PHASE` in the shader.
        const LATE_PHASE = 2;
        /// Whether processing indexed meshes (vs non-indexed).
        ///
        /// This `#define`'s `INDEXED` in the shader.
        const INDEXED = 4;
    }
}

/// The compute shader bind group for the mesh preprocessing pass for each
/// render phase.
///
/// This goes on the view. It maps the [`core::any::TypeId`] of a render phase
/// (e.g.  [`bevy_core_pipeline::core_3d::Opaque3d`]) to the
/// [`PhasePreprocessBindGroups`] for that phase.
#[derive(Component, Clone, Deref, DerefMut)]
pub struct PreprocessBindGroups(pub TypeIdMap<PhasePreprocessBindGroups>);

/// The compute shader bind group for the mesh preprocessing step for a single
/// render phase on a single view.
#[derive(Clone)]
pub enum PhasePreprocessBindGroups {
    /// The bind group used for the single invocation of the compute shader when
    /// indirect drawing is *not* being used.
    ///
    /// Because direct drawing doesn't require splitting the meshes into indexed
    /// and non-indexed meshes, there's only one bind group in this case.
    Direct(BindGroup),

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, there are two bind groups here.
    IndirectFrustumCulling {
        /// The bind group for indexed meshes.
        indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes.
        non_indexed: Option<BindGroup>,
    },

    /// The bind groups used for the compute shader when indirect drawing is
    /// being used, but occlusion culling isn't being used.
    ///
    /// Because indirect drawing requires splitting the meshes into indexed and
    /// non-indexed meshes, and because occlusion culling requires splitting
    /// this phase into early and late versions, there are four bind groups
    /// here.
    IndirectOcclusionCulling {
        /// The bind group for indexed meshes during the early mesh
        /// preprocessing phase.
        early_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the early mesh
        /// preprocessing phase.
        early_non_indexed: Option<BindGroup>,
        /// The bind group for indexed meshes during the late mesh preprocessing
        /// phase.
        late_indexed: Option<BindGroup>,
        /// The bind group for non-indexed meshes during the late mesh
        /// preprocessing phase.
        late_non_indexed: Option<BindGroup>,
    },
}

/// The bind groups for the compute shaders that reset indirect draw counts and
/// build indirect parameters.
///
/// Keyed by (phase TypeId, RetainedViewEntity) to support multi-view rendering.
/// Each view has its own output indices, so build_indirect_params must run
/// per-(phase, view) with view-specific material_indirect_geometry buffers.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct BuildIndirectParametersBindGroups(
    pub bevy_platform::collections::HashMap<
        (core::any::TypeId, RetainedViewEntity),
        PhaseBuildIndirectParametersBindGroups,
    >,
);

impl BuildIndirectParametersBindGroups {
    /// Creates a new, empty [`BuildIndirectParametersBindGroups`] table.
    pub fn new() -> BuildIndirectParametersBindGroups {
        Self::default()
    }
}

/// The per-phase set of bind groups for the compute shaders that reset indirect
/// draw counts and build indirect parameters.
pub struct PhaseBuildIndirectParametersBindGroups {
    /// The bind group for the `reset_indirect_batch_sets.wgsl` shader, for
    /// indexed meshes.
    reset_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `reset_indirect_batch_sets.wgsl` shader, for
    /// non-indexed meshes.
    reset_non_indexed_indirect_batch_sets: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wgsl` shader, for indexed
    /// meshes.
    build_indexed_indirect: Option<BindGroup>,
    /// The bind group for the `build_indirect_params.wgsl` shader, for
    /// non-indexed meshes.
    build_non_indexed_indirect: Option<BindGroup>,
}

/// Stops the `GpuPreprocessNode` attempting to generate the buffer for this view
/// useful to avoid duplicating effort if the bind group is shared between views
#[derive(Component, Default)]
pub struct SkipGpuPreprocess;

impl Plugin for GpuMeshPreprocessPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "mesh_preprocess.wgsl");
        embedded_asset!(app, "mesh_material_expand.wgsl");
        embedded_asset!(app, "reset_indirect_batch_sets.wgsl");
        embedded_asset!(app, "build_indirect_params.wgsl");
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // This plugin does nothing if GPU instance buffer building isn't in
        // use.
        let gpu_preprocessing_support = render_app.world().resource::<GpuPreprocessingSupport>();
        if !self.use_gpu_instance_buffer_builder || !gpu_preprocessing_support.is_available() {
            return;
        }

        render_app
            .init_resource::<PreprocessPipelines>()
            // MaterialExpansionBuffers is initialized in MeshRenderPlugin for both paths
            .init_resource::<SpecializedComputePipelines<PreprocessPipeline>>()
            .init_resource::<SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>>()
            .init_resource::<SpecializedComputePipelines<BuildIndirectParametersPipeline>>()
            .init_resource::<SpecializedComputePipelines<MaterialExpandPipeline>>()
            .add_systems(
                Render,
                (
                    // Invalidate submesh offsets for modified/removed meshes before
                    // meshes are collected. This mirrors MeshAllocator::free_meshes.
                    invalidate_submesh_offsets.in_set(RenderSystems::PrepareAssets),
                    prepare_preprocess_pipelines.in_set(RenderSystems::Prepare),
                    prepare_preprocess_bind_groups
                        .run_if(resource_exists::<BatchedInstanceBuffers<
                            MeshUniform,
                            MeshInputUniform
                        >>)
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_material_expand_bind_groups
                        .run_if(resource_exists::<BatchedInstanceBuffers<
                            MeshUniform,
                            MeshInputUniform
                        >>)
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(prepare_preprocess_bind_groups),
                    write_mesh_culling_data_buffer.in_set(RenderSystems::PrepareResourcesFlush),
                    write_submesh_buffer.in_set(RenderSystems::PrepareResourcesFlush),
                ),
            )
            .add_render_graph_node::<ClearIndirectParametersMetadataNode>(
                Core3d,
                NodePbr::ClearIndirectParametersMetadata
            )
            .add_render_graph_node::<EarlyGpuPreprocessNode>(Core3d, NodePbr::EarlyGpuPreprocess)
            .add_render_graph_node::<LateGpuPreprocessNode>(Core3d, NodePbr::LateGpuPreprocess)
            .add_render_graph_node::<EarlyPrepassBuildIndirectParametersNode>(
                Core3d,
                NodePbr::EarlyPrepassBuildIndirectParameters,
            )
            .add_render_graph_node::<LatePrepassBuildIndirectParametersNode>(
                Core3d,
                NodePbr::LatePrepassBuildIndirectParameters,
            )
            .add_render_graph_node::<MainBuildIndirectParametersNode>(
                Core3d,
                NodePbr::MainBuildIndirectParameters,
            )
            .add_render_graph_node::<MaterialExpandNode>(
                Core3d,
                NodePbr::MaterialExpand,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    NodePbr::ClearIndirectParametersMetadata,
                    NodePbr::EarlyGpuPreprocess,
                    // Stage 2: Material expansion runs after Stage 1 visibility culling,
                    // BEFORE build_indirect_params so draw_count is populated.
                    // Writes DrawData with material slots and increments draw_count.
                    NodePbr::MaterialExpand,
                    NodePbr::EarlyPrepassBuildIndirectParameters,
                    Node3d::EarlyPrepass,
                    Node3d::EarlyDeferredPrepass,
                    Node3d::EarlyDownsampleDepth,
                    NodePbr::LateGpuPreprocess,
                    NodePbr::LatePrepassBuildIndirectParameters,
                    Node3d::LatePrepass,
                    Node3d::LateDeferredPrepass,
                    NodePbr::MainBuildIndirectParameters,
                    Node3d::StartMainPass,
                ),
            ).add_render_graph_edges(
                Core3d,
                (
                    NodePbr::EarlyPrepassBuildIndirectParameters,
                    NodePbr::EarlyShadowPass,
                    Node3d::EarlyDownsampleDepth,
                )
            ).add_render_graph_edges(
                Core3d,
                (
                    NodePbr::LatePrepassBuildIndirectParameters,
                    NodePbr::LateShadowPass,
                    NodePbr::MainBuildIndirectParameters,
                )
            );
    }
}

impl Node for ClearIndirectParametersMetadataNode {
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(indirect_parameters_buffers) = world.get_resource::<IndirectParametersBuffers>()
        else {
            return Ok(());
        };

        // Clear out each indexed and non-indexed GPU-side buffer.
        for phase_indirect_parameters_buffers in indirect_parameters_buffers.values() {
            if let Some(indexed_gpu_metadata_buffer) = phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer()
            {
                render_context.command_encoder().clear_buffer(
                    indexed_gpu_metadata_buffer,
                    0,
                    Some(
                        phase_indirect_parameters_buffers.indexed.batch_count() as u64
                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                    ),
                );
            }

            if let Some(non_indexed_gpu_metadata_buffer) = phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer()
            {
                render_context.command_encoder().clear_buffer(
                    non_indexed_gpu_metadata_buffer,
                    0,
                    Some(
                        phase_indirect_parameters_buffers.non_indexed.batch_count() as u64
                            * size_of::<IndirectParametersGpuMetadata>() as u64,
                    ),
                );
            }
        }

        Ok(())
    }
}

impl FromWorld for EarlyGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
            main_view_query: QueryState::new(world),
        }
    }
}

impl Node for EarlyGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
        self.main_view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let diagnostics = render_context.diagnostic_recorder();

        // Grab the [`BatchedInstanceBuffers`].
        let batched_instance_buffers =
            world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("early_mesh_preprocessing"),
                    timestamp_writes: None,
                });
        let pass_span = diagnostics.pass_span(&mut compute_pass, "early_mesh_preprocessing");

        let mut all_views: SmallVec<[_; 8]> = SmallVec::new();
        all_views.push(graph.view_entity());
        if let Ok(shadow_cascade_views) =
            self.main_view_query.get_manual(world, graph.view_entity())
        {
            all_views.extend(shadow_cascade_views.lights.iter().copied());
        }

        // Run the compute passes.

        for view_entity in all_views {
            let Ok((
                view,
                bind_groups,
                view_uniform_offset,
                no_indirect_drawing,
                occlusion_culling,
            )) = self.view_query.get_manual(world, view_entity)
            else {
                continue;
            };

            let Some(bind_groups) = bind_groups else {
                continue;
            };
            let Some(view_uniform_offset) = view_uniform_offset else {
                continue;
            };

            // Select the right pipeline, depending on whether GPU culling is in
            // use.
            let maybe_pipeline_id = if no_indirect_drawing {
                preprocess_pipelines.direct_preprocess.pipeline_id
            } else if occlusion_culling {
                preprocess_pipelines
                    .early_gpu_occlusion_culling_preprocess
                    .pipeline_id
            } else {
                preprocess_pipelines
                    .gpu_frustum_culling_preprocess
                    .pipeline_id
            };

            // Fetch the pipeline.
            let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
                warn!("The build mesh uniforms pipeline wasn't ready");
                continue;
            };

            let Some(preprocess_pipeline) =
                pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
            else {
                // This will happen while the pipeline is being compiled and is fine.
                continue;
            };

            compute_pass.set_pipeline(preprocess_pipeline);

            // Loop over each render phase.
            for (phase_type_id, batched_phase_instance_buffers) in
                &batched_instance_buffers.phase_instance_buffers
            {
                // Grab the work item buffers for this view.
                let Some(work_item_buffers) = batched_phase_instance_buffers
                    .work_item_buffers
                    .get(&view.retained_view_entity)
                else {
                    continue;
                };

                // Fetch the bind group for the render phase.
                let Some(phase_bind_groups) = bind_groups.get(phase_type_id) else {
                    continue;
                };

                // Make sure the mesh preprocessing shader has access to the
                // view info it needs to do culling and motion vector
                // computation.
                let dynamic_offsets = [view_uniform_offset.offset];

                // Are we drawing directly or indirectly?
                match *phase_bind_groups {
                    PhasePreprocessBindGroups::Direct(ref bind_group) => {
                        // Invoke the mesh preprocessing shader to transform
                        // meshes only, but not cull.
                        let PreprocessWorkItemBuffers::Direct(work_item_buffer) = work_item_buffers
                        else {
                            continue;
                        };
                        compute_pass.set_bind_group(0, bind_group, &dynamic_offsets);
                        let workgroup_count = work_item_buffer.len().div_ceil(WORKGROUP_SIZE);
                        if workgroup_count > 0 {
                            compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                        }
                    }

                    PhasePreprocessBindGroups::IndirectFrustumCulling {
                        indexed: ref maybe_indexed_bind_group,
                        non_indexed: ref maybe_non_indexed_bind_group,
                    }
                    | PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        early_indexed: ref maybe_indexed_bind_group,
                        early_non_indexed: ref maybe_non_indexed_bind_group,
                        ..
                    } => {
                        // Invoke the mesh preprocessing shader to transform and
                        // cull the meshes.
                        let PreprocessWorkItemBuffers::Indirect {
                            indexed: indexed_buffer,
                            non_indexed: non_indexed_buffer,
                            ..
                        } = work_item_buffers
                        else {
                            continue;
                        };

                        // Transform and cull indexed meshes if there are any.
                        if let Some(indexed_bind_group) = maybe_indexed_bind_group {
                            if let PreprocessWorkItemBuffers::Indirect {
                                gpu_occlusion_culling:
                                    Some(GpuOcclusionCullingWorkItemBuffers {
                                        late_indirect_parameters_indexed_offset,
                                        ..
                                    }),
                                ..
                            } = *work_item_buffers
                            {
                                compute_pass.set_push_constants(
                                    0,
                                    bytemuck::bytes_of(&late_indirect_parameters_indexed_offset),
                                );
                            }

                            compute_pass.set_bind_group(0, indexed_bind_group, &dynamic_offsets);
                            let workgroup_count = indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }

                        // Transform and cull non-indexed meshes if there are any.
                        if let Some(non_indexed_bind_group) = maybe_non_indexed_bind_group {
                            if let PreprocessWorkItemBuffers::Indirect {
                                gpu_occlusion_culling:
                                    Some(GpuOcclusionCullingWorkItemBuffers {
                                        late_indirect_parameters_non_indexed_offset,
                                        ..
                                    }),
                                ..
                            } = *work_item_buffers
                            {
                                compute_pass.set_push_constants(
                                    0,
                                    bytemuck::bytes_of(
                                        &late_indirect_parameters_non_indexed_offset,
                                    ),
                                );
                            }

                            compute_pass.set_bind_group(
                                0,
                                non_indexed_bind_group,
                                &dynamic_offsets,
                            );
                            let workgroup_count = non_indexed_buffer.len().div_ceil(WORKGROUP_SIZE);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
                            }
                        }
                    }
                }
            }
        }

        pass_span.end(&mut compute_pass);

        Ok(())
    }
}

impl FromWorld for EarlyPrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for LatePrepassBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for MainBuildIndirectParametersNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl FromWorld for LateGpuPreprocessNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for LateGpuPreprocessNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let diagnostics = render_context.diagnostic_recorder();

        // Grab the [`BatchedInstanceBuffers`].
        let batched_instance_buffers =
            world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("late_mesh_preprocessing"),
                    timestamp_writes: None,
                });
        let pass_span = diagnostics.pass_span(&mut compute_pass, "late_mesh_preprocessing");

        // Run the compute passes.
        for (view, bind_groups, view_uniform_offset) in self.view_query.iter_manual(world) {
            let maybe_pipeline_id = preprocess_pipelines
                .late_gpu_occlusion_culling_preprocess
                .pipeline_id;

            // Fetch the pipeline.
            let Some(preprocess_pipeline_id) = maybe_pipeline_id else {
                warn!("The build mesh uniforms pipeline wasn't ready");
                return Ok(());
            };

            let Some(preprocess_pipeline) =
                pipeline_cache.get_compute_pipeline(preprocess_pipeline_id)
            else {
                // This will happen while the pipeline is being compiled and is fine.
                return Ok(());
            };

            compute_pass.set_pipeline(preprocess_pipeline);

            // Loop over each phase. Because we built the phases in parallel,
            // each phase has a separate set of instance buffers.
            for (phase_type_id, batched_phase_instance_buffers) in
                &batched_instance_buffers.phase_instance_buffers
            {
                let UntypedPhaseBatchedInstanceBuffers {
                    ref work_item_buffers,
                    ref late_indexed_indirect_parameters_buffer,
                    ref late_non_indexed_indirect_parameters_buffer,
                    ..
                } = *batched_phase_instance_buffers;

                // Grab the work item buffers for this view.
                let Some(phase_work_item_buffers) =
                    work_item_buffers.get(&view.retained_view_entity)
                else {
                    continue;
                };

                let (
                    PreprocessWorkItemBuffers::Indirect {
                        gpu_occlusion_culling:
                            Some(GpuOcclusionCullingWorkItemBuffers {
                                late_indirect_parameters_indexed_offset,
                                late_indirect_parameters_non_indexed_offset,
                                ..
                            }),
                        ..
                    },
                    Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
                        late_indexed: maybe_late_indexed_bind_group,
                        late_non_indexed: maybe_late_non_indexed_bind_group,
                        ..
                    }),
                    Some(late_indexed_indirect_parameters_buffer),
                    Some(late_non_indexed_indirect_parameters_buffer),
                ) = (
                    phase_work_item_buffers,
                    bind_groups.get(phase_type_id),
                    late_indexed_indirect_parameters_buffer.buffer(),
                    late_non_indexed_indirect_parameters_buffer.buffer(),
                )
                else {
                    continue;
                };

                let mut dynamic_offsets: SmallVec<[u32; 1]> = smallvec![];
                dynamic_offsets.push(view_uniform_offset.offset);

                // If there's no space reserved for work items, then don't
                // bother doing the dispatch, as there can't possibly be any
                // meshes of the given class (indexed or non-indexed) in this
                // phase.

                // Transform and cull indexed meshes if there are any.
                if let Some(late_indexed_bind_group) = maybe_late_indexed_bind_group {
                    compute_pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(late_indirect_parameters_indexed_offset),
                    );

                    compute_pass.set_bind_group(0, late_indexed_bind_group, &dynamic_offsets);
                    compute_pass.dispatch_workgroups_indirect(
                        late_indexed_indirect_parameters_buffer,
                        (*late_indirect_parameters_indexed_offset as u64)
                            * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
                    );
                }

                // Transform and cull non-indexed meshes if there are any.
                if let Some(late_non_indexed_bind_group) = maybe_late_non_indexed_bind_group {
                    compute_pass.set_push_constants(
                        0,
                        bytemuck::bytes_of(late_indirect_parameters_non_indexed_offset),
                    );

                    compute_pass.set_bind_group(0, late_non_indexed_bind_group, &dynamic_offsets);
                    compute_pass.dispatch_workgroups_indirect(
                        late_non_indexed_indirect_parameters_buffer,
                        (*late_indirect_parameters_non_indexed_offset as u64)
                            * (size_of::<LatePreprocessWorkItemIndirectParameters>() as u64),
                    );
                }
            }
        }

        pass_span.end(&mut compute_pass);

        Ok(())
    }
}

impl Node for EarlyPrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        // Get the current view's RetainedViewEntity for filtering.
        let Ok((extracted_view, _)) = self.view_query.get_manual(world, graph.view_entity()) else {
            return Ok(());
        };

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.early_phase,
            "early_prepass_indirect_parameters_building",
            extracted_view.retained_view_entity,
        )
    }
}

impl Node for LatePrepassBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        // Get the current view's RetainedViewEntity for filtering.
        let Ok((extracted_view, _)) = self.view_query.get_manual(world, graph.view_entity()) else {
            return Ok(());
        };

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.late_phase,
            "late_prepass_indirect_parameters_building",
            extracted_view.retained_view_entity,
        )
    }
}

impl Node for MainBuildIndirectParametersNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();

        // Get the current view's RetainedViewEntity for filtering.
        let Ok((extracted_view, _)) = self.view_query.get_manual(world, graph.view_entity()) else {
            return Ok(());
        };

        run_build_indirect_parameters_node(
            render_context,
            world,
            &preprocess_pipelines.main_phase,
            "main_indirect_parameters_building",
            extracted_view.retained_view_entity,
        )
    }
}

impl FromWorld for MaterialExpandNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
        }
    }
}

impl Node for MaterialExpandNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
    }

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        // If there are no views to process, nothing to do
        if self.view_query.iter_manual(world).next().is_none() {
            return Ok(());
        }

        let diagnostics = render_context.diagnostic_recorder();
        let pipeline_cache = world.resource::<PipelineCache>();
        let preprocess_pipelines = world.resource::<PreprocessPipelines>();
        let batched_instance_buffers =
            world.resource::<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>();

        // Get pipelines
        let direct_pipeline = preprocess_pipelines
            .material_expand_direct
            .pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let indirect_indexed_pipeline = preprocess_pipelines
            .material_expand_indirect_indexed
            .pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let indirect_non_indexed_pipeline = preprocess_pipelines
            .material_expand_indirect_non_indexed
            .pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let late_indexed_pipeline = preprocess_pipelines
            .material_expand_late_indexed
            .pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));
        let late_non_indexed_pipeline = preprocess_pipelines
            .material_expand_late_non_indexed
            .pipeline_id
            .and_then(|id| pipeline_cache.get_compute_pipeline(id));

        if direct_pipeline.is_none() && indirect_indexed_pipeline.is_none() {
            // Pipelines still compiling
            return Ok(());
        }

        let mut compute_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("material_expansion"),
                    timestamp_writes: None,
                });
        let pass_span = diagnostics.pass_span(&mut compute_pass, "material_expansion");

        // Process each view
        for (_view, bind_groups) in self.view_query.iter_manual(world) {
            // Loop over each render phase
            for (phase_type_id, phase_bind_groups) in bind_groups.iter() {
                // Get the late indirect parameters buffer for this phase (needed for late dispatch)
                let phase_instance_buffers = batched_instance_buffers
                    .phase_instance_buffers
                    .get(phase_type_id);

                match phase_bind_groups {
                    PhaseMaterialExpandBindGroups::Direct {
                        indexed,
                        non_indexed,
                        indexed_work_item_count,
                        non_indexed_work_item_count,
                    } => {
                        let Some(pipeline) = direct_pipeline else {
                            continue;
                        };
                        compute_pass.set_pipeline(pipeline);

                        // Process indexed meshes
                        if let Some(bind_group) = indexed {
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }

                        // Process non-indexed meshes
                        if let Some(bind_group) = non_indexed {
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                non_indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }
                    }

                    PhaseMaterialExpandBindGroups::IndirectFrustumCulling {
                        indexed,
                        non_indexed,
                        indexed_work_item_count,
                        non_indexed_work_item_count,
                    } => {
                        // Process indexed meshes with indexed pipeline
                        if let (Some(bind_group), Some(pipeline)) =
                            (indexed, indirect_indexed_pipeline)
                        {
                            compute_pass.set_pipeline(pipeline);
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }

                        // Process non-indexed meshes with non-indexed pipeline
                        if let (Some(bind_group), Some(pipeline)) =
                            (non_indexed, indirect_non_indexed_pipeline)
                        {
                            compute_pass.set_pipeline(pipeline);
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                non_indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }
                    }

                    PhaseMaterialExpandBindGroups::IndirectOcclusionCulling {
                        early_indexed,
                        early_non_indexed,
                        late_indexed,
                        late_non_indexed,
                        early_indexed_work_item_count,
                        early_non_indexed_work_item_count,
                        late_indirect_parameters_indexed_offset,
                        late_indirect_parameters_non_indexed_offset,
                    } => {
                        // Early phase - use indexed/non-indexed pipelines with direct dispatch
                        // Process early indexed meshes with indexed pipeline
                        if let (Some(bind_group), Some(pipeline)) =
                            (early_indexed, indirect_indexed_pipeline)
                        {
                            compute_pass.set_pipeline(pipeline);
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                early_indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }

                        // Process early non-indexed meshes with non-indexed pipeline
                        if let (Some(bind_group), Some(pipeline)) =
                            (early_non_indexed, indirect_non_indexed_pipeline)
                        {
                            compute_pass.set_pipeline(pipeline);
                            compute_pass.set_bind_group(0, bind_group, &[]);
                            let workgroup_count =
                                early_non_indexed_work_item_count.div_ceil(WORKGROUP_SIZE as u32);
                            if workgroup_count > 0 {
                                compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                            }
                        }

                        // Late phase - use indirect dispatch from late_indirect_parameters_buffer
                        // The late work items are GPU-populated, so we need indirect dispatch
                        if let Some(phase_buffers) = phase_instance_buffers {
                            let UntypedPhaseBatchedInstanceBuffers {
                                ref late_indexed_indirect_parameters_buffer,
                                ref late_non_indexed_indirect_parameters_buffer,
                                ..
                            } = *phase_buffers;

                            // Process late indexed meshes via indirect dispatch
                            if let (Some(bind_group), Some(pipeline), Some(indirect_buffer)) = (
                                late_indexed,
                                late_indexed_pipeline,
                                late_indexed_indirect_parameters_buffer.buffer(),
                            ) {
                                compute_pass.set_pipeline(pipeline);
                                compute_pass.set_bind_group(0, bind_group, &[]);
                                compute_pass.dispatch_workgroups_indirect(
                                    indirect_buffer,
                                    (*late_indirect_parameters_indexed_offset as u64)
                                        * (size_of::<LatePreprocessWorkItemIndirectParameters>()
                                            as u64),
                                );
                            }

                            // Process late non-indexed meshes via indirect dispatch
                            if let (Some(bind_group), Some(pipeline), Some(indirect_buffer)) = (
                                late_non_indexed,
                                late_non_indexed_pipeline,
                                late_non_indexed_indirect_parameters_buffer.buffer(),
                            ) {
                                compute_pass.set_pipeline(pipeline);
                                compute_pass.set_bind_group(0, bind_group, &[]);
                                compute_pass.dispatch_workgroups_indirect(
                                    indirect_buffer,
                                    (*late_indirect_parameters_non_indexed_offset as u64)
                                        * (size_of::<LatePreprocessWorkItemIndirectParameters>()
                                            as u64),
                                );
                            }
                        }
                    }
                }
            }
        }

        pass_span.end(&mut compute_pass);

        Ok(())
    }
}

fn run_build_indirect_parameters_node(
    render_context: &mut RenderContext,
    world: &World,
    preprocess_phase_pipelines: &PreprocessPhasePipelines,
    label: &'static str,
    view_filter: RetainedViewEntity,
) -> Result<(), NodeRunError> {
    let Some(build_indirect_params_bind_groups) =
        world.get_resource::<BuildIndirectParametersBindGroups>()
    else {
        return Ok(());
    };

    let diagnostics = render_context.diagnostic_recorder();

    let pipeline_cache = world.resource::<PipelineCache>();
    let indirect_parameters_buffers = world.resource::<IndirectParametersBuffers>();

    let mut compute_pass =
        render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
    let pass_span = diagnostics.pass_span(&mut compute_pass, label);

    // Fetch the pipeline.
    let (
        Some(reset_indirect_batch_sets_pipeline_id),
        Some(build_indexed_indirect_params_pipeline_id),
        Some(build_non_indexed_indirect_params_pipeline_id),
    ) = (
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .pipeline_id,
    )
    else {
        warn!("The build indirect parameters pipelines weren't ready");
        pass_span.end(&mut compute_pass);
        return Ok(());
    };

    let (
        Some(reset_indirect_batch_sets_pipeline),
        Some(build_indexed_indirect_params_pipeline),
        Some(build_non_indexed_indirect_params_pipeline),
    ) = (
        pipeline_cache.get_compute_pipeline(reset_indirect_batch_sets_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_indexed_indirect_params_pipeline_id),
        pipeline_cache.get_compute_pipeline(build_non_indexed_indirect_params_pipeline_id),
    )
    else {
        // This will happen while the pipeline is being compiled and is fine.
        pass_span.end(&mut compute_pass);
        return Ok(());
    };

    // Loop over each phase that has bind groups for the current view.
    // We filter by view_filter to ensure we only process this view's bind groups,
    // avoiding O(V²) dispatches when the render graph node runs per-view.
    for ((phase_type_id, bind_group_view), phase_build_indirect_params_bind_groups) in
        build_indirect_params_bind_groups.iter()
    {
        // Skip bind groups that aren't for this view.
        if *bind_group_view != view_filter {
            continue;
        }
        // Note: indirect_parameters_buffers is keyed by phase only (shared across views).
        // The per-view material_indirect_geometry is baked into the bind group.
        let Some(phase_indirect_parameters_buffers) =
            indirect_parameters_buffers.get(phase_type_id)
        else {
            continue;
        };

        // Build indexed indirect parameters.
        if let (
            Some(reset_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(true)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_indexed_params_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .indexed
                .batch_count()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }

        // Build non-indexed indirect parameters.
        if let (
            Some(reset_non_indexed_indirect_batch_sets_bind_group),
            Some(build_indirect_non_indexed_params_bind_group),
        ) = (
            &phase_build_indirect_params_bind_groups.reset_non_indexed_indirect_batch_sets,
            &phase_build_indirect_params_bind_groups.build_non_indexed_indirect,
        ) {
            compute_pass.set_pipeline(reset_indirect_batch_sets_pipeline);
            compute_pass.set_bind_group(0, reset_non_indexed_indirect_batch_sets_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .batch_set_count(false)
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            compute_pass.set_pipeline(build_non_indexed_indirect_params_pipeline);
            compute_pass.set_bind_group(0, build_indirect_non_indexed_params_bind_group, &[]);
            let workgroup_count = phase_indirect_parameters_buffers
                .non_indexed
                .batch_count()
                .div_ceil(WORKGROUP_SIZE);
            if workgroup_count > 0 {
                compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }
        }
    }

    pass_span.end(&mut compute_pass);

    Ok(())
}

impl PreprocessPipelines {
    /// Returns true if the preprocessing and indirect parameters pipelines have
    /// been loaded or false otherwise.
    pub(crate) fn pipelines_are_loaded(
        &self,
        pipeline_cache: &PipelineCache,
        preprocessing_support: &GpuPreprocessingSupport,
    ) -> bool {
        match preprocessing_support.max_supported_mode {
            GpuPreprocessingMode::None => false,
            GpuPreprocessingMode::PreprocessingOnly => {
                self.direct_preprocess.is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_preprocess
                        .is_loaded(pipeline_cache)
            }
            GpuPreprocessingMode::Culling => {
                self.direct_preprocess.is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .early_gpu_occlusion_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .late_gpu_occlusion_culling_preprocess
                        .is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_build_indexed_indirect_params
                        .is_loaded(pipeline_cache)
                    && self
                        .gpu_frustum_culling_build_non_indexed_indirect_params
                        .is_loaded(pipeline_cache)
                    && self.early_phase.is_loaded(pipeline_cache)
                    && self.late_phase.is_loaded(pipeline_cache)
                    && self.main_phase.is_loaded(pipeline_cache)
            }
        }
    }
}

impl PreprocessPhasePipelines {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.reset_indirect_batch_sets.is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_indexed_indirect_params
                .is_loaded(pipeline_cache)
            && self
                .gpu_occlusion_culling_build_non_indexed_indirect_params
                .is_loaded(pipeline_cache)
    }
}

impl PreprocessPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl BuildIndirectParametersPipeline {
    /// Returns true if this pipeline has been loaded into the pipeline cache or
    /// false otherwise.
    fn is_loaded(&self, pipeline_cache: &PipelineCache) -> bool {
        self.pipeline_id
            .is_some_and(|pipeline_id| pipeline_cache.get_compute_pipeline(pipeline_id).is_some())
    }
}

impl SpecializedComputePipeline for PreprocessPipeline {
    type Key = PreprocessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec!["WRITE_INDIRECT_PARAMETERS_METADATA".into()];
        if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
            shader_defs.push("INDIRECT".into());
            shader_defs.push("FRUSTUM_CULLING".into());
        }
        if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
            if key.contains(PreprocessPipelineKey::EARLY_PHASE) {
                shader_defs.push("EARLY_PHASE".into());
            } else {
                shader_defs.push("LATE_PHASE".into());
            }
        }

        ComputePipelineDescriptor {
            label: Some(
                format!(
                    "mesh preprocessing ({})",
                    if key.contains(
                        PreprocessPipelineKey::OCCLUSION_CULLING
                            | PreprocessPipelineKey::EARLY_PHASE
                    ) {
                        "early GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                        "late GPU occlusion culling"
                    } else if key.contains(PreprocessPipelineKey::FRUSTUM_CULLING) {
                        "GPU frustum culling"
                    } else {
                        "direct"
                    }
                )
                .into(),
            ),
            layout: vec![self.bind_group_layout.clone()],
            push_constant_ranges: if key.contains(PreprocessPipelineKey::OCCLUSION_CULLING) {
                vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..4,
                }]
            } else {
                vec![]
            },
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

impl FromWorld for PreprocessPipelines {
    fn from_world(world: &mut World) -> Self {
        // GPU culling bind group parameters are a superset of those in the CPU
        // culling (direct) shader.
        let direct_bind_group_layout_entries = preprocess_direct_bind_group_layout_entries();
        let gpu_frustum_culling_bind_group_layout_entries = gpu_culling_bind_group_layout_entries();
        let gpu_early_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries().extend_with_indices(((
                11,
                storage_buffer::<PreprocessWorkItem>(/*has_dynamic_offset=*/ false),
            ),));
        let gpu_late_occlusion_culling_bind_group_layout_entries =
            gpu_occlusion_culling_bind_group_layout_entries();

        let reset_indirect_batch_sets_bind_group_layout_entries =
            DynamicBindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (storage_buffer::<IndirectBatchSet>(false),),
            );

        // Indexed and non-indexed bind group parameters share all the bind
        // group layout entries except the final one. Binding 4 is reserved for
        // the indirect parameters data buffer (indexed or non-indexed).
        let build_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_with_indices(((4, storage_buffer::<IndirectParametersIndexed>(false)),));
        let build_non_indexed_indirect_params_bind_group_layout_entries =
            build_indirect_params_bind_group_layout_entries()
                .extend_with_indices(((4, storage_buffer::<IndirectParametersNonIndexed>(false)),));

        // Create the bind group layouts.
        let direct_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms direct bind group layout",
            &direct_bind_group_layout_entries,
        );
        let gpu_frustum_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU frustum culling bind group layout",
            &gpu_frustum_culling_bind_group_layout_entries,
        );
        let gpu_early_occlusion_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU early occlusion culling bind group layout",
            &gpu_early_occlusion_culling_bind_group_layout_entries,
        );
        let gpu_late_occlusion_culling_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build mesh uniforms GPU late occlusion culling bind group layout",
            &gpu_late_occlusion_culling_bind_group_layout_entries,
        );
        let reset_indirect_batch_sets_bind_group_layout = BindGroupLayoutDescriptor::new(
            "reset indirect batch sets bind group layout",
            &reset_indirect_batch_sets_bind_group_layout_entries,
        );
        let build_indexed_indirect_params_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build indexed indirect parameters bind group layout",
            &build_indexed_indirect_params_bind_group_layout_entries,
        );
        let build_non_indexed_indirect_params_bind_group_layout = BindGroupLayoutDescriptor::new(
            "build non-indexed indirect parameters bind group layout",
            &build_non_indexed_indirect_params_bind_group_layout_entries,
        );

        // Stage 2 material expansion bind group layouts
        let material_expand_direct_bind_group_layout_entries =
            material_expand_direct_bind_group_layout_entries();
        let material_expand_indirect_bind_group_layout_entries =
            material_expand_indirect_bind_group_layout_entries();
        let material_expand_late_bind_group_layout_entries =
            material_expand_late_bind_group_layout_entries();
        let material_expand_direct_bind_group_layout = BindGroupLayoutDescriptor::new(
            "material expand direct bind group layout",
            &material_expand_direct_bind_group_layout_entries,
        );
        let material_expand_indirect_bind_group_layout = BindGroupLayoutDescriptor::new(
            "material expand indirect bind group layout",
            &material_expand_indirect_bind_group_layout_entries,
        );
        let material_expand_late_bind_group_layout = BindGroupLayoutDescriptor::new(
            "material expand late bind group layout",
            &material_expand_late_bind_group_layout_entries,
        );

        let preprocess_shader = load_embedded_asset!(world, "mesh_preprocess.wgsl");
        let material_expand_shader = load_embedded_asset!(world, "mesh_material_expand.wgsl");
        let reset_indirect_batch_sets_shader =
            load_embedded_asset!(world, "reset_indirect_batch_sets.wgsl");
        let build_indirect_params_shader =
            load_embedded_asset!(world, "build_indirect_params.wgsl");

        let preprocess_phase_pipelines = PreprocessPhasePipelines {
            reset_indirect_batch_sets: ResetIndirectBatchSetsPipeline {
                bind_group_layout: reset_indirect_batch_sets_bind_group_layout.clone(),
                shader: reset_indirect_batch_sets_shader,
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                shader: build_indirect_params_shader.clone(),
                pipeline_id: None,
            },
            gpu_occlusion_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    shader: build_indirect_params_shader.clone(),
                    pipeline_id: None,
                },
        };

        PreprocessPipelines {
            direct_preprocess: PreprocessPipeline {
                bind_group_layout: direct_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_frustum_culling_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            early_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_early_occlusion_culling_bind_group_layout,
                shader: preprocess_shader.clone(),
                pipeline_id: None,
            },
            late_gpu_occlusion_culling_preprocess: PreprocessPipeline {
                bind_group_layout: gpu_late_occlusion_culling_bind_group_layout,
                shader: preprocess_shader,
                pipeline_id: None,
            },
            gpu_frustum_culling_build_indexed_indirect_params: BuildIndirectParametersPipeline {
                bind_group_layout: build_indexed_indirect_params_bind_group_layout.clone(),
                shader: build_indirect_params_shader.clone(),
                pipeline_id: None,
            },
            gpu_frustum_culling_build_non_indexed_indirect_params:
                BuildIndirectParametersPipeline {
                    bind_group_layout: build_non_indexed_indirect_params_bind_group_layout.clone(),
                    shader: build_indirect_params_shader,
                    pipeline_id: None,
                },
            early_phase: preprocess_phase_pipelines.clone(),
            late_phase: preprocess_phase_pipelines.clone(),
            main_phase: preprocess_phase_pipelines.clone(),

            // Stage 2 material expansion pipelines
            material_expand_direct: MaterialExpandPipeline {
                bind_group_layout: material_expand_direct_bind_group_layout,
                shader: material_expand_shader.clone(),
                pipeline_id: None,
            },
            material_expand_indirect_indexed: MaterialExpandPipeline {
                bind_group_layout: material_expand_indirect_bind_group_layout.clone(),
                shader: material_expand_shader.clone(),
                pipeline_id: None,
            },
            material_expand_indirect_non_indexed: MaterialExpandPipeline {
                bind_group_layout: material_expand_indirect_bind_group_layout,
                shader: material_expand_shader.clone(),
                pipeline_id: None,
            },
            material_expand_late_indexed: MaterialExpandPipeline {
                bind_group_layout: material_expand_late_bind_group_layout.clone(),
                shader: material_expand_shader.clone(),
                pipeline_id: None,
            },
            material_expand_late_non_indexed: MaterialExpandPipeline {
                bind_group_layout: material_expand_late_bind_group_layout,
                shader: material_expand_shader,
                pipeline_id: None,
            },
        }
    }
}

fn preprocess_direct_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            // `view`
            (
                0,
                uniform_buffer::<ViewUniform>(/* has_dynamic_offset= */ true),
            ),
            // `current_input`
            (3, storage_buffer_read_only::<MeshInputUniform>(false)),
            // `previous_input`
            (4, storage_buffer_read_only::<MeshInputUniform>(false)),
            // `indices`
            (5, storage_buffer_read_only::<PreprocessWorkItem>(false)),
            // `output`
            (6, storage_buffer::<MeshUniform>(false)),
            // `visibility_flags` - written by Stage 1 for Stage 2 material expansion
            (13, storage_buffer::<u32>(false)),
        ),
    )
}

// Returns the bind group layout entries shared between all invocations
// of the indirect parameters building shader.
fn build_indirect_params_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            (0, storage_buffer_read_only::<MeshInputUniform>(false)),
            (
                1,
                storage_buffer_read_only::<IndirectParametersCpuMetadata>(false),
            ),
            (
                2,
                storage_buffer_read_only::<IndirectParametersGpuMetadata>(false),
            ),
            (3, storage_buffer::<IndirectBatchSet>(false)),
            // binding 4 is the indirect parameters data buffer (indexed or non-indexed)
            // binding 5 is the submesh buffer
            (5, storage_buffer_read_only::<SubMeshDescriptor>(false)),
            // binding 6 is the material indirect geometry buffer (for MATERIAL_EXPANSION)
            (6, storage_buffer_read_only::<MaterialIndirectGeometry>(false)),
        ),
    )
}

/// A system that specializes the `mesh_preprocess.wgsl` and
/// `build_indirect_params.wgsl` pipelines if necessary.
fn gpu_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    // GPU culling bind group parameters are a superset of those in the CPU
    // culling (direct) shader.
    preprocess_direct_bind_group_layout_entries().extend_with_indices((
        // `indirect_parameters_cpu_metadata`
        (
            7,
            storage_buffer_read_only::<IndirectParametersCpuMetadata>(
                /* has_dynamic_offset= */ false,
            ),
        ),
        // `indirect_parameters_gpu_metadata`
        (
            8,
            storage_buffer::<IndirectParametersGpuMetadata>(/* has_dynamic_offset= */ false),
        ),
        // `mesh_culling_data`
        (
            9,
            storage_buffer_read_only::<MeshCullingData>(/* has_dynamic_offset= */ false),
        ),
    ))
}

fn gpu_occlusion_culling_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    gpu_culling_bind_group_layout_entries().extend_with_indices((
        (
            2,
            uniform_buffer::<PreviousViewData>(/*has_dynamic_offset=*/ false),
        ),
        (
            10,
            texture_2d(TextureSampleType::Float { filterable: true }),
        ),
        (
            12,
            storage_buffer::<LatePreprocessWorkItemIndirectParameters>(
                /*has_dynamic_offset=*/ false,
            ),
        ),
    ))
}

/// Bind group layout entries for Stage 2 material expansion (direct mode).
///
/// Dispatches directly over MaterialEntry buffer after visibility is resolved.
/// Each thread checks visibility and writes draw_data.
fn material_expand_direct_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::COMPUTE,
        (
            // `visibility_flags` - per-mesh visibility from Stage 1
            (0, storage_buffer_read_only::<u32>(false)),
            // `material_entries` - flat list of material draws
            (1, storage_buffer_read_only::<MaterialEntry>(false)),
            // `draw_data` - per-draw output with material slot
            (2, storage_buffer::<DrawData>(false)),
        ),
    )
}

/// Bind group layout entries for Stage 2 material expansion (indirect mode).
///
/// Adds geometry output, metadata access, and mesh/submesh buffers for GPU geometry lookup.
fn material_expand_indirect_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    material_expand_direct_bind_group_layout_entries().extend_with_indices((
        // `material_indirect_geometry` - per-batch geometry slices
        (3, storage_buffer::<MaterialIndirectGeometry>(false)),
        // `indirect_parameters_gpu_metadata` - read/write for draw_count and mesh_index
        (4, storage_buffer::<IndirectParametersGpuMetadata>(false)),
        // `indirect_parameters_cpu_metadata` - read-only for base_output_index
        (5, storage_buffer_read_only::<IndirectParametersCpuMetadata>(false)),
        // `current_input` - MeshInput buffer for geometry lookup
        (6, storage_buffer_read_only::<MeshInputUniform>(false)),
        // `submesh_buffer` - SubMeshDescriptor buffer for geometry lookup
        (7, storage_buffer_read_only::<SubMeshDescriptor>(false)),
    ))
}

/// Bind group layout entries for Stage 2 material expansion (late phase).
///
/// Note: With the new approach, we dispatch once over all MaterialEntries after
/// both early and late visibility are resolved. This layout is kept for compatibility
/// but may be consolidated with indirect in the future.
fn material_expand_late_bind_group_layout_entries() -> DynamicBindGroupLayoutEntries {
    // Same as indirect - late phase uses same bindings now
    material_expand_indirect_bind_group_layout_entries()
}

/// A system that specializes the `mesh_preprocess.wgsl` pipelines if necessary.
pub fn prepare_preprocess_pipelines(
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    mut specialized_preprocess_pipelines: ResMut<SpecializedComputePipelines<PreprocessPipeline>>,
    mut specialized_reset_indirect_batch_sets_pipelines: ResMut<
        SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    >,
    mut specialized_build_indirect_parameters_pipelines: ResMut<
        SpecializedComputePipelines<BuildIndirectParametersPipeline>,
    >,
    mut specialized_material_expand_pipelines: ResMut<
        SpecializedComputePipelines<MaterialExpandPipeline>,
    >,
    preprocess_pipelines: ResMut<PreprocessPipelines>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    let preprocess_pipelines = preprocess_pipelines.into_inner();

    preprocess_pipelines.direct_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::empty(),
    );
    preprocess_pipelines.gpu_frustum_culling_preprocess.prepare(
        &pipeline_cache,
        &mut specialized_preprocess_pipelines,
        PreprocessPipelineKey::FRUSTUM_CULLING,
    );

    // Prepare Stage 2 material expansion pipelines
    preprocess_pipelines.material_expand_direct.prepare(
        &pipeline_cache,
        &mut specialized_material_expand_pipelines,
        MaterialExpandPipelineKey::empty(),
    );
    preprocess_pipelines
        .material_expand_indirect_indexed
        .prepare(
            &pipeline_cache,
            &mut specialized_material_expand_pipelines,
            MaterialExpandPipelineKey::INDIRECT | MaterialExpandPipelineKey::INDEXED,
        );
    preprocess_pipelines
        .material_expand_indirect_non_indexed
        .prepare(
            &pipeline_cache,
            &mut specialized_material_expand_pipelines,
            MaterialExpandPipelineKey::INDIRECT,
        );

    if gpu_preprocessing_support.is_culling_supported() {
        preprocess_pipelines
            .early_gpu_occlusion_culling_preprocess
            .prepare(
                &pipeline_cache,
                &mut specialized_preprocess_pipelines,
                PreprocessPipelineKey::FRUSTUM_CULLING
                    | PreprocessPipelineKey::OCCLUSION_CULLING
                    | PreprocessPipelineKey::EARLY_PHASE,
            );
        preprocess_pipelines
            .late_gpu_occlusion_culling_preprocess
            .prepare(
                &pipeline_cache,
                &mut specialized_preprocess_pipelines,
                PreprocessPipelineKey::FRUSTUM_CULLING | PreprocessPipelineKey::OCCLUSION_CULLING,
            );
        // Prepare late material expansion pipelines for occlusion culling.
        preprocess_pipelines
            .material_expand_late_indexed
            .prepare(
                &pipeline_cache,
                &mut specialized_material_expand_pipelines,
                MaterialExpandPipelineKey::INDIRECT
                    | MaterialExpandPipelineKey::LATE_PHASE
                    | MaterialExpandPipelineKey::INDEXED,
            );
        preprocess_pipelines
            .material_expand_late_non_indexed
            .prepare(
                &pipeline_cache,
                &mut specialized_material_expand_pipelines,
                MaterialExpandPipelineKey::INDIRECT | MaterialExpandPipelineKey::LATE_PHASE,
            );
    }

    let mut build_indirect_parameters_pipeline_key = BuildIndirectParametersPipelineKey::empty();

    // If the GPU and driver support `multi_draw_indirect_count`, tell the
    // shader that.
    if render_device
        .wgpu_device()
        .features()
        .contains(WgpuFeatures::MULTI_DRAW_INDIRECT_COUNT)
    {
        build_indirect_parameters_pipeline_key
            .insert(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED);
    }

    // Stage 2 material expansion is active, so use draw_count for instance counts.
    build_indirect_parameters_pipeline_key
        .insert(BuildIndirectParametersPipelineKey::MATERIAL_EXPANSION);

    preprocess_pipelines
        .gpu_frustum_culling_build_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key | BuildIndirectParametersPipelineKey::INDEXED,
        );
    preprocess_pipelines
        .gpu_frustum_culling_build_non_indexed_indirect_params
        .prepare(
            &pipeline_cache,
            &mut specialized_build_indirect_parameters_pipelines,
            build_indirect_parameters_pipeline_key,
        );

    if !gpu_preprocessing_support.is_culling_supported() {
        return;
    }

    for (preprocess_phase_pipelines, build_indirect_parameters_phase_pipeline_key) in [
        (
            &mut preprocess_pipelines.early_phase,
            BuildIndirectParametersPipelineKey::EARLY_PHASE,
        ),
        (
            &mut preprocess_pipelines.late_phase,
            BuildIndirectParametersPipelineKey::LATE_PHASE,
        ),
        (
            &mut preprocess_pipelines.main_phase,
            BuildIndirectParametersPipelineKey::MAIN_PHASE,
        ),
    ] {
        preprocess_phase_pipelines
            .reset_indirect_batch_sets
            .prepare(
                &pipeline_cache,
                &mut specialized_reset_indirect_batch_sets_pipelines,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::INDEXED
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
        preprocess_phase_pipelines
            .gpu_occlusion_culling_build_non_indexed_indirect_params
            .prepare(
                &pipeline_cache,
                &mut specialized_build_indirect_parameters_pipelines,
                build_indirect_parameters_pipeline_key
                    | build_indirect_parameters_phase_pipeline_key
                    | BuildIndirectParametersPipelineKey::OCCLUSION_CULLING,
            );
    }
}

impl PreprocessPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<PreprocessPipeline>,
        key: PreprocessPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let preprocess_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(preprocess_pipeline_id);
    }
}

impl SpecializedComputePipeline for ResetIndirectBatchSetsPipeline {
    type Key = ();

    fn specialize(&self, _: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("reset indirect batch sets".into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            ..default()
        }
    }
}

impl SpecializedComputePipeline for BuildIndirectParametersPipeline {
    type Key = BuildIndirectParametersPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
            shader_defs.push("INDEXED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MULTI_DRAW_INDIRECT_COUNT_SUPPORTED) {
            shader_defs.push("MULTI_DRAW_INDIRECT_COUNT_SUPPORTED".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
            shader_defs.push("OCCLUSION_CULLING".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
            shader_defs.push("EARLY_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
            shader_defs.push("LATE_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MAIN_PHASE) {
            shader_defs.push("MAIN_PHASE".into());
        }
        if key.contains(BuildIndirectParametersPipelineKey::MATERIAL_EXPANSION) {
            shader_defs.push("MATERIAL_EXPANSION".into());
        }

        let label = format!(
            "{} build {}indexed indirect parameters",
            if !key.contains(BuildIndirectParametersPipelineKey::OCCLUSION_CULLING) {
                "frustum culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::EARLY_PHASE) {
                "early occlusion culling"
            } else if key.contains(BuildIndirectParametersPipelineKey::LATE_PHASE) {
                "late occlusion culling"
            } else {
                "main occlusion culling"
            },
            if key.contains(BuildIndirectParametersPipelineKey::INDEXED) {
                ""
            } else {
                "non-"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

impl SpecializedComputePipeline for MaterialExpandPipeline {
    type Key = MaterialExpandPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.contains(MaterialExpandPipelineKey::INDIRECT) {
            shader_defs.push("INDIRECT".into());
            // Required for atomic<u32> fields in IndirectParametersGpuMetadata
            shader_defs.push("WRITE_INDIRECT_PARAMETERS_METADATA".into());
        }
        if key.contains(MaterialExpandPipelineKey::LATE_PHASE) {
            shader_defs.push("LATE_PHASE".into());
        }
        if key.contains(MaterialExpandPipelineKey::INDEXED) {
            shader_defs.push("INDEXED".into());
        }

        let label = format!(
            "material expansion ({} {} {})",
            if key.contains(MaterialExpandPipelineKey::INDIRECT) {
                "indirect"
            } else {
                "direct"
            },
            if key.contains(MaterialExpandPipelineKey::INDEXED) {
                "indexed"
            } else {
                "non-indexed"
            },
            if key.contains(MaterialExpandPipelineKey::LATE_PHASE) {
                "late"
            } else {
                "early"
            }
        );

        ComputePipelineDescriptor {
            label: Some(label.into()),
            layout: vec![self.bind_group_layout.clone()],
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

impl ResetIndirectBatchSetsPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let reset_indirect_batch_sets_pipeline_id = pipelines.specialize(pipeline_cache, self, ());
        self.pipeline_id = Some(reset_indirect_batch_sets_pipeline_id);
    }
}

impl BuildIndirectParametersPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<BuildIndirectParametersPipeline>,
        key: BuildIndirectParametersPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let build_indirect_parameters_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(build_indirect_parameters_pipeline_id);
    }
}

impl MaterialExpandPipeline {
    fn prepare(
        &mut self,
        pipeline_cache: &PipelineCache,
        pipelines: &mut SpecializedComputePipelines<MaterialExpandPipeline>,
        key: MaterialExpandPipelineKey,
    ) {
        if self.pipeline_id.is_some() {
            return;
        }

        let material_expand_pipeline_id = pipelines.specialize(pipeline_cache, self, key);
        self.pipeline_id = Some(material_expand_pipeline_id);
    }
}

/// A system that attaches the mesh uniform buffers to the bind groups for the
/// variants of the mesh preprocessing compute shader.
#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of arguments"
)]
pub fn prepare_preprocess_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView)>,
    view_depth_pyramids: Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    mesh_culling_data_buffer: Res<MeshCullingDataBuffer>,
    submesh_buffer: Res<SubMeshBuffer>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    pipelines: Res<PreprocessPipelines>,
    material_expansion_buffers: Res<MaterialExpansionBuffers>,
) {
    // Grab the `BatchedInstanceBuffers`.
    let BatchedInstanceBuffers {
        current_input_buffer: current_input_buffer_vec,
        previous_input_buffer: previous_input_buffer_vec,
        phase_instance_buffers,
    } = batched_instance_buffers.into_inner();

    let (Some(current_input_buffer), Some(previous_input_buffer)) = (
        current_input_buffer_vec.buffer().buffer(),
        previous_input_buffer_vec.buffer().buffer(),
    ) else {
        return;
    };

    // Record whether we have any meshes that are to be drawn indirectly. If we
    // don't, then we can skip building indirect parameters.
    let mut any_indirect = false;

    // Loop over each view.
    for (view_entity, view) in &views {
        let mut bind_groups = TypeIdMap::default();

        // Loop over each phase.
        for (phase_type_id, phase_instance_buffers) in phase_instance_buffers {
            let UntypedPhaseBatchedInstanceBuffers {
                data_buffer: ref data_buffer_vec,
                ref work_item_buffers,
                ref late_indexed_indirect_parameters_buffer,
                ref late_non_indexed_indirect_parameters_buffer,
                ..  // material_work_item_info not needed here
            } = *phase_instance_buffers;

            let Some(data_buffer) = data_buffer_vec.buffer() else {
                continue;
            };

            // Grab the indirect parameters buffers for this phase.
            let Some(phase_indirect_parameters_buffers) =
                indirect_parameters_buffers.get(phase_type_id)
            else {
                continue;
            };

            let Some(work_item_buffers) = work_item_buffers.get(&view.retained_view_entity) else {
                continue;
            };

            // Create the `PreprocessBindGroupBuilder`.
            let preprocess_bind_group_builder = PreprocessBindGroupBuilder {
                view: view_entity,
                late_indexed_indirect_parameters_buffer,
                late_non_indexed_indirect_parameters_buffer,
                render_device: &render_device,
                pipeline_cache: &pipeline_cache,
                phase_indirect_parameters_buffers,
                mesh_culling_data_buffer: &mesh_culling_data_buffer,
                view_uniforms: &view_uniforms,
                previous_view_uniforms: &previous_view_uniforms,
                pipelines: &pipelines,
                current_input_buffer,
                previous_input_buffer,
                data_buffer,
                visibility_flags_buffer: material_expansion_buffers.visibility_flags.buffer(),
            };

            // Depending on the type of work items we have, construct the
            // appropriate bind groups.
            let (was_indirect, bind_group) = match *work_item_buffers {
                PreprocessWorkItemBuffers::Direct(ref work_item_buffer) => (
                    false,
                    preprocess_bind_group_builder
                        .create_direct_preprocess_bind_groups(work_item_buffer),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: Some(ref gpu_occlusion_culling_work_item_buffers),
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_occlusion_culling_preprocess_bind_groups(
                            &view_depth_pyramids,
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                            gpu_occlusion_culling_work_item_buffers,
                        ),
                ),

                PreprocessWorkItemBuffers::Indirect {
                    indexed: ref indexed_work_item_buffer,
                    non_indexed: ref non_indexed_work_item_buffer,
                    gpu_occlusion_culling: None,
                } => (
                    true,
                    preprocess_bind_group_builder
                        .create_indirect_frustum_culling_preprocess_bind_groups(
                            indexed_work_item_buffer,
                            non_indexed_work_item_buffer,
                        ),
                ),
            };

            // Write that bind group in.
            if let Some(bind_group) = bind_group {
                any_indirect = any_indirect || was_indirect;
                bind_groups.insert(*phase_type_id, bind_group);
            }
        }

        // Save the bind groups.
        commands
            .entity(view_entity)
            .insert(PreprocessBindGroups(bind_groups));
    }

    // Now, if there were any indirect draw commands, create the bind groups for
    // the indirect parameters building shader.
    if any_indirect {
        create_build_indirect_parameters_bind_groups(
            &mut commands,
            &render_device,
            &pipeline_cache,
            &pipelines,
            current_input_buffer,
            &indirect_parameters_buffers,
            phase_instance_buffers,
            &submesh_buffer,
            &material_expansion_buffers,
        );
    }
}

/// A system that creates bind groups for the Stage 2 material expansion shader.
///
/// This system creates per-view, per-phase bind groups for `mesh_material_expand.wgsl`.
/// The GPU-driven approach reads PreprocessWorkItem directly and looks up
/// material slots from a GPU-visible table.
///
/// This follows the same pattern as [`prepare_preprocess_bind_groups`]: it iterates
/// over views and phases, creating bind groups specific to each view's work items.
pub fn prepare_material_expand_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedView)>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    batched_instance_buffers: Res<BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    material_expansion_buffers: Res<MaterialExpansionBuffers>,
    indirect_parameters_buffers: Res<IndirectParametersBuffers>,
    submesh_buffer: Res<SubMeshBuffer>,
    pipelines: Res<PreprocessPipelines>,
) {
    // Get global buffers
    let Some(visibility_flags_buffer) = material_expansion_buffers.visibility_flags.buffer() else {
        return;
    };

    // Get current_input buffer for indirect mode geometry lookup
    let current_input_buffer = batched_instance_buffers
        .current_input_buffer
        .buffer()
        .buffer();
    let submesh_buffer_gpu = submesh_buffer.buffer();

    // Get bind group layouts (indexed and non-indexed share the same layout)
    let direct_bind_group_layout = pipeline_cache
        .get_bind_group_layout(&pipelines.material_expand_direct.bind_group_layout);
    let indirect_bind_group_layout = pipeline_cache
        .get_bind_group_layout(&pipelines.material_expand_indirect_indexed.bind_group_layout);
    let _late_bind_group_layout = pipeline_cache
        .get_bind_group_layout(&pipelines.material_expand_late_indexed.bind_group_layout);

    // Loop over each view
    for (view_entity, view) in &views {
        let mut bind_groups = TypeIdMap::default();

        // Loop over each phase
        for (&phase_type_id, phase_buffers) in &batched_instance_buffers.phase_instance_buffers {
            // Get per-(phase, view) buffers (material entries, draw_data, material_indirect_geometry)
            let phase_view_key = (phase_type_id, view.retained_view_entity);
            let phase_expansion = material_expansion_buffers.per_phase_view.get(&phase_view_key);

            // Get per-phase draw_data buffer, falling back to legacy global buffer
            let Some(draw_data_buffer) = phase_expansion
                .and_then(|p| p.draw_data.buffer())
                .or_else(|| material_expansion_buffers.draw_data.buffer())
            else {
                continue;
            };

            // Get per-phase material entries buffers
            let material_entries_indexed_buffer =
                phase_expansion.and_then(|p| p.material_entries_indexed.buffer());
            let material_entries_non_indexed_buffer =
                phase_expansion.and_then(|p| p.material_entries_non_indexed.buffer());

            // Get per-phase entry counts (default to 0 if no phase buffers)
            let indexed_entry_count = phase_expansion
                .map(|p| p.material_entries_indexed.len() as u32)
                .unwrap_or(0);
            let non_indexed_entry_count = phase_expansion
                .map(|p| p.material_entries_non_indexed.len() as u32)
                .unwrap_or(0);

            // Get the work item buffers for this view
            let Some(work_item_buffers) = phase_buffers
                .work_item_buffers
                .get(&view.retained_view_entity)
            else {
                continue;
            };

            // Get indirect parameters buffers for this phase
            let indirect_buffers = indirect_parameters_buffers.get(&phase_type_id);

            // Create bind groups based on the type of work items
            let phase_bind_groups = match work_item_buffers {
                PreprocessWorkItemBuffers::Direct(_work_item_buffer) => {
                    // Direct mode - dispatch over MaterialEntries
                    // Create separate bind groups for indexed and non-indexed meshes

                    // Indexed bind group
                    let indexed_bind_group = material_entries_indexed_buffer.map(|entries_buffer| {
                        render_device.create_bind_group(
                            "material_expand_direct_indexed_bind_group",
                            &direct_bind_group_layout,
                            &BindGroupEntries::with_indices((
                                (0, visibility_flags_buffer.as_entire_binding()),
                                (1, entries_buffer.as_entire_binding()),
                                (2, draw_data_buffer.as_entire_binding()),
                            )),
                        )
                    });

                    // Non-indexed bind group
                    let non_indexed_bind_group =
                        material_entries_non_indexed_buffer.map(|entries_buffer| {
                            render_device.create_bind_group(
                                "material_expand_direct_non_indexed_bind_group",
                                &direct_bind_group_layout,
                                &BindGroupEntries::with_indices((
                                    (0, visibility_flags_buffer.as_entire_binding()),
                                    (1, entries_buffer.as_entire_binding()),
                                    (2, draw_data_buffer.as_entire_binding()),
                                )),
                            )
                        });

                    PhaseMaterialExpandBindGroups::Direct {
                        indexed: indexed_bind_group,
                        non_indexed: non_indexed_bind_group,
                        indexed_work_item_count: indexed_entry_count,
                        non_indexed_work_item_count: non_indexed_entry_count,
                    }
                }

                PreprocessWorkItemBuffers::Indirect {
                    indexed: _,
                    non_indexed: _,
                    gpu_occlusion_culling: None,
                } => {
                    // Indirect frustum culling - dispatch over MaterialEntries
                    let Some(phase_indirect_buffers) = indirect_buffers else {
                        continue;
                    };

                    // Get separate material_indirect_geometry buffers for indexed and non-indexed
                    let phase_expansion = material_expansion_buffers.per_phase_view.get(&phase_view_key);
                    let material_indirect_geometry_indexed_buffer =
                        phase_expansion.and_then(|p| p.material_indirect_geometry_indexed.buffer());
                    let material_indirect_geometry_non_indexed_buffer =
                        phase_expansion.and_then(|p| p.material_indirect_geometry_non_indexed.buffer());

                    let indexed_cpu_metadata = phase_indirect_buffers.indexed.cpu_metadata_buffer();
                    let indexed_gpu_metadata = phase_indirect_buffers.indexed.gpu_metadata_buffer();
                    let non_indexed_cpu_metadata =
                        phase_indirect_buffers.non_indexed.cpu_metadata_buffer();
                    let non_indexed_gpu_metadata =
                        phase_indirect_buffers.non_indexed.gpu_metadata_buffer();

                    // Indexed bind group (includes current_input and submesh_buffer for geometry lookup)
                    let indexed_bind_group = material_entries_indexed_buffer
                        .zip(material_indirect_geometry_indexed_buffer)
                        .zip(indexed_gpu_metadata)
                        .zip(indexed_cpu_metadata)
                        .zip(current_input_buffer)
                        .zip(submesh_buffer_gpu)
                        .map(
                            |(((((entries, mat_geo), gpu_meta), cpu_meta), input), submesh)| {
                                render_device.create_bind_group(
                                    "material_expand_indirect_indexed_bind_group",
                                    &indirect_bind_group_layout,
                                    &BindGroupEntries::with_indices((
                                        (0, visibility_flags_buffer.as_entire_binding()),
                                        (1, entries.as_entire_binding()),
                                        (2, draw_data_buffer.as_entire_binding()),
                                        (3, mat_geo.as_entire_binding()),
                                        (4, gpu_meta.as_entire_binding()),
                                        (5, cpu_meta.as_entire_binding()),
                                        (6, input.as_entire_binding()),
                                        (7, submesh.as_entire_binding()),
                                    )),
                                )
                            },
                        );

                    // Non-indexed bind group (includes current_input and submesh_buffer for geometry lookup)
                    let non_indexed_bind_group = material_entries_non_indexed_buffer
                        .zip(material_indirect_geometry_non_indexed_buffer)
                        .zip(non_indexed_gpu_metadata)
                        .zip(non_indexed_cpu_metadata)
                        .zip(current_input_buffer)
                        .zip(submesh_buffer_gpu)
                        .map(
                            |(((((entries, mat_geo), gpu_meta), cpu_meta), input), submesh)| {
                                render_device.create_bind_group(
                                    "material_expand_indirect_non_indexed_bind_group",
                                    &indirect_bind_group_layout,
                                    &BindGroupEntries::with_indices((
                                        (0, visibility_flags_buffer.as_entire_binding()),
                                        (1, entries.as_entire_binding()),
                                        (2, draw_data_buffer.as_entire_binding()),
                                        (3, mat_geo.as_entire_binding()),
                                        (4, gpu_meta.as_entire_binding()),
                                        (5, cpu_meta.as_entire_binding()),
                                        (6, input.as_entire_binding()),
                                        (7, submesh.as_entire_binding()),
                                    )),
                                )
                            },
                        );

                    PhaseMaterialExpandBindGroups::IndirectFrustumCulling {
                        indexed: indexed_bind_group,
                        non_indexed: non_indexed_bind_group,
                        indexed_work_item_count: indexed_entry_count,
                        non_indexed_work_item_count: non_indexed_entry_count,
                    }
                }

                PreprocessWorkItemBuffers::Indirect {
                    indexed: _,
                    non_indexed: _,
                    gpu_occlusion_culling: Some(occlusion_culling_buffers),
                } => {
                    // Indirect occlusion culling - dispatch over MaterialEntries after visibility is resolved
                    // With the new design, we dispatch once after both early and late visibility are set.
                    let Some(phase_indirect_buffers) = indirect_buffers else {
                        continue;
                    };

                    // Get separate material_indirect_geometry buffers for indexed and non-indexed
                    let phase_expansion = material_expansion_buffers.per_phase_view.get(&phase_view_key);
                    let material_indirect_geometry_indexed_buffer =
                        phase_expansion.and_then(|p| p.material_indirect_geometry_indexed.buffer());
                    let material_indirect_geometry_non_indexed_buffer =
                        phase_expansion.and_then(|p| p.material_indirect_geometry_non_indexed.buffer());

                    let indexed_cpu_metadata = phase_indirect_buffers.indexed.cpu_metadata_buffer();
                    let indexed_gpu_metadata = phase_indirect_buffers.indexed.gpu_metadata_buffer();
                    let non_indexed_cpu_metadata =
                        phase_indirect_buffers.non_indexed.cpu_metadata_buffer();
                    let non_indexed_gpu_metadata =
                        phase_indirect_buffers.non_indexed.gpu_metadata_buffer();

                    // Indexed bind group (includes current_input and submesh_buffer for geometry lookup)
                    let indexed_bind_group = material_entries_indexed_buffer
                        .zip(material_indirect_geometry_indexed_buffer)
                        .zip(indexed_gpu_metadata)
                        .zip(indexed_cpu_metadata)
                        .zip(current_input_buffer)
                        .zip(submesh_buffer_gpu)
                        .map(
                            |(((((entries, mat_geo), gpu_meta), cpu_meta), input), submesh)| {
                                render_device.create_bind_group(
                                    "material_expand_indexed_bind_group",
                                    &indirect_bind_group_layout,
                                    &BindGroupEntries::with_indices((
                                        (0, visibility_flags_buffer.as_entire_binding()),
                                        (1, entries.as_entire_binding()),
                                        (2, draw_data_buffer.as_entire_binding()),
                                        (3, mat_geo.as_entire_binding()),
                                        (4, gpu_meta.as_entire_binding()),
                                        (5, cpu_meta.as_entire_binding()),
                                        (6, input.as_entire_binding()),
                                        (7, submesh.as_entire_binding()),
                                    )),
                                )
                            },
                        );

                    // Non-indexed bind group (includes current_input and submesh_buffer for geometry lookup)
                    let non_indexed_bind_group = material_entries_non_indexed_buffer
                        .zip(material_indirect_geometry_non_indexed_buffer)
                        .zip(non_indexed_gpu_metadata)
                        .zip(non_indexed_cpu_metadata)
                        .zip(current_input_buffer)
                        .zip(submesh_buffer_gpu)
                        .map(
                            |(((((entries, mat_geo), gpu_meta), cpu_meta), input), submesh)| {
                                render_device.create_bind_group(
                                    "material_expand_non_indexed_bind_group",
                                    &indirect_bind_group_layout,
                                    &BindGroupEntries::with_indices((
                                        (0, visibility_flags_buffer.as_entire_binding()),
                                        (1, entries.as_entire_binding()),
                                        (2, draw_data_buffer.as_entire_binding()),
                                        (3, mat_geo.as_entire_binding()),
                                        (4, gpu_meta.as_entire_binding()),
                                        (5, cpu_meta.as_entire_binding()),
                                        (6, input.as_entire_binding()),
                                        (7, submesh.as_entire_binding()),
                                    )),
                                )
                            },
                        );

                    // With the new design, we use the same bind groups for early and late
                    // since we dispatch once after all visibility is resolved
                    PhaseMaterialExpandBindGroups::IndirectOcclusionCulling {
                        early_indexed: indexed_bind_group.clone(),
                        early_non_indexed: non_indexed_bind_group.clone(),
                        late_indexed: indexed_bind_group,
                        late_non_indexed: non_indexed_bind_group,
                        early_indexed_work_item_count: indexed_entry_count,
                        early_non_indexed_work_item_count: non_indexed_entry_count,
                        late_indirect_parameters_indexed_offset: occlusion_culling_buffers
                            .late_indirect_parameters_indexed_offset,
                        late_indirect_parameters_non_indexed_offset: occlusion_culling_buffers
                            .late_indirect_parameters_non_indexed_offset,
                    }
                }
            };

            bind_groups.insert(phase_type_id, phase_bind_groups);
        }

        // Insert the bind groups as a component on the view entity
        if !bind_groups.is_empty() {
            commands
                .entity(view_entity)
                .insert(MaterialExpandBindGroups(bind_groups));
        }
    }
}

/// A temporary structure that stores all the information needed to construct
/// bind groups for the mesh preprocessing shader.
struct PreprocessBindGroupBuilder<'a> {
    /// The render-world entity corresponding to the current view.
    view: Entity,
    /// The indirect compute dispatch parameters buffer for indexed meshes in
    /// the late prepass.
    late_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The indirect compute dispatch parameters buffer for non-indexed meshes
    /// in the late prepass.
    late_non_indexed_indirect_parameters_buffer:
        &'a RawBufferVec<LatePreprocessWorkItemIndirectParameters>,
    /// The device.
    render_device: &'a RenderDevice,
    /// The pipeline cache
    pipeline_cache: &'a PipelineCache,
    /// The buffers that store indirect draw parameters.
    phase_indirect_parameters_buffers: &'a UntypedPhaseIndirectParametersBuffers,
    /// The GPU buffer that stores the information needed to cull each mesh.
    mesh_culling_data_buffer: &'a MeshCullingDataBuffer,
    /// The GPU buffer that stores information about the view.
    view_uniforms: &'a ViewUniforms,
    /// The GPU buffer that stores information about the view from last frame.
    previous_view_uniforms: &'a PreviousViewUniforms,
    /// The pipelines for the mesh preprocessing shader.
    pipelines: &'a PreprocessPipelines,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// current frame.
    current_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshInputUniform`]s for the
    /// previous frame.
    previous_input_buffer: &'a Buffer,
    /// The GPU buffer containing the list of [`MeshUniform`]s for the current
    /// frame.
    ///
    /// This is the buffer containing the mesh's final transforms that the
    /// shaders will write to.
    data_buffer: &'a Buffer,
    /// The GPU buffer for per-mesh visibility flags, written by Stage 1 for
    /// Stage 2 material expansion.
    visibility_flags_buffer: Option<&'a Buffer>,
}

impl<'a> PreprocessBindGroupBuilder<'a> {
    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// and GPU occlusion culling are both disabled.
    fn create_direct_preprocess_bind_groups(
        &self,
        work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        // Don't use `as_entire_binding()` here; the shader reads the array
        // length and the underlying buffer may be longer than the actual size
        // of the vector.
        let work_item_buffer_size = NonZero::<u64>::try_from(
            work_item_buffer.len() as u64 * u64::from(PreprocessWorkItem::min_size()),
        )
        .ok();

        // Visibility flags buffer for Stage 2 material expansion
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        Some(PhasePreprocessBindGroups::Direct(
            self.render_device.create_bind_group(
                "preprocess_direct_bind_group",
                &self
                    .pipeline_cache
                    .get_bind_group_layout(&self.pipelines.direct_preprocess.bind_group_layout),
                &BindGroupEntries::with_indices((
                    (0, self.view_uniforms.uniforms.binding()?),
                    (3, self.current_input_buffer.as_entire_binding()),
                    (4, self.previous_input_buffer.as_entire_binding()),
                    (
                        5,
                        BindingResource::Buffer(BufferBinding {
                            buffer: work_item_buffer.buffer()?,
                            offset: 0,
                            size: work_item_buffer_size,
                        }),
                    ),
                    (6, self.data_buffer.as_entire_binding()),
                    (13, visibility_flags_buffer.as_entire_binding()),
                )),
            ),
        ))
    }

    /// Creates the bind groups for mesh preprocessing when GPU occlusion
    /// culling is enabled.
    fn create_indirect_occlusion_culling_preprocess_bind_groups(
        &self,
        view_depth_pyramids: &Query<(&ViewDepthPyramid, &PreviousViewUniformOffset)>,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        gpu_occlusion_culling_work_item_buffers: &GpuOcclusionCullingWorkItemBuffers,
    ) -> Option<PhasePreprocessBindGroups> {
        let GpuOcclusionCullingWorkItemBuffers {
            late_indexed: ref late_indexed_work_item_buffer,
            late_non_indexed: ref late_non_indexed_work_item_buffer,
            ..
        } = *gpu_occlusion_culling_work_item_buffers;

        let (view_depth_pyramid, previous_view_uniform_offset) =
            view_depth_pyramids.get(self.view).ok()?;

        Some(PhasePreprocessBindGroups::IndirectOcclusionCulling {
            early_indexed: self.create_indirect_occlusion_culling_early_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                indexed_work_item_buffer,
                late_indexed_work_item_buffer,
            ),

            early_non_indexed: self.create_indirect_occlusion_culling_early_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                non_indexed_work_item_buffer,
                late_non_indexed_work_item_buffer,
            ),

            late_indexed: self.create_indirect_occlusion_culling_late_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_indexed_work_item_buffer,
            ),

            late_non_indexed: self.create_indirect_occlusion_culling_late_non_indexed_bind_group(
                view_depth_pyramid,
                previous_view_uniform_offset,
                late_non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            indexed_work_item_buffer.buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(indexed_work_item_gpu_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .early_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                11,
                                BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the first phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_early_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_work_item_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_early_non_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .early_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                11,
                                BufferBinding {
                                    buffer: late_non_indexed_work_item_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            late_indexed_work_item_buffer.buffer(),
            self.late_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(late_indexed_work_item_gpu_buffer),
                Some(late_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let late_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .late_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: late_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: late_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for the second phase of mesh preprocessing of
    /// non-indexed meshes when GPU occlusion culling is enabled.
    fn create_indirect_occlusion_culling_late_non_indexed_bind_group(
        &self,
        view_depth_pyramid: &ViewDepthPyramid,
        previous_view_uniform_offset: &PreviousViewUniformOffset,
        late_non_indexed_work_item_buffer: &UninitBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let previous_view_buffer = self.previous_view_uniforms.uniforms.buffer()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            late_non_indexed_work_item_buffer.buffer(),
            self.late_non_indexed_indirect_parameters_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
                Some(late_non_indexed_indirect_parameters_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    late_non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_late_non_indexed_gpu_occlusion_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .late_gpu_occlusion_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (10, &view_depth_pyramid.all_mips),
                            (
                                2,
                                BufferBinding {
                                    buffer: previous_view_buffer,
                                    offset: previous_view_uniform_offset.offset as u64,
                                    size: NonZeroU64::new(size_of::<PreviousViewData>() as u64),
                                },
                            ),
                            (
                                12,
                                BufferBinding {
                                    buffer: late_non_indexed_indirect_parameters_buffer,
                                    offset: 0,
                                    size: NonZeroU64::new(
                                        late_non_indexed_indirect_parameters_buffer.size(),
                                    ),
                                },
                            ),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind groups for mesh preprocessing when GPU frustum culling
    /// is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_preprocess_bind_groups(
        &self,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<PhasePreprocessBindGroups> {
        Some(PhasePreprocessBindGroups::IndirectFrustumCulling {
            indexed: self
                .create_indirect_frustum_culling_indexed_bind_group(indexed_work_item_buffer),
            non_indexed: self.create_indirect_frustum_culling_non_indexed_bind_group(
                non_indexed_work_item_buffer,
            ),
        })
    }

    /// Creates the bind group for mesh preprocessing of indexed meshes when GPU
    /// frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_indexed_bind_group(
        &self,
        indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .indexed
                .gpu_metadata_buffer(),
            indexed_work_item_buffer.buffer(),
        ) {
            (
                Some(indexed_cpu_metadata_buffer),
                Some(indexed_gpu_metadata_buffer),
                Some(indexed_work_item_gpu_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_indexed_frustum_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .gpu_frustum_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }

    /// Creates the bind group for mesh preprocessing of non-indexed meshes when
    /// GPU frustum culling is enabled, but GPU occlusion culling is disabled.
    fn create_indirect_frustum_culling_non_indexed_bind_group(
        &self,
        non_indexed_work_item_buffer: &RawBufferVec<PreprocessWorkItem>,
    ) -> Option<BindGroup> {
        let mesh_culling_data_buffer = self.mesh_culling_data_buffer.buffer()?;
        let view_uniforms_binding = self.view_uniforms.uniforms.binding()?;
        let visibility_flags_buffer = self.visibility_flags_buffer?;

        match (
            self.phase_indirect_parameters_buffers
                .non_indexed
                .cpu_metadata_buffer(),
            self.phase_indirect_parameters_buffers
                .non_indexed
                .gpu_metadata_buffer(),
            non_indexed_work_item_buffer.buffer(),
        ) {
            (
                Some(non_indexed_cpu_metadata_buffer),
                Some(non_indexed_gpu_metadata_buffer),
                Some(non_indexed_work_item_gpu_buffer),
            ) => {
                // Don't use `as_entire_binding()` here; the shader reads the array
                // length and the underlying buffer may be longer than the actual size
                // of the vector.
                let non_indexed_work_item_buffer_size = NonZero::<u64>::try_from(
                    non_indexed_work_item_buffer.len() as u64
                        * u64::from(PreprocessWorkItem::min_size()),
                )
                .ok();

                Some(
                    self.render_device.create_bind_group(
                        "preprocess_gpu_non_indexed_frustum_culling_bind_group",
                        &self.pipeline_cache.get_bind_group_layout(
                            &self
                                .pipelines
                                .gpu_frustum_culling_preprocess
                                .bind_group_layout,
                        ),
                        &BindGroupEntries::with_indices((
                            (3, self.current_input_buffer.as_entire_binding()),
                            (4, self.previous_input_buffer.as_entire_binding()),
                            (
                                5,
                                BindingResource::Buffer(BufferBinding {
                                    buffer: non_indexed_work_item_gpu_buffer,
                                    offset: 0,
                                    size: non_indexed_work_item_buffer_size,
                                }),
                            ),
                            (6, self.data_buffer.as_entire_binding()),
                            (7, non_indexed_cpu_metadata_buffer.as_entire_binding()),
                            (8, non_indexed_gpu_metadata_buffer.as_entire_binding()),
                            (9, mesh_culling_data_buffer.as_entire_binding()),
                            (0, view_uniforms_binding.clone()),
                            (13, visibility_flags_buffer.as_entire_binding()),
                        )),
                    ),
                )
            }
            _ => None,
        }
    }
}

/// A system that creates bind groups from the indirect parameters metadata and
/// data buffers for the indirect batch set reset shader and the indirect
/// parameter building shader.
///
/// Creates per-(phase, view) bind groups to support multi-view rendering where
/// each view has its own `material_indirect_geometry` buffer.
fn create_build_indirect_parameters_bind_groups(
    commands: &mut Commands,
    render_device: &RenderDevice,
    pipeline_cache: &PipelineCache,
    pipelines: &PreprocessPipelines,
    current_input_buffer: &Buffer,
    indirect_parameters_buffers: &IndirectParametersBuffers,
    phase_instance_buffers: &TypeIdMap<UntypedPhaseBatchedInstanceBuffers<MeshUniform>>,
    submesh_buffer: &SubMeshBuffer,
    material_expansion_buffers: &MaterialExpansionBuffers,
) {
    let mut build_indirect_parameters_bind_groups = BuildIndirectParametersBindGroups::new();

    // Iterate over (phase, view) pairs
    for (phase_type_id, phase_indirect_parameters_buffer) in indirect_parameters_buffers.iter() {
        // Get views for this phase from material_work_item_info keys
        let Some(phase_buffers) = phase_instance_buffers.get(phase_type_id) else {
            continue;
        };

        for &view_entity in phase_buffers.material_work_item_info.keys() {
            let phase_view_key = (*phase_type_id, view_entity);

            build_indirect_parameters_bind_groups.insert(
                phase_view_key,
                PhaseBuildIndirectParametersBindGroups {
                    reset_indexed_indirect_batch_sets: match (phase_indirect_parameters_buffer
                        .indexed
                        .batch_sets_buffer(),)
                    {
                        (Some(indexed_batch_sets_buffer),) => Some(
                            render_device.create_bind_group(
                                "reset_indexed_indirect_batch_sets_bind_group",
                                // The early bind group is good for the main phase and late
                                // phase too. They bind the same buffers.
                                &pipeline_cache.get_bind_group_layout(
                                    &pipelines
                                        .early_phase
                                        .reset_indirect_batch_sets
                                        .bind_group_layout,
                                ),
                                &BindGroupEntries::sequential((
                                    indexed_batch_sets_buffer.as_entire_binding(),
                                )),
                            ),
                        ),
                        _ => None,
                    },

                    reset_non_indexed_indirect_batch_sets: match (phase_indirect_parameters_buffer
                        .non_indexed
                        .batch_sets_buffer(),)
                    {
                        (Some(non_indexed_batch_sets_buffer),) => Some(
                            render_device.create_bind_group(
                                "reset_non_indexed_indirect_batch_sets_bind_group",
                                // The early bind group is good for the main phase and late
                                // phase too. They bind the same buffers.
                                &pipeline_cache.get_bind_group_layout(
                                    &pipelines
                                        .early_phase
                                        .reset_indirect_batch_sets
                                        .bind_group_layout,
                                ),
                                &BindGroupEntries::sequential((
                                    non_indexed_batch_sets_buffer.as_entire_binding(),
                                )),
                            ),
                        ),
                        _ => None,
                    },

                    build_indexed_indirect: match (
                        phase_indirect_parameters_buffer
                            .indexed
                            .cpu_metadata_buffer(),
                        phase_indirect_parameters_buffer
                            .indexed
                            .gpu_metadata_buffer(),
                        phase_indirect_parameters_buffer.indexed.data_buffer(),
                        phase_indirect_parameters_buffer.indexed.batch_sets_buffer(),
                        submesh_buffer.buffer(),
                        material_expansion_buffers
                            .per_phase_view
                            .get(&phase_view_key)
                            .and_then(|p| p.material_indirect_geometry_indexed.buffer()),
                    ) {
                    (
                        Some(indexed_indirect_parameters_cpu_metadata_buffer),
                        Some(indexed_indirect_parameters_gpu_metadata_buffer),
                        Some(indexed_indirect_parameters_data_buffer),
                        Some(indexed_batch_sets_buffer),
                        Some(submesh_gpu_buffer),
                        Some(material_indirect_geometry_indexed_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .gpu_frustum_culling_build_indexed_indirect_params
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::with_indices((
                                (0, current_input_buffer.as_entire_binding()),
                                // Don't use `as_entire_binding` here; the shader reads
                                // the length and `RawBufferVec` overallocates.
                                (
                                    1,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: indexed_indirect_parameters_cpu_metadata_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            phase_indirect_parameters_buffer.indexed.batch_count()
                                                as u64
                                                * size_of::<IndirectParametersCpuMetadata>() as u64,
                                        ),
                                    }),
                                ),
                                (
                                    2,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: indexed_indirect_parameters_gpu_metadata_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            phase_indirect_parameters_buffer.indexed.batch_count()
                                                as u64
                                                * size_of::<IndirectParametersGpuMetadata>() as u64,
                                        ),
                                    }),
                                ),
                                (3, indexed_batch_sets_buffer.as_entire_binding()),
                                (4, indexed_indirect_parameters_data_buffer.as_entire_binding()),
                                (5, submesh_gpu_buffer.as_entire_binding()),
                                (6, material_indirect_geometry_indexed_buffer.as_entire_binding()),
                            )),
                        ),
                    ),
                    _ => None,
                },

                build_non_indexed_indirect: match (
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .cpu_metadata_buffer(),
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .gpu_metadata_buffer(),
                    phase_indirect_parameters_buffer.non_indexed.data_buffer(),
                    phase_indirect_parameters_buffer
                        .non_indexed
                        .batch_sets_buffer(),
                    submesh_buffer.buffer(),
                    material_expansion_buffers
                        .per_phase_view
                        .get(&phase_view_key)
                        .and_then(|p| p.material_indirect_geometry_non_indexed.buffer()),
                ) {
                    (
                        Some(non_indexed_indirect_parameters_cpu_metadata_buffer),
                        Some(non_indexed_indirect_parameters_gpu_metadata_buffer),
                        Some(non_indexed_indirect_parameters_data_buffer),
                        Some(non_indexed_batch_sets_buffer),
                        Some(submesh_gpu_buffer),
                        Some(material_indirect_geometry_non_indexed_buffer),
                    ) => Some(
                        render_device.create_bind_group(
                            "build_non_indexed_indirect_parameters_bind_group",
                            // The frustum culling bind group is good for occlusion culling
                            // too. They bind the same buffers.
                            &pipeline_cache.get_bind_group_layout(
                                &pipelines
                                    .gpu_frustum_culling_build_non_indexed_indirect_params
                                    .bind_group_layout,
                            ),
                            &BindGroupEntries::with_indices((
                                (0, current_input_buffer.as_entire_binding()),
                                // Don't use `as_entire_binding` here; the shader reads
                                // the length and `RawBufferVec` overallocates.
                                (
                                    1,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: non_indexed_indirect_parameters_cpu_metadata_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            phase_indirect_parameters_buffer.non_indexed.batch_count()
                                                as u64
                                                * size_of::<IndirectParametersCpuMetadata>() as u64,
                                        ),
                                    }),
                                ),
                                (
                                    2,
                                    BindingResource::Buffer(BufferBinding {
                                        buffer: non_indexed_indirect_parameters_gpu_metadata_buffer,
                                        offset: 0,
                                        size: NonZeroU64::new(
                                            phase_indirect_parameters_buffer.non_indexed.batch_count()
                                                as u64
                                                * size_of::<IndirectParametersGpuMetadata>() as u64,
                                        ),
                                    }),
                                ),
                                (3, non_indexed_batch_sets_buffer.as_entire_binding()),
                                (4, non_indexed_indirect_parameters_data_buffer.as_entire_binding()),
                                (5, submesh_gpu_buffer.as_entire_binding()),
                                (6, material_indirect_geometry_non_indexed_buffer.as_entire_binding()),
                            )),
                        ),
                    ),
                    _ => None,
                },
            },
        );
        }
    }

    commands.insert_resource(build_indirect_parameters_bind_groups);
}

/// Writes the information needed to do GPU mesh culling to the GPU.
pub fn write_mesh_culling_data_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
) {
    mesh_culling_data_buffer.write_buffer(&render_device, &render_queue);
}
