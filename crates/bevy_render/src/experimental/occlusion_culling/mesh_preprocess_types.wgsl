// Types needed for GPU mesh uniform building.

#define_import_path bevy_pbr::mesh_preprocess_types

// Per-frame data that the CPU supplies to the GPU.
struct MeshInput {
    // The model transform.
    world_from_local: mat3x4<f32>,
    // The lightmap UV rect, packed into 64 bits.
    lightmap_uv_rect: vec2<u32>,
    // Various flags.
    flags: u32,
    previous_input_index: u32,
    first_vertex_index: u32,
    first_index_index: u32,
    index_count: u32,
    current_skin_index: u32,
    // Index of the lightmap in the binding array.
    // Material slot is NOT in MeshInput - provided via MaterialWorkItem in Stage 2.
    lightmap_bind_group_slot: u32,
    timestamp: u32,
    // User supplied index to identify the mesh instance
    tag: u32,
    // Packed submesh offset (low 16 bits) and count (high 16 bits).
    //
    // Slot 0 = full mesh (geometry from this struct's first_vertex_index, etc.)
    // Slots 1+ are 1-indexed: submesh_buffer[offset + slot - 1]
    // count = 0 means single-material (only slot 0, no buffer allocation)
    submesh_offset_count: u32,
}

// Unpacks submesh offset from packed value.
fn unpack_submesh_offset(packed: u32) -> u32 {
    return packed & 0xFFFFu;
}

// Unpacks submesh count from packed value.
fn unpack_submesh_count(packed: u32) -> u32 {
    return packed >> 16u;
}

// The `wgpu` indirect parameters structure. This is a union of two structures.
// For more information, see the corresponding comment in
// `gpu_preprocessing.rs`.
struct IndirectParametersIndexed {
    // `vertex_count` or `index_count`.
    index_count: u32,
    // `instance_count` in both structures.
    instance_count: u32,
    // `first_vertex` or `first_index`.
    first_index: u32,
    // `base_vertex` or `first_instance`.
    base_vertex: u32,
    // A read-only copy of `instance_index`.
    first_instance: u32,
}

struct IndirectParametersNonIndexed {
    vertex_count: u32,
    instance_count: u32,
    base_vertex: u32,
    first_instance: u32,
}

struct IndirectParametersCpuMetadata {
    base_output_index: u32,
    batch_set_index: u32,
    // Submesh slot index (low 16 bits). High 16 bits are padding.
    // Slot 0 = full mesh. Slots 1+ are 1-indexed into submesh_buffer.
    submesh_index_and_pad: u32,
}

// Extracts submesh_index from IndirectParametersCpuMetadata.
fn get_submesh_index(cpu_metadata: IndirectParametersCpuMetadata) -> u32 {
    return cpu_metadata.submesh_index_and_pad & 0xFFFFu;
}

struct IndirectParametersGpuMetadata {
    mesh_index: u32,
#ifdef WRITE_INDIRECT_PARAMETERS_METADATA
    early_instance_count: atomic<u32>,
    late_instance_count: atomic<u32>,
    // Stage 2 (material expansion) draw count - separate from early_instance_count
    // to avoid double-counting when both stages run.
    draw_count: atomic<u32>,
#else   // WRITE_INDIRECT_PARAMETERS_METADATA
    early_instance_count: u32,
    late_instance_count: u32,
    draw_count: u32,
#endif  // WRITE_INDIRECT_PARAMETERS_METADATA
}

struct IndirectBatchSet {
    indirect_parameters_count: atomic<u32>,
    indirect_parameters_base: u32,
}

// Describes a submesh - a portion of a mesh's geometry that can be drawn independently.
//
// Meshes can have multiple submeshes for multi-material rendering. Each submesh
// references a contiguous range of indices (or vertices for non-indexed meshes).
//
// Slot 0 = full mesh (geometry comes from MeshInput, not stored here).
// Slots 1+ are 1-indexed: stored at submesh_buffer[offset + slot - 1].
// A mesh with count=0 has only slot 0 (single-material, no buffer allocation).
//
// Interpretation depends on indexed vs non-indexed (known from pipeline/phase):
// - Indexed: first = first_index, count = index_count, base_vertex used
// - Non-indexed: first = first_vertex, count = vertex_count, base_vertex = 0
struct SubMeshDescriptor {
    // For indexed: first index (relative to mesh's first_index_index).
    // For non-indexed: first vertex (relative to mesh's first_vertex_index).
    first: u32,
    // For indexed: number of indices. For non-indexed: number of vertices.
    count: u32,
    // For indexed: offset added to index values before vertex lookup.
    // For non-indexed: unused (0).
    base_vertex: i32,
    // Padding to align to 16 bytes.
    pad: u32,
}
