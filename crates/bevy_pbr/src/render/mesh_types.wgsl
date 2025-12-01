#define_import_path bevy_pbr::mesh_types

struct Mesh {
    // Affine 4x3 matrices transposed to 3x4
    // Use bevy_render::maths::affine3_to_square to unpack
    world_from_local: mat3x4<f32>,
    previous_world_from_local: mat3x4<f32>,
    // 3x3 matrix packed in mat2x4 and f32 as:
    // [0].xyz, [1].x,
    // [1].yz, [2].xy
    // [2].z
    // Use bevy_pbr::mesh_functions::mat2x4_f32_to_mat3x3_unpack to unpack
    local_from_world_transpose_a: mat2x4<f32>,
    local_from_world_transpose_b: f32,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
    lightmap_uv_rect: vec2<u32>,
    // The index of the mesh's first vertex in the vertex buffer.
    first_vertex_index: u32,
    current_skin_index: u32,
    // Index of the lightmap in the binding array.
    // Note: Material slot is now in DrawData, written by Stage 2.
    lightmap_bind_group_slot: u32,
    // User supplied index to identify the mesh instance
    tag: u32,
    pad: u32,
};

#ifdef SKINNED
struct SkinnedMesh {
    data: array<mat4x4<f32>, 256u>,
};
#endif

#ifdef MORPH_TARGETS
struct MorphWeights {
    weights: array<vec4<f32>, 64u>, // 64 = 256 / 4 (256 = MAX_MORPH_WEIGHTS)
};
#endif

// [2^0, 2^16)
const MESH_FLAGS_VISIBILITY_RANGE_INDEX_BITS: u32     = (1u << 16u) - 1u;
const MESH_FLAGS_NO_FRUSTUM_CULLING_BIT: u32          = 1u << 28u;
const MESH_FLAGS_SHADOW_RECEIVER_BIT: u32             = 1u << 29u;
const MESH_FLAGS_TRANSMITTED_SHADOW_RECEIVER_BIT: u32 = 1u << 30u;
// if the flag is set, the sign is positive, else it is negative
const MESH_FLAGS_SIGN_DETERMINANT_MODEL_3X3_BIT: u32  = 1u << 31u;

// Per-draw data written by Stage 2 (material expansion).
// Keyed by output/instance index. Draw shaders read this to get material binding.
// Note: mesh[] is indexed directly by instance_index, so no mesh indirection needed here.
struct DrawData {
    // Index of the material inside the bind group data.
    material_bind_group_slot: u32,
}

// Entry in flat material list for Stage 2 material expansion.
// One per (mesh, material) pair. Separate buffers for indexed vs non-indexed.
// Geometry is resolved on GPU from SubMeshBuffer using submesh_slot.
struct MaterialEntry {
    // Index into MeshInput buffer (for visibility check and geometry lookup)
    mesh_input_index: u32,
    // Material bind group slot
    material_bind_group_slot: u32,
    // Submesh slot (0 = full mesh, 1+ = submesh indices)
    submesh_slot: u32,
    // Output destination (direct: output index, indirect: indirect_parameters_index)
    indirect_parameters_index: u32,
}

// Per-material geometry slice for indirect draws.
// Written by Stage 2, read by build_indirect_params.
// Keyed by indirect_parameters_index.
struct MaterialIndirectGeometry {
    first_index: u32,
    index_count: u32,
    base_vertex: u32,
    first_vertex: u32,
    vertex_count: u32,
    // Flags: bit 0 = indexed (1) or non-indexed (0)
    flags: u32,
}
