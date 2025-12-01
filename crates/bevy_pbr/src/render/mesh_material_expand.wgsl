// Stage 2: Material Expansion Pass (GPU-driven)
//
// This compute shader expands visible meshes into per-draw material bindings.
// It dispatches directly over MaterialEntry buffers (one per material/submesh)
// and checks visibility_flags to early-out for culled meshes.
//
// Runs after Stage 1 visibility (both early and late) is complete.
// Separate dispatches for indexed and non-indexed entries.
//
// Submesh slots: slot 0 = full mesh (from MeshInput), slots 1+ are 1-indexed
// into submesh_buffer. count=0 means single-material (only slot 0).
//
// Input:
//   - visibility_flags: Per-mesh visibility (1 = visible, 0 = culled) from Stage 1
//   - material_entries: Flat list of material entries (one per draw)
//   - (Indirect mode) current_input: MeshInput buffer for geometry lookup
//   - (Indirect mode) submesh_buffer: SubMeshDescriptor buffer for geometry
//
// Output:
//   - draw_data: Per-draw buffer with material_bind_group_slot
//   - (Indirect mode) material_indirect_geometry: Per-batch geometry
//   - (Indirect mode) Updates indirect_parameters_gpu_metadata draw counts + mesh_index

#import bevy_pbr::mesh_types::{DrawData, MaterialEntry, MaterialIndirectGeometry}

#ifdef INDIRECT
#import bevy_pbr::mesh_preprocess_types::{
    IndirectParametersCpuMetadata,
    IndirectParametersGpuMetadata,
    MeshInput,
    SubMeshDescriptor,
    unpack_submesh_offset,
    unpack_submesh_count,
}
#endif

// Stage 1 outputs (read-only)
@group(0) @binding(0) var<storage, read> visibility_flags: array<u32>;

// Material entries (one per material/submesh draw)
@group(0) @binding(1) var<storage, read> material_entries: array<MaterialEntry>;

// Per-draw output (write)
@group(0) @binding(2) var<storage, read_write> draw_data: array<DrawData>;

#ifdef INDIRECT
// Per-material geometry slices for indirect draws
@group(0) @binding(3) var<storage, read_write> material_indirect_geometry: array<MaterialIndirectGeometry>;

@group(0) @binding(4) var<storage, read_write> indirect_parameters_gpu_metadata: array<IndirectParametersGpuMetadata>;

@group(0) @binding(5) var<storage, read> indirect_parameters_cpu_metadata: array<IndirectParametersCpuMetadata>;

// MeshInput buffer for geometry lookup (only needed for indirect mode)
@group(0) @binding(6) var<storage, read> current_input: array<MeshInput>;

// Global submesh descriptor buffer for geometry lookup (only needed for indirect mode)
@group(0) @binding(7) var<storage, read> submesh_buffer: array<SubMeshDescriptor>;
#endif

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let entry_index = global_invocation_id.x;

    if entry_index >= arrayLength(&material_entries) {
        return;
    }

    let entry = material_entries[entry_index];

    // Early-out if mesh was culled (visibility set by Stage 1 early+late)
    if visibility_flags[entry.mesh_input_index] == 0u {
        return;
    }

#ifdef INDIRECT
    let indirect_idx = entry.indirect_parameters_index;

    // Write mesh_index for this batch (all writers have valid value - benign race)
    indirect_parameters_gpu_metadata[indirect_idx].mesh_index = entry.mesh_input_index;

    // Atomic increment to get our output slot
    let batch_offset = atomicAdd(
        &indirect_parameters_gpu_metadata[indirect_idx].draw_count,
        1u
    );
    let output_index = indirect_parameters_cpu_metadata[indirect_idx].base_output_index + batch_offset;

    // Resolve geometry: slot 0 = full mesh from MeshInput, slots 1+ from submesh_buffer
    let mesh = current_input[entry.mesh_input_index];
    let submesh_offset = unpack_submesh_offset(mesh.submesh_offset_count);
    let submesh_count = unpack_submesh_count(mesh.submesh_offset_count);

    // Bounds check: slot 0 always valid, slots 1+ must be <= submesh_count
    if entry.submesh_slot > 0u && entry.submesh_slot > submesh_count {
        return;
    }

    // Write geometry for this batch
    if entry.submesh_slot == 0u {
        // Slot 0: full mesh geometry from MeshInput
#ifdef INDEXED
        material_indirect_geometry[indirect_idx].first_index = mesh.first_index_index;
        material_indirect_geometry[indirect_idx].index_count = mesh.index_count;
        material_indirect_geometry[indirect_idx].base_vertex = mesh.first_vertex_index;
        material_indirect_geometry[indirect_idx].first_vertex = 0u;
        material_indirect_geometry[indirect_idx].vertex_count = 0u;
        material_indirect_geometry[indirect_idx].flags = 1u;  // FLAG_INDEXED
#else
        material_indirect_geometry[indirect_idx].first_index = 0u;
        material_indirect_geometry[indirect_idx].index_count = 0u;
        material_indirect_geometry[indirect_idx].base_vertex = 0u;
        material_indirect_geometry[indirect_idx].first_vertex = mesh.first_vertex_index;
        material_indirect_geometry[indirect_idx].vertex_count = mesh.index_count;
        material_indirect_geometry[indirect_idx].flags = 0u;  // Non-indexed
#endif
    } else {
        // Slots 1+: 1-indexed lookup
        let submesh = submesh_buffer[submesh_offset + entry.submesh_slot - 1u];
#ifdef INDEXED
        material_indirect_geometry[indirect_idx].first_index = mesh.first_index_index + submesh.first;
        material_indirect_geometry[indirect_idx].index_count = submesh.count;
        material_indirect_geometry[indirect_idx].base_vertex = mesh.first_vertex_index + u32(submesh.base_vertex);
        material_indirect_geometry[indirect_idx].first_vertex = 0u;
        material_indirect_geometry[indirect_idx].vertex_count = 0u;
        material_indirect_geometry[indirect_idx].flags = 1u;  // FLAG_INDEXED
#else
        material_indirect_geometry[indirect_idx].first_index = 0u;
        material_indirect_geometry[indirect_idx].index_count = 0u;
        material_indirect_geometry[indirect_idx].base_vertex = 0u;
        material_indirect_geometry[indirect_idx].first_vertex = mesh.first_vertex_index + submesh.first;
        material_indirect_geometry[indirect_idx].vertex_count = submesh.count;
        material_indirect_geometry[indirect_idx].flags = 0u;  // Non-indexed
#endif
    }
#else
    // Direct mode: indirect_parameters_index is the output index
    // Geometry already resolved on CPU via work items
    let output_index = entry.indirect_parameters_index;
#endif

    // Write material slot for this draw
    draw_data[output_index].material_bind_group_slot = entry.material_bind_group_slot;
}
