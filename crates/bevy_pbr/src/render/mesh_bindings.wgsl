#define_import_path bevy_pbr::mesh_bindings

#import bevy_pbr::mesh_types::{Mesh, DrawData}

#ifndef MESHLET_MESH_MATERIAL_PASS
#ifdef PER_OBJECT_BUFFER_BATCH_SIZE
@group(2) @binding(0) var<uniform> mesh: array<Mesh, #{PER_OBJECT_BUFFER_BATCH_SIZE}u>;
#else
@group(2) @binding(0) var<storage> mesh: array<Mesh>;
#endif // PER_OBJECT_BUFFER_BATCH_SIZE

// Per-draw data written by Stage 2 material expansion.
// Contains material slot and mesh index for each draw instance.
// Uses binding 8 to avoid conflicts with skinning (1), morph weights (2), etc.
@group(2) @binding(8) var<storage> draw_data: array<DrawData>;

#endif  // MESHLET_MESH_MATERIAL_PASS
