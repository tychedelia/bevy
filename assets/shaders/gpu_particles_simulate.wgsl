// GPU particles simulation shader.
//
// Writes per-slot `world_from_local` transforms and per-slot
// `MeshCullingData` into bevy's GPU preprocessing input buffers at
// offsets `[base_input_index .. base_input_index + count)`.
//
// The struct layout matches `bevy_pbr::mesh_preprocess_types::MeshInput`
// exactly — we only overwrite `world_from_local`; the remainder of the
// struct (mesh allocator offsets, flags, etc.) was written by the CPU
// at reservation time and must be preserved.

struct MeshInput {
    world_from_local: mat3x4<f32>,
    lightmap_uv_rect: vec2<u32>,
    flags: u32,
    previous_input_index: u32,
    first_vertex_index: u32,
    first_index_index: u32,
    index_count: u32,
    current_skin_index: u32,
    material_and_lightmap_bind_group_slot: u32,
    timestamp: u32,
    tag: u32,
    morph_descriptor_index: u32,
}

struct MeshCullingData {
    // Model-space AABB center. Bevy's frustum test transforms this via
    // `world_from_local` before testing, so for a particle mesh centered
    // on its own origin this is `vec3(0)`.
    aabb_center: vec4<f32>,
    // Model-space half-extents. Slightly larger than the mesh bounds.
    aabb_half_extents: vec4<f32>,
}

struct Params {
    base_input_index: u32,
    count: u32,
    time: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> mesh_input_buffer: array<MeshInput>;
@group(0) @binding(1) var<storage, read_write> mesh_culling_buffer: array<MeshCullingData>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }

    let slot = params.base_input_index + i;

    // Trivial simulation: particles drift in a rotating double helix.
    // This is intentionally uninteresting — the example's focus is on the
    // render integration, not on particle behavior.
    let t = f32(i);
    let angle = t * 0.3 + params.time * 0.4;
    let helix = t * 0.02 - 1.5;
    let radius = 2.0 + 0.3 * sin(t * 0.1 + params.time);
    let side = select(1.0, -1.0, (i & 1u) == 0u);
    let pos = vec3<f32>(
        cos(angle) * radius * side,
        helix,
        sin(angle) * radius * side,
    );

    // Build a translation matrix as mat3x4: three columns of vec4. After
    // bevy's `affine3_to_square` (which transposes), row 0 = col 0 etc.,
    // producing the standard 4x4 translation.
    mesh_input_buffer[slot].world_from_local = mat3x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, pos.x),
        vec4<f32>(0.0, 1.0, 0.0, pos.y),
        vec4<f32>(0.0, 0.0, 1.0, pos.z),
    );

    // Per-slot culling AABB in model space. The particle mesh is a small
    // cube centered on its own origin, so center = 0 and half_extents is
    // slightly larger than the mesh's actual half-extent.
    mesh_culling_buffer[slot].aabb_center = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    mesh_culling_buffer[slot].aabb_half_extents = vec4<f32>(0.2, 0.2, 0.2, 0.0);
}
