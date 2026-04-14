// GPU particles simulation shader.
//
// Writes per-slot `world_from_local` transforms into bevy's GPU
// preprocessing input buffer at offsets `[base_input_index ..
// base_input_index + count)`. Also writes per-slot culling data with
// a finite AABB (i.e. alive). No per-frame state — produces a static
// scatter pattern that the example renders every frame.
//
// Bind layout (group 0):
//   @binding(0) read_write: mesh_input_buffer (storage)
//   @binding(1) read_write: mesh_culling_buffer (storage)
//   @binding(2) uniform:    Params { base_input_index, count, time }
//
// Each `MeshInputUniform` is 64 bytes. The first 48 bytes are
// `world_from_local: [Vec4; 3]` (affine 4x3 packed into 3x4). We only
// touch those 48 bytes per slot, leaving the rest of the struct intact
// (the CPU wrote the template once at reservation time).

struct MeshInputUniform {
    // Affine 4x3 transposed to 3x4: three columns of 4 floats each.
    world_from_local_c0: vec4<f32>,
    world_from_local_c1: vec4<f32>,
    world_from_local_c2: vec4<f32>,
    // Remainder of the struct (48..64 bytes). We don't modify these.
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
    aabb_center: vec4<f32>,
    aabb_half_extents: vec4<f32>,
}

struct Params {
    base_input_index: u32,
    count: u32,
    time: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> mesh_input_buffer: array<MeshInputUniform>;
@group(0) @binding(1) var<storage, read_write> mesh_culling_buffer: array<MeshCullingData>;
@group(0) @binding(2) var<uniform> params: Params;

// Build an affine transform (3x4 column-major) representing a
// translation by `t`, with identity rotation and unit scale.
fn translation_world_from_local(t: vec3<f32>) -> array<vec4<f32>, 3> {
    return array<vec4<f32>, 3>(
        vec4<f32>(1.0, 0.0, 0.0, t.x),
        vec4<f32>(0.0, 1.0, 0.0, t.y),
        vec4<f32>(0.0, 0.0, 1.0, t.z),
    );
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }

    let slot = params.base_input_index + i;

    // Scatter in a helix: 16 particles per turn, rising 0.25 per turn,
    // radius growing slowly with i. The time param lets us rotate the
    // whole structure so something visibly animates.
    let t = f32(i);
    let angle = t * 0.4 + params.time * 0.5;
    let radius = 1.5 + t * 0.01;
    let y = -0.5 + t * 0.02;
    let pos = vec3<f32>(
        cos(angle) * radius,
        y,
        sin(angle) * radius,
    );

    let cols = translation_world_from_local(pos);
    mesh_input_buffer[slot].world_from_local_c0 = cols[0];
    mesh_input_buffer[slot].world_from_local_c1 = cols[1];
    mesh_input_buffer[slot].world_from_local_c2 = cols[2];

    // Per-slot culling AABB. The center is in *model* space — bevy's
    // frustum test transforms it via `world_from_local` before testing —
    // so for our centered particle mesh, center = origin. Half-extents
    // cover the mesh bounds (slightly loose).
    mesh_culling_buffer[slot].aabb_center = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    mesh_culling_buffer[slot].aabb_half_extents = vec4<f32>(0.1, 0.1, 0.1, 0.0);
}
