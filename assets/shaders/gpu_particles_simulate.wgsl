// GPU particles simulation: mouse attractor with per-particle lifespans.

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
    aabb_center: vec3<f32>,
    _pad: f32,
    aabb_half_extents: vec3<f32>,
    // 0.0 = alive, nonzero = skip in preprocessing.
    dead: f32,
}

struct ParticleState {
    // xyz = world position, w = age (s). w < 0 = uninitialized.
    pos: vec4<f32>,
    vel: vec4<f32>,
}

struct Params {
    base_input_index: u32,
    count: u32,
    time: f32,
    dt: f32,
    mouse_world_pos: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> mesh_input_buffer: array<MeshInput>;
@group(0) @binding(1) var<storage, read_write> mesh_culling_buffer: array<MeshCullingData>;
@group(0) @binding(2) var<storage, read_write> particle_state: array<ParticleState>;
@group(0) @binding(3) var<uniform> params: Params;

fn hash(n: u32) -> u32 {
    var x = n;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn hash_float(n: u32) -> f32 {
    return f32(hash(n)) / f32(0xffffffffu);
}

fn hash_vec3(n: u32) -> vec3<f32> {
    return vec3<f32>(
        hash_float(n * 3u + 0u) * 2.0 - 1.0,
        hash_float(n * 3u + 1u) * 2.0 - 1.0,
        hash_float(n * 3u + 2u) * 2.0 - 1.0,
    );
}

const ALIVE_MIN: f32 = 1.5;
const ALIVE_RANGE: f32 = 2.5;
const RESPAWN_DELAY: f32 = 0.4;

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count {
        return;
    }

    let slot = params.base_input_index + i;
    var state = particle_state[i];

    let lifespan = ALIVE_MIN + hash_float(i ^ 0xabcd1234u) * ALIVE_RANGE;

    if state.pos.w < 0.0 || state.pos.w >= lifespan + RESPAWN_DELAY {
        let seed_key = i ^ u32(params.time * 997.0);
        let seed = hash_vec3(seed_key) * 9.0;
        state.pos = vec4<f32>(seed, 0.0);
        state.vel = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let dead = state.pos.w >= lifespan;

    if !dead {
        let to_mouse = params.mouse_world_pos.xyz - state.pos.xyz;
        let dist = sqrt(dot(to_mouse, to_mouse) + 0.5);
        let force_mag = min(14.0 / dist, 40.0);
        let accel = (to_mouse / dist) * force_mag;

        let damping = exp(-1.0 * params.dt);
        state.vel = vec4<f32>(state.vel.xyz * damping + accel * params.dt, 0.0);
        state.pos = vec4<f32>(
            state.pos.xyz + state.vel.xyz * params.dt,
            state.pos.w + params.dt,
        );
    } else {
        state.pos.w += params.dt;
    }

    particle_state[i] = state;

    let to_mouse = params.mouse_world_pos.xyz - state.pos.xyz;
    let dist = max(length(to_mouse), 1e-4);
    let forward = to_mouse / dist;
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    var right: vec3<f32>;
    if abs(dot(forward, world_up)) > 0.999 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        right = normalize(cross(world_up, forward));
    }
    let up = cross(forward, right);

    mesh_input_buffer[slot].world_from_local = mat3x4<f32>(
        vec4<f32>(right.x, up.x, forward.x, state.pos.x),
        vec4<f32>(right.y, up.y, forward.y, state.pos.y),
        vec4<f32>(right.z, up.z, forward.z, state.pos.z),
    );

    mesh_culling_buffer[slot].aabb_center = vec3<f32>(0.0, 0.0, 0.0);
    mesh_culling_buffer[slot].aabb_half_extents = vec3<f32>(0.2, 0.2, 0.2);
    mesh_culling_buffer[slot].dead = select(0.0, 1.0, dead);
}
