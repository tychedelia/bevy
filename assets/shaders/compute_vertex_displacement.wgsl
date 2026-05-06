// Displaces vertex positions with a sine wave.
//
// Because position is in its own vertex buffer binding, this shader
// only needs access to the position buffer — it doesn't touch
// normals, UVs, or any other attribute data.

struct Params {
    position_start: u32,
    vertex_count: u32,
    time: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vertex_index = global_id.x;
    if vertex_index >= params.vertex_count {
        return;
    }

    // Each vertex is 3 consecutive f32s (x, y, z).
    let base = params.position_start + vertex_index * 3u;
    let x = positions[base];
    let z = positions[base + 2u];

    // Displace Y with a time-varying sine wave.
    let wave = sin(x * 1.5 + params.time * 2.0) * cos(z * 1.5 + params.time * 1.5) * 0.5;
    positions[base + 1u] = wave;
}
