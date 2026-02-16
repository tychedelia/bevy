#ifdef WIREFRAME_WIDE

#import bevy_pbr::{
    mesh_bindings::mesh,
    mesh_view_bindings::view,
    view_transformations::position_world_to_clip,
}
#import bevy_render::maths::affine3_to_square

struct Immediates {
    color: vec4<f32>,
    line_width: f32,
    smoothing: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    _pad0: f32,
    _pad1: f32,
#endif
}

var<immediate> immediates: Immediates;

struct VertexPullParams {
    index_offset: u32,
    vertex_stride: u32,
    position_offset: u32,
}

@group(3) @binding(0) var<storage, read> vertex_data: array<u32>;
@group(3) @binding(1) var<storage, read> index_data: array<u32>;
@group(3) @binding(2) var<uniform> vp_params: VertexPullParams;

struct WireframeVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(linear) edge_distance: vec3<f32>,
}

// finds the local-space vertex position for a given vertex index by reading from the vertex buffer using the provided
// parameters. the vertex buffer is interpreted as an array of u32s, so the parameters allow us to calculate the
// correct offset for the position attribute of the vertex at the given index.
fn read_local_position(first_vertex: u32, vertex_index: u32) -> vec3<f32> {
    let base = (first_vertex + vertex_index) * vp_params.vertex_stride
               + vp_params.position_offset;
    return vec3<f32>(
        bitcast<f32>(vertex_data[base]),
        bitcast<f32>(vertex_data[base + 1u]),
        bitcast<f32>(vertex_data[base + 2u]),
    );
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> WireframeVertexOutput {
    var out: WireframeVertexOutput;

    let first_vertex = mesh[instance_index].first_vertex_index;
    let draw_id = vertex_index - first_vertex;
    let corner = draw_id % 3u;
    let tri_base = draw_id - corner;

    let idx0 = index_data[vp_params.index_offset + tri_base];
    let idx1 = index_data[vp_params.index_offset + tri_base + 1u];
    let idx2 = index_data[vp_params.index_offset + tri_base + 2u];

    let p0 = read_local_position(first_vertex, idx0);
    let p1 = read_local_position(first_vertex, idx1);
    let p2 = read_local_position(first_vertex, idx2);

    let world_from_local = affine3_to_square(mesh[instance_index].world_from_local);
    let clip0 = position_world_to_clip((world_from_local * vec4(p0, 1.0)).xyz);
    let clip1 = position_world_to_clip((world_from_local * vec4(p1, 1.0)).xyz);
    let clip2 = position_world_to_clip((world_from_local * vec4(p2, 1.0)).xyz);

    let viewport_size = view.viewport.zw;
    let screen0 = (clip0.xy / clip0.w) * viewport_size * 0.5;
    let screen1 = (clip1.xy / clip1.w) * viewport_size * 0.5;
    let screen2 = (clip2.xy / clip2.w) * viewport_size * 0.5;

    let area2 = abs((screen1.x - screen0.x) * (screen2.y - screen0.y)
                   - (screen2.x - screen0.x) * (screen1.y - screen0.y));

    let len01 = length(screen1 - screen0);
    let len12 = length(screen2 - screen1);
    let len20 = length(screen0 - screen2);

    // altitudes: h_i = 2 * area / opposite_edge_length
    let h0 = area2 / max(len12, 0.001);
    let h1 = area2 / max(len20, 0.001);
    let h2 = area2 / max(len01, 0.001);

#ifdef WIREFRAME_QUADS
    // detect and suppress the shared diagonal between two triangles forming a quad.
    let quad_base = (draw_id / 6u) * 6u;
    let tri_in_quad = (draw_id / 3u) % 2u;
    let other_tri_base = quad_base + (1u - tri_in_quad) * 3u;

    let j0 = index_data[vp_params.index_offset + other_tri_base];
    let j1 = index_data[vp_params.index_offset + other_tri_base + 1u];
    let j2 = index_data[vp_params.index_offset + other_tri_base + 2u];

    let in_other_0 = (idx0 == j0) || (idx0 == j1) || (idx0 == j2);
    let in_other_1 = (idx1 == j0) || (idx1 == j1) || (idx1 == j2);
    let in_other_2 = (idx2 == j0) || (idx2 == j1) || (idx2 == j2);

    // edge is diagonal if both endpoints appear in the other triangle
    let diag_01 = in_other_0 && in_other_1;
    let diag_12 = in_other_1 && in_other_2;
    let diag_20 = in_other_2 && in_other_0;
    let mask = vec3<f32>(
        select(1.0, 0.0, diag_12),
        select(1.0, 0.0, diag_20),
        select(1.0, 0.0, diag_01),
    );
    let suppress = vec3<f32>(1.0) - mask;

    if corner == 0u {
        out.edge_distance = vec3<f32>(h0, 0.0, 0.0) * mask + suppress * 1e6;
        out.position = clip0;
    } else if corner == 1u {
        out.edge_distance = vec3<f32>(0.0, h1, 0.0) * mask + suppress * 1e6;
        out.position = clip1;
    } else {
        out.edge_distance = vec3<f32>(0.0, 0.0, h2) * mask + suppress * 1e6;
        out.position = clip2;
    }
#else // WIREFRAME_QUADS
    if corner == 0u {
        out.edge_distance = vec3<f32>(h0, 0.0, 0.0);
        out.position = clip0;
    } else if corner == 1u {
        out.edge_distance = vec3<f32>(0.0, h1, 0.0);
        out.position = clip1;
    } else {
        out.edge_distance = vec3<f32>(0.0, 0.0, h2);
        out.position = clip2;
    }
#endif // WIREFRAME_QUADS

    return out;
}

@fragment
fn fragment(in: WireframeVertexOutput) -> @location(0) vec4<f32> {
    let d = min(in.edge_distance.x, min(in.edge_distance.y, in.edge_distance.z));

    let width = immediates.line_width;
    let smoothing = immediates.smoothing;

    let effective_width = max(width, 1.0);
    let alpha_scale = min(width, 1.0);

    let half_width = effective_width * 0.5;
    // solid line up to half_width, then anti-alias over `smoothing` pixels at the outer edge
    let alpha = (1.0 - smoothstep(half_width, half_width + smoothing, d))
              * alpha_scale
              * immediates.color.a;

    if alpha < 0.01 {
        discard;
    }

    return vec4<f32>(immediates.color.rgb, alpha);
}

#else // WIREFRAME_WIDE

// the fast path for thin wireframes that render as lines

#import bevy_pbr::forward_io::VertexOutput

struct Immediates {
    color: vec4<f32>,
}

var<immediate> immediates: Immediates;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return immediates.color;
}

#endif // WIREFRAME_WIDE
