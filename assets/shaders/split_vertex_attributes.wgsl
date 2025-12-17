#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world}
#import bevy_pbr::view_transformations::position_world_to_clip

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) height_offset: f32,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    var position = vertex.position;
    position.y += vertex.height_offset;

    let world_from_local = get_world_from_local(vertex.instance_index);
    out.world_position = mesh_position_local_to_world(world_from_local, vec4(position, 1.0));
    out.position = position_world_to_clip(out.world_position.xyz);
    out.world_normal = mesh_normal_local_to_world(vertex.normal, vertex.instance_index);
    out.instance_index = vertex.instance_index;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3(1.0, 1.0, 1.0));
    let diffuse = max(dot(normalize(in.world_normal), light_dir), 0.2);
    return vec4(vec3(0.4, 0.6, 0.9) * diffuse, 1.0);
}
