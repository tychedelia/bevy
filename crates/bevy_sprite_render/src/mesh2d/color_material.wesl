import bevy_sprite_render::mesh2d::{
    vertex_output::VertexOutput,
    view_bindings::view,
};

@if(TONEMAP_IN_SHADER)
import bevy_core_pipeline::tonemapping;
@if(SRGB_OUTPUT)
import bevy_render::color_operations::linear_to_srgb;
@if(OKLAB_OUTPUT)
import bevy_render::color_operations::linear_rgb_to_oklab;

struct ColorMaterial {
    color: vec4<f32>,
    uv_transform: mat3x3<f32>,
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32,
    alpha_cutoff: f32,
};

const COLOR_MATERIAL_FLAGS_TEXTURE_BIT: u32              = 1u;
const COLOR_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS: u32 = 3221225472u; // (0b11u32 << 30)
const COLOR_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE: u32        = 0u;          // (0u32 << 30)
const COLOR_MATERIAL_FLAGS_ALPHA_MODE_MASK: u32          = 1073741824u; // (1u32 << 30)
const COLOR_MATERIAL_FLAGS_ALPHA_MODE_BLEND: u32         = 2147483648u; // (2u32 << 30)

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> material: ColorMaterial;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var texture_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    var output_color: vec4<f32> = material.color;

@if(VERTEX_COLORS)
    output_color = output_color * mesh.color;

    let uv = (material.uv_transform * vec3(mesh.uv, 1.0)).xy;

    if ((material.flags & COLOR_MATERIAL_FLAGS_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(texture, texture_sampler, uv);
    }

    output_color = alpha_discard(material, output_color);

@if(TONEMAP_IN_SHADER)
    output_color = tonemapping::tone_mapping(output_color, view.color_grading);
@if(SRGB_OUTPUT)
    output_color = vec4(linear_to_srgb(output_color.rgb), output_color.a);
@if(OKLAB_OUTPUT)
    output_color = vec4(linear_rgb_to_oklab(output_color.rgb), output_color.a);
    return output_color;
}

fn alpha_discard(material: ColorMaterial, output_color: vec4<f32>) -> vec4<f32> {
    var color = output_color;
    let alpha_mode = material.flags & COLOR_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if alpha_mode == COLOR_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE {
        // NOTE: If rendering as opaque, alpha should be ignored so set to 1.0
        color.a = 1.0;
    }
    else {
        @if(MAY_DISCARD)
        if alpha_mode == COLOR_MATERIAL_FLAGS_ALPHA_MODE_MASK {
           if color.a >= material.alpha_cutoff {
                // NOTE: If rendering as masked alpha and >= the cutoff, render as fully opaque
                color.a = 1.0;
            } else {
                // NOTE: output_color.a < in.material.alpha_cutoff should not be rendered
                discard;
            }
        }
    }

    return color;
}