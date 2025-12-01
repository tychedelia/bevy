// Corruption shader - liquid mercury / molten metal effect

#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::{view, globals}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> color_a: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> color_b: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> glitch_speed: f32;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var noise_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var noise_sampler: sampler;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    let t = globals.time;
    let uv = mesh.uv;

    // Sample noise with slow flowing distortion
    let flow1 = vec2(sin(t * 0.1) * 0.03, cos(t * 0.08) * 0.03);
    let flow2 = vec2(cos(t * 0.12) * 0.02, sin(t * 0.09) * 0.02);
    let mask1 = textureSample(noise_texture, noise_sampler, uv + flow1).r;
    let mask2 = textureSample(noise_texture, noise_sampler, uv * 1.3 + flow2).r;

    // Blend for smooth, blobby shapes
    let blob = smoothstep(0.3, 0.7, mask1 * 0.6 + mask2 * 0.4);

    // View-dependent effects for that liquid metal look
    let V = normalize(view.world_position - mesh.world_position.xyz);
    let N = faceForward(normalize(mesh.world_normal), -V, normalize(mesh.world_normal));
    let NdotV = max(dot(N, V), 0.0);

    // Fake environment reflection - use noise as fake reflection
    let reflect_uv = uv + N.xy * 0.3;
    let fake_reflect = textureSample(noise_texture, noise_sampler, reflect_uv).r;

    // Mercury base - highly reflective silver with color tints
    let silver = vec3(0.85, 0.87, 0.9);
    let tinted_silver = mix(silver, color_a.rgb, 0.2);

    // Specular highlight simulation
    let spec_fake = pow(fake_reflect, 3.0);
    let highlight = mix(tinted_silver, vec3(1.0), spec_fake * 0.5);

    // Dark pools where the "mercury" is deeper
    let depth_color = mix(color_b.rgb * 0.3, vec3(0.1), 0.5);
    var final_color = mix(depth_color, highlight, blob);

    // Add iridescent sheen based on view angle
    let iridescence = sin(NdotV * 6.0 + t * 0.2) * 0.5 + 0.5;
    let iri_color = mix(color_a.rgb, color_b.rgb, iridescence);
    final_color = mix(final_color, iri_color, (1.0 - NdotV) * 0.4);

    // Strong fresnel for that liquid surface tension look
    let fresnel = pow(1.0 - NdotV, 4.0);
    final_color += silver * fresnel * 0.8;

    // Stronger, faster ripple brightness
    let ripple = sin(blob * 30.0 + t * 2.0) * 0.25 + 0.85;
    final_color *= ripple;

    // Alpha from original mask, capped at 0.8 but boosted by fresnel
    let base_mask = textureSample(noise_texture, noise_sampler, uv).r;
    let base_alpha = smoothstep(0.0, 0.2, base_mask) * 0.8;
    let alpha = min(base_alpha + fresnel * 0.3, 1.0);

    return vec4(final_color, alpha);
}
