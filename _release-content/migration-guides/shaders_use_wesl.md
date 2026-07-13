---
title: Shaders use WESL instead of the naga_oil preprocessor
pull_requests: []
---

All of Bevy's shaders are now written in [WESL](https://wesl-lang.dev), and the
naga_oil preprocessor has been removed. Custom shaders written in the naga_oil
dialect (`#import`, `#ifdef`, `#define_import_path`, `#{...}`) must be
translated to WESL and renamed from `.wgsl` to `.wesl`. Plain, spec-compliant
WGSL files with no preprocessor directives keep working unchanged.

To translate a shader:

- `#import bevy_pbr::forward_io::VertexOutput` becomes
  `import bevy_pbr::render::forward_io::VertexOutput;` (the trailing semicolon
  is required).
  Bevy's module names now match the shader's path within its crate, so most
  have changed: `bevy_pbr::mesh_view_bindings` is
  `bevy_pbr::render::mesh_view_bindings`, `bevy_pbr::prepass_utils` is
  `bevy_pbr::prepass::utils`, `bevy_sprite::mesh2d_functions` is
  `bevy_sprite_render::mesh2d::functions`, and so on. Imports must appear at
  the top of the file, before any declaration or `enable` directive. Quoted
  asset-path imports (`#import "shaders/util.wgsl"::foo`) become relative
  imports (`import super::util::foo;`).
- `#ifdef FLAG` / `#else` / `#endif` become `@if(FLAG)` / `@else` translate-time
  attributes attached to whole declarations, struct members, function
  parameters, imports, or statements. `#else ifdef` becomes `@elif(...)`.
  Shader defs passed from `RenderPipelineDescriptor::shader_defs` and friends
  still work; boolean defs are mapped to WESL conditional compilation flags.
- `#{SHADER_DEF}` value interpolation becomes an ordinary expression reference
  to the virtual `constants` module: `@group(#{MATERIAL_BIND_GROUP})` becomes
  `@group(constants::MATERIAL_BIND_GROUP)`. Every `Int`/`UInt` shader def is
  available under `constants::` and also enables a flag of the same name.
- `#define_import_path bevy_foo::bar` has been removed. A shader loaded from
  the `embedded://` asset source is importable under the crate name and file
  path it is embedded at (`embedded://bevy_foo/bar.wesl` becomes
  `bevy_foo::bar`), and any other shader under its asset path. Registering a
  shader library is just `load_shader_library!(app, "bar.wesl");`.

The `shader_format_wesl` cargo feature has been removed: WESL support is
always enabled. `Shader::from_wgsl` no longer performs any preprocessing.
GLSL shader support (`shader_format_glsl`, `Shader::from_glsl`) has been
removed, as GLSL shaders could not import Bevy shader code without the
preprocessor. SPIR-V passthrough is unchanged.

Hot reloading now works for imported `.wesl` modules, including embedded
shader libraries with the `embedded_watcher` feature.
