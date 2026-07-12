import bevy_sprite::mesh2d_types::Mesh2d;
import bevy_render::mesh_metadata_types::MeshMetadata;

@if(PER_OBJECT_BUFFER_BATCH_SIZE)
@group(1) @binding(0) var<uniform> mesh: array<Mesh2d, u32(constants::PER_OBJECT_BUFFER_BATCH_SIZE)>;
@else
@group(1) @binding(0) var<storage> mesh: array<Mesh2d>;

@if(METADATA_USE_UNIFORM_BUFFERS)
@group(1) @binding(1) var<uniform> metadata: MeshMetadata;
@else
@group(1) @binding(1) var<storage> metadata: array<MeshMetadata>;
