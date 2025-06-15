use std::hash::Hash;
use std::marker::PhantomData;
use bevy_app::{App, Plugin};
use bevy_asset::{Asset, AssetApp, Handle};
use bevy_render::mesh::MeshVertexBufferLayoutRef;
use bevy_render::render_resource::{AsBindGroup, BindGroupLayout, FragmentState, RenderPipelineDescriptor, Shader, ShaderRef, SpecializedMeshPipelineError, SpecializedRenderPipeline};
use crate::fullscreen::fullscreen_shader_vertex_state;

pub trait FullscreenMaterial: Asset + AsBindGroup + Clone + Sized {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    fn sort_bias(&self) -> f32 {
        0.0
    }

    fn specialize(
        pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: FullscreenMaterialPipelineKey<Self>,
    ) -> bevy_ecs::error::Result<(), SpecializedMeshPipelineError> {
        Ok(())
    }
}

pub struct FullscreenMaterialPlugin<M: FullscreenMaterial>;

impl <M: FullscreenMaterial> Plugin for FullscreenMaterialPlugin<M>
    where M::Data: PartialEq + Eq + Hash + Clone
{
    fn build(&self, app: &mut App) {
        app.init_asset::<M>()
            .init_resource::<EntitiesNeedingSpecialization<M>>()

    }
}

pub struct FullscreenMaterialPipelineKey<M: FullscreenMaterial> {
    pub bind_group_data: M::Data,
}


impl<M: FullscreenMaterial> PartialEq for FullscreenMaterialPipelineKey<M>
where
    M::Data: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.bind_group_data == other.bind_group_data
    }
}

impl<M: FullscreenMaterial> Eq for FullscreenMaterialPipelineKey<M>
where
    M::Data: Eq,
{
}

impl<M: FullscreenMaterial> Hash for FullscreenMaterialPipelineKey<M>
where
    M::Data: Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bind_group_data.hash(state);
    }
}

impl <M: FullscreenMaterial> Clone for FullscreenMaterialPipelineKey<M>
where
    M::Data: Clone,
{
    fn clone(&self) -> Self {
        FullscreenMaterialPipelineKey {
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

pub struct FullscreenMaterialPipeline<M: FullscreenMaterial> {
    pub shader: Handle<Shader>,
    pub layout: BindGroupLayout,
    pub key: FullscreenMaterialPipelineKey<M>,
    pub bindless: bool,
    pub marker: PhantomData<M>,
}

impl <M: FullscreenMaterial> SpecializedRenderPipeline for FullscreenMaterialPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = FullscreenMaterialPipelineKey<M>;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: None,
            layout: [self.layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Default::default(),
                targets: vec![],
            }),
            push_constant_ranges: vec![],
            multisample: Default::default(),
            primitive: Default::default(),
            depth_stencil: None,
            zero_initialize_workgroup_memory: false,
        }
    }
}
