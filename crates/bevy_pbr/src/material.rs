use self::{irradiance_volume::IrradianceVolume, prelude::EnvironmentMapLight};
use crate::material_bind_groups::{MaterialBindGroupAllocator, MaterialBindingId};
#[cfg(feature = "meshlet")]
use crate::meshlet::{
    prepare_material_meshlet_meshes_main_opaque_pass, queue_material_meshlet_meshes,
    InstanceManager,
};
use crate::*;
use bevy_asset::prelude::AssetChanged;
use bevy_asset::{Asset, AssetId, AssetServer};
use bevy_core_pipeline::core_2d::{AlphaMask2d, Opaque2d, Transparent2d};
use bevy_core_pipeline::{
    core_3d::{
        AlphaMask3d, Camera3d, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey,
        ScreenSpaceTransmissionQuality, Transmissive3d, Transparent3d,
    },
    oit::OrderIndependentTransparencySettings,
    prepass::{
        DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
        OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey,
    },
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::query::{QueryFilter, QueryItem};
use bevy_ecs::{
    prelude::*,
    system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_render::mesh::Mesh2d;
use bevy_render::render_phase::specialization::{
    SpecializationPlugin,
    SpecializedMaterialPipelineCache,
};
use bevy_render::sync_world::MainEntity;
use bevy_render::{
    camera::TemporalJitter,
    extract_resource::ExtractResource,
    mesh::{Mesh3d, MeshVertexBufferLayoutRef, RenderMesh},
    render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
    render_phase::*,
    render_resource::*,
    renderer::RenderDevice,
    view::{ExtractedView, Msaa, RenderVisibilityRanges, ViewVisibility},
    Extract,
};
use bevy_render::{mesh::allocator::MeshAllocator, sync_world::MainEntityHashMap};
use bevy_render::{texture::FallbackImage, view::RenderVisibleEntities};
use bevy_utils::{hashbrown::hash_map::Entry, tracing::error};
use core::{hash::Hash, marker::PhantomData};

/// Materials are used alongside [`MaterialPlugin`], [`Mesh3d`], and [`MeshMaterial3d`]
/// to spawn entities that are rendered with a specific [`Material`] type. They serve as an easy to use high level
/// way to render [`Mesh3d`] entities with custom shader logic.
///
/// Materials must implement [`AsBindGroup`] to define how data will be transferred to the GPU and bound in shaders.
/// [`AsBindGroup`] can be derived, which makes generating bindings straightforward. See the [`AsBindGroup`] docs for details.
///
/// # Example
///
/// Here is a simple [`Material`] implementation. The [`AsBindGroup`] derive has many features. To see what else is available,
/// check out the [`AsBindGroup`] documentation.
///
/// ```
/// # use bevy_pbr::{Material, MeshMaterial3d};
/// # use bevy_ecs::prelude::*;
/// # use bevy_image::Image;
/// # use bevy_reflect::TypePath;
/// # use bevy_render::{mesh::{Mesh, Mesh3d}, render_resource::{AsBindGroup, ShaderRef}};
/// # use bevy_color::LinearRgba;
/// # use bevy_color::palettes::basic::RED;
/// # use bevy_asset::{Handle, AssetServer, Assets, Asset};
/// # use bevy_math::primitives::Capsule3d;
/// #
/// #[derive(AsBindGroup, Debug, Clone, Asset, TypePath)]
/// pub struct CustomMaterial {
///     // Uniform bindings must implement `ShaderType`, which will be used to convert the value to
///     // its shader-compatible equivalent. Most core math types already implement `ShaderType`.
///     #[uniform(0)]
///     color: LinearRgba,
///     // Images can be bound as textures in shaders. If the Image's sampler is also needed, just
///     // add the sampler attribute with a different binding index.
///     #[texture(1)]
///     #[sampler(2)]
///     color_texture: Handle<Image>,
/// }
///
/// // All functions on `Material` have default impls. You only need to implement the
/// // functions that are relevant for your material.
/// impl Material for CustomMaterial {
///     fn fragment_shader() -> ShaderRef {
///         "shaders/custom_material.wgsl".into()
///     }
/// }
///
/// // Spawn an entity with a mesh using `CustomMaterial`.
/// fn setup(
///     mut commands: Commands,
///     mut meshes: ResMut<Assets<Mesh>>,
///     mut materials: ResMut<Assets<CustomMaterial>>,
///     asset_server: Res<AssetServer>
/// ) {
///     commands.spawn((
///         Mesh3d(meshes.add(Capsule3d::default())),
///         MeshMaterial3d(materials.add(CustomMaterial {
///             color: RED.into(),
///             color_texture: asset_server.load("some_image.png"),
///         })),
///     ));
/// }
/// ```
///
/// In WGSL shaders, the material's binding would look like this:
///
/// ```wgsl
/// @group(2) @binding(0) var<uniform> color: vec4<f32>;
/// @group(2) @binding(1) var color_texture: texture_2d<f32>;
/// @group(2) @binding(2) var color_sampler: sampler;
/// ```
pub trait Material: Asset + AsBindGroup + Clone + Sized {
    /// Returns this material's vertex shader. If [`ShaderRef::Default`] is returned, the default mesh vertex shader
    /// will be used.
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's fragment shader. If [`ShaderRef::Default`] is returned, the default mesh fragment shader
    /// will be used.
    #[allow(unused_variables)]
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's [`AlphaMode`]. Defaults to [`AlphaMode::Opaque`].
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    /// Returns if this material should be rendered by the deferred or forward renderer.
    /// for `AlphaMode::Opaque` or `AlphaMode::Mask` materials.
    /// If `OpaqueRendererMethod::Auto`, it will default to what is selected in the `DefaultOpaqueRendererMethod` resource.
    #[inline]
    fn opaque_render_method(&self) -> OpaqueRendererMethod {
        OpaqueRendererMethod::Forward
    }

    #[inline]
    /// Add a bias to the view depth of the mesh which can be used to force a specific render order.
    /// for meshes with similar depth, to avoid z-fighting.
    /// The bias is in depth-texture units so large values may be needed to overcome small depth differences.
    fn depth_bias(&self) -> f32 {
        0.0
    }

    #[inline]
    /// Returns whether the material would like to read from [`ViewTransmissionTexture`](bevy_core_pipeline::core_3d::ViewTransmissionTexture).
    ///
    /// This allows taking color output from the [`Opaque3d`] pass as an input, (for screen-space transmission) but requires
    /// rendering to take place in a separate [`Transmissive3d`] pass.
    fn reads_view_transmission_texture(&self) -> bool {
        false
    }

    /// Returns this material's prepass vertex shader. If [`ShaderRef::Default`] is returned, the default prepass vertex shader
    /// will be used.
    ///
    /// This is used for the various [prepasses](bevy_core_pipeline::prepass) as well as for generating the depth maps
    /// required for shadow mapping.
    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's prepass fragment shader. If [`ShaderRef::Default`] is returned, the default prepass fragment shader
    /// will be used.
    ///
    /// This is used for the various [prepasses](bevy_core_pipeline::prepass) as well as for generating the depth maps
    /// required for shadow mapping.
    #[allow(unused_variables)]
    fn prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's deferred vertex shader. If [`ShaderRef::Default`] is returned, the default deferred vertex shader
    /// will be used.
    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's deferred fragment shader. If [`ShaderRef::Default`] is returned, the default deferred fragment shader
    /// will be used.
    #[allow(unused_variables)]
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's [`crate::meshlet::MeshletMesh`] fragment shader. If [`ShaderRef::Default`] is returned,
    /// the default meshlet mesh fragment shader will be used.
    ///
    /// This is part of an experimental feature, and is unnecessary to implement unless you are using `MeshletMesh`'s.
    ///
    /// See [`crate::meshlet::MeshletMesh`] for limitations.
    #[allow(unused_variables)]
    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's [`crate::meshlet::MeshletMesh`] prepass fragment shader. If [`ShaderRef::Default`] is returned,
    /// the default meshlet mesh prepass fragment shader will be used.
    ///
    /// This is part of an experimental feature, and is unnecessary to implement unless you are using `MeshletMesh`'s.
    ///
    /// See [`crate::meshlet::MeshletMesh`] for limitations.
    #[allow(unused_variables)]
    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_prepass_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Returns this material's [`crate::meshlet::MeshletMesh`] deferred fragment shader. If [`ShaderRef::Default`] is returned,
    /// the default meshlet mesh deferred fragment shader will be used.
    ///
    /// This is part of an experimental feature, and is unnecessary to implement unless you are using `MeshletMesh`'s.
    ///
    /// See [`crate::meshlet::MeshletMesh`] for limitations.
    #[allow(unused_variables)]
    #[cfg(feature = "meshlet")]
    fn meshlet_mesh_deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Default
    }

    /// Customizes the default [`RenderPipelineDescriptor`] for a specific entity using the entity's
    /// [`MaterialPipelineKey`] and [`MeshVertexBufferLayoutRef`] as input.
    #[allow(unused_variables)]
    #[inline]
    fn specialize(
        pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        Ok(())
    }
}

#[derive(QueryFilter)]
pub struct MeshMaterial3dNeedsSpecialization<M: Material> {
    _mesh: Or<(Changed<Mesh3d>, AssetChanged<Mesh3d>)>,
    _material: Or<(Changed<MeshMaterial3d<M>>, AssetChanged<MeshMaterial3d<M>>)>,
}

// impl<M> SpecializedMaterial for MeshMaterial3d<M>
// where
//     M: Material,
//     M::Data: PartialEq + Eq + Hash + Clone,
// {
//     type Param = (
//         (
//             SRes<ViewBinnedRenderPhases<Opaque3d>>,
//             SRes<ViewBinnedRenderPhases<AlphaMask3d>>,
//             SRes<ViewSortedRenderPhases<Transmissive3d>>,
//             SRes<ViewSortedRenderPhases<Transparent3d>>,
//         ),
//         (
//             SRes<RenderMaterialInstances<M>>,
//             SRes<RenderAssets<PreparedMaterial<M>>>,
//             SRes<RenderMeshInstances>,
//             SRes<RenderAssets<RenderMesh>>,
//             SRes<MaterialBindGroupAllocator<M>>,
//             SRes<RenderVisibilityRanges>,
//         ),
//     );
//     type Pipeline = MaterialPipeline<M>;
//     type ViewKey = MeshPipelineKey;
//     type VisibilityClass = Mesh3d;
//
//     fn should_specialize_view(
//         view_entity: Entity,
//         (
//             (
//                 opaque_render_phases,
//                 alpha_mask_render_phases,
//                 transmissive_render_phases,
//                 transparent_render_phases,
//             ),
//             _,
//         ): &SystemParamItem<Self::Param>,
//     ) -> bool {
//         opaque_render_phases.contains_key(&view_entity)
//             || alpha_mask_render_phases.contains_key(&view_entity)
//             || transmissive_render_phases.contains_key(&view_entity)
//             || transparent_render_phases.contains_key(&view_entity)
//     }
//
//     fn get_mesh_and_pipeline_key<'w>(
//         (_entity, main_entity): (Entity, MainEntity),
//         view_key: &Self::ViewKey,
//         (
//             _,
//             (
//                 render_material_instances,
//                 render_materials,
//                 render_mesh_instances,
//                 render_meshes,
//                 material_bind_group_allocator,
//                 render_visibility_ranges,
//             ),
//         ): &'w SystemParamItem<Self::Param>,
//     ) -> Option<(
//         &'w RenderMesh,
//         <Self::Pipeline as SpecializedMeshPipeline>::Key,
//     )> {
//         let Some(material_asset_id) = render_material_instances.get(&main_entity) else {
//             return None;
//         };
//         let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity) else {
//             return None;
//         };
//         let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
//             return None;
//         };
//         let Some(material) = render_materials.get(*material_asset_id) else {
//             return None;
//         };
//         let Some(material_bind_group) = material_bind_group_allocator.get(material.binding.group)
//         else {
//             return None;
//         };
//
//         let mut mesh_pipeline_key_bits = material.properties.mesh_pipeline_key_bits;
//         mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
//             material.properties.alpha_mode,
//             match view_key.msaa_samples() {
//                 1 => &Msaa::Off,
//                 2 => &Msaa::Sample2,
//                 4 => &Msaa::Sample4,
//                 8 => &Msaa::Sample8,
//                 _ => unreachable!("Unsupported MSAA sample count"),
//             },
//         ));
//         let mut mesh_key = *view_key
//             | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
//             | mesh_pipeline_key_bits;
//
//         if mesh_instance.shared.lightmap_slab_index.is_some() {
//             mesh_key |= MeshPipelineKey::LIGHTMAPPED;
//         }
//
//         if render_visibility_ranges.entity_has_crossfading_visibility_ranges(main_entity) {
//             mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
//         }
//
//         if view_key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
//             // If the previous frame have skins or morph targets, note that.
//             if mesh_instance
//                 .flags
//                 .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN)
//             {
//                 mesh_key |= MeshPipelineKey::HAS_PREVIOUS_SKIN;
//             }
//             if mesh_instance
//                 .flags
//                 .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH)
//             {
//                 mesh_key |= MeshPipelineKey::HAS_PREVIOUS_MORPH;
//             }
//         }
//
//         let key = MaterialPipelineKey {
//             mesh_key,
//             bind_group_data: material_bind_group
//                 .get_extra_data(material.binding.slot)
//                 .clone(),
//         };
//
//         Some((mesh, key))
//     }
// }

/// Adds the necessary ECS resources and render logic to enable rendering entities using the given [`Material`]
/// asset type.
pub struct MaterialPlugin<M: Material> {
    /// Controls if the prepass is enabled for the Material.
    /// For more information about what a prepass is, see the [`bevy_core_pipeline::prepass`] docs.
    ///
    /// When it is enabled, it will automatically add the [`PrepassPlugin`]
    /// required to make the prepass work on this Material.
    pub prepass_enabled: bool,
    /// Controls if shadows are enabled for the Material.
    pub shadows_enabled: bool,
    pub _marker: PhantomData<M>,
}

impl<M: Material> Default for MaterialPlugin<M> {
    fn default() -> Self {
        Self {
            prepass_enabled: true,
            shadows_enabled: true,
            _marker: Default::default(),
        }
    }
}

impl<M: Material> Plugin for MaterialPlugin<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        app.init_asset::<M>()
            .register_type::<MeshMaterial3d<M>>()
            .add_plugins((
                RenderAssetPlugin::<PreparedMaterial<M>>::default(),
                // SpecializationPlugin::<MeshMaterial3d<M>>::default(),
            ));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Shadow>>()
                .init_resource::<RenderMaterialInstances<M>>()
                .add_render_command::<Shadow, DrawPrepass<M>>()
                .add_render_command::<Transmissive3d, DrawMaterial<M>>()
                .add_render_command::<Transparent3d, DrawMaterial<M>>()
                .add_render_command::<Opaque3d, DrawMaterial<M>>()
                .add_render_command::<AlphaMask3d, DrawMaterial<M>>()
                .init_resource::<SpecializedMeshPipelines<MaterialPipeline<M>>>()
                .add_systems(ExtractSchedule, extract_mesh_materials::<M>)
                .add_systems(
                    Render,
                    queue_material_meshes::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>),
                )
                .add_systems(
                    Render,
                    prepare_material_bind_groups::<M>
                        .in_set(RenderSet::PrepareBindGroups)
                        .after(prepare_assets::<PreparedMaterial<M>>),
                );

            if self.shadows_enabled {
                render_app.add_systems(
                    Render,
                    queue_shadows::<M>
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<M>>),
                );
            }

            #[cfg(feature = "meshlet")]
            render_app.add_systems(
                Render,
                queue_material_meshlet_meshes::<M>
                    .in_set(RenderSet::QueueMeshes)
                    .run_if(resource_exists::<InstanceManager>),
            );

            #[cfg(feature = "meshlet")]
            render_app.add_systems(
                Render,
                prepare_material_meshlet_meshes_main_opaque_pass::<M>
                    .in_set(RenderSet::QueueMeshes)
                    .after(prepare_assets::<PreparedMaterial<M>>)
                    .before(queue_material_meshlet_meshes::<M>)
                    .run_if(resource_exists::<InstanceManager>),
            );
        }

        if self.shadows_enabled || self.prepass_enabled {
            // PrepassPipelinePlugin is required for shadow mapping and the optional PrepassPlugin
            app.add_plugins(PrepassPipelinePlugin::<M>::default());
        }

        if self.prepass_enabled {
            app.add_plugins(PrepassPlugin::<M>::default());
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MaterialPipeline<M>>()
                .init_resource::<MaterialBindGroupAllocator<M>>();
        }
    }
}

/// A key uniquely identifying a specialized [`MaterialPipeline`].
pub struct MaterialPipelineKey<M: Material> {
    pub mesh_key: MeshPipelineKey,
    pub bind_group_data: M::Data,
}

impl<M: Material> Eq for MaterialPipelineKey<M> where M::Data: PartialEq {}

impl<M: Material> PartialEq for MaterialPipelineKey<M>
where
    M::Data: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.mesh_key == other.mesh_key && self.bind_group_data == other.bind_group_data
    }
}

impl<M: Material> Clone for MaterialPipelineKey<M>
where
    M::Data: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mesh_key: self.mesh_key,
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

impl<M: Material> Hash for MaterialPipelineKey<M>
where
    M::Data: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.mesh_key.hash(state);
        self.bind_group_data.hash(state);
    }
}

#[derive(Clone, Copy)]
pub enum RenderPhaseType {
    Opaque,
    AlphaMask,
    Transmissive,
    Transparent,
}

/// Render pipeline data for a given [`Material`].
#[derive(Resource)]
pub struct MaterialPipeline<M: Material> {
    pub mesh_pipeline: MeshPipeline,
    pub material_layout: BindGroupLayout,
    pub vertex_shader: Option<Handle<Shader>>,
    pub fragment_shader: Option<Handle<Shader>>,
    /// Whether this material *actually* uses bindless resources, taking the
    /// platform support (or lack thereof) of bindless resources into account.
    pub bindless: bool,
    pub marker: PhantomData<M>,
}

impl<M: Material> Clone for MaterialPipeline<M> {
    fn clone(&self) -> Self {
        Self {
            mesh_pipeline: self.mesh_pipeline.clone(),
            material_layout: self.material_layout.clone(),
            vertex_shader: self.vertex_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),
            bindless: self.bindless,
            marker: PhantomData,
        }
    }
}

impl<M: Material> SpecializedMeshPipeline for MaterialPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = MaterialPipelineKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key.mesh_key, layout)?;
        if let Some(vertex_shader) = &self.vertex_shader {
            descriptor.vertex.shader = vertex_shader.clone();
        }

        if let Some(fragment_shader) = &self.fragment_shader {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader.clone();
        }

        descriptor.layout.insert(2, self.material_layout.clone());

        M::specialize(self, &mut descriptor, layout, key)?;

        // If bindless mode is on, add a `BINDLESS` define.
        if self.bindless {
            descriptor.vertex.shader_defs.push("BINDLESS".into());
            if let Some(ref mut fragment) = descriptor.fragment {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }

        Ok(descriptor)
    }
}

impl<M: Material> FromWorld for MaterialPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        MaterialPipeline {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            material_layout: M::bind_group_layout(render_device),
            vertex_shader: match M::vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            fragment_shader: match M::fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            bindless: material_bind_groups::material_uses_bindless_resources::<M>(render_device),
            marker: PhantomData,
        }
    }
}

type DrawMaterial<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetMaterialBindGroup<M, 2>,
    DrawMesh,
);

/// Sets the bind group for a given [`Material`] at the configured `I` index.
pub struct SetMaterialBindGroup<M: Material, const I: usize>(PhantomData<M>);
impl<P: PhaseItem, M: Material, const I: usize> RenderCommand<P> for SetMaterialBindGroup<M, I> {
    type Param = (
        SRes<RenderAssets<PreparedMaterial<M>>>,
        SRes<RenderMaterialInstances<M>>,
        SRes<MaterialBindGroupAllocator<M>>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (materials, material_instances, material_bind_group_allocator): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let materials = materials.into_inner();
        let material_instances = material_instances.into_inner();
        let material_bind_group_allocator = material_bind_group_allocator.into_inner();

        let Some(material_asset_id) = material_instances.get(&item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        let Some(material) = materials.get(*material_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(material_bind_group) = material_bind_group_allocator.get(material.binding.group)
        else {
            return RenderCommandResult::Skip;
        };
        let Some(bind_group) = material_bind_group.get_bind_group() else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, bind_group, &[]);
        RenderCommandResult::Success
    }
}

/// Stores all extracted instances of a [`Material`] in the render world.
#[derive(Resource, Deref, DerefMut)]
pub struct RenderMaterialInstances<M: Material>(pub MainEntityHashMap<AssetId<M>>);

impl<M: Material> Default for RenderMaterialInstances<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

pub const fn alpha_mode_pipeline_key(alpha_mode: AlphaMode, msaa: &Msaa) -> MeshPipelineKey {
    match alpha_mode {
        // Premultiplied and Add share the same pipeline key
        // They're made distinct in the PBR shader, via `premultiply_alpha()`
        AlphaMode::Premultiplied | AlphaMode::Add => MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA,
        AlphaMode::Blend => MeshPipelineKey::BLEND_ALPHA,
        AlphaMode::Multiply => MeshPipelineKey::BLEND_MULTIPLY,
        AlphaMode::Mask(_) => MeshPipelineKey::MAY_DISCARD,
        AlphaMode::AlphaToCoverage => match *msaa {
            Msaa::Off => MeshPipelineKey::MAY_DISCARD,
            _ => MeshPipelineKey::BLEND_ALPHA_TO_COVERAGE,
        },
        _ => MeshPipelineKey::NONE,
    }
}

pub const fn tonemapping_pipeline_key(tonemapping: Tonemapping) -> MeshPipelineKey {
    match tonemapping {
        Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
        Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
        Tonemapping::ReinhardLuminance => MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE,
        Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
        Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
        Tonemapping::SomewhatBoringDisplayTransform => {
            MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
        }
        Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
        Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
    }
}

pub const fn screen_space_specular_transmission_pipeline_key(
    screen_space_transmissive_blur_quality: ScreenSpaceTransmissionQuality,
) -> MeshPipelineKey {
    match screen_space_transmissive_blur_quality {
        ScreenSpaceTransmissionQuality::Low => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_LOW
        }
        ScreenSpaceTransmissionQuality::Medium => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_MEDIUM
        }
        ScreenSpaceTransmissionQuality::High => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_HIGH
        }
        ScreenSpaceTransmissionQuality::Ultra => {
            MeshPipelineKey::SCREEN_SPACE_SPECULAR_TRANSMISSION_ULTRA
        }
    }
}

fn extract_mesh_materials<M: Material>(
    mut material_instances: ResMut<RenderMaterialInstances<M>>,
    mut material_ids: ResMut<RenderMeshMaterialIds>,
    mut material_bind_group_allocator: ResMut<MaterialBindGroupAllocator<M>>,
    query: Extract<Query<(Entity, &ViewVisibility, &MeshMaterial3d<M>)>>,
) {
    material_instances.clear();

    for (entity, view_visibility, material) in &query {
        if view_visibility.get() {
            material_instances.insert(entity.into(), material.id());

            // Allocate a slot for this material in the bind group.
            let material_id = material.id().untyped();
            material_ids
                .mesh_to_material
                .insert(entity.into(), material_id);
            if let Entry::Vacant(entry) = material_ids.material_to_binding.entry(material_id) {
                entry.insert(material_bind_group_allocator.allocate());
            }
        }
    }
}

/// For each view, iterates over all the meshes visible from that view and adds
/// them to [`BinnedRenderPhase`]s or [`SortedRenderPhase`]s as appropriate.
#[allow(clippy::too_many_arguments)]
pub fn queue_material_meshes<M: Material>(
    specialized_pipeline_cache: Res<SpecializedMaterialPipelineCache<MeshMaterial3d<M>>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    mesh_allocator: Res<MeshAllocator>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    mut alpha_mask_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    mut transmissive_render_phases: ResMut<ViewSortedRenderPhases<Transmissive3d>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(Entity, &ExtractedView, &RenderVisibleEntities)>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (view_entity, view, visible_entities) in &views {
        let (
            Some(opaque_phase),
            Some(alpha_mask_phase),
            Some(transmissive_phase),
            Some(transparent_phase),
        ) = (
            opaque_render_phases.get_mut(&view_entity),
            alpha_mask_render_phases.get_mut(&view_entity),
            transmissive_render_phases.get_mut(&view_entity),
            transparent_render_phases.get_mut(&view_entity),
        )
        else {
            continue;
        };

        let rangefinder = view.rangefinder3d();
        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            let Some(pipeline_id) = specialized_pipeline_cache.get(&(view_entity, *visible_entity))
            else {
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };

            match material.properties.render_phase_type {
                RenderPhaseType::Transmissive => {
                    let distance = rangefinder.distance_translation(&mesh_instance.translation)
                        + material.properties.depth_bias;
                    transmissive_phase.add(Transmissive3d {
                        entity: (*render_entity, *visible_entity),
                        draw_function: material.properties.draw_function_id,
                        pipeline: *pipeline_id,
                        distance,
                        batch_range: 0..1,
                        extra_index: PhaseItemExtraIndex::None,
                    });
                }
                RenderPhaseType::Opaque => {
                    if material.properties.render_method == OpaqueRendererMethod::Forward {
                        let (vertex_slab, index_slab) =
                            mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);
                        let bin_key = Opaque3dBinKey {
                            batch_set_key: Opaque3dBatchSetKey {
                                draw_function: material.properties.draw_function_id,
                                pipeline: *pipeline_id,
                                material_bind_group_index: Some(material.binding.group.0),
                                vertex_slab: vertex_slab.unwrap_or_default(),
                                index_slab,
                                lightmap_slab: mesh_instance
                                    .shared
                                    .lightmap_slab_index
                                    .map(|index| *index),
                            },
                            asset_id: mesh_instance.mesh_asset_id.into(),
                        };
                        opaque_phase.add(
                            bin_key,
                            (*render_entity, *visible_entity),
                            BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                        );
                    }
                }
                // Alpha mask
                RenderPhaseType::AlphaMask => {
                    let bin_key = OpaqueNoLightmap3dBinKey {
                        batch_set_key: OpaqueNoLightmap3dBatchSetKey {
                            draw_function: material.properties.draw_function_id,
                            pipeline: *pipeline_id,
                            material_bind_group_index: Some(material.binding.group.0),
                        },
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    };
                    alpha_mask_phase.add(
                        bin_key,
                        (*render_entity, *visible_entity),
                        BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                    );
                }
                RenderPhaseType::Transparent => {
                    let distance = rangefinder.distance_translation(&mesh_instance.translation)
                        + material.properties.depth_bias;
                    transparent_phase.add(Transparent3d {
                        entity: (*render_entity, *visible_entity),
                        draw_function: material.properties.draw_function_id,
                        pipeline: *pipeline_id,
                        distance,
                        batch_range: 0..1,
                        extra_index: PhaseItemExtraIndex::None,
                    });
                }
            }
        }
    }
}

/// Default render method used for opaque materials.
#[derive(Default, Resource, Clone, Debug, ExtractResource, Reflect)]
#[reflect(Resource, Default, Debug)]
pub struct DefaultOpaqueRendererMethod(OpaqueRendererMethod);

impl DefaultOpaqueRendererMethod {
    pub fn forward() -> Self {
        DefaultOpaqueRendererMethod(OpaqueRendererMethod::Forward)
    }

    pub fn deferred() -> Self {
        DefaultOpaqueRendererMethod(OpaqueRendererMethod::Deferred)
    }

    pub fn set_to_forward(&mut self) {
        self.0 = OpaqueRendererMethod::Forward;
    }

    pub fn set_to_deferred(&mut self) {
        self.0 = OpaqueRendererMethod::Deferred;
    }
}

/// Render method used for opaque materials.
///
/// The forward rendering main pass draws each mesh entity and shades it according to its
/// corresponding material and the lights that affect it. Some render features like Screen Space
/// Ambient Occlusion require running depth and normal prepasses, that are 'deferred'-like
/// prepasses over all mesh entities to populate depth and normal textures. This means that when
/// using render features that require running prepasses, multiple passes over all visible geometry
/// are required. This can be slow if there is a lot of geometry that cannot be batched into few
/// draws.
///
/// Deferred rendering runs a prepass to gather not only geometric information like depth and
/// normals, but also all the material properties like base color, emissive color, reflectance,
/// metalness, etc, and writes them into a deferred 'g-buffer' texture. The deferred main pass is
/// then a fullscreen pass that reads data from these textures and executes shading. This allows
/// for one pass over geometry, but is at the cost of not being able to use MSAA, and has heavier
/// bandwidth usage which can be unsuitable for low end mobile or other bandwidth-constrained devices.
///
/// If a material indicates `OpaqueRendererMethod::Auto`, `DefaultOpaqueRendererMethod` will be used.
#[derive(Default, Clone, Copy, Debug, PartialEq, Reflect)]
pub enum OpaqueRendererMethod {
    #[default]
    Forward,
    Deferred,
    Auto,
}

/// Common [`Material`] properties, calculated for a specific material instance.
pub struct MaterialProperties {
    /// Is this material should be rendered by the deferred renderer when.
    /// [`AlphaMode::Opaque`] or [`AlphaMode::Mask`]
    pub render_method: OpaqueRendererMethod,
    /// The [`AlphaMode`] of this material.
    pub alpha_mode: AlphaMode,
    /// The bits in the [`MeshPipelineKey`] for this material.
    ///
    /// These are precalculated so that we can just "or" them together in
    /// [`queue_material_meshes`].
    pub mesh_pipeline_key_bits: MeshPipelineKey,
    /// Add a bias to the view depth of the mesh which can be used to force a specific render order
    /// for meshes with equal depth, to avoid z-fighting.
    /// The bias is in depth-texture units so large values may be needed to overcome small depth differences.
    pub depth_bias: f32,
    /// Whether the material would like to read from [`ViewTransmissionTexture`](bevy_core_pipeline::core_3d::ViewTransmissionTexture).
    ///
    /// This allows taking color output from the [`Opaque3d`] pass as an input, (for screen-space transmission) but requires
    /// rendering to take place in a separate [`Transmissive3d`] pass.
    pub reads_view_transmission_texture: bool,
    pub render_phase_type: RenderPhaseType,
    pub draw_function_id: DrawFunctionId,
}

/// Data prepared for a [`Material`] instance.
pub struct PreparedMaterial<M: Material> {
    pub binding: MaterialBindingId,
    pub properties: MaterialProperties,
    pub phantom: PhantomData<M>,
}

impl<M: Material> RenderAsset for PreparedMaterial<M> {
    type SourceAsset = M;

    type Param = (
        SRes<RenderDevice>,
        SRes<MaterialPipeline<M>>,
        SRes<DefaultOpaqueRendererMethod>,
        SRes<RenderMeshMaterialIds>,
        SResMut<MaterialBindGroupAllocator<M>>,
        SRes<DrawFunctions<Opaque3d>>,
        SRes<DrawFunctions<AlphaMask3d>>,
        SRes<DrawFunctions<Transmissive3d>>,
        SRes<DrawFunctions<Transparent3d>>,
        M::Param,
    );

    fn prepare_asset(
        material: Self::SourceAsset,
        material_id: AssetId<Self::SourceAsset>,
        (
            render_device,
            pipeline,
            default_opaque_render_method,
            mesh_material_ids,
            ref mut bind_group_allocator,
            opaque_draw_functions,
            alpha_mask_draw_functions,
            transmissive_draw_functions,
            transparent_draw_functions,
            ref mut material_param,
        ): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        // Fetch the material binding ID, so that we can write it in to the
        // `PreparedMaterial`.
        let Some(material_binding_id) = mesh_material_ids
            .material_to_binding
            .get(&material_id.untyped())
        else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let draw_opaque_pbr = opaque_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_alpha_mask_pbr = alpha_mask_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_transmissive_pbr = transmissive_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_transparent_pbr = transparent_draw_functions.read().id::<DrawMaterial<M>>();

        let render_method = match material.opaque_render_method() {
            OpaqueRendererMethod::Forward => OpaqueRendererMethod::Forward,
            OpaqueRendererMethod::Deferred => OpaqueRendererMethod::Deferred,
            OpaqueRendererMethod::Auto => default_opaque_render_method.0,
        };

        let mut mesh_pipeline_key_bits = MeshPipelineKey::empty();
        mesh_pipeline_key_bits.set(
            MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE,
            material.reads_view_transmission_texture(),
        );
        let reads_view_transmission_texture =
            mesh_pipeline_key_bits.contains(MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE);

        let forward = match render_method {
            OpaqueRendererMethod::Forward => true,
            OpaqueRendererMethod::Deferred => false,
            OpaqueRendererMethod::Auto => unreachable!(),
        };

        let render_phase_type = match material.alpha_mode() {
            AlphaMode::Opaque => {
                if reads_view_transmission_texture {
                    RenderPhaseType::Transmissive
                } else if forward {
                    RenderPhaseType::Opaque
                } else {
                    panic!("Invalid opaque configuration");
                }
            }
            AlphaMode::Mask(_) => {
                if reads_view_transmission_texture {
                    RenderPhaseType::Transmissive
                } else if forward {
                    RenderPhaseType::AlphaMask
                } else {
                    panic!("Invalid alpha mask configuration");
                }
            }
            AlphaMode::Blend
            | AlphaMode::Premultiplied
            | AlphaMode::Add
            | AlphaMode::Multiply
            | AlphaMode::AlphaToCoverage => RenderPhaseType::Transparent,
        };

        let draw_function_id = match render_phase_type {
            RenderPhaseType::Opaque => draw_opaque_pbr,
            RenderPhaseType::AlphaMask => draw_alpha_mask_pbr,
            RenderPhaseType::Transmissive => draw_transmissive_pbr,
            RenderPhaseType::Transparent => draw_transparent_pbr,
        };

        match material.unprepared_bind_group(
            &pipeline.material_layout,
            render_device,
            material_param,
            false,
        ) {
            Ok(unprepared) => {
                bind_group_allocator.init(render_device, *material_binding_id, unprepared);

                Ok(PreparedMaterial {
                    binding: *material_binding_id,
                    properties: MaterialProperties {
                        alpha_mode: material.alpha_mode(),
                        depth_bias: material.depth_bias(),
                        reads_view_transmission_texture,
                        render_phase_type,
                        render_method,
                        mesh_pipeline_key_bits,
                        draw_function_id,
                    },
                    phantom: PhantomData,
                })
            }

            Err(AsBindGroupError::RetryNextUpdate) => {
                Err(PrepareAssetError::RetryNextUpdate(material))
            }

            Err(AsBindGroupError::CreateBindGroupDirectly) => {
                // This material has opted out of automatic bind group creation
                // and is requesting a fully-custom bind group. Invoke
                // `as_bind_group` as requested, and store the resulting bind
                // group in the slot.
                match material.as_bind_group(
                    &pipeline.material_layout,
                    render_device,
                    material_param,
                ) {
                    Ok(prepared_bind_group) => {
                        // Store the resulting bind group directly in the slot.
                        bind_group_allocator.init_custom(
                            *material_binding_id,
                            prepared_bind_group.bind_group,
                            prepared_bind_group.data,
                        );

                        Ok(PreparedMaterial {
                            binding: *material_binding_id,
                            properties: MaterialProperties {
                                alpha_mode: material.alpha_mode(),
                                depth_bias: material.depth_bias(),
                                reads_view_transmission_texture: mesh_pipeline_key_bits
                                    .contains(MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE),
                                render_method,
                                mesh_pipeline_key_bits,
                                draw_function_id,
                                render_phase_type,
                            },
                            phantom: PhantomData,
                        })
                    }

                    Err(AsBindGroupError::RetryNextUpdate) => {
                        Err(PrepareAssetError::RetryNextUpdate(material))
                    }

                    Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
                }
            }

            Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
        }
    }

    fn unload_asset(
        asset_id: AssetId<Self::SourceAsset>,
        (_, _, _, mesh_material_ids, ref mut bind_group_allocator, _, _, _, _, _): &mut SystemParamItem<
            Self::Param,
        >,
    ) {
        // Mark this material's slot in the binding array as free.

        let Some(material_binding_id) = mesh_material_ids
            .material_to_binding
            .get(&asset_id.untyped())
        else {
            return;
        };

        bind_group_allocator.free(*material_binding_id);
    }
}

#[derive(Component, Clone, Copy, Default, PartialEq, Eq, Deref, DerefMut)]
pub struct MaterialBindGroupId(pub Option<BindGroupId>);

impl MaterialBindGroupId {
    pub fn new(id: BindGroupId) -> Self {
        Self(Some(id))
    }
}

impl From<BindGroup> for MaterialBindGroupId {
    fn from(value: BindGroup) -> Self {
        Self::new(value.id())
    }
}

/// A system that creates and/or recreates any bind groups that contain
/// materials that were modified this frame.
pub fn prepare_material_bind_groups<M>(
    mut allocator: ResMut<MaterialBindGroupAllocator<M>>,
    render_device: Res<RenderDevice>,
    fallback_image: Res<FallbackImage>,
    fallback_resources: Res<FallbackBindlessResources>,
) where
    M: Material,
{
    allocator.prepare_bind_groups(&render_device, &fallback_image, &fallback_resources);
}
