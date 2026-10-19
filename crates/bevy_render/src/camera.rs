use core::mem;

use crate::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    extract_component::ExtractComponent,
    extract_resource::{extract_resource, ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::TextureView,
    renderer::{RenderAdapter, RenderDevice},
    sync_component::{SyncComponent, SyncComponentPlugin},
    sync_world::{MainEntity, MainEntityHashSet, RenderEntity, SyncToRenderWorld},
    texture::{GpuImage, ManualTextureViews},
    view::{
        texture_format_features, texture_format_is_hdr, validate_main_texture_format, ColorGrading,
        ExtractedView, ExtractedWindow, MainTextureAllocation, Msaa, NoIndirectDrawing,
        RenderExtractedVisibleEntities, RenderVisibleEntities, RenderVisibleEntitiesClass,
        ResolvedCompositingSpace, RetainedViewEntity, Tonemapping, ViewMainTextureUsages,
        ViewTarget, ViewUniformOffset, VisibilityExtractionSystemParam,
    },
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};

use bevy_app::{App, Plugin, PostStartup, PostUpdate};
use bevy_asset::{AssetEvent, AssetEventSystems, AssetId, Assets};
use bevy_camera::{
    primitives::Frustum,
    visibility::{self, RenderLayers, VisibleEntities},
    Camera, Camera2d, Camera3d, CameraMainTexture, CameraOutputMode, CameraUpdateSystems,
    ClearColor, ClearColorConfig, CompositingSpace, Exposure, Hdr, MainTexture, MainTextureFilter,
    ManualTextureViewHandle, MsaaWriteback, NormalizedRenderTarget, PostProcessCamera, Projection,
    RenderTarget, RenderTargetInfo, TonemappingPass, Viewport,
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    change_detection::DetectChanges,
    component::Component,
    entity::{ContainsEntity, Entity, EntityHashMap, EntityHashSet},
    error::BevyError,
    lifecycle::HookContext,
    message::MessageReader,
    prelude::With,
    query::{Has, QueryItem},
    reflect::ReflectComponent,
    resource::Resource,
    schedule::{InternedScheduleLabel, IntoScheduleConfigs, ScheduleLabel, SystemSet},
    system::{Commands, Local, Query, Res, ResMut},
    world::DeferredWorld,
};
use bevy_image::Image;
use bevy_log::warn;
use bevy_log::warn_once;
use bevy_math::{uvec2, vec2, Mat4, URect, UVec2, UVec4, Vec2};
use bevy_platform::collections::{HashMap, HashSet};
use bevy_reflect::prelude::*;
use bevy_transform::components::GlobalTransform;
use bevy_window::{PrimaryWindow, Window, WindowCreated, WindowResized, WindowScaleFactorChanged};
use itertools::Either;
use wgpu::{FilterMode, TextureFormat, TextureFormatFeatureFlags};

/// Main-pass color [`TextureFormat`] keyed by camera render entity.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct CameraMainPassTextureFormats(pub EntityHashMap<TextureFormat>);

#[derive(Default)]
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Camera, Msaa>()
            .register_required_components::<Camera, SyncToRenderWorld>()
            .register_required_components::<Camera3d, ColorGrading>()
            .register_required_components::<PostProcessCamera, ColorGrading>()
            .register_required_components::<Camera3d, Exposure>()
            .add_plugins((
                ExtractResourcePlugin::<ClearColor>::default(),
                SyncComponentPlugin::<MainTexture>::default(),
            ))
            .add_systems(PostStartup, camera_system.in_set(CameraUpdateSystems))
            .add_systems(
                PostUpdate,
                camera_system
                    .in_set(CameraUpdateSystems)
                    .before(AssetEventSystems)
                    .before(visibility::update_frusta),
            );
        app.world_mut()
            .register_component_hooks::<Camera>()
            .on_add(warn_on_no_render_graph);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<CameraMainPassTextureFormats>()
                .init_resource::<SortedCameras>()
                .init_resource::<DirtySpecializations>()
                .init_resource::<DirtyWireframeSpecializations>()
                .allow_ambiguous_resource::<DirtySpecializations>()
                .allow_ambiguous_resource::<DirtyWireframeSpecializations>()
                .configure_sets(
                    ExtractSchedule,
                    (
                        DirtySpecializationSystems::Clear
                            .before_weak(DirtySpecializationSystems::CheckForChanges),
                        DirtySpecializationSystems::CheckForChanges
                            .before_weak(DirtySpecializationSystems::CheckForRemovals),
                    ),
                )
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_cameras
                            .after(extract_resource::<ManualTextureViews, RenderApp, ()>),
                        clear_dirty_specializations.in_set(DirtySpecializationSystems::Clear),
                        clear_dirty_wireframe_specializations
                            .in_set(DirtySpecializationSystems::Clear),
                        expire_specializations_for_views.in_set(RenderSystems::Cleanup),
                        expire_wireframe_specializations_for_views.in_set(RenderSystems::Cleanup),
                    ),
                )
                .add_systems(Render, sort_cameras.in_set(RenderSystems::CreateViews));
        }
    }
}

fn warn_on_no_render_graph(world: DeferredWorld, HookContext { entity, caller, .. }: HookContext) {
    if !world.entity(entity).contains::<CameraRenderGraph>() {
        warn!("{}Entity {entity} has a `Camera` component, but it doesn't have a render graph configured. Usually, adding a `Camera2d` or `Camera3d` component will work.
        However, you may instead need to enable `bevy_core_pipeline`, or may want to manually add a `CameraRenderGraph` component to create a custom render graph.", caller.map(|location|format!("{location}: ")).unwrap_or_default());
    }
}

impl ExtractResource<RenderApp> for ClearColor {
    type Source = Self;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

// Removing a `MainTexture` frees its textures.
impl SyncComponent<RenderApp> for MainTexture {
    type Target = MainTextureAllocation;
}

impl SyncComponent<RenderApp> for Camera2d {
    type Target = Self;
}

impl ExtractComponent<RenderApp> for Camera2d {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    type Out = Self;

    fn extract_component(item: QueryItem<Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

impl SyncComponent<RenderApp> for Camera3d {
    type Target = Self;
}

impl ExtractComponent<RenderApp> for Camera3d {
    type QueryData = &'static Self;
    type QueryFilter = With<Camera>;
    type Out = Self;

    fn extract_component(item: QueryItem<Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Configures the render schedule to be run for a given [`Camera`] entity.
#[derive(Component, Debug, Deref, DerefMut, Reflect, Clone)]
#[reflect(opaque)]
#[reflect(Component, Debug, Clone)]
pub struct CameraRenderGraph(pub InternedScheduleLabel);

impl CameraRenderGraph {
    /// Creates a new [`CameraRenderGraph`] from a schedule label.
    #[inline]
    pub fn new<T: ScheduleLabel>(schedule: T) -> Self {
        Self(schedule.intern())
    }

    /// Sets the schedule.
    #[inline]
    pub fn set<T: ScheduleLabel>(&mut self, schedule: T) {
        self.0 = schedule.intern();
    }
}

pub trait NormalizedRenderTargetExt {
    fn get_texture_view<'a>(
        &self,
        windows: &'a Query<(MainEntity, &ExtractedWindow)>,
        images: &'a RenderAssets<GpuImage>,
        manual_texture_views: &'a ManualTextureViews,
    ) -> Option<&'a TextureView>;

    /// Retrieves the [`TextureFormat`] of this render target, if it exists.
    fn get_texture_view_format<'a>(
        &self,
        windows: &'a Query<(MainEntity, &ExtractedWindow)>,
        images: &'a RenderAssets<GpuImage>,
        manual_texture_views: &'a ManualTextureViews,
    ) -> Option<TextureFormat>;

    fn get_render_target_info<'a>(
        &self,
        resolutions: impl IntoIterator<Item = (Entity, &'a Window)>,
        images: &Assets<Image>,
        manual_texture_views: &ManualTextureViews,
    ) -> Result<RenderTargetInfo, MissingRenderTargetInfoError>;

    // Check if this render target is contained in the given changed windows or images.
    fn is_changed(
        &self,
        changed_window_ids: &EntityHashSet,
        changed_image_handles: &HashSet<&AssetId<Image>>,
    ) -> bool;
}

impl NormalizedRenderTargetExt for NormalizedRenderTarget {
    fn get_texture_view<'a>(
        &self,
        windows: &'a Query<(MainEntity, &ExtractedWindow)>,
        images: &'a RenderAssets<GpuImage>,
        manual_texture_views: &'a ManualTextureViews,
    ) -> Option<&'a TextureView> {
        match self {
            NormalizedRenderTarget::Window(window_ref) => windows
                .iter()
                .find(|(e, _)| *e == window_ref.entity())
                .and_then(|(_, window)| window.swap_chain_texture_view.as_ref()),
            NormalizedRenderTarget::Image(image_target) => images
                .get(&image_target.handle)
                .map(|image| &image.texture_view),
            NormalizedRenderTarget::TextureView(id) => {
                manual_texture_views.get(id).map(|tex| &tex.texture_view)
            }
            NormalizedRenderTarget::None { .. } => None,
        }
    }

    /// Retrieves the texture view's [`TextureFormat`] of this render target, if it exists.
    fn get_texture_view_format<'a>(
        &self,
        windows: &'a Query<(MainEntity, &ExtractedWindow)>,
        images: &'a RenderAssets<GpuImage>,
        manual_texture_views: &'a ManualTextureViews,
    ) -> Option<TextureFormat> {
        match self {
            NormalizedRenderTarget::Window(window_ref) => windows
                .iter()
                .find(|(e, _)| *e == window_ref.entity())
                .and_then(|(_, window)| window.swap_chain_texture_view_format),
            NormalizedRenderTarget::Image(image_target) => {
                images.get(&image_target.handle).map(GpuImage::view_format)
            }
            NormalizedRenderTarget::TextureView(id) => {
                manual_texture_views.get(id).map(|tex| tex.view_format)
            }
            NormalizedRenderTarget::None { .. } => None,
        }
    }

    fn get_render_target_info<'a>(
        &self,
        resolutions: impl IntoIterator<Item = (Entity, &'a Window)>,
        images: &Assets<Image>,
        manual_texture_views: &ManualTextureViews,
    ) -> Result<RenderTargetInfo, MissingRenderTargetInfoError> {
        match self {
            NormalizedRenderTarget::Window(window_ref) => resolutions
                .into_iter()
                .find(|(entity, _)| *entity == window_ref.entity())
                .map(|(_, window)| RenderTargetInfo {
                    physical_size: window.physical_size(),
                    scale_factor: window.resolution.scale_factor(),
                })
                .ok_or(MissingRenderTargetInfoError::Window {
                    window: window_ref.entity(),
                }),
            NormalizedRenderTarget::Image(image_target) => images
                .get(&image_target.handle)
                .map(|image| RenderTargetInfo {
                    physical_size: image.size(),
                    scale_factor: image_target.scale_factor,
                })
                .ok_or(MissingRenderTargetInfoError::Image {
                    image: image_target.handle.id(),
                }),
            NormalizedRenderTarget::TextureView(id) => manual_texture_views
                .get(id)
                .map(|tex| RenderTargetInfo {
                    physical_size: tex.size,
                    scale_factor: 1.0,
                })
                .ok_or(MissingRenderTargetInfoError::TextureView { texture_view: *id }),
            NormalizedRenderTarget::None { width, height } => Ok(RenderTargetInfo {
                physical_size: uvec2(*width, *height),
                scale_factor: 1.0,
            }),
        }
    }

    // Check if this render target is contained in the given changed windows or images.
    fn is_changed(
        &self,
        changed_window_ids: &EntityHashSet,
        changed_image_handles: &HashSet<&AssetId<Image>>,
    ) -> bool {
        match self {
            NormalizedRenderTarget::Window(window_ref) => {
                changed_window_ids.contains(&window_ref.entity())
            }
            NormalizedRenderTarget::Image(image_target) => {
                changed_image_handles.contains(&image_target.handle.id())
            }
            NormalizedRenderTarget::TextureView(_) | NormalizedRenderTarget::None { .. } => true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MissingRenderTargetInfoError {
    #[error("RenderTarget::Window missing ({window:?}): Make sure the provided entity has a Window component.")]
    Window { window: Entity },
    #[error("RenderTarget::Image missing ({image:?}): Make sure the Image's usages include RenderAssetUsages::MAIN_WORLD.")]
    Image { image: AssetId<Image> },
    #[error("RenderTarget::TextureView missing ({texture_view:?}): make sure the texture view handle was not removed.")]
    TextureView {
        texture_view: ManualTextureViewHandle,
    },
}

/// System in charge of updating a [`Camera`] when its window or projection changes.
///
/// The system detects window creation, resize, and scale factor change events to update the camera
/// [`Projection`] if needed.
///
/// ## World Resources
///
/// [`Res<Assets<Image>>`](Assets<Image>) -- For cameras that render to an image, this resource is used to
/// inspect information about the render target. This system will not access any other image assets.
///
/// [`OrthographicProjection`]: bevy_camera::OrthographicProjection
/// [`PerspectiveProjection`]: bevy_camera::PerspectiveProjection
pub fn camera_system(
    mut window_resized_reader: MessageReader<WindowResized>,
    mut window_created_reader: MessageReader<WindowCreated>,
    mut window_scale_factor_changed_reader: MessageReader<WindowScaleFactorChanged>,
    mut image_asset_event_reader: MessageReader<AssetEvent<Image>>,
    primary_window: Query<Entity, With<PrimaryWindow>>,
    windows: Query<(Entity, &Window)>,
    images: Res<Assets<Image>>,
    manual_texture_views: Res<ManualTextureViews>,
    mut cameras: Query<(
        &mut Camera,
        &RenderTarget,
        Option<&mut Projection>,
        Option<&CameraMainTexture>,
    )>,
    main_textures: Query<&MainTexture>,
    render_device: Option<Res<RenderDevice>>,
) -> Result<(), BevyError> {
    let max_texture_dimension = render_device.map_or(u32::MAX, |render_device| {
        render_device.limits().max_texture_dimension_2d
    });

    let primary_window = primary_window.iter().next();

    let mut changed_window_ids = EntityHashSet::default();
    changed_window_ids.extend(window_created_reader.read().map(|event| event.window));
    changed_window_ids.extend(window_resized_reader.read().map(|event| event.window));
    let scale_factor_changed_window_ids: EntityHashSet = window_scale_factor_changed_reader
        .read()
        .map(|event| event.window)
        .collect();
    changed_window_ids.extend(scale_factor_changed_window_ids.clone());

    let changed_image_handles: HashSet<&AssetId<Image>> = image_asset_event_reader
        .read()
        .filter_map(|event| match event {
            AssetEvent::Modified { id } | AssetEvent::Added { id } => Some(id),
            _ => None,
        })
        .collect();

    for (mut camera, render_target, mut camera_projection, camera_main_texture) in &mut cameras {
        let mut viewport_size = camera
            .viewport
            .as_ref()
            .map(|viewport| viewport.physical_size);

        if let Some(normalized_target) = render_target.normalize(primary_window)
            && (normalized_target.is_changed(&changed_window_ids, &changed_image_handles)
                || camera.is_added()
                || camera_projection
                    .as_ref()
                    .is_some_and(DetectChanges::is_changed)
                || camera.computed.old_viewport_size != viewport_size
                || camera.computed.old_sub_camera_view != camera.sub_camera_view)
        {
            let new_computed_target_info = normalized_target.get_render_target_info(
                windows,
                &images,
                &manual_texture_views,
            )?;
            // Check for the scale factor changing, and resize the viewport if needed.
            // This can happen when the window is moved between monitors with different DPIs.
            // Without this, the viewport will take a smaller portion of the window moved to
            // a higher DPI monitor.
            if normalized_target.is_changed(&scale_factor_changed_window_ids, &HashSet::default())
                && let Some(old_scale_factor) = camera
                    .computed
                    .target_info
                    .as_ref()
                    .map(|info| info.scale_factor)
            {
                let resize_factor = new_computed_target_info.scale_factor / old_scale_factor;
                if let Some(ref mut viewport) = camera.viewport {
                    let resize = |vec: UVec2| (vec.as_vec2() * resize_factor).as_uvec2();
                    viewport.physical_position = resize(viewport.physical_position);
                    viewport.physical_size = resize(viewport.physical_size);
                    viewport_size = Some(viewport.physical_size);
                }
            }
            // This check is needed because when changing WindowMode to Fullscreen, the viewport may have invalid
            // arguments due to a sudden change on the window size to a lower value.
            // If the size of the window is lower, the viewport will match that lower value.
            if let Some(viewport) = &mut camera.viewport {
                viewport.clamp_to_size(new_computed_target_info.physical_size);
            }
            camera.computed.target_info = Some(new_computed_target_info);
            if let Some(camera_projection) = camera_projection.as_mut()
                && let Some(size) = camera.logical_viewport_size()
                && size.x != 0.0
                && size.y != 0.0
            {
                camera_projection.update(size.x, size.y);
                camera.computed.clip_from_view = match &camera.sub_camera_view {
                    Some(sub_view) => camera_projection.get_clip_from_view_for_sub(sub_view),
                    None => camera_projection.get_clip_from_view(),
                }
            }
        }

        let render_size = camera_main_texture
            .and_then(|main_texture| main_textures.get(main_texture.0).ok())
            .zip(camera.physical_target_size())
            .and_then(|(main_texture, target_size)| {
                main_texture
                    .size
                    .resolve(target_size, max_texture_dimension)
                    .ok()
            })
            .filter(|render_size| Some(*render_size) != camera.physical_target_size());
        if camera.computed.render_size != render_size {
            camera.computed.render_size = render_size;
        }

        if camera.computed.old_viewport_size != viewport_size {
            camera.computed.old_viewport_size = viewport_size;
        }

        if camera.computed.old_sub_camera_view != camera.sub_camera_view {
            camera.computed.old_sub_camera_view = camera.sub_camera_view;
        }
    }
    Ok(())
}

/// Describes a [`Camera`] in the render world.
///
/// Every `ExtractedCamera` also has an [`ExtractedView`], but not every
/// view comes from a camera. For example, views can come from lights,
/// for drawing shadow maps.
#[derive(Component, Debug)]
#[require(RenderVisibleEntities)]
pub struct ExtractedCamera {
    pub target: Option<NormalizedRenderTarget>,
    /// The size of [`ExtractedCamera::render_viewport`], or of the whole main texture when there
    /// is no viewport.
    pub physical_render_viewport_size: Option<UVec2>,
    /// The size the camera renders at, which is the size of its main texture. This is the size of
    /// the render target, unless the camera has a [`MainTexture`] with another
    /// [`MainTextureSize`](bevy_camera::MainTextureSize).
    pub physical_render_size: Option<UVec2>,
    /// The camera's viewport within its main texture.
    pub render_viewport: Option<Viewport>,
    /// The camera's viewport within its render target. It differs from
    /// [`ExtractedCamera::render_viewport`] when the main texture and the render target have
    /// different sizes.
    pub target_viewport: Option<Viewport>,
    pub schedule: InternedScheduleLabel,
    pub order: isize,
    pub output_mode: CameraOutputMode,
    pub msaa_writeback: MsaaWriteback,
    pub clear_color: ClearColorConfig,
    /// The camera's position among all cameras, in the order they render.
    pub sorted_camera_index: usize,
    pub sorted_camera_index_for_target: usize,
    pub exposure: f32,
    pub hdr: bool,
    /// Whether the camera tonemaps in its material shaders instead of in the
    /// tonemapping pass.
    ///
    /// This is true for SDR cameras with tonemapping enabled, unless the camera has
    /// [`TonemappingPass`], a non-linear [`CompositingSpace`], or a non-window render
    /// target. Those exceptions don't apply when the render target is shared by
    /// multiple cameras.
    ///
    /// A camera with a [`CameraMainTexture`] doesn't follow these rules, unless it decides the
    /// format of the texture. It runs the tonemapping pass when the format is an HDR format, and
    /// tonemaps in its material shaders otherwise.
    pub tonemap_in_shader: bool,
}

/// The [`MainTexture`] a camera view renders into, for cameras with a [`CameraMainTexture`].
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtractedCameraMainTexture {
    /// The render world entity of the [`MainTexture`].
    pub entity: Entity,
    /// How the texture is sampled when it is written to the render target. This is
    /// [`FilterMode::Nearest`] when the texture and the render target have the same size.
    pub filter: FilterMode,
}

type ExtractedCameraComponents = (
    ExtractedCamera,
    ExtractedView,
    ResolvedCompositingSpace,
    TemporalJitter,
    MipBias,
    RenderLayers,
    Projection,
    NoIndirectDrawing,
    ViewUniformOffset,
    ViewMainTextureUsages,
);

/// The components removed from a camera that is skipped this frame.
type SkippedCameraComponents = (
    ExtractedCameraComponents,
    ExtractedCameraMainTexture,
    RenderVisibleEntities,
    ViewTarget,
);

pub fn extract_cameras(
    mut commands: Commands,
    mut main_pass_formats: ResMut<CameraMainPassTextureFormats>,
    query: Extract<
        Query<(
            Entity,
            RenderEntity,
            &Camera,
            &RenderTarget,
            &CameraRenderGraph,
            &GlobalTransform,
            &VisibleEntities,
            &Frustum,
            (
                Has<Hdr>,
                Option<&Tonemapping>,
                Option<&CompositingSpace>,
                Option<&ColorGrading>,
                Option<&Exposure>,
                Option<&TemporalJitter>,
                Option<&MipBias>,
                Option<&RenderLayers>,
                Option<&Projection>,
                Has<NoIndirectDrawing>,
                Has<TonemappingPass>,
                Option<&CameraMainTexture>,
                Option<&Msaa>,
                Has<PostProcessCamera>,
            ),
        )>,
    >,
    main_textures: Extract<Query<(RenderEntity, &MainTexture)>>,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
    primary_window: Extract<Query<Entity, With<PrimaryWindow>>>,
    manual_texture_views: Res<ManualTextureViews>,
    images: Res<RenderAssets<GpuImage>>,
    mut existing_render_visible_entities_cpu_culling: Query<
        &mut RenderExtractedVisibleEntities,
        With<RenderVisibleEntities>,
    >,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    visibility_extraction_system_param: VisibilityExtractionSystemParam,
    extracted_swap_chains: Query<(MainEntity, &ExtractedWindow)>,
    mut active_cameras_per_target: Local<HashMap<NormalizedRenderTarget, usize>>,
    mut main_texture_deciders: Local<EntityHashMap<MainTextureCamera>>,
) {
    main_pass_formats.clear();
    let primary_window = primary_window.iter().next();

    // Count cameras per render target to make sure tonemapping isn't duplicated.
    // The map is a `Local` so it keeps its allocation between frames.
    active_cameras_per_target.clear();
    for (_, _, camera, render_target, ..) in query.iter() {
        if !camera.is_active {
            continue;
        }
        if let Some(target) = render_target.normalize(primary_window) {
            *active_cameras_per_target.entry(target).or_default() += 1;
        }
    }

    // What a camera's main texture would be without a `MainTexture` format.
    let derived_main_texture =
        |render_target: &RenderTarget,
         hdr: bool,
         tonemapping: Option<&Tonemapping>,
         tonemapping_pass: bool,
         compositing_space: Option<CompositingSpace>| {
            let target = render_target.normalize(primary_window);
            let output_texture_format = target
                .as_ref()
                .and_then(|target| {
                    target
                        .get_texture_view_format(
                            &extracted_swap_chains,
                            &images,
                            &manual_texture_views,
                        )
                        .map(|format| normalize_bgra8(target, format))
                })
                .unwrap_or(TextureFormat::Rgba8UnormSrgb);
            let shares_target = target
                .as_ref()
                .and_then(|t| active_cameras_per_target.get(t))
                .is_some_and(|&count| count > 1);
            resolve_main_texture(
                None,
                hdr,
                tonemapping.is_some_and(Tonemapping::is_enabled),
                tonemapping_pass,
                compositing_space,
                target.as_ref(),
                shares_target,
                output_texture_format,
            )
        };

    deciding_cameras(
        query.iter().filter_map(
            |(
                main_entity,
                _,
                camera,
                render_target,
                ..,
                (
                    hdr,
                    tonemapping,
                    compositing_space,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    tonemapping_pass,
                    camera_main_texture,
                    ..,
                ),
            )| {
                let size = camera.physical_render_size()?;
                if !camera.is_active || size.x == 0 || size.y == 0 {
                    return None;
                }
                let camera_main_texture = camera_main_texture?;
                let (_, main_texture) = main_textures.get(camera_main_texture.0).ok()?;
                let compositing_space = main_texture
                    .compositing_space
                    .or(compositing_space.copied());
                Some(MainTextureCamera {
                    camera: main_entity,
                    main_texture: camera_main_texture.0,
                    order: camera.order,
                    size,
                    resolved: derived_main_texture(
                        render_target,
                        hdr,
                        tonemapping,
                        tonemapping_pass,
                        compositing_space,
                    ),
                })
            },
        ),
        &mut main_texture_deciders,
    );

    for (
        main_entity,
        render_entity,
        camera,
        render_target,
        camera_render_graph,
        transform,
        visible_entities,
        frustum,
        (
            hdr,
            tonemapping,
            compositing_space,
            color_grading,
            exposure,
            temporal_jitter,
            mip_bias,
            render_layers,
            projection,
            no_indirect_drawing,
            tonemapping_pass,
            camera_main_texture,
            msaa,
            post_process_camera,
        ),
    ) in query.iter()
    {
        if !camera.is_active {
            // Note: `RenderVisibleEntities` is here because several other retained data `ViewBinnedRenderPhase<Opaque3d>` will be removed when camera is not active
            commands.entity(render_entity).remove::<(
                ExtractedCameraComponents,
                ExtractedCameraMainTexture,
                RenderVisibleEntities,
            )>();
            continue;
        }

        let color_grading = color_grading.unwrap_or(&ColorGrading::default()).clone();

        if let (
            Some(URect {
                min: viewport_origin,
                ..
            }),
            Some(viewport_size),
            Some(render_size),
        ) = (
            camera.physical_render_viewport_rect(),
            camera.physical_render_viewport_size(),
            camera.physical_render_size(),
        ) {
            if render_size.x == 0 || render_size.y == 0 {
                commands
                    .entity(render_entity)
                    .remove::<(ExtractedCameraComponents, ExtractedCameraMainTexture)>();
                continue;
            }

            let main_texture =
                camera_main_texture.and_then(|main_texture| main_textures.get(main_texture.0).ok());
            let compositing_space = main_texture
                .and_then(|(_, main_texture)| main_texture.compositing_space)
                .or(compositing_space.copied());

            let output_viewport = camera.viewport.clone();
            let viewport = camera.render_viewport();

            let target = render_target.normalize(primary_window);
            let tonemapping_enabled = tonemapping.is_some_and(Tonemapping::is_enabled);
            let explicit_format = main_texture.and_then(|(_, main_texture)| main_texture.format);
            let deciding = camera_main_texture
                .and_then(|main_texture| main_texture_deciders.get(&main_texture.0));

            if let Some(deciding) = deciding
                && deciding.size != render_size
            {
                warn_once!(
                    "Camera {main_entity} is not rendered: it renders at {render_size:?}, but its \
                    MainTexture is {:?}. Cameras that share a MainTexture must render at the same \
                    size.",
                    deciding.size,
                );
                commands
                    .entity(render_entity)
                    .remove::<SkippedCameraComponents>();
                continue;
            }

            // Every camera adapts to an explicit format. Without one, every camera but the
            // deciding one adapts.
            let adapts_format = explicit_format.is_some()
                || deciding.is_some_and(|deciding| deciding.camera != main_entity);
            let resolved = match (explicit_format, deciding) {
                (Some(format), _) => resolve_main_texture(
                    Some(format),
                    hdr,
                    tonemapping_enabled,
                    tonemapping_pass,
                    compositing_space,
                    target.as_ref(),
                    false,
                    format,
                ),
                (None, Some(deciding)) if deciding.camera != main_entity => {
                    let format = deciding.resolved.format;
                    ResolvedMainTexture {
                        format,
                        hdr: deciding.resolved.hdr,
                        tonemap_in_shader: tonemapping_enabled && !texture_format_is_hdr(format),
                    }
                }
                _ => derived_main_texture(
                    render_target,
                    hdr,
                    tonemapping,
                    tonemapping_pass,
                    compositing_space,
                ),
            };
            let target_format = resolved.format;
            let in_shader = resolved.tonemap_in_shader;

            if let Some((_, main_texture)) = main_texture
                && let Some(Err(error)) = camera.physical_target_size().map(|size| {
                    main_texture
                        .size
                        .resolve(size, render_device.limits().max_texture_dimension_2d)
                })
            {
                warn_once!(
                    "Camera {main_entity} is not rendered: its MainTexture size is invalid \
                    ({error})."
                );
                commands
                    .entity(render_entity)
                    .remove::<SkippedCameraComponents>();
                continue;
            }
            let format_features = main_texture
                .map(|_| texture_format_features(&render_device, &render_adapter, target_format));
            if let (Some(format), Some(features)) = (explicit_format, format_features)
                && let Err(error) = validate_main_texture_format(
                    format,
                    msaa.copied().unwrap_or_default().samples(),
                    features,
                )
            {
                warn_once!("Camera {main_entity} is not rendered: {error}.");
                commands
                    .entity(render_entity)
                    .remove::<SkippedCameraComponents>();
                continue;
            }
            if adapts_format {
                let format = target_format;
                if hdr && !resolved.hdr {
                    warn_once!(
                        "Camera {main_entity} has Hdr, but the format {format:?} of its \
                        MainTexture is not an HDR format. Hdr is ignored, and effects that need \
                        it don't run on this camera."
                    );
                }
                if tonemapping_pass && tonemapping_enabled && !resolved.hdr {
                    warn_once!(
                        "Camera {main_entity} has TonemappingPass, but the format {format:?} of \
                        its MainTexture is not an HDR format, so the camera tonemaps in its \
                        material shaders instead."
                    );
                }
                if format.is_srgb() && compositing_space == Some(CompositingSpace::Srgb) {
                    warn_once!(
                        "Camera {main_entity} composites in CompositingSpace::Srgb, but its \
                        MainTexture format {format:?} is already sRGB. Set MainTexture::format \
                        to {:?}.",
                        format.remove_srgb_suffix(),
                    );
                }
            }
            let hdr = resolved.hdr;
            main_pass_formats.insert(render_entity, target_format);

            let mut render_visible_entities_cpu_culling =
                match existing_render_visible_entities_cpu_culling.get_mut(render_entity) {
                    Ok(ref mut existing_render_visible_entities_cpu_culling) => {
                        mem::take(&mut **existing_render_visible_entities_cpu_culling)
                    }
                    Err(_) => RenderExtractedVisibleEntities::default(),
                };

            for (visibility_class, visible_mesh_entities) in visible_entities.entities.iter() {
                let render_view_visible_entities = render_visible_entities_cpu_culling
                    .classes
                    .entry(*visibility_class)
                    .or_default();
                render_view_visible_entities.entities.clear();
                for main_entity in visible_mesh_entities {
                    let render_entity =
                        match visibility_extraction_system_param.mapper.get(*main_entity) {
                            Ok(render_entity) => render_entity.entity(),
                            Err(_) => Entity::PLACEHOLDER,
                        };
                    render_view_visible_entities
                        .entities
                        .push((render_entity, MainEntity::from(*main_entity)));
                }
            }

            // Don't delete "unused" visibility classes from
            // `RenderVisibleEntities`. Even if a visibility class seems empty
            // *now*, phases need to be able to find the entities that were just
            // removed from it.

            let mut commands = commands.entity(render_entity);
            commands.insert((
                ExtractedCamera {
                    target,
                    render_viewport: viewport,
                    target_viewport: output_viewport,
                    physical_render_viewport_size: Some(viewport_size),
                    physical_render_size: Some(render_size),
                    schedule: camera_render_graph.0,
                    order: camera.order,
                    output_mode: camera.output_mode,
                    msaa_writeback: camera.msaa_writeback,
                    // A post-process camera works on what other cameras left in the texture.
                    clear_color: if post_process_camera {
                        ClearColorConfig::None
                    } else {
                        camera.clear_color
                    },
                    // this will be set in sort_cameras
                    sorted_camera_index: 0,
                    sorted_camera_index_for_target: 0,
                    exposure: exposure
                        .map(Exposure::exposure)
                        .unwrap_or_else(|| Exposure::default().exposure()),
                    hdr,
                    tonemap_in_shader: in_shader,
                },
                ResolvedCompositingSpace(compositing_space),
                ExtractedView {
                    retained_view_entity: RetainedViewEntity::new(main_entity.into(), None, 0),
                    clip_from_view: camera.clip_from_view(),
                    world_from_view: *transform,
                    clip_from_world: None,
                    target_format,
                    viewport: UVec4::new(
                        viewport_origin.x,
                        viewport_origin.y,
                        viewport_size.x,
                        viewport_size.y,
                    ),
                    color_grading,
                    invert_culling: camera.invert_culling,
                },
                render_visible_entities_cpu_culling,
                *frustum,
                ViewMainTextureUsages::default(),
            ));

            if let Some((entity, main_texture)) = main_texture {
                // The texture is only filtered when it is scaled.
                let filter = match main_texture.filter {
                    MainTextureFilter::Linear
                        if camera.physical_render_size() != camera.physical_target_size() =>
                    {
                        if format_features.is_some_and(|features| {
                            features
                                .flags
                                .contains(TextureFormatFeatureFlags::FILTERABLE)
                        }) {
                            FilterMode::Linear
                        } else {
                            warn_once!(
                                "Camera {main_entity} has a MainTexture with \
                                MainTextureFilter::Linear, but the device can't filter the \
                                format {target_format:?}. Falling back to \
                                MainTextureFilter::Nearest."
                            );
                            FilterMode::Nearest
                        }
                    }
                    _ => FilterMode::Nearest,
                };
                commands.insert(ExtractedCameraMainTexture { entity, filter });
            } else {
                commands.remove::<ExtractedCameraMainTexture>();
            }

            if let Some(temporal_jitter) = temporal_jitter {
                commands.insert(temporal_jitter.clone());
            } else {
                commands.remove::<TemporalJitter>();
            }

            if let Some(mip_bias) = mip_bias {
                commands.insert(mip_bias.clone());
            } else {
                commands.remove::<MipBias>();
            }

            if let Some(render_layers) = render_layers {
                commands.insert(render_layers.clone());
            } else {
                commands.remove::<RenderLayers>();
            }

            if let Some(projection) = projection {
                commands.insert(projection.clone());
            } else {
                commands.remove::<Projection>();
            }

            if no_indirect_drawing
                || !matches!(
                    gpu_preprocessing_support.max_supported_mode,
                    GpuPreprocessingMode::Culling
                )
            {
                commands.insert(NoIndirectDrawing);
            } else {
                commands.remove::<NoIndirectDrawing>();
            }
        };
    }
}

/// Bgra8 needs an optional feature to support storage binding, and only supports write-only.
/// We force Rgba8 so that we can always use storage bindings, and rely on the final blit to
/// convert at the end if needed. See <https://github.com/gpuweb/gpuweb/issues/2748>
/// Checking just `Bgra8UnormSrgb` and not `Bgra8Unorm` is fine here, because this is the texture
/// view we already guaranteed to be srgb space if possible. See `ExtractedWindow::set_swapchain_texture`
fn normalize_bgra8(target: &NormalizedRenderTarget, format: TextureFormat) -> TextureFormat {
    if matches!(target, NormalizedRenderTarget::Window(_))
        && format == TextureFormat::Bgra8UnormSrgb
    {
        return TextureFormat::Rgba8UnormSrgb;
    }
    format
}

/// Whether a camera tonemaps in its material shaders. See
/// [`ExtractedCamera::tonemap_in_shader`].
fn tonemaps_in_shader(
    hdr: bool,
    tonemapping_enabled: bool,
    tonemapping_pass: bool,
    compositing_space: Option<CompositingSpace>,
    target: Option<&NormalizedRenderTarget>,
    shares_target: bool,
) -> bool {
    tonemapping_enabled
        && !hdr
        && (shares_target
            || (!tonemapping_pass
                && compositing_space.is_none_or(|s| s == CompositingSpace::Linear)
                && matches!(target, Some(NormalizedRenderTarget::Window(_)))))
}

/// The main texture settings that follow from a camera and its [`MainTexture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResolvedMainTexture {
    /// The format of the main texture.
    format: TextureFormat,
    /// Whether the main texture is an HDR texture. See [`ExtractedCamera::hdr`].
    hdr: bool,
    /// See [`ExtractedCamera::tonemap_in_shader`].
    tonemap_in_shader: bool,
}

/// Resolves a camera's main texture settings.
///
/// With an explicit format, the camera runs the tonemapping pass if the format is an HDR format,
/// and tonemaps in its material shaders otherwise. Without one, the format is picked from where
/// the camera tonemaps.
fn resolve_main_texture(
    explicit_format: Option<TextureFormat>,
    hdr: bool,
    tonemapping_enabled: bool,
    tonemapping_pass: bool,
    compositing_space: Option<CompositingSpace>,
    target: Option<&NormalizedRenderTarget>,
    shares_target: bool,
    output_texture_format: TextureFormat,
) -> ResolvedMainTexture {
    if let Some(format) = explicit_format {
        let hdr = texture_format_is_hdr(format);
        return ResolvedMainTexture {
            format,
            hdr,
            tonemap_in_shader: tonemapping_enabled && !hdr,
        };
    }
    let tonemap_in_shader = tonemaps_in_shader(
        hdr,
        tonemapping_enabled,
        tonemapping_pass,
        compositing_space,
        target,
        shares_target,
    );
    ResolvedMainTexture {
        format: main_texture_format(
            hdr,
            tonemapping_enabled && !tonemap_in_shader,
            compositing_space,
            output_texture_format,
        ),
        hdr,
        tonemap_in_shader,
    }
}

/// The main texture format for a camera view.
fn main_texture_format(
    hdr: bool,
    tonemapping_pass_runs: bool,
    compositing_space: Option<CompositingSpace>,
    output_texture_format: TextureFormat,
) -> TextureFormat {
    if hdr || tonemapping_pass_runs {
        // The tonemapping pass needs values above 1.0, which an 8-bit texture clamps.
        TextureFormat::Rgba16Float
    } else if compositing_space == Some(CompositingSpace::Srgb) {
        TextureFormat::Rgba8Unorm
    } else {
        output_texture_format
    }
}

/// A camera that renders into a [`MainTexture`], used to pick the camera that decides the
/// texture's size and format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MainTextureCamera {
    /// The main world entity of the camera.
    camera: Entity,
    /// The main world entity of the [`MainTexture`].
    main_texture: Entity,
    /// The camera's [`Camera::order`].
    order: isize,
    /// The size the camera renders at.
    size: UVec2,
    /// The main texture the camera would have on its own.
    resolved: ResolvedMainTexture,
}

/// For each [`MainTexture`], finds the camera with the lowest order, with the entity breaking a
/// tie. That camera decides the size of the texture and, when [`MainTexture::format`] is `None`,
/// its format.
fn deciding_cameras(
    cameras: impl IntoIterator<Item = MainTextureCamera>,
    deciding: &mut EntityHashMap<MainTextureCamera>,
) {
    deciding.clear();
    for camera in cameras {
        deciding
            .entry(camera.main_texture)
            .and_modify(|deciding| {
                if (camera.order, camera.camera) < (deciding.order, deciding.camera) {
                    *deciding = camera;
                }
            })
            .or_insert(camera);
    }
}

/// Cameras sorted by their order field. This is updated in the [`sort_cameras`] system.
#[derive(Resource, Default)]
pub struct SortedCameras(pub Vec<SortedCamera>);

pub struct SortedCamera {
    pub entity: Entity,
    pub order: isize,
    pub target: Option<NormalizedRenderTarget>,
    pub output_mode: CameraOutputMode,
}

pub fn sort_cameras(
    mut sorted_cameras: ResMut<SortedCameras>,
    mut cameras: Query<(Entity, &mut ExtractedCamera)>,
) {
    sorted_cameras.0.clear();
    for (entity, camera) in cameras.iter() {
        sorted_cameras.0.push(SortedCamera {
            entity,
            order: camera.order,
            target: camera.target.clone(),
            output_mode: camera.output_mode,
        });
    }
    // sort by order and ensure within an order, RenderTargets of the same type are packed together
    sorted_cameras
        .0
        .sort_by(|c1, c2| (c1.order, &c1.target).cmp(&(c2.order, &c2.target)));
    let mut previous_order_target = None;
    let mut ambiguities = <HashSet<_>>::default();
    let mut target_counts = <HashMap<_, _>>::default();
    for (index, sorted_camera) in sorted_cameras.0.iter_mut().enumerate() {
        if let Ok((_, mut camera)) = cameras.get_mut(sorted_camera.entity) {
            camera.sorted_camera_index = index;
        }
        let new_order_target = (sorted_camera.order, sorted_camera.target.clone());
        if let Some(previous_order_target) = previous_order_target
            && previous_order_target == new_order_target
        {
            ambiguities.insert(new_order_target.clone());
        }
        if let Some(target) = &sorted_camera.target {
            // Cameras that share a render target are indexed bottom to top. The index
            // does not affect render graph ordering. It is read at the end of
            // rendering, when each camera's image is written out to the target. By
            // default the bottom camera replaces what is there and every camera above
            // it alpha-blends on top.
            //
            // The map has to be keyed by only the target, and not by anything else.
            // Splitting the count by a camera setting such as `Hdr` would leave a
            // mixed stack with two bottom cameras, and the top one would overwrite
            // the base instead of blending over it. So we don't do that.
            let count = target_counts.entry(target.clone()).or_insert(0usize);
            let (_, mut camera) = cameras.get_mut(sorted_camera.entity).unwrap();
            camera.sorted_camera_index_for_target = *count;
            *count += 1;
        }
        previous_order_target = Some(new_order_target);
    }

    if !ambiguities.is_empty() {
        warn_once!(
            "Camera order ambiguities detected for active cameras with the following priorities: {:?}. \
            To fix this, ensure there is exactly one Camera entity spawned with a given order for a given RenderTarget. \
            Ambiguities should be resolved because either (1) multiple active cameras were spawned accidentally, which will \
            result in rendering multiple instances of the scene or (2) for cases where multiple active cameras is intentional, \
            ambiguities could result in unpredictable render results.",
            ambiguities
        );
    }
}

/// A subpixel offset to jitter a camera's frustum by.
///
/// Useful for temporal rendering techniques.
#[derive(Component, Clone, Default, Reflect)]
#[reflect(Default, Component, Clone)]
pub struct TemporalJitter {
    /// Offset is in range [-0.5, 0.5].
    pub offset: Vec2,
}

impl TemporalJitter {
    pub fn jitter_projection(&self, clip_from_view: &mut Mat4, view_size: Vec2) {
        // https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/d7531ae47d8b36a5d4025663e731a47a38be882f/docs/techniques/media/super-resolution-temporal/jitter-space.svg
        let jitter = (self.offset * vec2(2.0, -2.0)) / view_size;

        if clip_from_view.w_axis.w == 1.0 {
            // orthographic: w is 1 at every depth, so translate in clip space
            // by the same screen offset the perspective path gives (-jitter)
            clip_from_view.w_axis.x -= jitter.x;
            clip_from_view.w_axis.y -= jitter.y;
        } else {
            // perspective: w is the view depth, so the z_axis term is a constant shift
            clip_from_view.z_axis.x += jitter.x;
            clip_from_view.z_axis.y += jitter.y;
        }
    }
}

/// Camera component specifying a mip bias to apply when sampling from material textures.
///
/// Often used in conjunction with antialiasing post-process effects to reduce textures blurriness.
#[derive(Component, Reflect, Clone)]
#[reflect(Default, Component)]
pub struct MipBias(pub f32);

impl Default for MipBias {
    fn default() -> Self {
        Self(-1.0)
    }
}

/// Stores information about all entities that have changed in such a way as to
/// potentially require their pipelines to be re-specialized.
///
/// This is conservative; there's no harm, other than performance, in having an
/// entity in this list that doesn't actually need to be re-specialized. Note
/// that the presence of an entity in this list doesn't mean that a new shader
/// will necessarily be compiled; the pipeline cache is checked first.
///
/// This handles 2D meshes, 3D meshes, and sprites. For 2D and 3D wireframes,
/// see [`DirtyWireframeSpecializations`]. The reason for having two separate
/// lists is that a single entity can have both a mesh and a wireframe.
#[derive(Clone, Resource, Default)]
pub struct DirtySpecializations {
    /// All renderable objects that must be re-specialized this frame.
    pub changed_renderables: MainEntityHashSet,

    /// All renderable objects that need their specializations removed this
    /// frame.
    ///
    /// Note that this may include entities in [`Self::changed_renderables`].
    /// This is fine, as old specializations are removed before new ones are
    /// added.
    pub removed_renderables: MainEntityHashSet,

    /// Views that must be respecialized this frame.
    ///
    /// The presence of a view in this list causes all entities that it renders
    /// to be re-specialized.
    pub views: HashSet<RetainedViewEntity>,
}

impl DirtySpecializations {
    /// Returns true if the view has changed in such a way that all specialized
    /// pipelines for entities visible from it must be regenerated.
    pub fn must_wipe_specializations_for_view(&self, view: RetainedViewEntity) -> bool {
        self.views.contains(&view)
    }

    /// Given a main entity known to be visible, returns it alongside any render
    /// entity it corresponds to.
    ///
    /// If no render entity corresponds to the given main entity, the render
    /// entity returned will be [`Entity::PLACEHOLDER`].
    fn entity_pair_from_visible_main_entity<'a>(
        &'a self,
        render_visible_mesh_entities: &'a RenderVisibleEntitiesClass,
        main_entity: &'a MainEntity,
    ) -> Option<(&'a Entity, &'a MainEntity)> {
        // Check entities with CPU culling.
        if let Ok(index) = render_visible_mesh_entities
            .entities_cpu_culling
            .binary_search_by_key(main_entity, |(_, main_entity)| *main_entity)
        {
            let (key, value) = &render_visible_mesh_entities.entities_cpu_culling[index];
            return Some((key, value));
        }

        // Check entities that opted out of CPU culling.
        if let Some(entity) = render_visible_mesh_entities
            .entities_gpu_culling
            .get(main_entity)
        {
            return Some((entity, main_entity));
        }

        // We didn't find the entity, so return `None`.
        None
    }

    /// Iterates over all entities that need their specializations cleared in
    /// this frame.
    pub fn iter_to_despecialize<'a>(&'a self) -> impl Iterator<Item = &'a MainEntity> {
        // Entities that changed or were removed must be
        // de-specialized.
        self.changed_renderables
            .iter()
            .chain(self.removed_renderables.iter())
    }

    /// Iterates over all entities that need to have their pipelines
    /// re-specialized this frame.
    ///
    /// `last_frame_view_pending_queues` should be the contents of the
    /// [`ViewPendingQueues::prev_frame`] list.
    pub fn iter_to_specialize<'a>(
        &'a self,
        view: RetainedViewEntity,
        render_view_visible_mesh_entities: &'a RenderVisibleEntitiesClass,
        last_frame_view_pending_queues: &'a HashSet<(Entity, MainEntity)>,
    ) -> impl Iterator<Item = (&'a Entity, &'a MainEntity)> {
        (if self.must_wipe_specializations_for_view(view) {
            Either::Left(render_view_visible_mesh_entities.iter_visible())
        } else {
            Either::Right(
                render_view_visible_mesh_entities
                    .added_entities()
                    .iter()
                    .map(|(entity, main_entity)| (entity, main_entity))
                    .chain(self.changed_renderables.iter().filter_map(|main_entity| {
                        self.entity_pair_from_visible_main_entity(
                            render_view_visible_mesh_entities,
                            main_entity,
                        )
                    })),
            )
        })
        .chain(
            last_frame_view_pending_queues
                .iter()
                .filter_map(|(_, main_entity)| {
                    // Resolve pending entries against the current visible entities.
                    self.entity_pair_from_visible_main_entity(
                        render_view_visible_mesh_entities,
                        main_entity,
                    )
                }),
        )
    }

    /// Iterates over all renderables that should be removed from the phase.
    ///
    /// This includes renderables that became invisible this frame, renderables
    /// that are in [`DirtySpecializations::changed_renderables`], and
    /// renderables that are in [`DirtySpecializations::removed_renderables`].
    /// If this view must itself be re-specialized, this will iterate over all
    /// visible entities in addition to those that became invisible.
    pub fn iter_to_dequeue<'a>(
        &'a self,
        view: RetainedViewEntity,
        render_visible_mesh_entities: &'a RenderVisibleEntitiesClass,
    ) -> impl Iterator<Item = &'a MainEntity> {
        render_visible_mesh_entities
            .removed_entities
            .iter()
            .map(|(_, main_entity)| main_entity)
            .chain(if self.must_wipe_specializations_for_view(view) {
                // All visible entities must be removed.
                // Note that this includes potentially-invisible entities, but
                // that's OK as they shouldn't be in the caller's bins in the
                // first place.
                Either::Left(
                    render_visible_mesh_entities
                        .iter_visible()
                        .map(|(_, main_entity)| main_entity),
                )
            } else {
                // Only entities that changed must be removed.
                Either::Right(
                    self.changed_renderables
                        .iter()
                        .chain(self.removed_renderables.iter()),
                )
            })
    }

    /// Iterates over all renderables that potentially need to be re-queued.
    ///
    /// This includes both renderables that became visible and those that are in
    /// [`DirtySpecializations::changed_renderables`]. If this view must itself
    /// be re-specialized, this will iterate over all visible renderables.
    ///
    /// `last_frame_view_pending_queues` should be the contents of the
    /// [`ViewPendingQueues::prev_frame`] list.
    /// `mesh_instances_queued_this_iteration_scratch_space` should be a
    /// `Local<MainEntityHashSet>`; it's used internally to avoid yielding the
    /// same mesh instance multiple times.
    pub fn iter_to_queue<'a>(
        &'a self,
        view: RetainedViewEntity,
        render_visible_mesh_entities: &'a RenderVisibleEntitiesClass,
        last_frame_view_pending_queues: &'a HashSet<(Entity, MainEntity)>,
        mesh_instances_queued_this_iteration_scratch_space: &'a mut MainEntityHashSet,
    ) -> impl Iterator<Item = (&'a Entity, &'a MainEntity)> {
        mesh_instances_queued_this_iteration_scratch_space.clear();

        // Use `mesh_instances_queued_this_iteration_scratch_space` to avoid
        // yielding the same mesh instance twice.
        // Yielding a mesh instance twice would result in binning it twice,
        // which is illegal.
        (if self.must_wipe_specializations_for_view(view) {
            Either::Left(render_visible_mesh_entities.iter_visible())
        } else {
            Either::Right(
                render_visible_mesh_entities
                    .added_entities()
                    .iter()
                    .map(|(entity, main_entity)| (entity, main_entity))
                    .chain(self.changed_renderables.iter().filter_map(|main_entity| {
                        self.entity_pair_from_visible_main_entity(
                            render_visible_mesh_entities,
                            main_entity,
                        )
                    })),
            )
        })
        .chain(
            last_frame_view_pending_queues
                .iter()
                .filter_map(|(_, main_entity)| {
                    // Resolve pending entries against the current visible entities.
                    self.entity_pair_from_visible_main_entity(
                        render_visible_mesh_entities,
                        main_entity,
                    )
                }),
        )
        .filter(|(_, main_entity)| {
            mesh_instances_queued_this_iteration_scratch_space.insert(**main_entity)
        })
    }
}

/// Stores information about all entities that have changed in such a way as to
/// potentially require their wireframe pipelines to be re-specialized.
///
/// This is separate from [`DirtySpecializations`] because a single entity can
/// have both a mesh and a wireframe on it, and the pipelines are treated
/// separately.
///
/// See [`DirtySpecializations`] for more information.
#[derive(Clone, Resource, Default, Deref, DerefMut)]
pub struct DirtyWireframeSpecializations(pub DirtySpecializations);

/// Clears out the [`DirtySpecializations`] resource in preparation for a new
/// frame.
pub fn clear_dirty_specializations(mut dirty_specializations: ResMut<DirtySpecializations>) {
    dirty_specializations.changed_renderables.clear();
    dirty_specializations.removed_renderables.clear();
    dirty_specializations.views.clear();
}

/// Clears out the [`DirtyWireframeSpecializations`] resource in preparation for
/// a new frame.
pub fn clear_dirty_wireframe_specializations(
    mut dirty_wireframe_specializations: ResMut<DirtyWireframeSpecializations>,
) {
    dirty_wireframe_specializations.changed_renderables.clear();
    dirty_wireframe_specializations.removed_renderables.clear();
    dirty_wireframe_specializations.views.clear();
}

/// A system that removes views that don't exist any longer from
/// [`DirtySpecializations`].
pub fn expire_specializations_for_views(
    views: Query<&ExtractedView>,
    mut dirty_specializations: ResMut<DirtySpecializations>,
) {
    let all_live_retained_view_entities: HashSet<_> =
        views.iter().map(|view| view.retained_view_entity).collect();
    dirty_specializations.views.retain(|retained_view_entity| {
        all_live_retained_view_entities.contains(retained_view_entity)
    });
}

/// A system that removes views that don't exist any longer from
/// [`DirtyWireframeSpecializations`].
pub fn expire_wireframe_specializations_for_views(
    views: Query<&ExtractedView>,
    mut dirty_wireframe_specializations: ResMut<DirtyWireframeSpecializations>,
) {
    let all_live_retained_view_entities: HashSet<_> =
        views.iter().map(|view| view.retained_view_entity).collect();
    dirty_wireframe_specializations
        .views
        .retain(|retained_view_entity| {
            all_live_retained_view_entities.contains(retained_view_entity)
        });
}

/// A [`SystemSet`] that contains all systems that mutate the
/// [`DirtySpecializations`] resource and other resources that wrap that type.
///
/// These systems must run in order.
#[derive(SystemSet, Clone, PartialEq, Eq, Debug, Hash)]
pub enum DirtySpecializationSystems {
    /// Systems that clear out [`DirtySpecializations`] types in preparation for
    /// a new frame.
    Clear,

    /// Systems that add entities that need to be re-specialized to
    /// [`DirtySpecializations`].
    CheckForChanges,

    /// Systems that determine which entities need to be removed from render
    /// phases and write the results to [`DirtySpecializations`].
    ///
    /// The set of entities that need to be removed from the render phases can
    /// only be determined after all systems in
    /// [`DirtySpecializationSystems::CheckForChanges`] have run. That's because
    /// these systems check `RemovedComponents` resources, and they have to be
    /// able to distinguish between the case in which an entity was truly made
    /// unrenderable and the case in which an entity appeared in a
    /// `RemovedComponents` table simply because its material *type* changed.
    CheckForRemovals,
}

/// Holds all entities that couldn't be specialized and/or queued because their
/// materials or other dependent resources hadn't loaded yet.
///
/// We might not be able to specialize and/or enqueue a renderable entity if a
/// dependent resource like a material isn't available. In that case, we add the
/// entity to the appropriate list so that we attempt to re-specialize and
/// re-queue it on subsequent frames.
///
/// This type is expected to be placed in a newtype wrapper and stored as a
/// resource: e.g. `PendingMeshMaterialQueues`.
#[derive(Default, Deref, DerefMut)]
pub struct PendingQueues(pub HashMap<RetainedViewEntity, ViewPendingQueues>);

/// Holds all entities that couldn't be specialized and/or queued because their
/// materials and/or other dependent resources hadn't loaded yet for a single
/// view.
///
/// See the documentation of [`PendingQueues`] for more information.
#[derive(Default)]
pub struct ViewPendingQueues {
    /// The entities that couldn't be specialized and/or queued this frame.
    ///
    /// We add to this list during pipeline specialization and queuing.
    pub current_frame: HashSet<(Entity, MainEntity)>,

    /// The entities that we need to re-examine in this frame.
    ///
    /// We attempt to specialize and queue entities in this list every frame, as
    /// long as those entities are still visible.
    pub prev_frame: HashSet<(Entity, MainEntity)>,
}

impl PendingQueues {
    /// Initializes the pending queues for a new frame.
    ///
    /// This method is called during specialization. It creates the queues for
    /// the view if necessary and initializes them.
    pub fn prepare_for_new_frame(
        &mut self,
        retained_view_entity: RetainedViewEntity,
    ) -> &mut ViewPendingQueues {
        let view_pending_queues = self.entry(retained_view_entity).or_default();
        mem::swap(
            &mut view_pending_queues.current_frame,
            &mut view_pending_queues.prev_frame,
        );
        view_pending_queues.current_frame.clear();
        view_pending_queues
    }

    /// Removes any pending queues that belong to views not in the supplied
    /// `all_views` table.
    ///
    /// Specialization systems for phases should call this before returning in
    /// order to clean up resources relating to views that no longer exist.
    pub fn expire_stale_views(&mut self, all_views: &HashSet<RetainedViewEntity>) {
        self.retain(|retained_view_entity, _| all_views.contains(retained_view_entity));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::Main;
    use bevy_camera::{CameraProjection, OrthographicProjection, PerspectiveProjection};
    use bevy_ecs::{system::RunSystemOnce, world::World};
    use bevy_math::{Rect, Vec4};
    use bevy_window::WindowRef;

    fn window_target() -> NormalizedRenderTarget {
        NormalizedRenderTarget::Window(
            WindowRef::Entity(Entity::from_raw_u32(0).unwrap())
                .normalize(None)
                .unwrap(),
        )
    }

    /// The inputs to [`tonemaps_in_shader`] and [`main_texture_format`] for one
    /// table case.
    #[derive(Clone, Copy)]
    struct CameraCase<'a> {
        hdr: bool,
        tonemapping_enabled: bool,
        tonemapping_pass: bool,
        compositing_space: Option<CompositingSpace>,
        target: Option<&'a NormalizedRenderTarget>,
        shares_target: bool,
    }

    /// A solo SDR camera that tonemaps in-shader, used as the baseline for the case
    /// tables.
    fn eligible_camera(target: &NormalizedRenderTarget) -> CameraCase<'_> {
        CameraCase {
            hdr: false,
            tonemapping_enabled: true,
            tonemapping_pass: false,
            compositing_space: None,
            target: Some(target),
            shares_target: false,
        }
    }

    const OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

    fn assert_case_table(table: &[(&str, CameraCase, TextureFormat, bool)]) {
        for (name, case, expected_format, expected_in_shader) in table {
            let in_shader = tonemaps_in_shader(
                case.hdr,
                case.tonemapping_enabled,
                case.tonemapping_pass,
                case.compositing_space,
                case.target,
                case.shares_target,
            );
            assert_eq!(in_shader, *expected_in_shader, "{name} (in-shader)");
            let format = main_texture_format(
                case.hdr,
                case.tonemapping_enabled && !in_shader,
                case.compositing_space,
                OUTPUT_FORMAT,
            );
            assert_eq!(format, *expected_format, "{name}");
        }
    }

    #[test]
    fn solo_camera_cases() {
        let window = window_target();
        let texture_view = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let base = eligible_camera(&window);

        let table = [
            ("eligible SDR camera", base, OUTPUT_FORMAT, true),
            (
                "explicit linear compositing",
                CameraCase {
                    compositing_space: Some(CompositingSpace::Linear),
                    ..base
                },
                OUTPUT_FORMAT,
                true,
            ),
            (
                "Hdr camera",
                CameraCase { hdr: true, ..base },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "TonemappingPass camera",
                CameraCase {
                    tonemapping_pass: true,
                    ..base
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "explicit sRGB compositing with tonemapping enabled",
                CameraCase {
                    compositing_space: Some(CompositingSpace::Srgb),
                    ..base
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "explicit Oklab compositing with tonemapping enabled",
                CameraCase {
                    compositing_space: Some(CompositingSpace::Oklab),
                    ..base
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "non-window target",
                CameraCase {
                    target: Some(&texture_view),
                    ..base
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "no render target",
                CameraCase {
                    target: None,
                    ..base
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "tonemapping disabled",
                CameraCase {
                    tonemapping_enabled: false,
                    ..base
                },
                OUTPUT_FORMAT,
                false,
            ),
            (
                "explicit sRGB compositing with tonemapping disabled",
                CameraCase {
                    tonemapping_enabled: false,
                    compositing_space: Some(CompositingSpace::Srgb),
                    ..base
                },
                TextureFormat::Rgba8Unorm,
                false,
            ),
            (
                "explicit Oklab compositing with tonemapping disabled",
                CameraCase {
                    tonemapping_enabled: false,
                    compositing_space: Some(CompositingSpace::Oklab),
                    ..base
                },
                OUTPUT_FORMAT,
                false,
            ),
        ];

        assert_case_table(&table);
    }

    /// Stacked and split screen cameras keep the in-shader path. Only solo
    /// cameras move to the tonemapping pass; see [`ExtractedCamera::tonemap_in_shader`].
    #[test]
    fn shared_target_cameras_keep_the_in_shader_path() {
        let window = window_target();
        let texture_view = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let shared = CameraCase {
            shares_target: true,
            ..eligible_camera(&window)
        };

        let table = [
            ("shared window target", shared, OUTPUT_FORMAT, true),
            (
                "shared texture target",
                CameraCase {
                    target: Some(&texture_view),
                    ..shared
                },
                OUTPUT_FORMAT,
                true,
            ),
            (
                "shared target with TonemappingPass",
                CameraCase {
                    tonemapping_pass: true,
                    ..shared
                },
                OUTPUT_FORMAT,
                true,
            ),
            (
                "Hdr camera on a shared target",
                CameraCase {
                    hdr: true,
                    ..shared
                },
                TextureFormat::Rgba16Float,
                false,
            ),
            (
                "sRGB compositing with tonemapping enabled on a shared target",
                CameraCase {
                    compositing_space: Some(CompositingSpace::Srgb),
                    ..shared
                },
                TextureFormat::Rgba8Unorm,
                true,
            ),
            (
                "misconfigured Oklab camera without Hdr on a shared target",
                CameraCase {
                    compositing_space: Some(CompositingSpace::Oklab),
                    ..shared
                },
                OUTPUT_FORMAT,
                true,
            ),
            (
                "tonemapping disabled on a shared target",
                CameraCase {
                    tonemapping_enabled: false,
                    ..shared
                },
                OUTPUT_FORMAT,
                false,
            ),
        ];

        assert_case_table(&table);
    }

    #[test]
    fn explicit_formats_decide_where_tonemapping_runs() {
        let window = window_target();
        let resolve = |format, hdr, tonemapping_enabled, tonemapping_pass| {
            resolve_main_texture(
                Some(format),
                hdr,
                tonemapping_enabled,
                tonemapping_pass,
                None,
                Some(&window),
                false,
                OUTPUT_FORMAT,
            )
        };
        let resolved = |format, hdr, tonemap_in_shader| ResolvedMainTexture {
            format,
            hdr,
            tonemap_in_shader,
        };

        // An HDR format runs the pass, with or without the `Hdr` component.
        for format in [
            TextureFormat::Rgba16Float,
            TextureFormat::Rgba32Float,
            TextureFormat::Rg11b10Ufloat,
        ] {
            assert_eq!(
                resolve(format, false, true, false),
                resolved(format, true, false)
            );
            assert_eq!(
                resolve(format, true, false, false),
                resolved(format, true, false)
            );
        }
        // Any other format tonemaps in shader, even with `Hdr` or `TonemappingPass`.
        let format = TextureFormat::Rgba8UnormSrgb;
        assert_eq!(
            resolve(format, true, true, true),
            resolved(format, false, true)
        );
        assert_eq!(
            resolve(format, false, false, false),
            resolved(format, false, false)
        );
    }

    #[test]
    fn lowest_order_camera_decides_the_main_texture() {
        let entity = |raw| Entity::from_raw_u32(raw).unwrap();
        let resolved = |format| ResolvedMainTexture {
            format,
            hdr: false,
            tonemap_in_shader: false,
        };
        let camera = |raw, texture, order, size: u32| MainTextureCamera {
            camera: entity(raw),
            main_texture: entity(texture),
            order,
            size: UVec2::splat(size),
            resolved: resolved(TextureFormat::Rgba8UnormSrgb),
        };
        let mut deciding = EntityHashMap::default();

        // The lowest order decides, whatever the iteration order.
        let first = camera(3, 100, -1, 64);
        let later = camera(1, 100, 0, 32);
        for cameras in [[first, later], [later, first]] {
            deciding_cameras(cameras, &mut deciding);
            assert_eq!(deciding[&entity(100)], first);
        }

        // Ties are broken the same way.
        let a = camera(1, 100, 0, 64);
        let b = camera(2, 100, 0, 32);
        deciding_cameras([a, b], &mut deciding);
        let winner = deciding[&entity(100)];
        deciding_cameras([b, a], &mut deciding);
        assert_eq!(deciding[&entity(100)], winner);

        // Each main texture has its own deciding camera.
        let other = camera(4, 101, 5, 16);
        deciding_cameras([first, other], &mut deciding);
        assert_eq!(deciding[&entity(100)], first);
        assert_eq!(deciding[&entity(101)], other);
    }

    fn extracted_camera(
        order: isize,
        hdr: bool,
        target: NormalizedRenderTarget,
    ) -> ExtractedCamera {
        ExtractedCamera {
            target: Some(target),
            physical_render_viewport_size: None,
            physical_render_size: None,
            render_viewport: None,
            target_viewport: None,
            schedule: Main.intern(),
            order,
            output_mode: CameraOutputMode::default(),
            msaa_writeback: MsaaWriteback::default(),
            clear_color: ClearColorConfig::Default,
            sorted_camera_index: 0,
            sorted_camera_index_for_target: 0,
            exposure: 1.0,
            hdr,
            tonemap_in_shader: false,
        }
    }

    #[test]
    fn sort_cameras_assigns_sequential_indices_for_mixed_hdr_shared_target() {
        let mut world = World::new();
        world.init_resource::<SortedCameras>();

        let shared = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0));
        let other = NormalizedRenderTarget::TextureView(ManualTextureViewHandle(1));
        // Spawn the upper camera first, so the indices prove camera order beats spawn order.
        let upper = world.spawn(extracted_camera(1, true, shared.clone())).id();
        let lower = world.spawn(extracted_camera(0, false, shared)).id();
        let solo = world.spawn(extracted_camera(0, true, other)).id();

        world.run_system_once(sort_cameras).unwrap();

        let index = |entity: Entity| {
            world
                .entity(entity)
                .get::<ExtractedCamera>()
                .unwrap()
                .sorted_camera_index_for_target
        };
        assert_eq!(index(lower), 0);
        assert_eq!(index(upper), 1);
        assert_eq!(index(solo), 0);
    }

    /// How far the jitter moves a point at view depth `z`, in pixels.
    fn jitter_in_pixels(clip_from_view: Mat4, jitter: &TemporalJitter, z: f32) -> Vec2 {
        let view_size = vec2(1200.0, 800.0);
        let mut jittered = clip_from_view;
        jitter.jitter_projection(&mut jittered, view_size);
        let point = Vec4::new(0.3, -0.2, z, 1.0);
        let before = clip_from_view * point;
        let after = jittered * point;
        let ndc_shift =
            after.truncate().truncate() / after.w - before.truncate().truncate() / before.w;
        ndc_shift * view_size * 0.5
    }

    #[test]
    fn orthographic_jitter_matches_perspective_at_every_depth() {
        let jitter = TemporalJitter {
            offset: vec2(0.25, -0.4),
        };
        let perspective = PerspectiveProjection::default().get_clip_from_view();
        let orthographic = OrthographicProjection {
            area: Rect::new(-12.0, -8.0, 12.0, 8.0),
            ..OrthographicProjection::default_3d()
        }
        .get_clip_from_view();

        for z in [-1.0, -10.0, -200.0] {
            let expected = jitter_in_pixels(perspective, &jitter, z);
            let actual = jitter_in_pixels(orthographic, &jitter, z);
            assert!(
                actual.abs_diff_eq(expected, 1e-3),
                "depth {z}: orthographic {actual}, perspective {expected}"
            );
        }
    }
}
