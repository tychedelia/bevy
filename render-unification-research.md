# Render Unification: 2D/3D/UI Research Notes

State-of-the-codebase survey for unifying 2D, 3D, and UI rendering onto shared
mid-level infrastructure. Based on `main` at 0.20.0-dev (July 2026). All
`file:line` references are current as of this writing.

---

## 1. Headline findings

**The unification is further along than the plan assumes.** Three of the four
layers the plan worries about are either already unified or have a working
prototype in-tree:

1. **The material *management* layer is already unified.** 2D materials no
   longer use "older, naive patterns" — `mesh2d/material.rs` has been rebuilt
   on the same type-erased architecture as 3D: `MaterialProperties`,
   `ErasedMaterialKey`/`ErasedMaterialPipelineKey`/`ErasedMeshPipelineKey`,
   erased `base_specialize`/`user_specialize` fn pointers, `ErasedRenderAsset`
   (`PreparedMaterial2d` is non-generic in the render world),
   `MaterialBindGroupAllocators` + bindless, `DirtySpecializations`-driven
   respecialization, binned `Opaque2d`/`AlphaMask2d`. The remaining divergence
   is almost entirely in the **mesh/instance-data plumbing**, not materials.

2. **The render graph is already unified.** The old `RenderGraph`/`Node2d`/
   `Node3d` model is gone; `Core2d`/`Core3d` are `ScheduleLabel`s sharing the
   same `Prepass → MainPass → EarlyPostProcess → PostProcess` system-set chain
   (`bevy_core_pipeline/src/schedule.rs:48,87`), the same `camera_driver`, and
   the same view-target/MSAA/HDR prep. 2D's graph divergence is purely "the 3D
   pass set it doesn't register" (prepass, deferred, transmission, OIT,
   skybox, occlusion culling) — adoptable incrementally.

3. **The crate split the plan proposes already half-exists.** `bevy_material`
   exists and is pure `wgpu-types` (no `bevy_render` dep, no concrete wgpu).
   `bevy_sprite_render` and `bevy_ui_render` are already split out.
   Commit `ff376dea7` (bevy_material, #22426) states the charter explicitly:
   "another step towards shared 2d and 3d rendering infrastructure
   deduplication." **`bevy_material_render` does not exist yet** — it is the
   missing middle layer, and its content is well-defined (§5).

4. **A sprite-unification prototype is already in-tree.** `sprite_mesh/`
   (`SpriteMesh` + `SpriteMaterial` on a shared unit-quad `Mesh2d`) expresses
   the full sprite feature set (flip/atlas/rect/scaling-mode/tiling/9-slice/
   anchor/alpha modes) as `Material2d` uniforms, and `tilemap_chunk/` is
   already a plain `Material2d` with a data texture. The bespoke paths that
   remain are `render/` (SpritePipeline), `text2d/` (parasitic on it), and
   `texture_slice/` (feeds it).

**Where the real work is:**

- **2D mesh instance data**: CPU batching with dynamic-offset
  `GpuArrayBuffer<Mesh2dUniform>`; no GPU preprocessing, no indirect/multidraw
  (deliberately stubbed — see §3). This is the core lift for "fold 2D into 3D."
- **Sprites/text**: per-batch texture binding + transient engine-owned
  instance buffers don't map onto the retained mesh/material model (§4).
- **UI**: five parallel pipelines duplicating identical scaffolding, a strict
  painter's-order contract, CPU vertex-clamp clipping, transient render
  entities, and a bolt-on view model (§6). Hardest, as the plan expects.
- **Extracting `bevy_material_render` without dragging 3D down the graph**:
  the "generic" queue/specialize systems in `bevy_pbr` reference lightmaps,
  shadows, `MeshPipeline`, and `MeshInputUniform` (§5).

---

## 2. The 3D architecture (target state) and where its erasure stops

Verified claim: from prepare down, a 3D material is data. `prepare_asset`
collapses `M: Material` into non-generic
`PreparedMaterial { binding: MaterialBindingId, properties: Arc<MaterialProperties> }`
(`bevy_pbr/src/material.rs:1479`), stored in `ErasedRenderAssets<PreparedMaterial>`.
Draw functions, shaders, layouts, phase assignment, and specialization
callbacks are all data in `MaterialProperties` (`bevy_material/src/lib.rs:53`).
The only monomorphized residue is fn pointers captured at prepare time
(`user_specialize::<M>`, `material.rs:1714`) plus per-`M` extraction systems.
Render-world systems (`specialize_material_meshes`, `queue_material_meshes`,
prepass, shadow) never name a concrete `M` — queue switches on
`properties.render_phase_type` and looks up draw functions by interned label.

**But the erasure is material-only. Hardwired 3D assumptions:**

| Assumption | Where |
|---|---|
| `Mesh3d` in visibility queries | `material.rs:1002,1197`; `light.rs:2414` |
| Exactly 4 phase types (Opaque/AlphaMask/Transmissive/Transparent) | `bevy_material/src/phase.rs:3`; queue writes only 3D phases |
| Preparing a material *requires* `DrawFunctions` for all 9 3D/prepass/deferred/shadow phases | `ErasedRenderAsset::Param`, `material.rs:1563` |
| Fixed bind group indices: 0/1 view, 2 mesh, 3 material | `MATERIAL_BIND_GROUP_INDEX = 3`, `material.rs:74`; layout inserted at index 3 in both main + prepass specializers |
| `MeshUniform` carries skinning, morphs, lightmap UVs, previous-frame transform | `render/mesh.rs:517,568` |
| Lightmaps threaded through specialize/queue (`RenderLightmaps`, `LIGHTMAPPED` bits) | `material.rs:946,1084,1316` |
| `MeshAllocator` slab IDs in batch-set keys | `queue_material_meshes` `mesh_slabs`; `Opaque3dBatchSetKey.slabs` |
| Every erased-key downcast assumes `MeshPipelineKey` (u64) | `material.rs:505,1075,1543` |
| View key built from 3D camera features (prepass markers, env maps, SSR, OIT…) | `render/mesh.rs:360` |

Any generalized `bevy_material_render` must parameterize or hook each of these.

---

## 3. 2D meshes: what's left after the material-layer unification

The material half of `mesh2d/` is a near-line-for-line sibling of `bevy_pbr`
(same `MaterialProperties` construction, same `DirtySpecializations` flow, same
cache shapes — even the same misleading "(tick, pipeline_id)" doc comment).
The divergent half:

**Instance-data model.** 2D: per-instance `Mesh2dUniform` in a CPU-filled
`GpuArrayBuffer` addressed by dynamic offset (`SetMesh2dBindGroup`,
`mesh2d/mesh.rs:901,949`). 3D: GPU-preprocessed storage buffers
(`MeshInputUniform` → compute → `MeshUniform`), indirect draws.
The 2D `GetFullBatchData` GPU methods are `error!()` stubs returning `None`
(`mesh2d/mesh.rs:448-471`, "not yet implemented for 2d meshes"), and
`extract_core_2d_camera_phases` hardcodes `GpuPreprocessingMode::None`
(`core_2d/mod.rs:410`). 2D also rebuilds `RenderMesh2dInstances` from scratch
every frame (`extract_mesh2d` clears the map, `mesh.rs:306`) vs 3D's
change-detected GPU-instance builder.

**Batch-set key asymmetry (deepest schema change).** 3D splits
`BatchSetKey` (pipeline, draw_function, material_bind_group_index, mesh
slabs, lightmap slab) from a minimal `BinKey { asset_id }`, enabling
multidraw across a slab. 2D collapses everything into the `BinKey`
(pipeline + draw_fn + asset + bind group, `core_2d/mod.rs:115,229`) with a
near-empty `BatchSetKey2d { indexed }` — explicitly because "2D meshes
presently can't be multidrawn" (`core_2d/mod.rs:164,187`). Even accessors
differ: `Opaque2d::draw_function()` reads the bin key, `Opaque3d`'s reads the
batch-set key. Unifying means moving 2D's key fields up and adding
`MeshSlabs`, or keeping a per-dimension key schema behind the shared systems.

**Phase wiring.** 2D hand-registers only the `no_gpu_preprocessing` batch
systems (`mesh2d/mesh.rs:112-124`) instead of using
`BinnedRenderPhasePlugin`/`SortedRenderPhasePlugin` (which register both CPU
and GPU paths run_if-gated). Exception: `Wireframe2d` *does* use
`BinnedRenderPhasePlugin` — the one existing example of 2D on the shared
plugin, though still CPU-batched. The `gpu_preprocess.rs` compute shaders
(`mesh_preprocess.wgsl` etc.) live in `bevy_pbr` and are 3D-only.

**Retention hacks.** 2D's retained-phase management drives binned
`add`/`remove` off `DirtySpecializations`, and sorted `Transparent2d` items
use `Entity::PLACEHOLDER` as render entity because `DirtySpecializations`
tracks only main entities (`mesh2d/material.rs:1132-1145`). Reconciling this
with 3D's plumbing around a single retained-bin API is required.

**2D-only semantics a unified path must preserve:**
- Transparent sort key is world-space Z + depth bias, precomputed at queue
  time (`material.rs:1154`); 3D recomputes camera-distance lazily via
  `ViewRangefinder3d` (no 2D rangefinder exists). The Z-ordering contract is
  user-visible behavior.
- `AlphaMode2d` ⊂ `AlphaMode`: no Premultiplied/Add/Multiply/AlphaToCoverage.
  Sharing `AlphaMode` means implementing or rejecting the extra modes in 2D.
- `CompositingSpace` (SRGB/OKLAB) pipeline-key bits (`mesh.rs:520-521`) have
  no 3D equivalent; a shared key must reserve room for them.
- Tonemapping LUTs are baked into the mesh2d *view bind group layout*
  (`mesh.rs:355-365`) and applied in the main-pass shader when `!hdr`; 3D
  tonemaps in a separate post-process system. Merging view layouts is
  nontrivial.
- No prepass ⇒ no motion vectors, no deferred, `OpaqueRendererMethod`
  hardcoded Forward, `shadows_enabled: false` (`material.rs:1261-1269`).

**Can the same specialization functions be reused?** Mostly yes at the
framework level — both sides already funnel through
`SpecializedMeshPipelines<Specializer>` with `Key = ErasedMaterialPipelineKey`
and identical system shapes. The blockers are that `MaterialPipeline` embeds
`MeshPipeline` while `Material2dPipeline` embeds `Mesh2dPipeline`, the
downcasts assume `MeshPipelineKey` vs `Mesh2dPipelineKey`, and the shader-def
sets differ. A "concrete mesh pipeline" trait parameter (or data-driven
equivalent) makes the specialize/queue skeletons shareable.

---

## 4. Sprites and text: the bespoke third path

`render/mod.rs` (SpritePipeline) is a fully independent pipeline: shader-
generated quads (no vertex buffer; corners from `vertex_index`), an 80-byte
`SpriteInstance` (affine + color + uv_offset_scale) in a per-frame
`RawBufferVec`, instance-rate vertex attributes, one `Transparent2d` item per
sprite via `add_transient`, batches formed in prepare by contiguous runs of
the same `AssetId<Image>` and broken by any interleaved non-sprite item
(`render/mod.rs:692-696`). Text2d has no pipeline at all — it synthesizes
`ExtractedSprite`/`ExtractedSlice` records into the same resources
(glyph-atlas quads, `TemporaryRenderEntity` per batch). `texture_slice/` is
CPU geometry expansion (one instance per slice/tile) feeding the same path.

Hazards ranked (for folding sprites into the unified mesh/material path):

1. **Per-batch texture binding vs one-material-per-draw.** Cross-image
   batching relies on swapping bind group 1 per batch. A unified path needs
   per-instance texture selection: bindless texture arrays, atlas arrays
   (as `tilemap_chunk` already uses), or accept per-image draw splits.
2. **Transient, engine-owned instance streams.** Glyphs and slices are a
   variable per-frame quad count with no retained entity per quad. The
   material path assumes retained per-entity instances and asset-owned
   meshes. A unified renderer needs a "dynamic instance stream" concept.
3. **Order-preserving batching across heterogeneous items** in one sorted
   transparent phase (sprites interleaved with mesh2d, etc.).
4. **CPU-side geometry math** (anchor/flip/rect/scaling/9-slice/tiling) —
   already proven expressible as material uniforms by `SpriteMaterial`
   (`sprite_mesh/sprite_material.rs:99-310`); the decision is CPU-expansion
   vs shader-side, and covering multi-quad expansion.
5. **Auxiliary geometry re-derivation**: rendering, `calculate_bounds_2d`,
   and sprite picking each independently compute sprite geometry; plus the
   temporary `SpriteMesh` vs `Sprite` bounds split (comments already
   anticipate merging the components).

`SpriteMaterial`'s known gaps vs the bespoke path: each unique
`(SpriteMesh, Anchor)` is a distinct material asset (bind group) — no
cross-sprite batching — and per-frame material mutation churn is untested at
sprite scale. Closing those gaps is essentially hazards 1–2.

---

## 5. Crate layering: `bevy_material_render`

Current facts:
- Only `bevy_render` depends on concrete `wgpu`; everything else reaches
  concrete types via `bevy_render::render_resource` re-exports. The plan's
  "refers to concrete wgpu types" test in practice means "imports
  `RenderDevice`/`TrackedRenderPass`/`PipelineCache`/`BindGroup` from
  bevy_render."
- `bevy_material` is pure `wgpu-types` and already holds: `MaterialProperties`,
  `AlphaMode`, `OpaqueRendererMethod`, `RenderPhaseType`, the erased key
  types, wgpu-free `RenderPipelineDescriptor`/`BindGroupLayoutDescriptor`,
  the erased specialize fn signatures, label interning,
  `BindGroupLayoutEntries` builders.
- The concrete-wgpu material machinery is already in `bevy_render`, not
  `bevy_pbr`: `AsBindGroup` + derive, `MaterialBindGroupAllocator(s)`,
  `RenderMaterialBindings`, bindless, `ErasedRenderAsset`,
  batching/GPU-preprocessing substrate, `PipelineCache`.
- **No render crate depends on `bevy_pbr`** (`bevy_pbr → bevy_core_pipeline`,
  not vice versa). 2D/UI literally cannot reach the generic material systems
  today because they live in pbr, above them in the graph.

So `bevy_material_render` sits above `bevy_render` + `bevy_material`, below
`bevy_pbr` and `bevy_sprite_render`. What it absorbs (from
`bevy_pbr/src/material.rs`, deduplicating the `mesh2d/material.rs` clone):

| Moves to bevy_material_render | Stays in bevy_pbr |
|---|---|
| `Material` trait core (see hazard below), `MaterialPlugin<M>` generic wiring | `MeshMaterial3d<M>`, `StandardMaterial`, extended materials |
| `RenderMaterialInstances` + extract/sweep systems | `MaterialPipeline` (embeds `MeshPipeline`) unless abstracted |
| `SpecializedMaterialPipelineCache`, `EntitiesNeedingSpecialization`, check/extract specialization systems | shadow path: `SpecializedShadowMaterialPipelineCache`, `specialize_shadows`, `queue_shadows` |
| `SetMaterialBindGroup<I>`, `add_material_bind_group_allocator::<M>`, `PreparedMaterial` + `ErasedRenderAsset` impl body, `base_specialize` skeleton | `queue_material_meshes` (lightmaps, `OpaqueNoLightmap3d*` keys, Shadow phase) — or split generic-core + pbr hooks |
| main-pass draw-function/shader labels | prepass/deferred/shadow/meshlet labels |

Layering hazards:

1. **The "generic" systems drag 3D along.** `specialize_material_meshes` /
   `queue_material_meshes` reference `RenderLightmaps`,
   `OpaqueNoLightmap3dBatchSetKey`, the `Shadow` phase, `MeshInputUniform`
   buffers, `MeshPipeline`. Moving them wholesale pulls lightmaps/shadows
   into every 2D/UI consumer. They must be split into a generic core with
   pbr-supplied hooks (extra key bits, extra phases, extra draw functions).
2. **`ErasedRenderAsset::Param` requires all 9 phase `DrawFunctions` to
   prepare any material.** The required-phase set must become data-driven for
   2D (3 phases) and UI (1 phase) materials to prepare at all.
3. **`Material` trait defaults are 3D-flavored** (prepass/deferred/shadow
   hooks). Either genericize, or keep per-domain traits (`Material`,
   `Material2d`, later `UiMaterial`) that all lower into the same
   `MaterialProperties` — the type-erasure principle means the traits are
   cheap surface syntax; only the lowering must be shared.
4. **`RenderPhaseType` (4 variants, 3D-shaped) and the fixed material bind
   group index 3** (2D uses 2) both need to become data on
   `MaterialProperties` or per-domain configuration.
5. **`MeshMaterial3d`/`MeshMaterial2d` extraction** is per-component but
   generic in shape; either a generic-over-marker extraction or two thin
   extraction systems feeding one shared `RenderMaterialInstances`.

---

## 6. UI: inventory and constraints (later phase)

Architecture today: one sorted phase `TransparentUi`
(`sort_key = FloatOrd(stack_index + sub_z_offset)`) fed by **five parallel
pipelines** — base nodes (bg/border/image/text/debug), `UiMaterial<M>`, box
shadow, gradient, texture slice — each duplicating the same six-part
scaffold: extracted-vec, `*Meta` (RawBufferVec vertices + view bind group),
`*Batch`, `*Pipeline`+key, extract/queue/prepare systems, `Draw*` chain. The
coexistence trick: every prepare system iterates all phase items but only
processes those whose `item.entity()` matches its own extracted records,
using `item.index` as a back-pointer (`lib.rs:1624`, repeated in all five).
Everything is `TemporaryRenderEntity`, rebuilt per frame; nothing is retained.
`UiMaterial` is the one place UI touches `RenderAsset`/`AsBindGroup`, but it
reimplements the whole extract→draw path and shares nothing with
`Material`/`Material2d`.

UI-only requirements a unified renderer must express (ranked):

1. **Strict painter's-algorithm total order** from `ComputedStackIndex` +
   hand-assigned fractional `stack_z_offsets` (shadow −0.1 … cursor 0.08,
   `lib.rs:108-120`). No reordering; batching only across *adjacent*
   same-key items. This rules out binning for UI — UI maps only onto the
   sorted-phase model, and any shared batcher must be order-stable.
2. **No depth, no MSAA; analytic SDF anti-aliasing** in-shader
   (`ANTI_ALIAS` def from per-camera `UiAntiAlias`), pass drawn to the main
   camera's unsampled color target after PostProcess, before upscaling, in
   *both* Core2d and Core3d schedules (`lib.rs:265-272`).
3. **Clipping via CPU per-vertex clamping** (`positions_diff` idiom,
   duplicated in all five prepares) — no scissor/stencil anywhere; known
   broken under rotation. A unified path wants real scissor or clip-rect-
   as-instance-data.
4. **Dedicated UI subview per camera**: top-left-origin ortho projection,
   `RetainedViewEntity` subview 1, bidirectional `UiCameraView`/
   `UiViewTarget` links, viewport borrowed from the main camera. Multi-window
   and camera-target routing via `ComputedUiTargetCamera`.
5. **One entity → many primitives at distinct sub-Z** (bg + 4 dedup'd border
   quads + outline + gradient segments + image + glyphs + decorations), all
   transient. Primitive granularity is finer than entity granularity.
6. **Shape parameters as first-class data**: border radius per corner,
   per-edge border widths + edge flags, inverted fills, 9 gradient color
   spaces, 9-slice/tiling — currently five incompatible vertex layouts
   (8/5/14/7/7 attrs), packed per-corner rather than per-instance.
7. **Physical-pixel coordinates + per-view scale factor** for `Val`
   resolution; flipped-Y clip space vs the 2D/3D convention.

The realistic near-term win for UI is not "UI becomes meshes" but: port the
five pipelines onto shared retained/sorted-phase machinery, a common
instance-stream batcher (the same one sprites need — hazard §4.2), and lower
`UiMaterial` into `MaterialProperties`. The sub-z/stack-order contract and
the UI view model likely survive unification as-is.

---

## 7. Trouble list (ranked) and sequencing

Ranked risk register for the "fold 2D into 3D" phase:

| # | Risk | Severity | Notes |
|---|---|---|---|
| 1 | Splitting generic queue/specialize cores out of `queue_material_meshes`/`specialize_material_meshes` without perf/behavior regression in 3D | High | Lightmap/shadow/mesh-uniform coupling; this is the heart of `bevy_material_render` |
| 2 | 2D GPU preprocessing + multidraw (batch-set key schema, `Mesh2dInputUniform`, compute path currently pbr-only) | High | Or explicitly decide 2D stays CPU-batched behind the shared interface — the shared systems already support both backends run_if-gated |
| 3 | Per-instance texture selection for sprites/text (bindless/atlas-array) + transient instance streams | High | Blocks retiring SpritePipeline; SpriteMaterial prototype solves features but not batching |
| 4 | Required-phase set / `RenderPhaseType` / bind-group-index rigidity in the erased path | Medium | Mechanical but wide: touches `ErasedRenderAsset::Param`, shaders (`MATERIAL_BIND_GROUP`), all specializers |
| 5 | Preserving 2D public semantics: Z-sort contract, `AlphaMode2d` subset, `CompositingSpace`, in-shader tonemapping | Medium | User-visible; needs conformance examples/tests before refactor |
| 6 | Two retention mechanisms (`DirtySpecializations` + placeholder-entity hack vs 3D plumbing) | Medium | Reconcile around one retained-bin API |
| 7 | View bind group layout merge (2D bakes tonemapping LUTs into mesh view layout) | Medium | |
| 8 | Wireframe2d (tick-cache, push-constant immediates, own phase) and other stragglers | Low | Migrate last; useful canary |
| 9 | UI phase (everything in §6) | High but deferred | Prerequisites are #3's instance streams + #1's generic core |

Suggested sequencing that falls out of the dependency structure:

1. **Carve out `bevy_material_render`** (generic core of specialize/queue/
   instances/caches/`SetMaterialBindGroup`/`PreparedMaterial`), with
   `bevy_pbr` supplying mesh-pipeline + lightmap/shadow/prepass hooks.
   No behavior change; 3D is the only consumer initially.
2. **Point `mesh2d/material.rs` at it**, deleting the clone. Requires the
   mesh-pipeline abstraction (hazard: `MaterialPipeline` vs
   `Material2dPipeline`) and data-driven phase/bind-group-index sets.
3. **2D mesh instance path**: adopt the phase plugins, decide GPU
   preprocessing (implement `Mesh2dInputUniform` or formalize the CPU
   backend), fix the batch-set key schema, unify retention.
4. **Sprites**: build the per-instance-texture + transient-instance-stream
   capability, then fold SpritePipeline/text2d/texture_slice into it (or
   finish the `SpriteMesh` route and close its batching gaps). Merge
   `SpriteMesh` into `Sprite`.
5. **UI**: port the five pipelines onto the shared sorted-phase/instance
   machinery and lower `UiMaterial` into `MaterialProperties`, preserving
   the stack-order and view-model contracts.
