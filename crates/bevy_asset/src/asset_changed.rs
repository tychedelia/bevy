//! Define the [`AssetChanged`] query filter.
//!
//! Like [`Changed`], but for [`Asset`]s.

use bevy_ecs::{
    archetype::Archetype,
    component::{ComponentId, Tick},
    prelude::{Entity, Resource, World},
    query::{FilteredAccess, QueryFilter, QueryItem, ReadFetch, WorldQuery},
    storage::{Table, TableRow},
    world::unsafe_world_cell::UnsafeWorldCell,
};
use bevy_utils::HashMap;
use disqualified::ShortName;
use std::marker::PhantomData;

use crate::{AsAssetId, Asset, AssetId};

#[derive(Resource)]
pub(crate) struct AssetChanges<A: Asset> {
    change_ticks: HashMap<AssetId<A>, Tick>,
    last_change_tick: Tick,
}

impl<A: Asset> AssetChanges<A> {
    pub(crate) fn insert(&mut self, asset_id: AssetId<A>, tick: Tick) {
        self.last_change_tick = tick;
        self.change_ticks.insert(asset_id, tick);
    }
    pub(crate) fn remove(&mut self, asset_id: &AssetId<A>) {
        self.change_ticks.remove(asset_id);
    }
}

impl<A: Asset> Default for AssetChanges<A> {
    fn default() -> Self {
        Self {
            change_ticks: Default::default(),
            last_change_tick: Tick::new(0),
        }
    }
}

struct AssetChangeCheck<'w, A: AsAssetId> {
    change_ticks: &'w HashMap<AssetId<A::Asset>, Tick>,
    last_run: Tick,
    this_run: Tick,
}

impl<A: AsAssetId> Clone for AssetChangeCheck<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: AsAssetId> Copy for AssetChangeCheck<'_, A> {}

impl<'w, A: AsAssetId> AssetChangeCheck<'w, A> {
    fn new(changes: &'w AssetChanges<A::Asset>, last_run: Tick, this_run: Tick) -> Self {
        Self {
            change_ticks: &changes.change_ticks,
            last_run,
            this_run,
        }
    }
    // TODO(perf): some sort of caching? Each check has two levels of indirection,
    // which is not optimal.
    fn has_changed(&self, handle: &A) -> bool {
        let is_newer = |tick: &Tick| tick.is_newer_than(self.last_run, self.this_run);
        let id = handle.as_asset_id();

        self.change_ticks.get(&id).is_some_and(is_newer)
    }
}

/// Filter that selects entities with a `A` for an asset that changed
/// after the system last ran, where `A` is a component that implements
/// [`AsAssetId`].
///
/// Unlike `Changed<A>`, this is true whenever the asset for the `A`
/// in `ResMut<Assets<A>>` changed. For example, when a mesh changed through the
/// [`Assets<Mesh>::get_mut`] method, `AssetChanged<Mesh>` will iterate over all
/// entities with the `Handle<Mesh>` for that mesh. Meanwhile, `Changed<Handle<Mesh>>`
/// will iterate over no entities.
///
/// Swapping the actual `A` component is a common pattern. So you
/// should check for _both_ `AssetChanged<A>` and `Changed<A>` with
/// `Or<(Changed<A>, AssetChanged<A>)>`.
///
/// # Quirks
///
/// - Asset changes are registered in the [`AssetEvents`] schedule.
/// - Removed assets are not detected.
///
/// The list of changed assets only gets updated in the
/// [`AssetEvents`] schedule which runs in `Last`. Therefore, `AssetChanged`
/// will only pick up asset changes in schedules following `AssetEvents` or the
/// next frame. Consider adding the system in the `Last` schedule after [`AssetEvents`] if you need
/// to react without frame delay to asset changes.
///
/// # Performance
///
/// When at least one `A` is updated, this will
/// read a hashmap once per entity with a `A` component. The
/// runtime of the query is proportional to how many entities with a `A`
/// it matches.
///
/// If no `A` asset updated since the last time the system ran, then no lookups occur.
///
/// [`AssetEvents`]: crate::AssetEvents
/// [`Assets<Mesh>::get_mut`]: crate::Assets::get_mut
pub struct AssetChanged<A: AsAssetId>(PhantomData<A>);

/// Fetch for [`AssetChanged`].
#[doc(hidden)]
pub struct AssetChangedFetch<'w, A: AsAssetId> {
    inner: Option<ReadFetch<'w, A>>,
    check: AssetChangeCheck<'w, A>,
}

impl<'w, A: AsAssetId> Clone for AssetChangedFetch<'w, A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner,
            check: self.check,
        }
    }
}

/// State for [`AssetChanged`].
#[doc(hidden)]
pub struct AssetChangedState<A: AsAssetId> {
    asset_id: ComponentId,
    resource_id: ComponentId,
    _asset: PhantomData<fn(A)>,
}

// SAFETY: `ROQueryFetch<Self>` is the same as `QueryFetch<Self>`
unsafe impl<A: AsAssetId> WorldQuery for AssetChanged<A> {
    type Item<'w> = ();
    type Fetch<'w> = AssetChangedFetch<'w, A>;

    type State = AssetChangedState<A>;

    fn shrink<'wlong: 'wshort, 'wshort>(_: QueryItem<'wlong, Self>) -> QueryItem<'wshort, Self> {}

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(fetch: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {
        fetch
    }

    unsafe fn init_fetch<'w>(
        world: UnsafeWorldCell<'w>,
        state: &Self::State,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        let err_msg = || {
            panic!(
                "AssetChanges<{ty}> resource was removed, please do not remove \
                AssetChanges<{ty}> when using the AssetChanged<{ty}> world query",
                ty = ShortName::of::<A>()
            )
        };
        // SAFETY: `AssetChanges` is private and only accessed mutably in the `AssetEvents` schedule
        let changes: &AssetChanges<_> = unsafe { world.get_resource().unwrap_or_else(err_msg) };
        let has_updates = changes.last_change_tick.is_newer_than(last_run, this_run);

        AssetChangedFetch {
            inner: has_updates.then(||
                    // SAFETY: We delegate to the inner `init_fetch` for `A`
                    unsafe {
                        <&A>::init_fetch(world, &state.asset_id, last_run, this_run)
                    }),
            check: AssetChangeCheck::new(changes, last_run, this_run),
        }
    }

    const IS_DENSE: bool = <&A>::IS_DENSE;

    unsafe fn set_archetype<'w>(
        fetch: &mut Self::Fetch<'w>,
        state: &Self::State,
        archetype: &'w Archetype,
        table: &'w Table,
    ) {
        if let Some(inner) = &mut fetch.inner {
            // SAFETY: We delegate to the inner `set_archetype` for `A`
            unsafe {
                <&A>::set_archetype(inner, &state.asset_id, archetype, table);
            }
        }
    }

    unsafe fn set_table<'w>(fetch: &mut Self::Fetch<'w>, state: &Self::State, table: &'w Table) {
        if let Some(inner) = &mut fetch.inner {
            // SAFETY: We delegate to the inner `set_table` for `A`
            unsafe {
                <&A>::set_table(inner, &state.asset_id, table);
            }
        }
    }

    unsafe fn fetch<'w>(_: &mut Self::Fetch<'w>, _: Entity, _: TableRow) -> Self::Item<'w> {}

    #[inline]
    fn update_component_access(state: &Self::State, access: &mut FilteredAccess<ComponentId>) {
        <&A>::update_component_access(&state.asset_id, access);
        access.add_resource_read(state.resource_id);
    }

    fn init_state(world: &mut World) -> AssetChangedState<A> {
        let resource_id = world.init_resource::<AssetChanges<A::Asset>>();
        let asset_id = world.component_id::<A>().unwrap();
        AssetChangedState {
            asset_id,
            resource_id,
            _asset: PhantomData,
        }
    }

    fn get_state<'w>(world: impl Into<UnsafeWorldCell<'w>>) -> Option<Self::State> {
        // SAFETY:
        // - `world` is a valid world
        // -  we only access our private `AssetChanges` resource
        let world = unsafe { world.into().world() };

        let resource_id = world.resource_id::<AssetChanges<A::Asset>>()?;
        let asset_id = world.component_id::<A>()?;
        Some(AssetChangedState {
            asset_id,
            resource_id,
            _asset: PhantomData,
        })
    }

    fn matches_component_set(
        state: &Self::State,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        set_contains_id(state.asset_id)
    }
}

/// SAFETY: read-only access
unsafe impl<A: AsAssetId> QueryFilter for AssetChanged<A> {
    const IS_ARCHETYPAL: bool = false;

    #[inline]
    unsafe fn filter_fetch(
        fetch: &mut Self::Fetch<'_>,
        entity: Entity,
        table_row: TableRow,
    ) -> bool {
        fetch.inner.as_mut().map_or(false, |inner| {
            // SAFETY: We delegate to the inner `fetch` for `A`
            unsafe {
                let handle = <&A>::fetch(inner, entity, table_row);
                fetch.check.has_changed(handle)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{self as bevy_asset, AssetEvents, AssetPlugin, Handle};
    use std::num::NonZero;

    use crate::{AssetApp, Assets};
    use bevy_app::{App, AppExit, Last, Startup, Update};
    use bevy_core::TaskPoolPlugin;
    use bevy_ecs::schedule::IntoSystemConfigs;
    use bevy_ecs::{
        component::Component,
        event::EventWriter,
        system::{Commands, IntoSystem, Local, Query, Res, ResMut, Resource},
    };
    use bevy_reflect::TypePath;

    use super::*;

    #[derive(Asset, TypePath, Debug)]
    struct MyAsset(usize, &'static str);

    #[derive(Component)]
    struct MyComponent(Handle<MyAsset>);

    impl AsAssetId for MyComponent {
        type Asset = MyAsset;

        fn as_asset_id(&self) -> AssetId<Self::Asset> {
            self.0.id()
        }
    }

    fn run_app<Marker>(system: impl IntoSystem<(), (), Marker>) {
        let mut app = App::new();
        app.add_plugins((TaskPoolPlugin::default(), AssetPlugin::default()))
            .init_asset::<MyAsset>()
            .add_systems(Update, system);
        app.update();
    }

    // According to a comment in QueryState::new in bevy_ecs, components on filter
    // position shouldn't conflict with components on query position.
    #[test]
    fn handle_filter_pos_ok() {
        fn compatible_filter(
            _query: Query<&mut MyComponent, AssetChanged<MyComponent>>,
            mut exit: EventWriter<AppExit>,
        ) {
            exit.send(AppExit::Error(NonZero::<u8>::MIN));
        }
        run_app(compatible_filter);
    }

    #[derive(Default, PartialEq, Debug, Resource)]
    struct Counter(Vec<u32>);

    fn count_update(
        mut counter: ResMut<Counter>,
        assets: Res<Assets<MyAsset>>,
        query: Query<&MyComponent, AssetChanged<MyComponent>>,
    ) {
        println!("counting updates");
        for handle in query.iter() {
            let asset = assets.get(&handle.0).unwrap();
            counter.0[asset.0] += 1;
        }
    }

    fn update_some(mut assets: ResMut<Assets<MyAsset>>, mut run_count: Local<u32>) {
        let mut update_index = |i| {
            let id = assets
                .iter()
                .find_map(|(h, a)| (a.0 == i).then_some(h))
                .unwrap();
            let asset = assets.get_mut(id).unwrap();
            println!("setting new value for {}", asset.0);
            asset.1 = "new_value";
        };
        match *run_count {
            0 | 1 => update_index(0),
            2 => {}
            3 => {
                update_index(0);
                update_index(1);
            }
            4.. => update_index(1),
        };
        *run_count += 1;
    }

    fn add_some(
        mut assets: ResMut<Assets<MyAsset>>,
        mut cmds: Commands,
        mut run_count: Local<u32>,
    ) {
        match *run_count {
            1 => {
                cmds.spawn(MyComponent(assets.add(MyAsset(0, "init"))));
            }
            0 | 2 => {}
            3 => {
                cmds.spawn(MyComponent(assets.add(MyAsset(1, "init"))));
                cmds.spawn(MyComponent(assets.add(MyAsset(2, "init"))));
            }
            4.. => {
                cmds.spawn(MyComponent(assets.add(MyAsset(3, "init"))));
            }
        };
        *run_count += 1;
    }

    #[track_caller]
    fn assert_counter(app: &App, assert: Counter) {
        assert_eq!(&assert, app.world().resource::<Counter>());
    }

    #[test]
    fn added() {
        let mut app = App::new();

        app.add_plugins((TaskPoolPlugin::default(), AssetPlugin::default()))
            .init_asset::<MyAsset>()
            .insert_resource(Counter(vec![0, 0, 0, 0]))
            .add_systems(Update, add_some)
            .add_systems(Last, count_update.after(AssetEvents));

        // First run of the app, `add_systems(Startup…)` runs.
        app.update(); // run_count == 0
        assert_counter(&app, Counter(vec![0, 0, 0, 0]));
        app.update(); // run_count == 1
        assert_counter(&app, Counter(vec![1, 0, 0, 0]));
        app.update(); // run_count == 2
        assert_counter(&app, Counter(vec![1, 0, 0, 0]));
        app.update(); // run_count == 3
        assert_counter(&app, Counter(vec![1, 1, 1, 0]));
        app.update(); // run_count == 4
        assert_counter(&app, Counter(vec![1, 1, 1, 1]));
    }

    #[test]
    fn changed() {
        let mut app = App::new();

        app.add_plugins((TaskPoolPlugin::default(), AssetPlugin::default()))
            .init_asset::<MyAsset>()
            .insert_resource(Counter(vec![0, 0]))
            .add_systems(
                Startup,
                |mut cmds: Commands, mut assets: ResMut<Assets<MyAsset>>| {
                    let asset0 = assets.add(MyAsset(0, "init"));
                    let asset1 = assets.add(MyAsset(1, "init"));
                    cmds.spawn(MyComponent(asset0.clone()));
                    cmds.spawn(MyComponent(asset0));
                    cmds.spawn(MyComponent(asset1.clone()));
                    cmds.spawn(MyComponent(asset1.clone()));
                    cmds.spawn(MyComponent(asset1));
                },
            )
            .add_systems(Update, update_some)
            .add_systems(Last, count_update.after(AssetEvents));

        // First run of the app, `add_systems(Startup…)` runs.
        app.update(); // run_count == 0

        // First run: We count the entities that were added in the `Startup` schedule
        assert_counter(&app, Counter(vec![2, 3]));

        // Second run: `update_once` updates the first asset, which is
        // associated with two entities, so `count_update` picks up two updates
        app.update(); // run_count == 1
        assert_counter(&app, Counter(vec![4, 3]));

        // Third run: `update_once` doesn't update anything, same values as last
        app.update(); // run_count == 2
        assert_counter(&app, Counter(vec![4, 3]));

        // Fourth run: We update the two assets (asset 0: 2 entities, asset 1: 3)
        app.update(); // run_count == 3
        assert_counter(&app, Counter(vec![6, 6]));

        // Fifth run: only update second asset
        app.update(); // run_count == 4
        assert_counter(&app, Counter(vec![6, 9]));
        // ibid
        app.update(); // run_count == 5
        assert_counter(&app, Counter(vec![6, 12]));
    }
}
