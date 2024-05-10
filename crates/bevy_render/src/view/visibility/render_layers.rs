use bevy_ecs::prelude::{Component, ReflectComponent};
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use smallvec::SmallVec;

pub const DEFAULT_LAYERS: &RenderLayers = &RenderLayers::layer(0);

/// An identifier for a rendering layer.
pub type Layer = usize;

/// Describes which rendering layers an entity belongs to.
///
/// Cameras with this component will only render entities with intersecting
/// layers.
///
/// There are 32 layers numbered `0` - [`TOTAL_LAYERS`](RenderLayers::TOTAL_LAYERS). Entities may
/// belong to one or more layers, or no layer at all.
///
/// The [`Default`] instance of `RenderLayers` contains layer `0`, the first layer.
///
/// An entity with this component without any layers is invisible.
///
/// Entities without this component belong to layer `0`.
#[derive(Component, Clone, Reflect, PartialEq, Eq, PartialOrd, Ord)]
#[reflect(Component, Default, PartialEq)]
pub struct RenderLayers(SmallVec<[u64; 1]>);

impl Default for &RenderLayers {
    fn default() -> Self {
        DEFAULT_LAYERS
    }
}

impl std::fmt::Debug for RenderLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RenderLayers")
            .field(&self.iter().collect::<Vec<_>>())
            .finish()
    }
}

impl FromIterator<Layer> for RenderLayers {
    fn from_iter<T: IntoIterator<Item = Layer>>(i: T) -> Self {
        i.into_iter().fold(Self::none(), |mask, g| mask.with(g))
    }
}

impl Default for RenderLayers {
    /// By default, this structure includes layer `0`, which represents the first layer.
    fn default() -> Self {
        let (_, bit) = Self::layer_info(0);
        RenderLayers(SmallVec::from_const([bit]))
    }
}

impl RenderLayers {
    /// Create a new `RenderLayers` belonging to the given layer.
    pub const fn layer(n: Layer) -> Self {
        let (buffer_index, bit) = Self::layer_info(n);
        assert!(
            buffer_index < 1,
            "layer is out of bounds for const construction"
        );
        RenderLayers(SmallVec::from_const([bit]))
    }

    /// Create a new `RenderLayers` that belongs to no layers.
    pub const fn none() -> Self {
        RenderLayers(SmallVec::from_const([0]))
    }

    /// Create a `RenderLayers` from a list of layers.
    pub fn from_layers(layers: &[Layer]) -> Self {
        layers.iter().copied().collect()
    }

    /// Add the given layer.
    ///
    /// This may be called multiple times to allow an entity to belong
    /// to multiple rendering layers. The maximum layer is `TOTAL_LAYERS - 1`.
    ///
    /// # Panics
    /// Panics when called with a layer greater than `TOTAL_LAYERS - 1`.
    #[must_use]
    pub fn with(mut self, layer: Layer) -> Self {
        let (buffer_index, bit) = Self::layer_info(layer);
        Self(SmallVec::new_const()).extend_buffer(buffer_index + 1);
        self.0[buffer_index] |= bit;
        self
    }

    /// Removes the given rendering layer.
    ///
    /// # Panics
    /// Panics when called with a layer greater than `TOTAL_LAYERS - 1`.
    #[must_use]
    pub fn without(mut self, layer: Layer) -> Self {
        let (buffer_index, bit) = Self::layer_info(layer);
        if buffer_index < self.0.len() {
            self.0[buffer_index] &= !bit;
        }
        self
    }

    /// Get an iterator of the layers.
    pub fn iter(&self) -> impl Iterator<Item = Layer> + '_ {
        self.0.iter().copied().flat_map(Self::iter_layers)
    }

    /// Determine if a `RenderLayers` intersects another.
    ///
    /// `RenderLayers`s intersect if they share any common layers.
    ///
    /// A `RenderLayers` with no layers will not match any other
    /// `RenderLayers`, even another with no layers.
    pub fn intersects(&self, other: &RenderLayers) -> bool {
        for (self_layer, other_layer) in self.0.iter().zip(other.0.iter()) {
            if (*self_layer & *other_layer) != 0 {
                return true;
            }
        }

        false
    }

    /// get the bitmask representation of the contained layers
    pub fn bits(&self) -> &[u64] {
        self.0.as_slice()
    }

    const fn layer_info(layer: usize) -> (usize, u64) {
        let buffer_index = layer / 64;
        let bit_index = layer % 64;
        let bit = 1u64 << bit_index;

        (buffer_index, bit)
    }

    fn extend_buffer(&mut self, other_len: usize) {
        let new_size = std::cmp::max(self.0.len(), other_len);
        self.0.reserve_exact(new_size - self.0.len());
        self.0.resize(new_size, 0u64);
    }

    fn iter_layers(mut buffer: u64) -> impl Iterator<Item = Layer> + 'static {
        let mut layer: usize = 0;
        std::iter::from_fn(move || {
            if buffer == 0 {
                return None;
            }
            let next = buffer.trailing_zeros() + 1;
            buffer >>= next;
            layer += next as usize;
            Some(layer - 1)
        })
    }
}

#[cfg(test)]
mod rendering_mask_tests {
    use super::{Layer, RenderLayers};
    use smallvec::SmallVec;

    #[test]
    fn rendering_mask_sanity() {
        let layer_0 = RenderLayers::layer(0);
        assert_eq!(layer_0.0.len(), 1, "layer 0 is one buffer");
        assert_eq!(layer_0.0[0], 1, "layer 0 is mask 1");
        let layer_1 = RenderLayers::layer(1);
        assert_eq!(layer_1.0.len(), 1, "layer 1 is one buffer");
        assert_eq!(layer_1.0[0], 2, "layer 1 is mask 2");
        let layer_0_1 = RenderLayers::layer(0).with(1);
        assert_eq!(layer_0_1.0.len(), 1, "layer 0 + 1 is one buffer");
        assert_eq!(layer_0_1.0[0], 3, "layer 0 + 1 is mask 3");
        let layer_0_1_without_0 = layer_0_1.without(0);
        assert_eq!(
            layer_0_1_without_0.0.len(),
            1,
            "layer 0 + 1 - 0 is one buffer"
        );
        assert_eq!(layer_0_1_without_0.0[0], 2, "layer 0 + 1 - 0 is mask 2");
        assert!(
            RenderLayers::layer(1).intersects(&RenderLayers::layer(1)),
            "layers match like layers"
        );
        assert!(
            RenderLayers::layer(0).intersects(&RenderLayers(SmallVec::from_const([1]))),
            "a layer of 0 means the mask is just 1 bit"
        );

        assert!(
            RenderLayers::layer(0)
                .with(3)
                .intersects(&RenderLayers::layer(3)),
            "a mask will match another mask containing any similar layers"
        );

        assert!(
            RenderLayers::default().intersects(&RenderLayers::default()),
            "default masks match each other"
        );

        assert!(
            !RenderLayers::layer(0).intersects(&RenderLayers::layer(1)),
            "masks with differing layers do not match"
        );
        assert!(
            !RenderLayers::none().intersects(&RenderLayers::none()),
            "empty masks don't match"
        );
        assert_eq!(
            RenderLayers::from_layers(&[0, 2, 16, 30])
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 2, 16, 30],
            "from_layers and get_layers should roundtrip"
        );
        assert_eq!(
            format!("{:?}", RenderLayers::from_layers(&[0, 1, 2, 3])).as_str(),
            "RenderLayers([0, 1, 2, 3])",
            "Debug instance shows layers"
        );
        assert_eq!(
            RenderLayers::from_layers(&[0, 1, 2]),
            <RenderLayers as FromIterator<Layer>>::from_iter(vec![0, 1, 2]),
            "from_layers and from_iter are equivalent"
        );
    }
}
