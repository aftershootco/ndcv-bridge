use std::{
    borrow::Borrow,
    marker::PhantomData,
    ops::{Add, Mul},
};

use bounding_box::{
    Aabb2,
    nalgebra::{Point2, Vector2},
};
use ndarray::ArrayView2;

pub mod channel;
pub mod color;
pub mod image;
pub mod paste_algos;
pub mod traits;

const DEFAULT_MASK: f32 = 0.;

#[derive(Debug, thiserror::Error)]
#[error("Paste Error")]
pub struct PasteError;

/// A position on the source image with an anchor describing which point of
/// the overlay image should align with it.
///
/// The [`pos`](AnchoredPos::pos) specifies the target location on the source
/// image. The [`anchor`](AnchoredPos::anchor) specifies which point on the
/// overlay image will be placed at that location.
///
/// # Example
///
/// ```ignore
/// // This means the center of your image will be placed at the center
/// // of the source image.
/// AnchoredPos {
///     pos: Pos::normalized(0.5, 0.5),
///     anchor: Anchor::Center,
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AnchoredPos {
    /// Target position on the source image.
    pub pos: Pos,

    /// Point on the overlay image that will align with [`pos`](AnchoredPos::pos).
    pub anchor: Anchor,
}

impl Default for AnchoredPos {
    fn default() -> Self {
        Self {
            pos: Pos::Abs(AbsolutePos::new(0, 0)),
            anchor: Anchor::TopLeft,
        }
    }
}

impl AnchoredPos {
    pub fn new(pos: Pos, anchor: Anchor) -> Self {
        Self { pos, anchor }
    }

    pub fn from_dim_norm(normalised_h: f64, normalised_w: f64, anchor: Anchor) -> Self {
        Self::new(Pos::norm(normalised_h, normalised_w), anchor)
    }

    pub fn from_dim_abs(h: i64, w: i64, anchor: Anchor) -> Self {
        Self::new(Pos::abs(h, w), anchor)
    }

    /// Returns the corresponding min vertex or "the top left point" for the "other" image.
    /// Treats height as x and width as y, hence places top left point on the origin (0, 0) and bottom
    /// right point on (height, width).
    pub fn get_min_vertex(
        &self,
        src_h: usize,
        src_w: usize,
        other_h: usize,
        other_w: usize,
    ) -> Point2<i128> {
        self.get_top_left_point(
            Bounds::from_dim(src_h, src_w),
            Bounds::from_dim(other_h, other_w),
        )
    }

    fn get_top_left_point(&self, src_bounds: Bounds, other_bounds: Bounds) -> Point2<i128> {
        let (h, w) = match self.pos {
            Pos::Norm(norm_pos) => {
                // TODO: use bounds.len
                let h = (src_bounds.h_max() - src_bounds.h_min()) as f64 * norm_pos.h
                    + src_bounds.h_min() as f64;
                let w = (src_bounds.w_max() - src_bounds.w_min()) as f64 * norm_pos.w
                    + src_bounds.w_min() as f64;

                (h, w)
            }
            Pos::Abs(abs_pos) => (abs_pos.h as f64, abs_pos.w as f64),
        };

        match self.anchor {
            Anchor::TopLeft => Point2::new(h.round() as i128, w.round() as i128),
            Anchor::Center => Point2::new(
                // TODO: use bounds.len
                (h - ((other_bounds.h_max() - other_bounds.h_min()) as f64) / 2.).round() as i128,
                (w - ((other_bounds.w_max() - other_bounds.w_min()) as f64) / 2.).round() as i128,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Anchor {
    TopLeft,
    Center,
}

#[derive(Debug, Clone, Copy)]
pub enum Pos {
    Norm(NormalizedPos),
    Abs(AbsolutePos),
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizedPos {
    pub h: f64,
    pub w: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct AbsolutePos {
    pub h: i64,
    pub w: i64,
}

impl Pos {
    pub fn norm(normalized_h: f64, normalized_w: f64) -> Self {
        Self::Norm(NormalizedPos::new(normalized_h, normalized_w))
    }

    pub fn abs(h: i64, w: i64) -> Self {
        Self::Abs(AbsolutePos::new(h, w))
    }
}

impl NormalizedPos {
    pub fn new(h: f64, w: f64) -> Self {
        Self { h, w }
    }

    pub fn mid_point() -> Self {
        Self::new(0.5, 0.5)
    }

    pub fn zero() -> Self {
        Self::new(0., 0.)
    }
}

impl AbsolutePos {
    pub fn new(h: i64, w: i64) -> Self {
        Self { h, w }
    }
}

/// Information describing the bounding box used for pasting.
///
/// This determines **where pasting is allowed to occur** in the source image.
/// The bounding box is first resolved from [`bbox_type`], then positioned using
/// [`pos`].
///
/// By default this represents the entire source image.
#[derive(Debug, Clone, Copy, Default)]
pub struct PasteRegionInfo {
    pub bbox_type: PasteRegion,

    /// Position of the bounding box within the source image.
    pub pos: AnchoredPos,
}

impl PasteRegionInfo {
    fn get_bounds(&self, this_bounds: Bounds) -> Bounds {
        let bounds = match self.bbox_type {
            PasteRegion::Abs(bbox) => Bounds(bbox.as_()),
            PasteRegion::Norm(norm) => {
                Bounds(norm.denormalize(this_bounds.0.as_().size()).round().as_())
            }
        };

        let translation = self.pos.get_top_left_point(this_bounds, bounds).coords;

        bounds.translate(translation)
    }
}

/// A bounding box describing the region where pasting may occur.
///
/// The bounding box can either be specified:
///
/// - in **absolute pixel coordinates**, or
/// - in **normalized coordinates** relative to the source image.
///
/// Normalized coordinates are in the range `[0, 1]`, where `(1, 1)` represents
/// the full size of the source image.
///
/// Height is on the **+x-axis** and width is on the **+y-axis**.
/// Recommended to use `norm` and `abs` functions to create bbox, as they abstract away the mapping
/// of height and width to the coordinate system
///
/// # Example
///
/// ```ignore
/// // to create a normalised bbox of half width and height of
/// let bbox = PasteBbox::norm(0.5, 0.5);
///
/// // to create an absolute bbox
/// let bbox = PasteBbox::abs(50, 100);
/// ```
#[derive(Debug, Clone, Copy)]
pub enum PasteRegion {
    /// Bounding box specified in absolute pixel coordinates.
    Abs(Aabb2<usize>),

    /// Bounding box specified in normalized coordinates relative to the source
    /// image dimensions.
    Norm(Aabb2<f64>),
}

impl Default for PasteRegion {
    fn default() -> Self {
        Self::Norm(Aabb2::new_point_size(
            Point2::origin(),
            Vector2::new(1., 1.),
        ))
    }
}

impl PasteRegion {
    /// Creates a normalized bounding box starting at the origin.
    ///
    /// `h` and `w` represent the normalized height and width of the region
    /// relative to the source image.
    pub fn norm(h: f64, w: f64) -> Self {
        let v1 = Point2::origin();
        let v2 = Point2::new(h, w);

        let (min_ver, max_vert) = v1.inf_sup(&v2);
        Self::Norm(Aabb2::new(min_ver, max_vert))
    }

    /// Creates an absolute bounding box starting at the origin.
    ///
    /// `h` and `w` represent the height and width in pixels.
    pub fn abs(h: usize, w: usize) -> Self {
        Self::Abs(Aabb2::new(Point2::origin(), Point2::new(h, w)))
    }
}

/// Options controlling how one image is pasted onto another.
///
/// `PasteOpts` configures:
///
/// - the position of the pasted image
/// - optional masking
/// - alpha/opacity of the pasted image
/// - the region where pasting is allowed
/// - the blending algorithm used
///
/// Most users can start with [`PasteOpts::new`] and customize the options
/// using the provided builder-style methods.
#[derive(Debug, Clone, Copy)]
pub struct PasteOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    /// Optional mask used to control blending.
    ///
    /// When present, the mask value is passed to the paste algorithm as
    /// `PasteInput::mask`.
    pub mask_info: Option<MaskInfo<'a, A>>,

    /// Global alpha applied to the `other` image during blending.
    pub alpha: f32,

    /// Position of the `other` image relative to the source image.
    pub pos: AnchoredPos,

    /// Bounding box restricting where pasting can occur.
    ///
    /// Pasting will only happen in the region where this bounding box
    /// intersects the source image.
    pub paste_region: PasteRegionInfo,

    /// The blending algorithm used to combine pixels.
    ///
    /// The function receives [`PasteInput`] containing the source pixel,
    /// overlay pixel, mask value, and alpha.
    pub paste_algo: F,

    phantom_t: PhantomData<T>,
}

/// Information describing a mask used during pasting.
///
/// The mask controls **per-pixel blending strength** for the pasted image.
#[derive(Debug, Clone, Copy)]
pub struct MaskInfo<'a, A> {
    /// Mask values.
    ///
    /// Typically expected to be in the range `[0, 1]`.
    pub mask: ArrayView2<'a, A>,

    /// Position of the mask relative to the source image.
    pub pos: AnchoredPos,
}

/// Input provided to the paste blending algorithm.
///
/// Each call represents the data required to compute a single output pixel.
pub struct PasteInput<T> {
    /// Pixel value from the source image.
    pub this: T,

    /// Pixel value from the image being pasted.
    pub other: T,

    /// Mask value at this pixel in the range `[0, 1]`.
    pub mask: f32,

    /// Global alpha applied to `other` in the range `[0, 1]`.
    pub alpha: f32,
}

impl<T> PasteInput<T> {
    pub fn into<O>(self) -> PasteInput<O>
    where
        T: Into<O>,
    {
        PasteInput {
            this: self.this.into(),
            other: self.other.into(),
            mask: self.mask,
            alpha: self.alpha,
        }
    }
}

impl<O, T> From<(O, O, f32, f32)> for PasteInput<T>
where
    O: Into<T>,
{
    fn from(value: (O, O, f32, f32)) -> Self {
        Self {
            this: value.0.into(),
            other: value.1.into(),
            mask: value.2,
            alpha: value.3,
        }
    }
}

impl<'a, A, T> PasteOpts<'a, A, T, fn(PasteInput<T>) -> T>
where
    T: Copy,
    T: Add<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Mul<f32, Output = T>,
{
    /// Creates a new set of paste options with sensible defaults.
    ///
    /// Defaults:
    ///
    /// - `alpha = 1.0`
    /// - no mask
    /// - position at the origin
    /// - bounding box covering the entire source image
    /// - default blending algorithm (`blend`)
    pub fn new() -> Self {
        Self {
            mask_info: None,
            alpha: 1.,
            pos: AnchoredPos::default(),
            paste_region: PasteRegionInfo::default(),
            paste_algo: paste_algos::blend,
            phantom_t: PhantomData::default(),
        }
    }
}

impl<'a, A, T, F> PasteOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    /// Adds a mask used during blending.
    ///
    /// The mask is positioned using the provided [`AnchoredPos`].
    pub fn with_mask(mut self, mask: ArrayView2<'a, A>, pos: AnchoredPos) -> Self {
        self.mask_info = Some(MaskInfo { mask: mask, pos });
        self
    }

    /// Sets the global alpha applied to the `other` image.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the position where the `other` image will be pasted.
    pub fn with_pos(mut self, pos: AnchoredPos) -> Self {
        self.pos = pos;
        self
    }

    /// Sets the bounding box restricting where pasting may occur.
    ///
    /// Pasting will only happen in the region where this bounding box
    /// intersects the source image.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray_ops::prelude::*;
    ///
    /// // Paste bbox of half width and height of source image, placed at its center
    /// let opts = PasteOpts::new().with_paste_region(
    ///     PasteRegion::norm(0.5, 0.5),
    ///     AnchoredPos::from_dim_norm(0.5, 0.5, Anchor::Center),
    /// );
    /// ```
    pub fn with_paste_region(
        mut self,
        bbox: PasteRegion,
        pos: impl Into<Option<AnchoredPos>>,
    ) -> Self {
        self.paste_region = PasteRegionInfo {
            bbox_type: bbox,
            pos: pos.into().unwrap_or_default(),
        };

        self
    }

    /// Change the algo which blends other with self
    ///
    /// # Example
    /// ```ignore
    /// use ndarray_ops::prelude::*;
    ///
    /// // algos::blend is the default paste algo
    /// let opts = PasteOpts::new().with_paste_algo(paste_algos::blend);
    ///
    /// // treat mask as background mask instead of subject mask
    /// let opts = PasteOpts::new().with_paste_algo(paste_algos::blend_inverted_mask);
    /// ```
    pub fn with_paste_algo<F2>(self, algo: F2) -> PasteOpts<'a, A, T, F2>
    where
        F2: Fn(PasteInput<T>) -> T + Send + Sync,
    {
        PasteOpts {
            mask_info: self.mask_info,
            alpha: self.alpha,
            pos: self.pos,
            paste_region: self.paste_region,
            paste_algo: algo,
            phantom_t: self.phantom_t,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Bounds(Aabb2<i128>);

impl Bounds {
    fn from_dim(height: usize, width: usize) -> Self {
        Self::new(0, 0, height, width)
    }

    /// If min values > max values, finds the correct min points for the bbox
    fn new(h_min: usize, w_min: usize, h_max: usize, w_max: usize) -> Self {
        let min_point = Point2::new(h_min as i128, w_min as i128);
        let max_point = Point2::new(h_max as i128, w_max as i128);

        let (min_point, max_point) = min_point.inf_sup(&max_point);

        let bbox = Aabb2::new(min_point, max_point);

        Self(bbox)
    }

    fn h_min(&self) -> i128 {
        self.0.min_vertex().x
    }

    fn h_max(&self) -> i128 {
        self.0.max_vertex().x
    }

    fn w_min(&self) -> i128 {
        self.0.min_vertex().y
    }

    fn w_max(&self) -> i128 {
        self.0.max_vertex().y
    }

    fn intersection(self, other: Self) -> Option<Self> {
        self.0.intersection(other.0).map(Self)
    }

    fn translate(&self, translation: Vector2<i128>) -> Self {
        Self(self.0.translate(translation))
    }
}

struct Sections<'a, A> {
    this_bounds: Bounds,
    other_translation: Vector2<i128>,
    mask_with_translation: Option<(ArrayView2<'a, A>, Vector2<i128>)>,
}

#[expect(dead_code)]
#[inline(always)]
fn convert_to_f32_4(x: [u8; 4]) -> [f32; 4] {
    std::array::from_fn(|i| convert_to_f32(x[i]))
}

#[inline(always)]
fn convert_to_u8_4(x: [f32; 4]) -> [u8; 4] {
    std::array::from_fn(|i| convert_to_u8(x[i]))
}

#[inline(always)]
fn convert_to_f32(x: u8) -> f32 {
    (x as f32) / 255.
}

#[inline(always)]
fn convert_to_u8(x: f32) -> u8 {
    x.clamp(0., 1.).mul(255.) as u8
}

fn from_iter_to_f32_4<T: Borrow<u8>>(it: impl IntoIterator<Item = T>) -> [f32; 4] {
    let mut it = it
        .into_iter()
        .map(|x| convert_to_f32(*x.borrow()))
        .chain(std::iter::repeat(0.));

    // Safety:
    // 'it' is an infinite iterator because its chained with repeat, so it will never return none
    std::array::from_fn(|_| unsafe { it.next().unwrap_unchecked() })
}

fn get_tri_intersection<'a, A>(
    this_bounds: Bounds,
    other_bounds: Bounds,
    mask_info: Option<MaskInfo<'a, A>>,
    pos: AnchoredPos,
    paste_bbox: PasteRegionInfo,
) -> Option<Sections<'a, A>> {
    let paste_bounds = paste_bbox.get_bounds(this_bounds);

    let other_translation = pos.get_top_left_point(this_bounds, other_bounds).coords;

    let other_translated = other_bounds.translate(other_translation);

    let (this_bounds, mask_info) = if let Some(MaskInfo {
        mask,
        pos: mask_position,
    }) = mask_info
    {
        let (mh, mw) = mask.dim();
        let mask_bounds = Bounds::from_dim(mh, mw);
        let mask_translation = mask_position
            .get_top_left_point(this_bounds, mask_bounds)
            .coords;

        let mask_translated = mask_bounds.translate(mask_translation);

        let img_bounds = this_bounds
            .intersection(other_translated)
            .and_then(|x| x.intersection(mask_translated))?;
        (img_bounds, Some((mask, mask_translation)))
    } else {
        let img_bounds = this_bounds.intersection(other_translated)?;
        (img_bounds, None)
    };

    let this_bounds = this_bounds.intersection(paste_bounds)?;

    Some(Sections {
        this_bounds,
        other_translation,
        mask_with_translation: mask_info,
    })
}
