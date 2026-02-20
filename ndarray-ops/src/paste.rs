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

pub mod channel_paster;
pub mod color_paster;
pub mod image_paster;
pub mod paste_algos;
pub mod traits;

#[derive(Debug, thiserror::Error)]
#[error("Paste Error")]
pub struct PasteError;

#[derive(Debug, Clone, Copy)]
pub struct AnchoredPos {
    pub normalised_pos: NormalisedPos,

    /// Anchor on your image which will correspond to the above position on the source image
    ///
    /// # Example
    ///
    /// ```ignore
    /// // This means the center of your image will be placed at the center of the source image
    /// AnchoredPos {
    ///   normalised_pos: NormalisedPos::mid_point(),
    ///   anchor: Ancrho::Center
    /// };
    /// ```
    pub anchor: Anchor,
}

impl Default for AnchoredPos {
    fn default() -> Self {
        Self {
            normalised_pos: NormalisedPos::zero(),
            anchor: Anchor::TopLeft,
        }
    }
}

impl AnchoredPos {
    pub fn new(normalised_pos: NormalisedPos, anchor: Anchor) -> Self {
        Self {
            normalised_pos,
            anchor,
        }
    }

    pub fn from_dim(normalised_h: f64, normalised_w: f64, anchor: Anchor) -> Self {
        Self::new(NormalisedPos::new(normalised_h, normalised_w), anchor)
    }

    fn get_top_left_pos(&self, src_bounds: Bounds, to_paste_bounds: Bounds) -> Point2<i128> {
        let h =
            // TODO: use bounds.len
            (src_bounds.h_max() - src_bounds.h_min()) as f64 * self.normalised_pos.h + src_bounds.h_min() as f64;
        let w = (src_bounds.w_max() - src_bounds.w_min()) as f64 * self.normalised_pos.w
            + src_bounds.w_min() as f64;

        match self.anchor {
            Anchor::TopLeft => Point2::new(h as i128, w as i128),
            Anchor::Center => Point2::new(
                // TODO: use bounds.len
                (h - ((to_paste_bounds.h_max() - to_paste_bounds.h_min()) as f64) / 2.).floor()
                    as i128,
                (w - ((to_paste_bounds.w_max() - to_paste_bounds.w_min()) as f64) / 2.).floor()
                    as i128,
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
pub struct NormalisedPos {
    pub h: f64,
    pub w: f64,
}

impl NormalisedPos {
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

#[derive(Debug, Clone, Copy)]
pub struct PasteOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub mask_info: Option<MaskInfo<'a, A>>,
    pub alpha: f32,
    pub pos: AnchoredPos,
    pub paste_algo: F,
    phantom_t: PhantomData<T>,
}

#[derive(Debug, Clone, Copy)]
pub struct MaskInfo<'a, A> {
    pub mask: ArrayView2<'a, A>,
    pub pos: AnchoredPos,
}

pub struct PasteInput<T> {
    pub this: T,
    pub other: T,
    pub mask: f32,
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
    pub fn new() -> Self {
        Self {
            mask_info: None,
            alpha: 1.,
            pos: AnchoredPos::default(),
            paste_algo: paste_algos::blend,
            phantom_t: PhantomData::default(),
        }
    }
}

impl<'a, A, T, F> PasteOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub fn with_mask(mut self, mask: ArrayView2<'a, A>, pos: AnchoredPos) -> Self {
        self.mask_info = Some(MaskInfo { mask: mask, pos });
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_pos(mut self, pos: AnchoredPos) -> Self {
        self.pos = pos;
        self
    }

    /// Change the algo which blends other with self
    ///
    /// # Example
    /// ```ignore
    /// // algos::blend is the default paste algo
    /// let opts = PasteOpts::new().with_paste_algo(ndarray_ops::paste::paste_algos::blend);
    /// ```
    pub fn with_paste_algo<F2>(self, algo: F2) -> PasteOpts<'a, A, T, F2>
    where
        F2: Fn(PasteInput<T>) -> T + Send + Sync,
    {
        PasteOpts {
            mask_info: self.mask_info,
            alpha: self.alpha,
            pos: self.pos,
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
) -> Option<Sections<'a, A>> {
    let other_translation = pos.get_top_left_pos(this_bounds, other_bounds).coords;

    let other_translated = other_bounds.translate(other_translation);

    let (this_bounds, mask_info) = if let Some(MaskInfo {
        mask,
        pos: mask_position,
    }) = mask_info
    {
        let (mh, mw) = mask.dim();
        let mask_bounds = Bounds::from_dim(mh, mw);
        let mask_translation = mask_position
            .get_top_left_pos(this_bounds, mask_bounds)
            .coords;

        let mask_translated = mask_bounds.translate(mask_translation);

        let img_bounds = this_bounds
            .intersection(other_translated)
            .and_then(|x| x.intersection(mask_translated));
        (img_bounds, Some((mask, mask_translation)))
    } else {
        let img_bounds = this_bounds.intersection(other_translated);
        (img_bounds, None)
    };

    this_bounds.map(|x| Sections {
        this_bounds: x,
        other_translation: other_translation,
        mask_with_translation: mask_info,
    })
}
