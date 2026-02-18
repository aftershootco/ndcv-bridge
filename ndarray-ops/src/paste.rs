use std::{
    borrow::Borrow,
    ops::{Div, Mul},
};

use bounding_box::{
    Aabb2,
    nalgebra::{Point2, Vector2},
};
use error_stack::ResultExt;
use ndarray::{Array3, ArrayView2, ArrayViewMut3, Axis, s};
use tap::{Pipe, TapOptional};

mod channel_paster;
mod color_paster;
mod image_paster;
mod traits;

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
            normalised_pos: NormalisedPos::mid_point(),
            anchor: Anchor::Center,
        }
    }
}

impl AnchoredPos {
    pub fn new(normalised_h: f64, normalised_w: f64, anchor: Anchor) -> Self {
        Self {
            normalised_pos: NormalisedPos::new(normalised_h, normalised_w),
            anchor,
        }
    }

    pub fn get_top_left_pos(&self, src_bounds: Bounds, to_paste_bounds: Bounds) -> Point2<i128> {
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
}

#[derive(Debug, Clone, Copy)]
struct Bounds(Aabb2<i128>);

impl Bounds {
    pub fn from_dim(height: usize, width: usize) -> Self {
        Self::new(0, 0, height, width)
    }

    /// Panics if min values > max values
    fn new(h_min: usize, w_min: usize, h_max: usize, w_max: usize) -> Self {
        let bbox = Aabb2::new(
            Point2::new(h_min as i128, w_min as i128),
            Point2::new(h_max as i128, w_max as i128),
        );

        Self(bbox)
    }

    fn try_new(h_min: usize, w_min: usize, h_max: usize, w_max: usize) -> Option<Self> {
        let bbox = Aabb2::try_new(
            Point2::new(h_min as i128, w_min as i128),
            Point2::new(h_max as i128, w_max as i128),
        )?;

        Some(Self(bbox))
    }

    pub fn h_min(&self) -> i128 {
        self.0.min_vertex().x
    }

    pub fn h_max(&self) -> i128 {
        self.0.max_vertex().x
    }

    pub fn w_min(&self) -> i128 {
        self.0.min_vertex().y
    }

    pub fn w_max(&self) -> i128 {
        self.0.max_vertex().y
    }

    pub fn intersection(self, other: Self) -> Option<Self> {
        self.0.intersection(other.0).map(Self)
    }

    pub fn translate(&self, translation: Vector2<i128>) -> Self {
        Self(self.0.translate(translation))
    }
}

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
        .map(|x| (*x.borrow() as f32) / 255.)
        .chain(std::iter::repeat(0.));

    // Safety:
    // it is infinite iterator because its chained with repeat, so it will never return none
    std::array::from_fn(|_| unsafe { it.next().unwrap_unchecked() })
}

#[inline(always)]
fn blend_f32_4(this: [f32; 4], other: [f32; 4], mask: f32, alpha: f32) -> [f32; 4] {
    let this = wide::f32x4::from(this);
    let other = wide::f32x4::from(other);
    let mask = wide::f32x4::splat(mask * alpha);

    (this * (1.0 - mask) + other * mask).to_array()
}

fn blend_f32(this: f32, other: f32, mask: f32, alpha: f32) -> f32 {
    let mask = mask * alpha;
    this * (1.0 - mask) + other * mask
}
