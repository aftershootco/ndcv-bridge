use std::borrow::Borrow;
use std::ops::Div;

use bounding_box::nalgebra::Point2;
use error_stack::ResultExt;
use ndarray::{Array3, ArrayView, ArrayView2, ArrayView3, ArrayViewMut3, AsArray, Axis, Zip, s};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use tap::Pipe;

use crate::paste::{AnchoredPos, Bounds};

use super::NormalisedPos;
use super::traits::{Paste, PasteConfig, PasteError};

#[derive(Debug, Clone, Copy)]
pub struct ImagePaster<'a> {
    // TODO: use getset?
    to_paste: ArrayView3<'a, u8>,
    // TODO: enum for position types?
    position: AnchoredPos,
    mask: Option<ArrayView2<'a, u8>>,
    mask_position: AnchoredPos,
    alpha: f32,
    pow: f32,
}

impl<'a> ImagePaster<'a> {
    pub fn new(to_paste: ArrayView3<'a, u8>) -> Self {
        Self {
            to_paste: to_paste,
            position: AnchoredPos::default(),
            mask: None,
            alpha: 1.,
            mask_position: AnchoredPos::default(),
            pow: 1.,
        }
    }

    /// On which channel of Array3, to paste the array on.
    /// If channel is out of bounds, a channel will be appeneded to the array
    /// with default values of 255_u8.
    ///
    /// # Example
    /// ```ignore
    /// // Paste at 4th channel
    /// let mask = ndarray::Array2::ones((100, 100)) * 255_u8;
    /// ChannelPaster::new(mask.view()).with_channel_idx(3);
    /// ```
    pub fn with_mask(mut self, mask: impl Into<Option<ArrayView2<'a, u8>>>) -> Self {
        self.mask = mask.into();
        self
    }

    pub fn with_mask_pos(mut self, pos: AnchoredPos) -> Self {
        self.mask_position = pos;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_pow(mut self, pow: f32) -> Self {
        self.pow = pow;
        self
    }

    /// Where should the center of the mask be on the image
    ///
    /// # Example
    /// ```ignore
    /// let mask = ndarray::Array2::ones((100, 100)) * 255_u8;
    /// ChannelPaster::new(mask.view()).with_center_position(NormalisedPos::mid_point());
    /// ```
    // TODO: trait for this?
    pub fn with_position(mut self, pos: AnchoredPos) -> Self {
        self.position = pos;
        self
    }
}

impl<'a, 'b> Paste<ImagePaster<'b>> for ArrayViewMut3<'a, u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    // TODO: this doesn't return error, so paste and try_paste?
    /// Paste an image over self, only works till 4 channels and rest are ignored.
    ///
    /// If the dimensions of other or mask don't match with self, their common intersection is
    /// pasted on self
    fn paste(mut self, other: ImagePaster) -> error_stack::Result<Self::Out, PasteError> {
        let ImagePaster {
            to_paste,
            position,
            mask,
            alpha,
            mask_position,
            pow,
        } = other;

        let (ih, iw, ic) = self.dim();
        let (oh, ow, oc) = to_paste.dim();

        // blend uses wide::f32x4 so only works till 4 channels
        let c = ic.min(oc).min(4);
        if c <= 0 {
            return Ok(self);
        }

        let img_bounds = Bounds::from_dim(ih, iw);
        let other_bounds = Bounds::from_dim(oh, ow);
        dbg!(img_bounds);
        dbg!(other_bounds);

        let other_translation = position.get_top_left_pos(img_bounds, other_bounds).coords;
        dbg!(other_translation);

        let other_translated = other_bounds.translate(other_translation);
        dbg!(other_translated);

        let img_bounds = if let Some(mask) = mask {
            let (oh, ow) = mask.dim();
            let mask_bounds = Bounds::from_dim(oh, ow);
            dbg!(mask_bounds);
            let mask_translation = mask_position
                .get_top_left_pos(img_bounds, mask_bounds)
                .coords;
            dbg!(mask_translation);

            let mask_translated = mask_bounds.translate(mask_translation);
            dbg!(mask_translated);

            img_bounds
                .intersection(other_translated)
                .and_then(|x| dbg!(x).intersection(mask_translated))
        } else {
            img_bounds.intersection(other_translated)
        };

        let Some(img_bounds) = img_bounds else {
            return Ok(self);
        };
        dbg!(img_bounds);

        let other_bounds = img_bounds.translate(-other_translation);

        // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
        // intersection with it will also have positive coords
        let mut cropped_src = self.slice_mut(s![
            img_bounds.h_min() as usize..img_bounds.h_max() as usize,
            img_bounds.w_min() as usize..img_bounds.w_max() as usize,
            0..c
        ]);
        dbg!(other_bounds.h_min() as usize..other_bounds.h_max() as usize);
        dbg!(other_bounds.w_min() as usize..other_bounds.w_max() as usize);
        let cropped_other = to_paste.slice(s![
            other_bounds.h_min() as usize..other_bounds.h_max() as usize,
            other_bounds.w_min() as usize..other_bounds.w_max() as usize,
            0..c
        ]);

        if let Some(mask) = mask {
            let (oh, ow) = mask.dim();
            let mask_bounds = Bounds::from_dim(oh, ow);
            let mask_translation = mask_position
                .get_top_left_pos(img_bounds, mask_bounds)
                .coords;

            let mask_bounds = img_bounds.translate(-mask_translation);

            let cropped_mask = mask.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_other.lanes(Axis(2)))
                .and(cropped_mask)
                .par_for_each(|mut this, other, mask| {
                    let res = super::blend(
                        super::from_iter_to_f32_4(&this),
                        super::from_iter_to_f32_4(other),
                        (*mask as f32).div(255.).powf(pow),
                        alpha,
                    )
                    .pipe(super::convert_to_u8_4);

                    this.assign(&ArrayView::from(&[res][..c]));
                });
        } else {
            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_other.lanes(Axis(2)))
                .par_for_each(|mut this, other| {
                    let res = super::blend(
                        super::from_iter_to_f32_4(&this),
                        super::from_iter_to_f32_4(other),
                        1.,
                        alpha,
                    )
                    .pipe(super::convert_to_u8_4);

                    this.assign(&ArrayView::from(&res[0..c]));
                });
        }

        Ok(self)
    }
}

// impl<'a, S, T> PasteConfig<S> for &'a T
// where
//     S: Paste<ImagePaster<'a>>,
//     for<'t> &'t T: AsArray<'t, u8, ndarray::Ix3>,
// {
//     type Out = ImagePaster<'a>;
//
//     fn with_opts(self) -> Self::Out {
//         let arr = self.into();
//         ImagePaster::new(arr)
//     }
// }

impl<'a, T: 'a> PasteConfig<'a> for T
where
    T: AsArray<'a, u8, ndarray::Ix3>,
{
    type Out = ImagePaster<'a>;

    fn with_opts(self) -> Self::Out {
        let arr = self.into();
        ImagePaster::new(arr)
    }
}

impl<'a> PasteConfig<'a> for ImagePaster<'a> {
    type Out = Self;

    fn with_opts(self) -> Self::Out {
        self
    }
}

impl<'a, T> Paste<T> for ArrayViewMut3<'a, u8>
where
    for<'b> &'b T: AsArray<'b, u8, ndarray::Ix3>,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: T) -> error_stack::Result<Self::Out, PasteError> {
        let arr: ArrayView3<u8> = other.borrow().into();
        self.paste(arr.with_opts())
    }
}

impl<'a, T> Paste<T> for &'a mut Array3<u8>
where
    T: PasteConfig<'a>,
    <T as PasteConfig<'a>>::Out: Into<ImagePaster<'a>>,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: T) -> error_stack::Result<Self::Out, PasteError> {
        let paster: ImagePaster = other.with_opts().into();
        self.view_mut().paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use std::{f64, ops::Mul};

    use super::*;
    use ndarray::Array2;
    use ndarray_image::NdarrayToImage;

    #[test]
    pub fn test_paste_equal_dim_centered() {
        let mut this = Array3::zeros((512, 512, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(other.with_opts().with_alpha(0.5)).unwrap();

        let img: image::RgbImage = this.view().to_image().unwrap();
        // img.save("test_paste_equal_dim_centered.jpg").unwrap();
    }

    #[test]
    pub fn test_paste_bg_pan_50() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(
            other
                .with_opts()
                .with_alpha(0.3)
                .with_position(AnchoredPos::new(
                    NormalisedPos::new(0., 0.),
                    crate::paste::Anchor::TopLeft,
                )),
        )
        .unwrap();

        let img: image::RgbImage = this.view().to_image().unwrap();
        // img.save("test_paste_bg_pan_50.jpg").unwrap();
    }

    #[test]
    pub fn test_paste_bg_pan_with_mask() {
        const VF: f64 = 10.;
        const HF: f64 = 10.;

        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;
        let (mh, mw) = (4096, 2048);

        let vf = mh as f64 / VF;
        let hf = mw as f64 / HF;

        let mask = Array2::from_shape_fn((4096, 2048), |(h, w)| {
            ((h as f64).div(vf).mul(f64::consts::PI).sin()
                + (w as f64).div(hf).mul(f64::consts::PI).cos())
            .mul(255.)
            .clamp(0., 255.)
            .floor() as u8
        });

        this.paste(
            other
                .with_opts()
                .with_alpha(0.3)
                .with_position(AnchoredPos::new(
                    NormalisedPos::new(0., 0.),
                    crate::paste::Anchor::TopLeft,
                ))
                .with_mask(mask.view()),
        )
        .unwrap();

        let img: image::RgbImage = this.view().to_image().unwrap();
        // img.save("test_paste_bg_pan_with_mask.jpg").unwrap();
    }
}
