use error_stack::Report;
use ndarray::{Array3, ArrayView, ArrayView2, ArrayView3, ArrayViewMut3, Axis, Zip, s};
use std::ops::Div;
use tap::Pipe;

use crate::paste::{AnchoredPos, Bounds, PasteError};

use super::traits::{Paste, PasteConfig};

#[derive(Debug, Clone, Copy)]
pub struct ImagePaster<'a> {
    // TODO: use getset?
    data: ArrayView3<'a, u8>,
    position: AnchoredPos,
    mask: Option<ArrayView2<'a, u8>>,
    mask_position: AnchoredPos,
    alpha: f32,
    pow: f32,
}

impl<'a> ImagePaster<'a> {
    pub fn new(data: ArrayView3<'a, u8>) -> Self {
        Self {
            data,
            position: AnchoredPos::default(),
            mask: None,
            alpha: 1.,
            mask_position: AnchoredPos::default(),
            pow: 1.,
        }
    }

    pub fn with_data(mut self, data: ArrayView3<'a, u8>) -> Self {
        self.data = data;
        self
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

    pub fn with_mask_position(mut self, pos: AnchoredPos) -> Self {
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
    fn paste(mut self, other: ImagePaster) -> Result<Self::Out, Report<PasteError>> {
        let ImagePaster {
            data,
            position,
            mask,
            alpha,
            mask_position,
            pow,
        } = other;

        let (ih, iw, ic) = self.dim();
        let (oh, ow, oc) = data.dim();

        // blend uses wide::f32x4 so only works till 4 channels
        let c = ic.min(oc).min(4);
        if c <= 0 {
            return Ok(self);
        }

        let img_bounds = Bounds::from_dim(ih, iw);
        let other_bounds = Bounds::from_dim(oh, ow);

        let other_translation = position.get_top_left_pos(img_bounds, other_bounds).coords;

        let other_translated = other_bounds.translate(other_translation);

        let img_bounds = if let Some(mask) = mask {
            let (oh, ow) = mask.dim();
            let mask_bounds = Bounds::from_dim(oh, ow);
            let mask_translation = mask_position
                .get_top_left_pos(img_bounds, mask_bounds)
                .coords;

            let mask_translated = mask_bounds.translate(mask_translation);

            img_bounds
                .intersection(other_translated)
                .and_then(|x| x.intersection(mask_translated))
        } else {
            img_bounds.intersection(other_translated)
        };

        let Some(img_bounds) = img_bounds else {
            return Ok(self);
        };

        let other_bounds = img_bounds.translate(-other_translation);

        // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
        // intersection with it will also have positive coords
        let mut cropped_src = self.slice_mut(s![
            img_bounds.h_min() as usize..img_bounds.h_max() as usize,
            img_bounds.w_min() as usize..img_bounds.w_max() as usize,
            0..c
        ]);

        let cropped_other = data.slice(s![
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
                    let res = super::blend_f32_4(
                        super::from_iter_to_f32_4(&this),
                        super::from_iter_to_f32_4(other),
                        (*mask as f32).div(255.).powf(pow),
                        alpha,
                    )
                    .pipe(super::convert_to_u8_4);

                    this.assign(&ArrayView::from(&res[..c]));
                });
        } else {
            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_other.lanes(Axis(2)))
                .par_for_each(|mut this, other| {
                    let res = super::blend_f32_4(
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

// PasteConfig impls
impl<'a> PasteConfig<'a> for ArrayView3<'a, u8> {
    type Out = ImagePaster<'a>;

    fn with_opts(self) -> Self::Out {
        ImagePaster::new(self)
    }
}

impl<'a> PasteConfig<'a> for &'a Array3<u8> {
    type Out = ImagePaster<'a>;

    fn with_opts(self) -> Self::Out {
        ImagePaster::new(self.view())
    }
}

// On &mut Array3
impl<'a, 'b> Paste<ArrayView3<'b, u8>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: ArrayView3<'b, u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster: ImagePaster = other.with_opts().into();
        self.view_mut().paste(paster)
    }
}

impl<'a, 'b> Paste<ImagePaster<'b>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: ImagePaster<'b>) -> Result<Self::Out, Report<PasteError>> {
        self.view_mut().paste(other)
    }
}

// On ArrayViewMut3
impl<'a, 'b> Paste<ArrayView3<'b, u8>> for ArrayViewMut3<'a, u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: ArrayView3<'b, u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster: ImagePaster = other.with_opts().into();
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::*;

    #[test]
    pub fn test_paste_equal_dim_centered() {
        let mut this = Array3::zeros((512, 512, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(other.with_opts().with_alpha(0.5)).unwrap();

        save_rgb(this.view(), "test_paste_equal_dim_centered.jpg");
    }

    #[test]
    pub fn test_paste_bg_pan_50() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(
            other
                .with_opts()
                .with_alpha(0.3)
                .with_position(AnchoredPos::from_dim(0., 0., crate::paste::Anchor::TopLeft)),
        )
        .unwrap();

        save_rgb(this.view(), "test_paste_bg_pan_50.jpg");
    }

    #[test]
    pub fn test_paste_bg_pan_with_mask() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((2048, 2048, 3)) * 255_u8;
        let mask = circular_wave_mask(4096, 2048, 30., 30.);

        this.paste(
            other
                .with_opts()
                .with_alpha(0.7)
                .with_position(AnchoredPos::from_dim(
                    0.5,
                    0.5,
                    crate::paste::Anchor::Center,
                ))
                .with_mask(mask.view())
                .with_mask_position(AnchoredPos::from_dim(
                    0.3,
                    0.5,
                    crate::paste::Anchor::Center,
                ))
                .with_pow(0.7),
        )
        .unwrap();

        save_rgb(this.view(), "test_paste_bg_pan_with_mask.jpg");
    }
}
