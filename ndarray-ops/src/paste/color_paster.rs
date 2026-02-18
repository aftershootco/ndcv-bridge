use error_stack::Report;
use ndarray::{Array3, ArrayView, ArrayView2, ArrayViewMut3, Axis, Zip, s};
use std::ops::Div;
use tap::Pipe;

use crate::Rgb;
use crate::paste::{AnchoredPos, Bounds, PasteError};

use super::traits::{Paste, PasteConfig};

#[derive(Debug, Clone, Copy)]
pub struct ColorPaster<'a> {
    data: Rgb<u8>,
    position: AnchoredPos,
    size: Bounds,
    mask: Option<ArrayView2<'a, u8>>,
    mask_position: AnchoredPos,
    alpha: f32,
    pow: f32,
}

impl<'a> ColorPaster<'a> {
    pub fn new(data: Rgb<u8>) -> Self {
        Self {
            data,
            position: AnchoredPos::default(),
            size: Bounds::from_dim(1, 1),
            mask: None,
            mask_position: AnchoredPos::default(),
            alpha: 1.,
            pow: 1.,
        }
    }

    pub fn with_data(mut self, data: Rgb<u8>) -> Self {
        self.data = data;
        self
    }

    pub fn with_size(mut self, h: usize, w: usize) -> Self {
        self.size = Bounds::from_dim(h, w);
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

impl<'a, 'b> Paste<ColorPaster<'b>> for ArrayViewMut3<'a, u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    // TODO: this doesn't return error, so paste and try_paste?
    /// Paste an image over self, only works till 4 channels and rest are ignored.
    ///
    /// If the dimensions of other or mask don't match with self, their common intersection is
    /// pasted on self
    fn paste(mut self, other: ColorPaster) -> Result<Self::Out, Report<PasteError>> {
        let ColorPaster {
            data,
            position,
            size,
            mask,
            mask_position,
            alpha,
            pow,
        } = other;

        let (ih, iw, ic) = self.dim();
        let oc = data.channels();

        // blend uses wide::f32x4 so only works till 4 channels
        let c = ic.min(oc).min(4);
        if c <= 0 {
            return Ok(self);
        }

        let img_bounds = Bounds::from_dim(ih, iw);
        let other_bounds = size;

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

        // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
        // intersection with it will also have positive coords
        let mut cropped_src = self.slice_mut(s![
            img_bounds.h_min() as usize..img_bounds.h_max() as usize,
            img_bounds.w_min() as usize..img_bounds.w_max() as usize,
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
                .and(cropped_mask)
                .par_for_each(|mut this, mask| {
                    let res = super::blend_f32_4(
                        super::from_iter_to_f32_4(&this),
                        super::from_iter_to_f32_4(data.0),
                        (*mask as f32).div(255.).powf(pow),
                        alpha,
                    )
                    .pipe(super::convert_to_u8_4);

                    this.assign(&ArrayView::from(&res[..c]));
                });
        } else {
            Zip::from(cropped_src.lanes_mut(Axis(2))).par_for_each(|mut this| {
                let res = super::blend_f32_4(
                    super::from_iter_to_f32_4(&this),
                    super::from_iter_to_f32_4(data.0),
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
impl<'a> PasteConfig<'a> for Rgb<u8> {
    type Out = ColorPaster<'a>;

    fn with_opts(self) -> Self::Out {
        ColorPaster::new(self)
    }
}

// On &mut Array3
impl<'a, 'b> Paste<Rgb<u8>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: Rgb<u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster: ColorPaster = other.with_opts().into();
        self.view_mut().paste(paster)
    }
}

impl<'a, 'b> Paste<ColorPaster<'b>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: ColorPaster<'b>) -> Result<Self::Out, Report<PasteError>> {
        self.view_mut().paste(other)
    }
}

// On ArrayViewMut3
impl<'a, 'b> Paste<Rgb<u8>> for ArrayViewMut3<'a, u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: Rgb<u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster: ColorPaster = other.with_opts().into();
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::*;

    #[test]
    pub fn test_color_equal_dim_centered() {
        let mut this = Array3::zeros((512, 512, 3));
        let other = Rgb::new(128, 0, 128);

        this.paste(other.with_opts().with_alpha(0.5).with_size(512, 512))
            .unwrap();

        save_rgb(this.view(), "test_color_equal_dim_centered.jpg");
    }

    #[test]
    pub fn test_color_pan_50() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Rgb::new(128, 0, 128);

        this.paste(
            other
                .with_opts()
                .with_alpha(0.3)
                .with_size(512, 512)
                .with_position(AnchoredPos::from_dim(0.5, 1., crate::paste::Anchor::Center)),
        )
        .unwrap();

        save_rgb(this.view(), "test_color_pan_50.jpg");
    }

    #[test]
    pub fn test_color_pan_with_mask() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Rgb::new(128, 0, 128);
        let mask = circular_wave_mask(4096, 2048, 30., 30.);

        this.paste(
            other
                .with_opts()
                .with_alpha(0.7)
                .with_size(1900, 2540)
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

        save_rgb(this.view(), "test_color_pan_with_mask.jpg");
    }
}
