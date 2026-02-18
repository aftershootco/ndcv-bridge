use error_stack::{Report, ResultExt};
use ndarray::{Array2, Array3, ArrayView2, Axis, Zip, s};

use super::traits::{Paste, PasteConfig};
use super::{AnchoredPos, Bounds, PasteError};

#[derive(Debug, Clone, Copy)]
pub struct ChannelPaster<'a> {
    data: ArrayView2<'a, u8>,
    channel_idx: usize,
    position: AnchoredPos,
    alpha: f32,
    pow: f32,
}

impl<'a> ChannelPaster<'a> {
    pub fn new(data: ArrayView2<'a, u8>) -> Self {
        Self {
            data,
            channel_idx: 3,
            position: AnchoredPos::default(),
            alpha: 1.,
            pow: 1.,
        }
    }

    pub fn with_data(mut self, data: ArrayView2<'a, u8>) -> Self {
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
    pub fn with_channel_idx(mut self, channel_idx: usize) -> Self {
        self.channel_idx = channel_idx;
        self
    }

    /// Where should the center of the mask be on the image
    ///
    /// # Example
    /// ```ignore
    /// let mask = ndarray::Array2::ones((100, 100)) * 255_u8;
    /// ChannelPaster::new(mask.view()).with_center_position(NormalisedPos::mid_point());
    /// ```
    pub fn with_position(mut self, pos: AnchoredPos) -> Self {
        self.position = pos;
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
}

impl<'a, 'b> Paste<ChannelPaster<'b>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = &'a mut Array3<u8>;

    fn paste(self, other: ChannelPaster) -> core::result::Result<Self::Out, Report<PasteError>> {
        let ChannelPaster {
            data,
            channel_idx,
            position,
            alpha,
            pow,
        } = other;

        let channel = channel_idx + 1;
        let (mh, mw) = data.dim();
        let (ih, iw, ic) = self.dim();

        let c = if channel > ic {
            let filled_3d_arrray = Array3::ones((ih as usize, iw as usize, 1)) * 255_u8;
            *self = ndarray::concatenate(Axis(2), &[self.view(), filled_3d_arrray.view()])
                .change_context(PasteError)?;

            self.dim().2
        } else {
            channel
        };

        let img_bounds = Bounds::from_dim(ih, iw);
        let mask_bounds = Bounds::from_dim(mh, mw);

        let mask_translation = position.get_top_left_pos(img_bounds, mask_bounds);

        let mask_translated = mask_bounds.translate(mask_translation.coords);

        let img_bounds = img_bounds.intersection(mask_translated);

        if let Some(img_bounds) = img_bounds {
            let mask_bounds = img_bounds.translate(-mask_translation.coords);

            // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
            // intersection with it will also have positive coords
            let cropped_src = self.slice_mut(s![
                img_bounds.h_min() as usize..img_bounds.h_max() as usize,
                img_bounds.w_min() as usize..img_bounds.w_max() as usize,
                c - 1
            ]);

            // safe to cast to usize as mask bounds (which were originally in 1st quadrant) were first translated by mask_translation,
            // any intersection with it will also be be moved by this translation,
            // so reversing this translation will result in positive coords
            let cropped_mask = data.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            Zip::from(cropped_src)
                .and(cropped_mask)
                .par_for_each(|this, other| {
                    let res = super::blend_f32(
                        super::convert_to_f32(*this),
                        super::convert_to_f32(*other).powf(pow),
                        1.,
                        alpha,
                    );

                    *this = super::convert_to_u8(res);
                });
        }

        Ok(self)
    }
}

// Paste Config impls
impl<'a> PasteConfig<'a> for ArrayView2<'a, u8> {
    type Out = ChannelPaster<'a>;

    fn with_opts(self) -> Self::Out {
        ChannelPaster::new(self)
    }
}

impl<'a> PasteConfig<'a> for &'a Array2<u8> {
    type Out = ChannelPaster<'a>;

    fn with_opts(self) -> Self::Out {
        ChannelPaster::new(self.view())
    }
}

// On &mut Array3
impl<'a> Paste<ArrayView2<'a, u8>> for &'a mut Array3<u8> {
    type Out = &'a mut Array3<u8>;

    fn paste(
        self,
        other: ArrayView2<'a, u8>,
    ) -> core::result::Result<Self::Out, Report<PasteError>> {
        let paster: ChannelPaster = other.with_opts();
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::*;

    #[test]
    pub fn test_alpha_chan_equal_dim_centered() {
        let mut this = Array3::ones((512, 512, 3)) * 255;
        let mask = circular_wave_mask(512, 512, 20., 20.);

        this.paste(mask.view()).unwrap();

        assert_eq!(this.dim().2, 4);

        save_rgba(
            this.as_standard_layout().view(),
            "test_alpha_chan_equal_dim_centered.png",
        );
    }

    #[test]
    pub fn test_panned_alpha_channel() {
        let mut this = Array3::ones((3072, 4096, 4)) * 255;
        let mask = circular_wave_mask(1900, 2540, 100., 100.);

        this.paste(
            mask.with_opts()
                .with_position(AnchoredPos::from_dim(
                    0.5,
                    0.5,
                    crate::paste::Anchor::TopLeft,
                ))
                .with_alpha(0.7)
                .with_pow(2.),
        )
        .unwrap();

        save_rgba(
            this.as_standard_layout().view(),
            "test_panned_alpha_channel.png",
        );
    }

    #[test]
    pub fn test_pasting_color_channel() {
        let mut this = Array3::ones((3072, 4096, 4)) * 255;
        let mask = circular_wave_mask(1900, 2540, 100., 100.);

        this.paste(
            mask.with_opts()
                .with_position(AnchoredPos::from_dim(
                    0.5,
                    0.5,
                    crate::paste::Anchor::TopLeft,
                ))
                .with_channel_idx(1)
                .with_alpha(0.7)
                .with_pow(2.),
        )
        .unwrap();

        save_rgba(
            this.as_standard_layout().view(),
            "test_pasting_color_channel.png",
        );
    }
}
