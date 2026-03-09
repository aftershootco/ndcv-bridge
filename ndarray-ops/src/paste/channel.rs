use std::ops::{Add, Div, Mul};

use error_stack::{Report, ResultExt};
use ndarray::{Array2, Array3, ArrayView2, Axis, Zip, s};

use crate::paste::{
    Bounds, DEFAULT_MASK, PasteError, PasteInput, PasteOpts, Sections,
    traits::{Paste, PasteConfig},
};

#[derive(Debug, Clone, Copy)]
pub struct ChannelOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub opts: PasteOpts<'a, A, T, F>,
    pub channel_idx: usize,
}

impl<'a, A, T> ChannelOpts<'a, A, T, fn(PasteInput<T>) -> T>
where
    T: Copy,
    T: Add<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Mul<f32, Output = T>,
{
    pub fn new() -> Self {
        Self {
            opts: PasteOpts::new(),
            channel_idx: 3,
        }
    }
}

impl<'a, A, T, F> ChannelOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub fn with_paste_opts<F2>(self, opts: PasteOpts<'a, A, T, F2>) -> ChannelOpts<'a, A, T, F2>
    where
        F2: Fn(PasteInput<T>) -> T + Send + Sync,
    {
        ChannelOpts {
            opts: opts,
            channel_idx: self.channel_idx,
        }
    }

    /// On which channel of Array3, to paste the array on.
    /// If channel is out of bounds, a channel will be appeneded to the array
    /// with default values of 255_u8.
    ///
    /// # Example
    /// ```ignore
    /// // Paste at 4th channel
    /// ChannelOpts::new().with_channel_idx(3);
    /// ```
    pub fn with_channel_idx(mut self, channel_idx: usize) -> Self {
        self.channel_idx = channel_idx;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChannelPaster<'a, 'b, F>
where
    F: Fn(PasteInput<f32>) -> f32 + Send + Sync,
    'a: 'b,
{
    data: ArrayView2<'a, u8>,
    opts: ChannelOpts<'b, u8, f32, F>,
}

impl<'a, 'b, F> ChannelPaster<'a, 'b, F>
where
    F: Fn(PasteInput<f32>) -> f32 + Send + Sync,
{
    pub fn new(data: ArrayView2<'a, u8>, opts: ChannelOpts<'b, u8, f32, F>) -> Self {
        Self { data, opts }
    }

    pub fn with_data(mut self, data: ArrayView2<'a, u8>) -> Self {
        self.data = data;
        self
    }
}

impl<'t, 'a, 'b, F> Paste<ChannelPaster<'a, 'b, F>> for &'t mut Array3<u8>
where
    F: Fn(PasteInput<f32>) -> f32 + Send + Sync,
    'a: 'b,
    't: 'a,
{
    type Out = &'t mut Array3<u8>;

    fn paste(self, other: ChannelPaster<'a, 'b, F>) -> Result<Self::Out, Report<PasteError>> {
        let ChannelPaster { data, opts } = other;

        let channel = opts.channel_idx + 1;
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
        let other_bounds = Bounds::from_dim(mh, mw);

        let Some(Sections {
            this_bounds: img_bounds,
            other_translation,
            mask_with_translation: mask_info,
        }) = super::get_tri_intersection(
            img_bounds,
            other_bounds,
            opts.opts.mask_info,
            opts.opts.pos,
        )
        else {
            return Ok(self);
        };
        let paste_algo = opts.opts.paste_algo;
        let alpha = opts.opts.alpha;

        let other_bounds = img_bounds.translate(-other_translation);

        // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
        // intersection with it will also have positive coords
        let cropped_src = self.slice_mut(s![
            img_bounds.h_min() as usize..img_bounds.h_max() as usize,
            img_bounds.w_min() as usize..img_bounds.w_max() as usize,
            c - 1
        ]);

        let cropped_other = data.slice(s![
            other_bounds.h_min() as usize..other_bounds.h_max() as usize,
            other_bounds.w_min() as usize..other_bounds.w_max() as usize,
        ]);

        if let Some((mask, mask_translation)) = mask_info {
            let mask_bounds = img_bounds.translate(-mask_translation);

            // safe to cast to usize as mask bounds (which were originally in 1st quadrant) were first translated by mask_translation,
            // any intersection with it will also be be moved by this translation,
            // so reversing this translation will result in positive coords
            let cropped_mask = mask.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            Zip::from(cropped_src)
                .and(cropped_other)
                .and(cropped_mask)
                .par_for_each(|this, other, mask| {
                    let res = paste_algo(
                        (
                            super::convert_to_f32(*this),
                            super::convert_to_f32(*other),
                            (*mask as f32).div(255.),
                            alpha,
                        )
                            .into(),
                    );

                    *this = super::convert_to_u8(res);
                });
        } else {
            Zip::from(cropped_src)
                .and(cropped_other)
                .par_for_each(|this, other| {
                    let res = paste_algo(
                        (
                            super::convert_to_f32(*this),
                            super::convert_to_f32(*other),
                            DEFAULT_MASK,
                            alpha,
                        )
                            .into(),
                    );

                    *this = super::convert_to_u8(res);
                });
        }

        Ok(self)
    }
}

// Paste Config impls
impl<'a, 'p, F> PasteConfig<ChannelOpts<'p, u8, f32, F>> for ArrayView2<'a, u8>
where
    F: Fn(PasteInput<f32>) -> f32 + Send + Sync,
    'a: 'p,
{
    type Out = ChannelPaster<'a, 'p, F>;

    fn with_opts(self, opts: ChannelOpts<'p, u8, f32, F>) -> Self::Out {
        ChannelPaster::new(self, opts)
    }
}

impl<'a, 'p, F> PasteConfig<ChannelOpts<'p, u8, f32, F>> for &'a Array2<u8>
where
    F: Fn(PasteInput<f32>) -> f32 + Send + Sync,
    'a: 'p,
{
    type Out = ChannelPaster<'a, 'p, F>;

    fn with_opts(self, opts: ChannelOpts<'p, u8, f32, F>) -> Self::Out {
        ChannelPaster::new(self.view(), opts)
    }
}

// On &mut Array3
impl<'a, 'b> Paste<ArrayView2<'b, u8>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = &'a mut Array3<u8>;

    fn paste(self, other: ArrayView2<'b, u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster = other.with_opts(ChannelOpts::new());
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paste::AnchoredPos;

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
            mask.with_opts(
                ChannelOpts::new().with_paste_opts(
                    PasteOpts::new()
                        .with_pos(AnchoredPos::from_dim(
                            0.5,
                            0.5,
                            crate::paste::Anchor::TopLeft,
                        ))
                        .with_alpha(0.7)
                        .with_paste_algo(|mut i| {
                            i.mask = i.mask.powf(2.);
                            crate::paste::paste_algos::blend(i)
                        }),
                ),
            ),
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
            mask.with_opts(
                ChannelOpts::new().with_channel_idx(1).with_paste_opts(
                    PasteOpts::new()
                        .with_alpha(0.7)
                        .with_pos(AnchoredPos::from_dim(
                            0.5,
                            0.5,
                            crate::paste::Anchor::TopLeft,
                        ))
                        .with_paste_algo(|mut i| {
                            i.mask = i.mask.powf(2.);
                            crate::paste::paste_algos::blend(i)
                        }),
                ),
            ),
        )
        .unwrap();

        save_rgba(
            this.as_standard_layout().view(),
            "test_pasting_color_channel.png",
        );
    }
}
