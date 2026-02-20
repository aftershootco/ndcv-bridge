use error_stack::Report;
use ndarray::{Array3, ArrayView, ArrayViewMut3, Axis, Zip, s};
use std::ops::{Add, Div, Mul};
use tap::Pipe;

use crate::{
    Rgb,
    paste::{
        Bounds, PasteError, PasteInput, PasteOpts, Sections, traits::Paste, traits::PasteConfig,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct ColorOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub opts: PasteOpts<'a, A, T, F>,
    pub size_h: usize,
    pub size_w: usize,
}

impl<'a, A, T> ColorOpts<'a, A, T, fn(PasteInput<T>) -> T>
where
    T: Copy,
    T: Add<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Mul<f32, Output = T>,
{
    pub fn new() -> Self {
        Self {
            opts: PasteOpts::new(),
            size_h: 1,
            size_w: 1,
        }
    }
}

impl<'a, A, T, F> ColorOpts<'a, A, T, F>
where
    F: Fn(PasteInput<T>) -> T + Send + Sync,
{
    pub fn with_paste_opts<F2>(self, opts: PasteOpts<'a, A, T, F2>) -> ColorOpts<'a, A, T, F2>
    where
        F2: Fn(PasteInput<T>) -> T + Send + Sync,
    {
        ColorOpts {
            opts: opts,
            size_h: self.size_h,
            size_w: self.size_w,
        }
    }

    pub fn with_size(mut self, h: usize, w: usize) -> Self {
        self.size_h = h;
        self.size_w = w;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorPaster<'b, F>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
{
    data: Rgb<u8>,
    opts: ColorOpts<'b, u8, wide::f32x4, F>,
}

impl<'b, F> ColorPaster<'b, F>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
{
    pub fn new(data: Rgb<u8>, opts: ColorOpts<'b, u8, wide::f32x4, F>) -> Self {
        Self { data, opts }
    }

    pub fn with_data(mut self, data: Rgb<u8>) -> Self {
        self.data = data;
        self
    }
}

impl<'t, 'b, F> Paste<ColorPaster<'b, F>> for ArrayViewMut3<'t, u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    't: 'b,
{
    type Out = ArrayViewMut3<'t, u8>;

    // TODO: this doesn't return error, so paste and try_paste?
    /// Paste an image over self, only works till 4 channels and rest are ignored.
    ///
    /// If the dimensions of other or mask don't match with self, their common intersection is
    /// pasted on self
    fn paste(mut self, other: ColorPaster<'b, F>) -> Result<Self::Out, Report<PasteError>> {
        let ColorPaster { data, opts } = other;

        let (ih, iw, ic) = self.dim();
        let oc = data.channels();

        // blend uses wide::f32x4 so only works till 4 channels
        let c = ic.min(oc).min(4);
        if c <= 0 {
            return Ok(self);
        }

        let img_bounds = Bounds::from_dim(ih, iw);
        let other_bounds = Bounds::from_dim(opts.size_h, opts.size_w);

        let Some(Sections {
            this_bounds: img_bounds,
            other_translation: _,
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

        // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
        // intersection with it will also have positive coords
        let mut cropped_src = self.slice_mut(s![
            img_bounds.h_min() as usize..img_bounds.h_max() as usize,
            img_bounds.w_min() as usize..img_bounds.w_max() as usize,
            0..c
        ]);

        if let Some((mask, mask_translation)) = mask_info {
            let mask_bounds = img_bounds.translate(-mask_translation);

            let cropped_mask = mask.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_mask)
                .par_for_each(|mut this, mask| {
                    let res = paste_algo(
                        (
                            super::from_iter_to_f32_4(&this),
                            super::from_iter_to_f32_4(data.0),
                            (*mask as f32).div(255.),
                            alpha,
                        )
                            .into(),
                    )
                    .pipe(|x| super::convert_to_u8_4(x.to_array()));

                    this.assign(&ArrayView::from(&res[..c]));
                });
        } else {
            Zip::from(cropped_src.lanes_mut(Axis(2))).par_for_each(|mut this| {
                let res = paste_algo(
                    (
                        super::from_iter_to_f32_4(&this),
                        super::from_iter_to_f32_4(data.0),
                        1.,
                        alpha,
                    )
                        .into(),
                )
                .pipe(|x| super::convert_to_u8_4(x.to_array()));

                this.assign(&ArrayView::from(&res[0..c]));
            });
        }

        Ok(self)
    }
}

// PasteConfig impls
impl<'p, F> PasteConfig<ColorOpts<'p, u8, wide::f32x4, F>> for Rgb<u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
{
    type Out = ColorPaster<'p, F>;

    fn with_opts(self, opts: ColorOpts<'p, u8, wide::f32x4, F>) -> Self::Out {
        ColorPaster::new(self, opts)
    }
}

// On &mut Array3
impl<'a> Paste<Rgb<u8>> for &'a mut Array3<u8> {
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: Rgb<u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster = other.with_opts(ColorOpts::new());
        self.view_mut().paste(paster)
    }
}

impl<'t, 'b, F> Paste<ColorPaster<'b, F>> for &'t mut Array3<u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    't: 'b,
{
    type Out = ArrayViewMut3<'t, u8>;

    fn paste(self, other: ColorPaster<'b, F>) -> Result<Self::Out, Report<PasteError>> {
        self.view_mut().paste(other)
    }
}

// On ArrayViewMut3
impl<'a> Paste<Rgb<u8>> for ArrayViewMut3<'a, u8> {
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: Rgb<u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster = other.with_opts(ColorOpts::new());
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paste::AnchoredPos;

    use crate::test_utils::*;

    #[test]
    pub fn test_color_equal_dim_centered() {
        let mut this = Array3::zeros((512, 512, 3));
        let other = Rgb::new(128, 0, 128);

        this.paste(
            other.with_opts(
                ColorOpts::new()
                    .with_size(512, 512)
                    .with_paste_opts(PasteOpts::new().with_alpha(0.5)),
            ),
        )
        .unwrap();

        save_rgb(this.view(), "test_color_equal_dim_centered.jpg");
    }

    #[test]
    pub fn test_color_pan_50() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Rgb::new(128, 0, 128);

        this.paste(
            other.with_opts(
                ColorOpts::new().with_size(512, 512).with_paste_opts(
                    PasteOpts::new()
                        .with_alpha(0.3)
                        .with_pos(AnchoredPos::from_dim(0.5, 1., crate::paste::Anchor::Center)),
                ),
            ),
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
            other.with_opts(
                ColorOpts::new().with_size(1900, 2540).with_paste_opts(
                    PasteOpts::new()
                        .with_alpha(0.7)
                        .with_pos(AnchoredPos::from_dim(
                            0.5,
                            0.8,
                            crate::paste::Anchor::Center,
                        ))
                        .with_mask(
                            mask.view(),
                            AnchoredPos::from_dim(0.3, 0.5, crate::paste::Anchor::Center),
                        )
                        .with_paste_algo(|mut input| {
                            input.mask = input.mask.powf(0.7);
                            crate::paste::paste_algos::blend(input)
                        }),
                ),
            ),
        )
        .unwrap();

        save_rgb(this.view(), "test_color_pan_with_mask.jpg");
    }
}
