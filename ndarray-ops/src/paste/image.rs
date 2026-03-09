use error_stack::Report;
use ndarray::{Array3, ArrayView, ArrayView3, ArrayViewMut3, Axis, Zip, s};
use std::ops::Div;
use tap::Pipe;

use crate::paste::{
    Bounds, PasteError, PasteInput, PasteOpts, Sections, traits::Paste, traits::PasteConfig,
};

#[derive(Debug, Clone, Copy)]
pub struct ImagePaster<'a, 'b, F>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    'a: 'b,
{
    data: ArrayView3<'a, u8>,
    opts: PasteOpts<'b, u8, wide::f32x4, F>,
}

impl<'a, 'b, F> ImagePaster<'a, 'b, F>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
{
    pub fn new(data: ArrayView3<'a, u8>, opts: PasteOpts<'b, u8, wide::f32x4, F>) -> Self {
        Self { data, opts }
    }

    pub fn with_data(mut self, data: ArrayView3<'a, u8>) -> Self {
        self.data = data;
        self
    }
}

impl<'t, 'a, 'b, F> Paste<ImagePaster<'a, 'b, F>> for ArrayViewMut3<'t, u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    'a: 'b,
    't: 'a,
{
    type Out = ArrayViewMut3<'t, u8>;

    // TODO: this doesn't return error, so paste and try_paste?
    /// Paste an image over self, only works till 4 channels and rest are ignored.
    ///
    /// If the dimensions of other or mask don't match with self, their common intersection is
    /// pasted on self
    fn paste(mut self, other: ImagePaster<'a, 'b, F>) -> Result<Self::Out, Report<PasteError>> {
        let ImagePaster { data, opts } = other;

        let (ih, iw, ic) = self.dim();
        let (oh, ow, oc) = data.dim();

        // blend uses wide::f32x4 so only works till 4 channels
        let c = ic.min(oc).min(4);
        if c <= 0 {
            return Ok(self);
        }

        let img_bounds = Bounds::from_dim(ih, iw);
        let other_bounds = Bounds::from_dim(oh, ow);

        let Some(Sections {
            this_bounds: img_bounds,
            other_translation,
            mask_with_translation: mask_info,
        }) = super::get_tri_intersection(img_bounds, other_bounds, opts.mask_info, opts.pos)
        else {
            return Ok(self);
        };
        let paste_algo = opts.paste_algo;
        let alpha = opts.alpha;

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

        if let Some((mask, mask_translation)) = mask_info {
            let mask_bounds = img_bounds.translate(-mask_translation);

            let cropped_mask = mask.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_other.lanes(Axis(2)))
                .and(cropped_mask)
                .par_for_each(|mut this, other, mask| {
                    let res = paste_algo(
                        (
                            super::from_iter_to_f32_4(&this),
                            super::from_iter_to_f32_4(other),
                            (*mask as f32).div(255.),
                            alpha,
                        )
                            .into(),
                    )
                    .pipe(|x| super::convert_to_u8_4(x.to_array()));

                    this.assign(&ArrayView::from(&res[..c]));
                });
        } else {
            Zip::from(cropped_src.lanes_mut(Axis(2)))
                .and(cropped_other.lanes(Axis(2)))
                .par_for_each(|mut this, other| {
                    let res = paste_algo(
                        (
                            super::from_iter_to_f32_4(&this),
                            super::from_iter_to_f32_4(other),
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
impl<'a, 'p, F> PasteConfig<PasteOpts<'p, u8, wide::f32x4, F>> for ArrayView3<'a, u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    'a: 'p,
{
    type Out = ImagePaster<'a, 'p, F>;

    fn with_opts(self, opts: PasteOpts<'p, u8, wide::f32x4, F>) -> Self::Out {
        ImagePaster::new(self, opts)
    }
}

impl<'a, 'p, F> PasteConfig<PasteOpts<'p, u8, wide::f32x4, F>> for &'a Array3<u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    'a: 'p,
{
    type Out = ImagePaster<'a, 'p, F>;

    fn with_opts(self, opts: PasteOpts<'p, u8, wide::f32x4, F>) -> Self::Out {
        ImagePaster::new(self.view(), opts)
    }
}

// On &mut Array3
impl<'a, 'b> Paste<ArrayView3<'b, u8>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = ArrayViewMut3<'a, u8>;

    fn paste(self, other: ArrayView3<'b, u8>) -> Result<Self::Out, Report<PasteError>> {
        let paster = other.with_opts(PasteOpts::new());
        self.view_mut().paste(paster)
    }
}

impl<'t, 'a, 'b, F> Paste<ImagePaster<'a, 'b, F>> for &'t mut Array3<u8>
where
    F: Fn(PasteInput<wide::f32x4>) -> wide::f32x4 + Send + Sync,
    'a: 'b,
    't: 'a,
{
    type Out = ArrayViewMut3<'t, u8>;
    fn paste(self, other: ImagePaster<'a, 'b, F>) -> Result<Self::Out, Report<PasteError>> {
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
        let paster = other.with_opts(PasteOpts::new());
        self.paste(paster)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paste::AnchoredPos;

    use crate::test_utils::*;

    #[test]
    pub fn test_paste_equal_dim_centered() {
        let mut this = Array3::zeros((512, 512, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(other.with_opts(PasteOpts::new().with_alpha(0.5)))
            .unwrap();

        save_rgb(this.view(), "test_paste_equal_dim_centered.jpg");
    }

    #[test]
    pub fn test_paste_bg_pan_50() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((512, 512, 3)) * 255_u8;

        this.paste(
            other.with_opts(
                PasteOpts::new()
                    .with_alpha(0.3)
                    .with_pos(AnchoredPos::from_dim(0., 0., crate::paste::Anchor::TopLeft)),
            ),
        )
        .unwrap();

        save_rgb(this.view(), "test_paste_bg_pan_50.jpg");
    }

    #[test]
    pub fn test_paste_bg_pan_with_mask() {
        let mut this = Array3::zeros((4096, 4096, 3));
        let other = Array3::ones((4096, 4096, 3)) * 255_u8;
        let mask = circular_wave_mask(4096, 4096, 1024., 1024.);

        this.paste(
            other.with_opts(
                PasteOpts::new()
                    .with_mask(
                        mask.view(),
                        AnchoredPos::from_dim(0., 0., crate::paste::Anchor::TopLeft),
                    )
                    .with_alpha(0.7)
                    .with_pos(AnchoredPos::from_dim(
                        1.2,
                        0.75,
                        crate::paste::Anchor::Center,
                    ))
                    .with_paste_algo(|mut input| {
                        input.mask = input.mask.powf(5.);
                        crate::paste::paste_algos::blend(input)
                    }),
            ),
        )
        .unwrap();

        save_rgb(this.view(), "test_paste_bg_pan_with_mask.jpg");
    }
}
