//! <https://docs.rs/opencv/latest/opencv/ximgproc/fn.guided_filter.html>
use crate::conversions::*;
use glam::USizeVec2;
use ndarray::*;
use opencv::core::MatTraitConst;

#[derive(Debug, thiserror::Error)]
pub enum GuidedFilterError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    /// Sizes are `(width, height)`.
    #[error(
        "guide is {}x{} (WxH) but the image is {}x{}",
        .guide.x, .guide.y, .image.x, .image.y
    )]
    GuideSizeMismatch { guide: USizeVec2, image: USizeVec2 },
    #[error("eps must be finite and greater than 0, got {0}")]
    InvalidEps(f64),
    #[error("OpenCV wrote to a buffer of its own instead of the output array")]
    OutputReallocated,
}

mod seal {
    pub trait Sealed {}
    // OpenCV's guided filter rejects every other depth, for the guide and the
    // image alike. The two may differ from each other.
    impl Sealed for u8 {}
    impl Sealed for f32 {}
}

/// He et al.'s guided filter, from OpenCV's `ximgproc` contrib module: each
/// output pixel is a local linear function of the guide, fitted over a
/// `(2 * radius + 1)`-wide box. Where the guide is flat the image is
/// averaged; where the guide has an edge, the edge is carried into the output.
///
/// Passing the image as its own guide makes it an edge-preserving smoother,
/// which is `cv2.ximgproc.guidedFilter(guide=img, src=img, ...)`.
pub trait NdCvGuidedFilter<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// - `guide`: same height and width as `self`, 1 or 3 channels, `u8` or
    ///   `f32` (independently of `self`). A size mismatch is a
    ///   [`GuidedFilterError::GuideSizeMismatch`]; any other channel count is
    ///   a [`GuidedFilterError::OpenCvError`].
    /// - `radius`: box radius in pixels. 0 is a 1x1 box, which returns the
    ///   image unchanged.
    /// - `eps`: regularisation, in squared units of the guide (so `200` on a
    ///   `u8` guide is `200 / 255^2`, about `0.0031`, on the same guide
    ///   scaled to `[0, 1]`). Larger smooths across stronger edges. It must be
    ///   finite and above 0, or the call is a [`GuidedFilterError::InvalidEps`]:
    ///   OpenCV divides by the guide's local variance plus `eps`, so 0 turns a
    ///   flat patch of a 1-channel `f32` guide into NaN, and a negative or NaN
    ///   `eps` corrupts the output without an error.
    ///
    ///   Positive is necessary but not sufficient. OpenCV computes that
    ///   variance in `f32`, and for a 1-channel guide whose values span `R`, an
    ///   `eps` below about `1e-7 * R^2` can still give NaN (f32) or wrong
    ///   values (u8) on flat patches: under `1e-8` for `[0, 1]`, around
    ///   `0.001` for a `u8` guide at 255. This is not checked, since the safe
    ///   floor depends on the data. Typical values (`200` on `u8`, `0.003` on
    ///   `[0, 1]`) are three orders of magnitude above it.
    ///
    /// The output has `self`'s shape and element type. `self` may have any
    /// number of channels; each is filtered against the same guide.
    fn guided_filter<G, GS, GD>(
        &self,
        guide: &ArrayBase<GS, GD>,
        radius: u16,
        eps: f64,
    ) -> Result<ndarray::Array<T, D>, GuidedFilterError>
    where
        G: bytemuck::Pod + seal::Sealed + crate::types::CvType,
        GS: ndarray::Data<Elem = G>,
        GD: ndarray::Dimension,
        ArrayBase<GS, GD>: crate::image::NdImage,
        for<'a> ArrayView<'a, G, GD>: NdAsImage<G, GD>;
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvGuidedFilter<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    for<'a> ndarray::ArrayView<'a, T, D>: crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn guided_filter<G, GS, GD>(
        &self,
        guide: &ArrayBase<GS, GD>,
        radius: u16,
        eps: f64,
    ) -> Result<ndarray::Array<T, D>, GuidedFilterError>
    where
        G: bytemuck::Pod + seal::Sealed + crate::types::CvType,
        GS: ndarray::Data<Elem = G>,
        GD: ndarray::Dimension,
        ArrayBase<GS, GD>: crate::image::NdImage,
        for<'a> ArrayView<'a, G, GD>: NdAsImage<G, GD>,
    {
        use crate::image::NdImage;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(GuidedFilterError::InvalidEps(eps));
        }
        let image_size = USizeVec2::new(self.width(), self.height());
        let guide_size = USizeVec2::new(guide.width(), guide.height());
        if guide_size != image_size {
            return Err(GuidedFilterError::GuideSizeMismatch {
                guide: guide_size,
                image: image_size,
            });
        }
        let mut dst = ndarray::Array::zeros(self.raw_dim());
        // OpenCV asserts on empty input; an empty image filters to itself.
        if self.is_empty() {
            return Ok(dst);
        }

        let src = standard_layout(self)?;
        let src = src.view();
        let cv_src = src.as_image_mat()?;
        // The image as its own guide, the usual call, would otherwise convert
        // (and, out of standard layout, copy) the same pixels twice. The
        // element types must match exactly, or a guide aliasing the image as a
        // different type of the same width would be read as `T`. Both are
        // `'static` through `Pod`.
        let self_guided = core::any::TypeId::of::<T>() == core::any::TypeId::of::<G>()
            && self.as_ptr().cast::<u8>() == guide.as_ptr().cast::<u8>()
            && self.shape() == guide.shape()
            && self.strides() == guide.strides();
        let guide = (!self_guided).then(|| standard_layout(guide)).transpose()?;
        let guide = guide.as_ref().map(CowArray::view);
        let cv_guide = guide.as_ref().map(NdAsImage::as_image_mat).transpose()?;
        let cv_guide = cv_guide.as_deref().unwrap_or(&*cv_src);
        let dst_data = dst.as_ptr().cast::<u8>();
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::ximgproc::guided_filter(
            cv_guide,
            &*cv_src,
            &mut *cv_dst,
            radius.into(),
            eps,
            -1,
            1.0,
        )?;
        // Had OpenCV wanted a different size or type it would have written into
        // a new buffer and left `dst` all zeros.
        if cv_dst.data() != dst_data {
            return Err(GuidedFilterError::OutputReallocated);
        }
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn texture(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            u8::try_from((y * 37 + x * 11 + ch * 71 + (x * y) % 13) % 256).unwrap()
        })
    }

    #[test]
    fn test_guided_filter_matches_cv2() {
        // cv2.ximgproc.guidedFilter(guide=t, src=t, radius=6, eps=200) on
        // OpenCV 5.0, the call body smoothening makes. Allow one level for
        // rounding differences between OpenCV versions.
        let t = texture(40, 40, 3);
        let res = t.guided_filter(&t, 6, 200.0).unwrap();
        assert_eq!(res.shape(), &[40, 40, 3]);
        for ((y, x), want) in [
            ((20, 20), [201u8, 21, 90]),
            ((0, 0), [6, 74, 145]),
            ((39, 17), [92, 162, 231]),
        ] {
            for (c, &w) in want.iter().enumerate() {
                let got = res[[y, x, c]];
                assert!(got.abs_diff(w) <= 1, "({y}, {x}, {c}): got {got}, want {w}");
            }
        }
        let sum: i64 = res.iter().map(|&v| i64::from(v)).sum();
        assert!((sum - 611_305).abs() < 40 * 40 * 3 / 10, "sum {sum}");
    }

    #[test]
    fn test_guided_filter_small_values_match_cv2() {
        // cv2 at radius 2, eps 200: first 3x3 of channel 0, 3- and 1-channel.
        let t = texture(12, 10, 3);
        let res = t.guided_filter(&t, 2, 200.0).unwrap();
        assert_eq!(
            res.slice(s![..3, ..3, 0]).to_owned(),
            ndarray::array![[3u8, 14, 25], [40, 51, 62], [74, 86, 98]]
        );

        let gray = t.slice(s![.., .., 0]).to_owned();
        let res = gray.guided_filter(&gray, 2, 200.0).unwrap();
        assert_eq!(
            res.slice(s![..3, ..3]).to_owned(),
            ndarray::array![[6u8, 16, 27], [40, 51, 62], [74, 86, 99]]
        );

        // f32 in [0, 1] with cv2's eps 0.01, on that scale. (The u8 calls'
        // eps 200 would be 200 / 255^2 here; see the next test.)
        let f = t.mapv(|v| f32::from(v) / 255.0);
        let res = f.guided_filter(&f, 2, 0.01).unwrap();
        let want = [
            [0.037_548_3f32, 0.078_044_05, 0.120_792_65],
            [0.172_656_34, 0.211_716_32, 0.252_906_02],
        ];
        for (y, row) in want.iter().enumerate() {
            for (x, &w) in row.iter().enumerate() {
                let got = res[[y, x, 0]];
                assert!((got - w).abs() < 1e-4, "({y}, {x}): got {got}, want {w}");
            }
        }
    }

    #[test]
    fn test_guided_filter_u8_and_f32_agree_once_eps_is_scaled() {
        // The documented conversion: eps on a u8 guide is eps / 255^2 on the
        // same guide in [0, 1]. The u8 path rounds, so allow one level.
        let t = texture(30, 30, 3);
        let res_u8 = t.guided_filter(&t, 4, 200.0).unwrap();
        let f = t.mapv(|v| f32::from(v) / 255.0);
        let res_f32 = f.guided_filter(&f, 4, 200.0 / (255.0 * 255.0)).unwrap();
        let worst = res_u8
            .iter()
            .zip(res_f32.iter())
            .map(|(&a, &b)| (f32::from(a) - b * 255.0).abs())
            .fold(0.0, f32::max);
        assert!(worst <= 1.0, "u8 and scaled f32 differ by {worst} levels");
    }

    #[test]
    fn test_guided_filter_rejects_bad_eps() {
        let t = texture(8, 8, 3);
        for eps in [0.0, -0.0, -1e-4, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    t.guided_filter(&t, 2, eps),
                    Err(GuidedFilterError::InvalidEps(_))
                ),
                "eps {eps}"
            );
        }
        // A positive eps alone is not enough: OpenCV computes the variance in
        // f32, and on a flat 1-channel guide in [0, 1] an eps of 1e-8 or less
        // still comes back as NaN. 1e-6 is clear of that.
        let flat = Array2::<f32>::from_elem((8, 8), 0.4);
        let res = flat.guided_filter(&flat, 2, 1e-6).unwrap();
        assert!(res.iter().all(|&v| (v - 0.4).abs() < 1e-6), "{res:?}");
    }

    #[test]
    fn test_guided_filter_self_guided_view_matches_separate_guide() {
        // The self-guided shortcut must give what a distinct guide with the
        // same pixels gives, in and out of standard layout.
        let t = texture(10, 12, 3);
        let other = t.clone();
        assert_eq!(
            t.guided_filter(&t, 2, 200.0).unwrap(),
            t.guided_filter(&other, 2, 200.0).unwrap()
        );
        let flipped = t.slice(s![.., ..;-1, ..]);
        let flipped_other = other.slice(s![.., ..;-1, ..]);
        assert_eq!(
            flipped.guided_filter(&flipped, 2, 200.0).unwrap(),
            flipped.guided_filter(&flipped_other, 2, 200.0).unwrap()
        );
        // Same pointer, different shape: a one-channel view of the image is
        // not the image.
        let first = t.slice(s![.., .., 0..1]);
        assert_eq!(
            first.guided_filter(&t, 2, 200.0).unwrap(),
            first.to_owned().guided_filter(&other, 2, 200.0).unwrap()
        );
    }

    #[test]
    fn test_guided_filter_preserves_an_edge_in_the_guide() {
        // A step far above sqrt(eps) stays sharp; the same step under a flat
        // guide is averaged away.
        let mut img = Array3::<u8>::zeros((20, 20, 3));
        img.slice_mut(s![..10, .., ..]).fill(200);

        let res = img.guided_filter(&img, 4, 100.0).unwrap();
        assert!(
            res[[9, 10, 0]] >= 195,
            "above the edge: {}",
            res[[9, 10, 0]]
        );
        assert!(
            res[[10, 10, 0]] <= 5,
            "below the edge: {}",
            res[[10, 10, 0]]
        );

        let flat = Array2::<u8>::from_elem((20, 20), 128);
        let res = img.guided_filter(&flat, 4, 100.0).unwrap();
        assert!(
            (80..=120).contains(&res[[9, 10, 0]]) && (80..=120).contains(&res[[10, 10, 0]]),
            "flat guide should blur across the step, got {} and {}",
            res[[9, 10, 0]],
            res[[10, 10, 0]]
        );
    }

    #[test]
    fn test_guided_filter_radius_zero_is_identity() {
        let t = texture(8, 9, 3);
        assert_eq!(t.guided_filter(&t, 0, 200.0).unwrap(), t);
    }

    #[test]
    fn test_guided_filter_mixed_guide_depth() {
        let t = texture(10, 10, 3);
        let guide = t.mapv(f32::from);
        let res = t.guided_filter(&guide, 3, 200.0).unwrap();
        assert_eq!(res, t.guided_filter(&t, 3, 200.0).unwrap());
    }

    #[test]
    fn test_guided_filter_non_standard_layout() {
        let t = texture(10, 12, 3);
        let flipped = t.slice(s![.., ..;-1, ..]);
        let res = flipped.guided_filter(&flipped, 2, 200.0).unwrap();
        let expected = flipped
            .to_owned()
            .guided_filter(&flipped.to_owned(), 2, 200.0)
            .unwrap();
        assert_eq!(res, expected);

        // A one-row slice of a larger image.
        let row = t.slice(s![3..4, .., ..]);
        let res = row.guided_filter(&row, 1, 200.0).unwrap();
        assert_eq!(res.shape(), &[1, 12, 3]);
    }

    #[test]
    fn test_guided_filter_rejects_bad_guides() {
        let t = texture(10, 10, 3);
        let short = texture(5, 10, 3);
        let err = t.guided_filter(&short, 2, 200.0).unwrap_err();
        assert!(
            matches!(
                err,
                GuidedFilterError::GuideSizeMismatch { guide, image }
                    if guide == USizeVec2::new(10, 5) && image == USizeVec2::new(10, 10)
            ),
            "{err}"
        );
        assert_eq!(
            err.to_string(),
            "guide is 10x5 (WxH) but the image is 10x10"
        );

        let four = texture(10, 10, 4);
        assert!(matches!(
            t.guided_filter(&four, 2, 200.0),
            Err(GuidedFilterError::OpenCvError(_))
        ));
    }

    #[test]
    fn test_guided_filter_empty() {
        let empty = Array3::<u8>::zeros((0, 0, 3));
        let res = empty.guided_filter(&empty, 2, 200.0).unwrap();
        assert_eq!(res.shape(), &[0, 0, 3]);
    }
}
