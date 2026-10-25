//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.stack_blur.html>
use crate::conversions::*;
use glam::U16Vec2;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum StackBlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

mod seal {
    pub trait Sealed {}
    // src: any number of channels, depth CV_8U, CV_16U, CV_16S or CV_32F.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
}

/// A Gaussian-like blur whose cost does not grow with the kernel: a separable
/// triangular kernel, weights `r + 1 - |i|` for radius `r = size / 2`, applied
/// along rows and then columns. Channels are filtered independently.
///
/// OpenCV supports only a replicate border here, so there is no `border_type`.
/// For `f32` the result matches the triangular kernel with a replicate border
/// to float precision, with one exception measured on OpenCV 4.12 and 5.0:
/// for kernels wider than 9, the row pass clamps its last few columns
/// differently, by up to about 0.03 on data in `[0, 1]` (3 columns at width 21,
/// 8 at width 41). `u8`, `u16` and `i16` use a fixed-point approximation that
/// can be a level or two from the exact filter.
pub trait NdCvStackBlur<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// `kernel_size` is `(width, height)`, as in [`crate::NdCvBlur::blur`].
    /// Both must be odd and positive; anything else comes back as a
    /// [`StackBlurError::OpenCvError`] rather than panicking. A size of 1 on an
    /// axis leaves that axis unfiltered.
    fn stack_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
    ) -> Result<ndarray::Array<T, D>, StackBlurError>;
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvStackBlur<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    for<'a> ndarray::CowArray<'a, T, D>: crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn stack_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
    ) -> Result<ndarray::Array<T, D>, StackBlurError> {
        let kernel_size = kernel_size.into();
        let mut dst = ndarray::Array::zeros(self.raw_dim());
        // OpenCV asserts on empty input; an empty image blurs to itself.
        if self.is_empty() {
            return Ok(dst);
        }
        // The Mat conversion keeps only the outer stride, so a view that is not
        // in standard layout would be read as if it were. A no-op borrow when
        // the input already is.
        let src = self.as_standard_layout();
        let cv_src = src.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::stack_blur(
            &*cv_src,
            &mut *cv_dst,
            opencv::core::Size::new(kernel_size.x.into(), kernel_size.y.into()),
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    /// The triangular kernel written out along one axis, with a replicate
    /// border.
    fn triangle(src: &Array3<f64>, size: usize, axis: usize) -> Array3<f64> {
        let r = i64::try_from(size / 2).unwrap();
        let n = i64::try_from(src.len_of(Axis(axis))).unwrap();
        let norm = f64::from(u32::try_from((r + 1) * (r + 1)).unwrap());
        Array3::from_shape_fn(src.dim(), |(y, x, c)| {
            let at = if axis == 0 { y } else { x };
            let at = i64::try_from(at).unwrap();
            (-r..=r)
                .map(|i| {
                    let j = usize::try_from((at + i).clamp(0, n - 1)).unwrap();
                    let v = if axis == 0 {
                        src[[j, x, c]]
                    } else {
                        src[[y, j, c]]
                    };
                    f64::from(u32::try_from(r + 1 - i.abs()).unwrap()) * v
                })
                .sum::<f64>()
                / norm
        })
    }

    fn reference(src: &Array3<f64>, width: usize, height: usize) -> Array3<f64> {
        triangle(&triangle(src, width, 1), height, 0)
    }

    fn texture(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            u8::try_from((y * 37 + x * 11 + ch * 71 + (x * y) % 13) % 256).unwrap()
        })
    }

    fn worst(a: ArrayView3<f32>, b: ArrayView3<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_stack_blur_matches_the_written_out_kernel() {
        // Different widths and heights, so swapping them would show up.
        let src = texture(9, 11, 3);
        let src_f64 = src.mapv(f64::from);
        for (w, h) in [(3u16, 3u16), (5, 7), (9, 3), (1, 5), (5, 1)] {
            let expected = reference(&src_f64, usize::from(w), usize::from(h));
            let res = src.mapv(f32::from).stack_blur((w, h)).unwrap();
            assert!(worst(res.view(), expected.view()) < 1e-3, "({w}, {h})");

            // The fixed-point path stays within two levels of the rounded filter.
            let res_u8 = src.stack_blur((w, h)).unwrap();
            let off = res_u8
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            assert!(off < 2.5, "({w}, {h}) u8 off by {off}");
        }
    }

    #[test]
    fn test_stack_blur_large_kernel_away_from_the_right_edge() {
        // Kernels wider than 9 take OpenCV's running-sum path. It is the same
        // filter, except over the last few columns (see the trait docs).
        let src = texture(40, 64, 1).mapv(f64::from);
        let res = src.mapv(|v| v as f32).stack_blur((21, 21)).unwrap();
        let expected = reference(&src, 21, 21);
        let interior = s![.., ..54, ..];
        assert!(worst(res.slice(interior), expected.slice(interior)) < 1e-2);
    }

    #[test]
    fn test_stack_blur_one_leaves_the_axis_alone() {
        let src = texture(6, 7, 3);
        assert_eq!(src.stack_blur((1, 1)).unwrap(), src);
        let rows_only = src.mapv(f32::from).stack_blur((5, 1)).unwrap();
        let expected = reference(&src.mapv(f64::from), 5, 1);
        assert!(worst(rows_only.view(), expected.view()) < 1e-3);
    }

    #[test]
    fn test_stack_blur_leaves_a_flat_image_flat() {
        let res = Array3::<u8>::from_elem((10, 10, 3), 77)
            .stack_blur((7, 7))
            .unwrap();
        assert!(res.iter().all(|&v| v == 77));
        let res = Array2::<f32>::from_elem((10, 10), 0.3)
            .stack_blur((21, 21))
            .unwrap();
        assert!(res.iter().all(|&v| (v - 0.3).abs() < 1e-6));
    }

    #[test]
    fn test_stack_blur_rejected_sizes_are_errors() {
        // Each is an OpenCV assertion: an error, not a panic.
        let arr = Array2::<f32>::ones((8, 8));
        for size in [(4, 3), (3, 4), (0, 3), (3, 0)] {
            let err = arr.stack_blur(size).unwrap_err();
            assert!(
                matches!(err, StackBlurError::OpenCvError(_)),
                "{size:?}: {err}"
            );
        }
    }

    #[test]
    fn test_stack_blur_empty_input() {
        for (h, w) in [(0, 5), (5, 0), (0, 0)] {
            let res = Array2::<f32>::zeros((h, w)).stack_blur((3, 3)).unwrap();
            assert_eq!(res.dim(), (h, w));
            let res = Array3::<u8>::zeros((h, w, 3)).stack_blur((3, 3)).unwrap();
            assert_eq!(res.dim(), (h, w, 3));
        }
    }

    #[test]
    fn test_stack_blur_non_standard_layouts_match_a_copy() {
        // Asymmetric content, so a transposition or flip would show up.
        let base = texture(12, 10, 3).mapv(f32::from);
        let check = |name: &str, view: ArrayView3<f32>| {
            assert_eq!(
                view.stack_blur((5, 3)).unwrap(),
                view.to_owned().stack_blur((5, 3)).unwrap(),
                "{name}"
            );
        };
        check("channel slice", base.slice(s![.., .., 1..]));
        check("column slice", base.slice(s![.., 2..8, ..]));
        check("step 2", base.slice(s![..;2, ..;2, ..]));
        check("columns reversed", base.slice(s![.., ..;-1, ..]));

        let plane = base.slice(s![.., .., 0]).to_owned();
        let transposed = plane.t();
        assert_eq!(
            transposed.stack_blur((5, 3)).unwrap(),
            transposed.to_owned().stack_blur((5, 3)).unwrap()
        );
    }
}
