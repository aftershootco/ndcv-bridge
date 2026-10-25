//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.stack_blur.html>
use crate::conversions::*;
use glam::U16Vec2;
use ndarray::*;
use opencv::core::MatTraitConst;

#[derive(Debug, thiserror::Error)]
pub enum StackBlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error(
        "kernel {width}x{height} is wider than {max}, where OpenCV's sums overflow for this type"
    )]
    KernelTooLarge { width: u16, height: u16, max: u16 },
    #[error("OpenCV wrote to a buffer of its own instead of the output array")]
    OutputReallocated,
}

mod seal {
    pub trait Sealed {
        /// The widest kernel whose sums fit OpenCV's accumulator for this type.
        const MAX_KERNEL: u16;
    }
    // OpenCV also accepts CV_16U and CV_16S, but its integer sums overflow for
    // them at any kernel past 3: a flat 65535 `u16` image blurs to 43775 at
    // size 5, and a flat 30000 `i16` one to 20963. They are left out.
    impl Sealed for u8 {
        // OpenCV sums `255 * (r + 1)^2` in an `int`, which overflows past
        // radius 2900: a flat 255 image comes back black from size 5803 up.
        const MAX_KERNEL: u16 = 5801;
    }
    impl Sealed for f32 {
        const MAX_KERNEL: u16 = u16::MAX;
    }
}

/// A Gaussian-like blur whose cost does not grow with the kernel: a separable
/// triangular kernel, weights `r + 1 - |i|` for radius `r = size / 2`, applied
/// along rows and then columns. Channels are filtered independently.
///
/// OpenCV supports only a replicate border here, so there is no `border_type`.
/// Measured against that filter, written out, on OpenCV 4.12 and 5.0:
///
/// - `f32` matches it to the precision of OpenCV's running sums (within about
///   5e-4 on data in `[0, 1]` at photo sizes), except at the right edge
///   whenever the row pass takes its running-sum path, which it does for
///   kernels wider than 9 or rows no wider than the kernel. There the last
///   columns are clamped differently: 1 column at width 13, 3 at 21, 8 at 41,
///   23 at 101, off by up to about 0.07.
/// - `u8` uses truncating fixed-point sums: within 2 levels of the rounded
///   filter and biased low by up to about 0.8 of a level, but up to 4 levels
///   in those right-edge columns and up to 7 once the kernel is wider than the
///   image.
/// - When a dimension is smaller than the kernel, OpenCV 4.12 and 5.0 disagree
///   with each other (by up to about 0.45 on data in `[0, 1]`); 4.12 is the one
///   that matches the written-out filter.
pub trait NdCvStackBlur<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// `kernel_size` is `(width, height)`, as in [`crate::NdCvBlur::blur`].
    /// Both must be odd and positive; anything else comes back as a
    /// [`StackBlurError::OpenCvError`] rather than panicking. For `u8` a size
    /// above 5801 is a [`StackBlurError::KernelTooLarge`]. A size of 1 on an
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
    for<'a> ndarray::ArrayView<'a, T, D>: crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn stack_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
    ) -> Result<ndarray::Array<T, D>, StackBlurError> {
        let kernel_size = kernel_size.into();
        if kernel_size.max_element() > T::MAX_KERNEL {
            return Err(StackBlurError::KernelTooLarge {
                width: kernel_size.x,
                height: kernel_size.y,
                max: T::MAX_KERNEL,
            });
        }
        let mut dst = ndarray::Array::zeros(self.raw_dim());
        // OpenCV asserts on empty input; an empty image blurs to itself.
        if self.is_empty() {
            return Ok(dst);
        }
        // The Mat conversion keeps only the outer stride, so input that is not
        // in standard layout is copied first. Contiguous input is borrowed, but
        // ndarray leaves the stride of a length-1 axis arbitrary (0 for a
        // one-row slice), which OpenCV can reject; a view over the same slice
        // has the default strides, without a copy.
        let src = self.as_standard_layout();
        let src = match src.as_slice() {
            Some(data) => {
                ArrayView::from_shape(self.raw_dim(), data).map_err(ConversionError::from)?
            }
            None => src.view(),
        };
        let cv_src = src.as_image_mat()?;
        let dst_data = dst.as_ptr().cast::<u8>();
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::stack_blur(
            &*cv_src,
            &mut *cv_dst,
            opencv::core::Size::new(kernel_size.x.into(), kernel_size.y.into()),
        )?;
        // Had OpenCV wanted a different size or type it would have written into
        // a new buffer and left `dst` all zeros. Not expected, but an error is
        // better than a black result.
        if cv_dst.data() != dst_data {
            return Err(StackBlurError::OutputReallocated);
        }
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

    #[test]
    fn test_stack_blur_u8_kernel_limit() {
        // The widest u8 kernel still sums without overflow, and one size wider
        // is an error rather than a black image.
        let flat = Array2::<u8>::from_elem((40, 40), 255);
        assert!(
            flat.stack_blur((5801, 5801))
                .unwrap()
                .iter()
                .all(|&v| v == 255)
        );
        for size in [(5803, 3), (3, 5803)] {
            let err = flat.stack_blur(size).unwrap_err();
            assert!(
                matches!(err, StackBlurError::KernelTooLarge { max: 5801, .. }),
                "{size:?}: {err}"
            );
        }
        // f32 has no such limit: the widest kernel stays flat to within float
        // precision of a sum over a 32768^2 window (4e-5 measured on 4.12).
        let flat = Array2::<f32>::from_elem((40, 40), 0.5);
        let res = flat.stack_blur((u16::MAX, u16::MAX)).unwrap();
        assert!(res.iter().all(|&v| (v - 0.5).abs() < 1e-4));
    }

    #[test]
    fn test_stack_blur_unit_length_axes() {
        // ndarray keeps an arbitrary stride on a length-1 axis: 0 for a one-row
        // slice, even after `to_owned`. Each must match the same values laid
        // out with default strides.
        let base = texture(8, 9, 3).mapv(f32::from);
        let fresh3 = |v: ArrayView3<f32>| {
            Array3::from_shape_vec(v.raw_dim(), v.iter().copied().collect()).unwrap()
        };
        let fresh2 = |v: ArrayView2<f32>| {
            Array2::from_shape_vec(v.raw_dim(), v.iter().copied().collect()).unwrap()
        };
        for view in [
            base.slice(s![3..4, .., ..]),
            base.slice(s![.., 4..5, ..]),
            base.slice(s![2..3, 5..6, ..]),
        ] {
            let expected = fresh3(view).stack_blur((5, 3)).unwrap();
            assert_eq!(
                view.stack_blur((5, 3)).unwrap(),
                expected,
                "{:?}",
                view.strides()
            );
            assert_eq!(view.to_owned().stack_blur((5, 3)).unwrap(), expected);
        }
        let plane = base.slice(s![.., .., 0]).to_owned();
        let row = plane.slice(s![3..4, ..]);
        let expected = fresh2(row).stack_blur((5, 3)).unwrap();
        assert_eq!(row.stack_blur((5, 3)).unwrap(), expected);
        assert_eq!(
            plane
                .row(3)
                .insert_axis(Axis(0))
                .stack_blur((5, 3))
                .unwrap(),
            expected
        );
    }
}
