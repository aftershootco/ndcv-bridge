//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.sobel.html>
use crate::conversions::*;
use crate::gaussian::BorderType;
use ndarray::*;
use opencv::core::MatTraitConst;

#[derive(Debug, thiserror::Error)]
pub enum SobelError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("OpenCV wrote to a buffer of its own instead of the output array")]
    OutputReallocated,
}

/// The derivative kernel, OpenCV's `ksize`.
///
/// OpenCV takes an `int` that is either an odd size or `-1` for Scharr; this
/// lists the values it accepts instead.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SobelKernel {
    /// The 3x3 Scharr kernel: a more accurate first derivative than
    /// [`SobelKernel::Size3`]. Only for a first derivative along one axis,
    /// `dx + dy == 1`.
    Scharr,
    /// `[-1 0 1]`, or `[1 -2 1]` for a second derivative, with no smoothing
    /// across it.
    Size1,
    /// The usual 3x3 Sobel kernel, cv2's default.
    Size3,
    Size5,
    Size7,
}

impl From<SobelKernel> for i32 {
    /// OpenCV's `ksize` for the kernel.
    fn from(kernel: SobelKernel) -> Self {
        match kernel {
            SobelKernel::Scharr => opencv::imgproc::FILTER_SCHARR,
            SobelKernel::Size1 => 1,
            SobelKernel::Size3 => 3,
            SobelKernel::Size5 => 5,
            SobelKernel::Size7 => 7,
        }
    }
}

pub(crate) mod seal {
    pub trait Sealed {}
    // src: any number of channels, filtered independently.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}

    /// An output depth `Self` that OpenCV accepts for input of depth `T` and
    /// that keeps the sign of the derivative. Left out on purpose: `u8` output,
    /// which zeroes every negative slope, and `i16` output for any input wider
    /// than `u8`, which saturates; OpenCV accepts both. Narrower-than-input
    /// floats (`f64` to `f32`) are rejected by OpenCV at run time, so they do
    /// not type-check here either.
    pub trait SobelOutput<T>: Copy + bytemuck::Pod + num::Zero + crate::types::CvType {}
    impl SobelOutput<u8> for i16 {}
    impl SobelOutput<u8> for f32 {}
    impl SobelOutput<u8> for f64 {}
    impl SobelOutput<u16> for f32 {}
    impl SobelOutput<u16> for f64 {}
    impl SobelOutput<i16> for f32 {}
    impl SobelOutput<i16> for f64 {}
    impl SobelOutput<f32> for f32 {}
    impl SobelOutput<f32> for f64 {}
    impl SobelOutput<f64> for f64 {}
}

/// The image derivative `d^(dx + dy) / dx^dx dy^dy`, by a separable Sobel or
/// Scharr kernel that also smooths across the derivative. Channels are filtered
/// independently, and the output has the input's shape.
///
/// The output depth `O` is a type parameter, OpenCV's `ddepth`, restricted to
/// the pairings that keep a signed derivative: `u8` input into `i16`, `f32` or
/// `f64`; `u16` or `i16` into `f32` or `f64`; `f32` into `f32` or `f64`; and
/// `f64` into `f64`. Any other pairing does not compile:
///
/// ```compile_fail
/// use ndcv_bridge::NdCvSobel;
/// let narrowing = ndarray::Array2::<f64>::zeros((4, 4));
/// let _ = narrowing.sobel_def::<f32>(1, 0);
/// ```
///
/// `u8` into `i16` still saturates at a strong edge under
/// [`SobelKernel::Size7`], whose gain on a unit step is 640. An unsupported
/// derivative order for the kernel, or `dx == dy == 0`, comes back as a
/// [`SobelError::OpenCvError`] rather than panicking.
pub trait NdCvSobel<T: bytemuck::Pod + seal::Sealed + crate::types::CvType, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// - `dx`, `dy`: derivative order along x (columns) and y (rows).
    /// - `scale`: multiplies the derivative; `1.0` leaves it as the kernel
    ///   gives it, which for [`SobelKernel::Size3`] is 8 times the slope.
    /// - `delta`: added after scaling.
    /// - `border_type`: how the kernel reads past the edge.
    ///   [`BorderType::BorderWrap`] and [`BorderType::BorderTransparent`] are
    ///   not supported.
    fn sobel<O: seal::SobelOutput<T>>(
        &self,
        dx: u8,
        dy: u8,
        kernel: SobelKernel,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<ndarray::Array<O, D>, SobelError>
    where
        ndarray::Array<O, D>: crate::conversions::NdAsImageMut<O, D>;

    /// `cv2.Sobel` defaults: a [`SobelKernel::Size3`] kernel, no scale or
    /// delta, and a `borderType` of [`BorderType::BorderDefault`].
    fn sobel_def<O: seal::SobelOutput<T>>(
        &self,
        dx: u8,
        dy: u8,
    ) -> Result<ndarray::Array<O, D>, SobelError>
    where
        ndarray::Array<O, D>: crate::conversions::NdAsImageMut<O, D>,
    {
        self.sobel::<O>(
            dx,
            dy,
            SobelKernel::Size3,
            1.0,
            0.0,
            BorderType::BorderDefault,
        )
    }
}

impl<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvSobel<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    for<'a> ndarray::ArrayView<'a, T, D>: crate::conversions::NdAsImage<T, D>,
{
    fn sobel<O: seal::SobelOutput<T>>(
        &self,
        dx: u8,
        dy: u8,
        kernel: SobelKernel,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<ndarray::Array<O, D>, SobelError>
    where
        ndarray::Array<O, D>: crate::conversions::NdAsImageMut<O, D>,
    {
        let mut dst = ndarray::Array::<O, D>::zeros(self.raw_dim());
        // OpenCV rejects empty input; an empty image has an empty derivative.
        if self.is_empty() {
            return Ok(dst);
        }
        // The Mat conversion keeps only the outer stride, so input that is not
        // in standard layout is copied first. Contiguous input is borrowed, but
        // ndarray leaves the stride of a length-1 axis arbitrary (0 for a
        // one-row slice), which OpenCV rejects; a view over the same slice has
        // the default strides, without a copy.
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
        opencv::imgproc::sobel(
            &*cv_src,
            &mut *cv_dst,
            O::cv_depth(),
            i32::from(dx),
            i32::from(dy),
            i32::from(kernel),
            scale,
            delta,
            // The discriminant of a fieldless `repr(C)` enum, which is OpenCV's
            // own constant; there is no `From` impl for it.
            border_type as i32,
        )?;
        // Had OpenCV wanted a different size or type it would have written into
        // a new buffer and left `dst` all zeros. Not expected, but an error is
        // better than a black result.
        if cv_dst.data() != dst_data {
            return Err(SobelError::OutputReallocated);
        }
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn reflect101(i: i64, n: i64) -> usize {
        let r = if i < 0 {
            -i
        } else if i >= n {
            2 * n - 2 - i
        } else {
            i
        };
        usize::try_from(r).unwrap()
    }

    /// The 3x3 Sobel kernel written out: `[-1 0 1]` along the derivative and
    /// `[1 2 1]` across it, with a reflect-101 border.
    fn reference_3x3(src: &Array3<f64>, dx: u8) -> Array3<f64> {
        let (h, w, c) = src.dim();
        let (hi, wi) = (i64::try_from(h).unwrap(), i64::try_from(w).unwrap());
        let deriv = [-1.0, 0.0, 1.0];
        let smooth = [1.0, 2.0, 1.0];
        let (kx, ky) = if dx == 1 {
            (deriv, smooth)
        } else {
            (smooth, deriv)
        };
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            let (y, x) = (i64::try_from(y).unwrap(), i64::try_from(x).unwrap());
            let mut acc = 0.0;
            for (i, wy) in ky.iter().enumerate() {
                for (j, wx) in kx.iter().enumerate() {
                    let sy = reflect101(y + i64::try_from(i).unwrap() - 1, hi);
                    let sx = reflect101(x + i64::try_from(j).unwrap() - 1, wi);
                    acc += wy * wx * src[[sy, sx, ch]];
                }
            }
            acc
        })
    }

    fn texture(h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            f32::from(u16::try_from((y * 37 + x * 11 + ch * 71 + (x * y) % 13) % 256).unwrap())
        })
    }

    /// `f(y, x) = x`, a unit slope along the columns.
    fn column_ramp() -> Array2<f32> {
        Array2::from_shape_fn((9, 12), |(_, x)| f32::from(u16::try_from(x).unwrap()))
    }

    #[test]
    fn test_sobel_matches_the_written_out_kernel() {
        let src = texture(7, 9, 3);
        for (dx, dy) in [(1, 0), (0, 1)] {
            let expected = reference_3x3(&src.mapv(f64::from), dx);
            let res: Array3<f32> = src.sobel_def(dx, dy).unwrap();
            let worst = res
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (f64::from(a) - b).abs())
                .fold(0.0, f64::max);
            assert!(worst < 1e-3, "dx={dx} dy={dy}: worst abs error {worst}");
        }
    }

    #[test]
    fn test_sobel_kernel_sizes_have_their_gains() {
        // On a unit slope each kernel returns its own gain, so a kernel passed
        // as the wrong size would show up. cv2 gives these exact values.
        let ramp = column_ramp();
        for (kernel, gain) in [
            (SobelKernel::Size1, 2.0),
            (SobelKernel::Size3, 8.0),
            (SobelKernel::Size5, 128.0),
            (SobelKernel::Size7, 2048.0),
            (SobelKernel::Scharr, 32.0),
        ] {
            let res: Array2<f32> = ramp
                .sobel(1, 0, kernel, 1.0, 0.0, BorderType::BorderDefault)
                .unwrap();
            assert_eq!(res[[4, 6]], gain, "{kernel:?}");
        }
    }

    #[test]
    fn test_sobel_scale_then_delta() {
        let res: Array2<f32> = column_ramp()
            .sobel(
                1,
                0,
                SobelKernel::Size3,
                0.125,
                3.0,
                BorderType::BorderDefault,
            )
            .unwrap();
        assert_eq!(res[[4, 6]], 4.0, "8 * 0.125 + 3");
    }

    #[test]
    fn test_sobel_u8_input_keeps_negative_slopes() {
        // The reason `u8` output is not offered: a falling ramp has a negative
        // derivative, which `i16` keeps.
        let falling = Array2::from_shape_fn((9, 12), |(_, x)| u8::try_from(11 - x).unwrap());
        let res: Array2<i16> = falling.sobel_def(1, 0).unwrap();
        assert_eq!(res[[4, 6]], -8);
        let res: Array2<f64> = falling.sobel_def(1, 0).unwrap();
        assert_eq!(res[[4, 6]], -8.0);
    }

    #[test]
    fn test_sobel_def_reflects_the_border() {
        // On a ramp, reflect-101 mirrors about the edge pixel, so the first
        // column's derivative is zero; replicate gives half a step.
        let ramp = column_ramp();
        let def: Array2<f32> = ramp.sobel_def(1, 0).unwrap();
        let reflect: Array2<f32> = ramp
            .sobel(
                1,
                0,
                SobelKernel::Size3,
                1.0,
                0.0,
                BorderType::BorderReflect101,
            )
            .unwrap();
        let replicate: Array2<f32> = ramp
            .sobel(
                1,
                0,
                SobelKernel::Size3,
                1.0,
                0.0,
                BorderType::BorderReplicate,
            )
            .unwrap();
        assert_eq!(def, reflect);
        assert_eq!(def[[4, 0]], 0.0);
        assert_eq!(replicate[[4, 0]], 4.0);
    }

    #[test]
    fn test_sobel_rejected_arguments_are_errors() {
        // Each of these is an OpenCV assertion: it must come back as an error,
        // not a panic.
        let ramp = column_ramp();
        let cases: [(&str, Result<Array2<f32>, SobelError>); 4] = [
            ("no derivative", ramp.sobel_def(0, 0)),
            (
                "Scharr on two axes",
                ramp.sobel(
                    1,
                    1,
                    SobelKernel::Scharr,
                    1.0,
                    0.0,
                    BorderType::BorderDefault,
                ),
            ),
            (
                "order too high for the kernel",
                ramp.sobel(
                    3,
                    0,
                    SobelKernel::Size3,
                    1.0,
                    0.0,
                    BorderType::BorderDefault,
                ),
            ),
            (
                "wrap border",
                ramp.sobel(1, 0, SobelKernel::Size3, 1.0, 0.0, BorderType::BorderWrap),
            ),
        ];
        for (name, res) in cases {
            assert!(
                matches!(res, Err(SobelError::OpenCvError(_))),
                "{name}: {res:?}"
            );
        }
    }

    #[test]
    fn test_sobel_empty_input() {
        for (h, w) in [(0, 5), (5, 0), (0, 0)] {
            let res: Array2<f32> = Array2::<f32>::zeros((h, w)).sobel_def(1, 0).unwrap();
            assert_eq!(res.dim(), (h, w));
            let res: Array3<i16> = Array3::<u8>::zeros((h, w, 3)).sobel_def(0, 1).unwrap();
            assert_eq!(res.dim(), (h, w, 3));
        }
    }

    #[test]
    fn test_sobel_non_standard_layouts_match_a_copy() {
        // Asymmetric content, so a transposition or flip would show up.
        let base = texture(12, 10, 3);
        let check = |name: &str, view: ArrayView3<f32>| {
            let a: Array3<f32> = view.sobel_def(1, 0).unwrap();
            let b: Array3<f32> = view.to_owned().sobel_def(1, 0).unwrap();
            assert_eq!(a, b, "{name}");
        };
        check("channel slice", base.slice(s![.., .., 1..]));
        check("column slice", base.slice(s![.., 2..8, ..]));
        check("step 2", base.slice(s![..;2, ..;2, ..]));
        check("columns reversed", base.slice(s![.., ..;-1, ..]));

        let plane = base.slice(s![.., .., 0]).to_owned();
        let transposed = plane.t();
        let a: Array2<f32> = transposed.sobel_def(0, 1).unwrap();
        let b: Array2<f32> = transposed.to_owned().sobel_def(0, 1).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_sobel_unit_length_axes() {
        // ndarray keeps an arbitrary stride on a length-1 axis: 0 for a one-row
        // slice, even after `to_owned`, which OpenCV used to reject here. Each
        // must match the same values laid out with default strides.
        let base = texture(8, 9, 3);
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
            let expected: Array3<f32> = fresh3(view).sobel_def(1, 0).unwrap();
            let a: Array3<f32> = view.sobel_def(1, 0).unwrap();
            let b: Array3<f32> = view.to_owned().sobel_def(1, 0).unwrap();
            assert_eq!(a, expected, "{:?}", view.strides());
            assert_eq!(b, expected, "{:?}", view.strides());
        }
        let plane = base.slice(s![.., .., 0]).to_owned();
        let row = plane.slice(s![3..4, ..]);
        let expected: Array2<f32> = fresh2(row).sobel_def(1, 0).unwrap();
        let a: Array2<f32> = row.sobel_def(1, 0).unwrap();
        let b: Array2<f32> = plane.row(3).insert_axis(Axis(0)).sobel_def(1, 0).unwrap();
        assert_eq!(a, expected);
        assert_eq!(b, expected);
    }
}
