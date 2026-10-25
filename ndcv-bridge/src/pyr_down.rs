//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.pyr_down.html>
use crate::conversions::*;
use crate::gaussian::BorderType;
use ndarray::*;
use opencv::core::MatTraitConst;

#[derive(Debug, thiserror::Error)]
pub enum PyrDownError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("OpenCV wrote to a buffer of its own instead of the output array")]
    OutputReallocated,
}

mod seal {
    pub trait Sealed {}
    // src: any number of channels, depth CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// One downsampling step of a Gaussian pyramid: blur with the 5x5 binomial
/// kernel `[1 4 6 4 1]^T [1 4 6 4 1] / 256`, then keep every even row and
/// column. An `h x w` image becomes `ceil(h / 2) x ceil(w / 2)`, with the same
/// channel count; channels are filtered independently.
///
/// OpenCV's `dstsize` is not exposed: the output is always the default half
/// size, which is the only size that keeps this a plain decimation.
pub trait NdCvPyrDown<T: bytemuck::Pod + seal::Sealed + crate::types::CvType, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// `border_type` decides how the kernel reads past the edge.
    /// [`BorderType::BorderConstant`] is not supported by OpenCV here and comes
    /// back as a [`PyrDownError::OpenCvError`], as does any other border it
    /// rejects.
    fn pyr_down(&self, border_type: BorderType) -> Result<ndarray::Array<T, D>, PyrDownError>;

    /// `cv2.pyrDown` defaults, whose `borderType` is
    /// [`BorderType::BorderDefault`].
    fn pyr_down_def(&self) -> Result<ndarray::Array<T, D>, PyrDownError> {
        self.pyr_down(BorderType::BorderDefault)
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvPyrDown<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    for<'a> ndarray::ArrayView<'a, T, D>: crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn pyr_down(&self, border_type: BorderType) -> Result<ndarray::Array<T, D>, PyrDownError> {
        // Halve the two spatial axes, rounding up; a channel axis is left alone.
        let mut dim = self.raw_dim();
        dim.slice_mut()
            .iter_mut()
            .take(2)
            .for_each(|n| *n = n.div_ceil(2));
        let mut dst = ndarray::Array::zeros(dim);
        // OpenCV rejects empty input; an empty image has an empty half.
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
        opencv::imgproc::pyr_down(
            &*cv_src,
            &mut *cv_dst,
            // An empty size asks for OpenCV's default, ((cols + 1) / 2, (rows +
            // 1) / 2): the shape `dst` was allocated with, so OpenCV writes
            // into it rather than reallocating a buffer of its own.
            opencv::core::Size::default(),
            // The discriminant of a fieldless `repr(C)` enum, which is OpenCV's
            // own constant; there is no `From` impl for it.
            border_type as i32,
        )?;
        // Had OpenCV wanted a different size or type it would have written into
        // a new buffer and left `dst` all zeros. Not expected, but an error is
        // better than a black result.
        if cv_dst.data() != dst_data {
            return Err(PyrDownError::OutputReallocated);
        }
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    const K: [i64; 5] = [1, 4, 6, 4, 1];

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

    /// `pyrDown` written out: the 5x5 binomial sum at every even pixel, with a
    /// reflect-101 border.
    fn reference(src: &Array3<f64>) -> Array3<f64> {
        let (h, w, c) = src.dim();
        let (hi, wi) = (i64::try_from(h).unwrap(), i64::try_from(w).unwrap());
        Array3::from_shape_fn((h.div_ceil(2), w.div_ceil(2), c), |(y, x, ch)| {
            let (y, x) = (i64::try_from(y).unwrap(), i64::try_from(x).unwrap());
            let mut acc = 0.0;
            for (i, ki) in K.iter().enumerate() {
                for (j, kj) in K.iter().enumerate() {
                    let sy = reflect101(2 * y + i64::try_from(i).unwrap() - 2, hi);
                    let sx = reflect101(2 * x + i64::try_from(j).unwrap() - 2, wi);
                    acc += f64::from(u32::try_from(ki * kj).unwrap()) * src[[sy, sx, ch]];
                }
            }
            acc / 256.0
        })
    }

    fn ramp(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            u8::try_from((y * 37 + x * 11 + ch * 71 + (x * y) % 13) % 256).unwrap()
        })
    }

    #[test]
    fn test_pyr_down_matches_the_written_out_filter() {
        // Odd sizes, so the last output row and column read past the edge and
        // the round-up of the output size is exercised.
        let src = ramp(9, 11, 3);
        let src_f32 = src.mapv(f32::from);
        let expected = reference(&src.mapv(f64::from));

        let res = src_f32.pyr_down_def().unwrap();
        assert_eq!(res.dim(), (5, 6, 3));
        let worst = res
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        assert!(worst < 1e-4, "worst abs error {worst}");

        // u8 rounds the same sum half up, exactly.
        let res_u8 = src.pyr_down_def().unwrap();
        let expected_u8 = expected.mapv(|v| u8::try_from((v + 0.5).floor() as i64).unwrap());
        assert_eq!(res_u8, expected_u8);
    }

    #[test]
    fn test_pyr_down_output_size_rounds_up() {
        for (h, w) in [(1, 1), (2, 3), (5, 4), (8, 8), (7, 1)] {
            let res = Array2::<f32>::ones((h, w)).pyr_down_def().unwrap();
            assert_eq!(res.dim(), (h.div_ceil(2), w.div_ceil(2)), "{h}x{w}");
            assert!(res.iter().all(|&v| (v - 1.0).abs() < 1e-6), "{h}x{w}");
        }
    }

    #[test]
    fn test_pyr_down_def_reflects_the_border() {
        // On a ramp the border mode decides the first output pixel: reflect-101
        // mirrors about the edge pixel, replicate repeats it.
        let src =
            Array2::from_shape_fn((6, 6), |(y, x)| f32::from(u8::try_from(y * 6 + x).unwrap()));
        let def = src.pyr_down_def().unwrap();
        let reflect = src.pyr_down(BorderType::BorderReflect101).unwrap();
        let replicate = src.pyr_down(BorderType::BorderReplicate).unwrap();
        assert_eq!(def, reflect);
        assert!((def[[0, 0]] - replicate[[0, 0]]).abs() > 0.1);
    }

    #[test]
    fn test_pyr_down_rejects_a_constant_border() {
        // Unsupported by OpenCV: an error, not a panic.
        let err = Array2::<f32>::ones((8, 8))
            .pyr_down(BorderType::BorderConstant)
            .unwrap_err();
        assert!(matches!(err, PyrDownError::OpenCvError(_)), "got: {err}");
    }

    #[test]
    fn test_pyr_down_empty_input() {
        for (h, w) in [(0, 5), (5, 0), (0, 0)] {
            let res = Array2::<f32>::zeros((h, w)).pyr_down_def().unwrap();
            assert_eq!(res.dim(), (h.div_ceil(2), w.div_ceil(2)));
            let res = Array3::<u8>::zeros((h, w, 3)).pyr_down_def().unwrap();
            assert_eq!(res.dim(), (h.div_ceil(2), w.div_ceil(2), 3));
        }
    }

    #[test]
    fn test_pyr_down_non_standard_layouts_match_a_copy() {
        // Asymmetric content, so a transposition or flip would show up.
        let base = ramp(12, 10, 3).mapv(f32::from);
        let check = |name: &str, view: ArrayView3<f32>| {
            assert_eq!(
                view.pyr_down_def().unwrap(),
                view.to_owned().pyr_down_def().unwrap(),
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
            transposed.pyr_down_def().unwrap(),
            transposed.to_owned().pyr_down_def().unwrap()
        );
    }

    #[test]
    fn test_pyr_down_unit_length_axes() {
        // ndarray keeps an arbitrary stride on a length-1 axis: 0 for a one-row
        // slice, even after `to_owned`. Each must match the same values laid
        // out with default strides.
        let base = ramp(8, 9, 3).mapv(f32::from);
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
            assert_eq!(
                view.pyr_down_def().unwrap(),
                fresh3(view).pyr_down_def().unwrap(),
                "{:?}",
                view.strides()
            );
            assert_eq!(
                view.to_owned().pyr_down_def().unwrap(),
                fresh3(view).pyr_down_def().unwrap()
            );
        }
        let plane = base.slice(s![.., .., 0]).to_owned();
        let row = plane.slice(s![3..4, ..]);
        assert_eq!(
            row.pyr_down_def().unwrap(),
            fresh2(row).pyr_down_def().unwrap()
        );
        let inserted = plane.row(3).insert_axis(Axis(0));
        assert_eq!(
            inserted.pyr_down_def().unwrap(),
            fresh2(row).pyr_down_def().unwrap()
        );
    }
}
