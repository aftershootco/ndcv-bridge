//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.gaussian_blur.html>
use crate::conversions::*;
use glam::{DVec2, U16Vec2};
use ndarray::*;
use opencv::core::AlgorithmHint as OpencvAlgorithmHint;
use opencv::core::MatTraitConst;
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum GaussianBlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub enum BorderType {
    #[default]
    BorderConstant = 0,
    BorderReplicate = 1,
    BorderReflect = 2,
    BorderWrap = 3,
    BorderReflect101 = 4,
    BorderTransparent = 5,
    BorderIsolated = 16,
}

impl BorderType {
    /// cv2's `BORDER_DEFAULT`, which is an alias for `BORDER_REFLECT_101` and
    /// the border every cv2 filter uses unless told otherwise -- except the
    /// morphology ones, which default to `BorderConstant`.
    ///
    /// Spelled like the variants rather than in the Rust constant convention,
    /// so `BorderType::BorderDefault` reads alongside
    /// `BorderType::BorderConstant` at a call site.
    #[allow(non_upper_case_globals)]
    pub const BorderDefault: Self = Self::BorderReflect101;
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub enum AlgorithmHint {
    #[default]
    AlgoHintDefault = OpencvAlgorithmHint::ALGO_HINT_DEFAULT as isize,
    AlgoHintAccurate = OpencvAlgorithmHint::ALGO_HINT_ACCURATE as isize,
    AlgoHintApprox = OpencvAlgorithmHint::ALGO_HINT_APPROX as isize,
}

impl AlgorithmHint {
    pub fn to_opencv(self) -> OpencvAlgorithmHint {
        match self {
            AlgorithmHint::AlgoHintDefault => OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
            AlgorithmHint::AlgoHintAccurate => OpencvAlgorithmHint::ALGO_HINT_ACCURATE,
            AlgorithmHint::AlgoHintApprox => OpencvAlgorithmHint::ALGO_HINT_APPROX,
        }
    }
}

mod seal {
    pub trait Sealed {}
    // src: input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}

    /// The types OpenCV filters on the calling thread alone. `u8` and `u16` take OpenCV's
    /// fixed-point path, which already runs on OpenCV's own thread pool; cut into strips they
    /// took 1.1 to 1.7 times as long, so `par_gaussian_blur` does not take them.
    pub trait OneThreadInOpenCv: Sealed {}
    impl OneThreadInOpenCv for i16 {}
    impl OneThreadInOpenCv for f32 {}
    impl OneThreadInOpenCv for f64 {}
}

pub trait NdCvGaussianBlur<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError>;
    /// `cv2.GaussianBlur` defaults, whose `borderType` is
    /// [`BorderType::BorderDefault`] -- not the constant border the morphology
    /// `_def` helpers pass, which is what cv2 defaults to for *those*.
    fn gaussian_blur_def(
        &self,
        kernel: impl Into<U16Vec2>,
        sigma: f64,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        self.gaussian_blur(kernel, (sigma, sigma), BorderType::BorderDefault)
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvGaussianBlur<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        let kernel_size = kernel_size.into();
        let sigma = sigma.into();
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::gaussian_blur(
            &*cv_self,
            &mut *cv_dst,
            opencv::core::Size {
                width: kernel_size.x as i32,
                height: kernel_size.y as i32,
            },
            sigma.x,
            sigma.y,
            border_type as i32,
            OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(dst)
    }
}

/// [`NdCvGaussianBlur`] spread over rayon's thread pool, for `i16`, `f32` and `f64`. OpenCV runs
/// `GaussianBlur` on the calling thread alone for those types, so on a large image this is
/// several times faster. (`u8` and `u16` are already parallel inside OpenCV and are left out.)
///
/// The image is cut into strips along its first axis, one per thread, and each strip is blurred
/// by its own OpenCV call straight into its rows of the output. Each call sees its strip as a
/// region of the whole image, so OpenCV reads the rows past the strip's edges from the image and
/// applies the border only at the image's own edges. The result is bit-identical to
/// [`NdCvGaussianBlur::gaussian_blur`], with no extra memory beyond a standard-layout copy of
/// input that is not in standard layout.
///
/// It falls back to that single call when there is nothing to gain or the split would change the
/// result:
///
/// - the image is too short to split: fewer rows than two strips of `max(16, 4 * radius)` rows;
/// - the border is not one of `BorderConstant`, `BorderReplicate`, `BorderReflect` or
///   `BorderReflect101`: [`BorderType::BorderWrap`] reads the opposite edge of the image, and
///   [`BorderType::BorderIsolated`] would put a border at every strip edge;
/// - the image is one column wide, which OpenCV filters differently as a region of a taller image;
/// - the vertical kernel size cannot be worked out (zero size and no positive sigma), so OpenCV
///   reports the error.
pub trait NdCvParGaussianBlur<
    T: bytemuck::Pod + seal::OneThreadInOpenCv + crate::types::CvType,
    D: ndarray::Dimension,
>: NdCvGaussianBlur<T, D>
{
    fn par_gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError>;
    /// See [`NdCvGaussianBlur::gaussian_blur_def`].
    fn par_gaussian_blur_def(
        &self,
        kernel: impl Into<U16Vec2>,
        sigma: f64,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        self.par_gaussian_blur(kernel, (sigma, sigma), BorderType::BorderDefault)
    }
}

/// Fewest rows a strip is cut to, so a small image is not split into slivers.
const MIN_STRIP_ROWS: usize = 16;

/// Rows OpenCV reads past each edge of a strip, which sets how short a strip can usefully be: at
/// least the kernel's vertical radius. A zero size is derived from sigma as OpenCV does, with the rule for floating-point images,
/// `cvRound(sigma * 8 + 1) | 1`, which is never smaller than the 8-bit one (`sigma * 6`), plus a
/// row for the rounding. `None` when OpenCV would derive no size.
fn halo_rows(kernel_height: u16, sigma: DVec2) -> Option<usize> {
    if kernel_height > 0 {
        return Some(usize::from(kernel_height) / 2);
    }
    // OpenCV's createGaussianKernels: a non-positive sigma_y takes sigma_x.
    let s = if sigma.y <= 0.0 { sigma.x } else { sigma.y };
    // A float-to-int `as` saturates, so a huge sigma becomes a huge halo and the call falls back.
    (s.is_finite() && s > 0.0).then(|| (s * 8.0 + 1.0).round() as usize / 2 + 1)
}

/// Rows per strip, or `None` for one call. A strip shorter than four halos would spend most of
/// its time on the rows past its edges, which every strip filters again.
fn strip_rows(
    rows: usize,
    cols: usize,
    kernel_height: u16,
    sigma: DVec2,
    border_type: BorderType,
    threads: usize,
) -> Option<usize> {
    // Only borders that read the image's own edge: BorderWrap reads the opposite edge, and
    // BorderIsolated would treat every strip edge as a border.
    if !matches!(
        border_type,
        BorderType::BorderConstant
            | BorderType::BorderReplicate
            | BorderType::BorderReflect
            | BorderType::BorderReflect101
    ) {
        return None;
    }
    // OpenCV filters a one-column region of a taller image differently from a one-column image:
    // the results differ in the last bit.
    if cols < 2 {
        return None;
    }
    let halo = halo_rows(kernel_height, sigma)?;
    let core = rows
        .div_ceil(threads.max(1))
        .max(halo.saturating_mul(4))
        .max(MIN_STRIP_ROWS);
    (core < rows).then_some(core)
}

impl<
    T: bytemuck::Pod + num::Zero + seal::OneThreadInOpenCv + crate::types::CvType + Send + Sync,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvParGaussianBlur<T, D> for ArrayBase<S, D>
where
    Self: NdCvGaussianBlur<T, D>,
    for<'a> ArrayView<'a, T, D>: NdCvGaussianBlur<T, D>,
    for<'a> ArrayViewMut<'a, T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn par_gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        let kernel_size = kernel_size.into();
        let sigma = sigma.into();
        let (rows, cols) = match self.shape() {
            [rows, cols, ..] => (*rows, *cols),
            _ => (0, 0),
        };
        // The Mat conversion keeps only the outer stride, so input that is not in standard layout
        // is copied, for the single call as well; standard input is borrowed. Standard layout also
        // makes every output strip one contiguous run of rows.
        let x = self.as_standard_layout();
        let threads = rayon::current_num_threads();
        let Some(core) = strip_rows(rows, cols, kernel_size.y, sigma, border_type, threads)
            .filter(|_| !x.is_empty())
        else {
            return x.view().gaussian_blur(kernel_size, sigma, border_type);
        };

        let row_len = x.len() / rows;
        let mut out = ndarray::Array::<T, D>::zeros(self.raw_dim());
        let out_data = out.as_slice_mut().ok_or(ConversionError::from(
            ConversionErrorKind::NonContiguousData,
        ))?;
        out_data
            .par_chunks_mut(core * row_len)
            .enumerate()
            .try_for_each(|(i, o)| -> Result<(), GaussianBlurError> {
                let height = o.len() / row_len;
                // The strip is a region of a Mat over the whole image, so OpenCV reads the rows
                // past its edges from the image and applies the border only at the image's own
                // edges, exactly as in one call. Rows and columns fit in an i32, as `whole`
                // exists.
                let view = x.view();
                let whole = view.as_image_mat()?;
                let region =
                    opencv::core::Rect::new(0, (i * core) as i32, whole.cols(), height as i32);
                let src = opencv::core::Mat::roi(&*whole, region)?;
                let mut dim = x.raw_dim();
                dim[0] = height;
                let mut dst = ArrayViewMut::from_shape(dim, o).map_err(ConversionError::from)?;
                let mut cv_dst = dst.as_image_mat_mut()?;
                opencv::imgproc::gaussian_blur(
                    &src,
                    &mut *cv_dst,
                    opencv::core::Size {
                        width: kernel_size.x as i32,
                        height: kernel_size.y as i32,
                    },
                    sigma.x,
                    sigma.y,
                    border_type as i32,
                    OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
                )?;
                Ok(())
            })?;
        Ok(out)
    }
}

// impl<
//         T: bytemuck::Pod + num::Zero + seal::Sealed,
//         S: ndarray::RawData + ndarray::Data<Elem = T>,
//     > NdCvGaussianBlur<T, Ix3> for ArrayBase<S, Ix3>
// {
//     fn gaussian_blur(
//         &self,
//         kernel_size: (u8, u8),
//         sigma_x: f64,
//         sigma_y: f64,
//         border_type: BorderType,
//     ) -> Result<ndarray::Array<T, Ix3>, GaussianBlurError> {
//         let mut dst = ndarray::Array::zeros(self.dim());
//         let cv_self = self.as_image_mat()?;
//         let mut cv_dst = dst.as_image_mat_mut()?;
//         opencv::imgproc::gaussian_blur(
//             &*cv_self,
//             &mut *cv_dst,
//             opencv::core::Size {
//                 width: kernel_size.0 as i32,
//                 height: kernel_size.1 as i32,
//             },
//             sigma_x,
//             sigma_y,
//             border_type as i32,
//         )
//
//         .attach("Failed to apply gaussian blur")?;
//         Ok(dst)
//     }
// }
//
// impl<
//         T: bytemuck::Pod + num::Zero + seal::Sealed,
//         S: ndarray::RawData + ndarray::Data<Elem = T>,
//     > NdCvGaussianBlur<T, Ix2> for ArrayBase<S, Ix2>
// {
//     fn gaussian_blur(
//         &self,
//         kernel_size: (u8, u8),
//         sigma_x: f64,
//         sigma_y: f64,
//         border_type: BorderType,
//     ) -> Result<ndarray::Array<T, Ix2>, GaussianBlurError> {
//         let mut dst = ndarray::Array::zeros(self.dim());
//         let cv_self = self.as_image_mat()?;
//         let mut cv_dst = dst.as_image_mat_mut()?;
//         opencv::imgproc::gaussian_blur(
//             &*cv_self,
//             &mut *cv_dst,
//             opencv::core::Size {
//                 width: kernel_size.0 as i32,
//                 height: kernel_size.1 as i32,
//             },
//             sigma_x,
//             sigma_y,
//             border_type as i32,
//         )
//
//         .attach("Failed to apply gaussian blur")?;
//         Ok(dst)
//     }
// }

/// For smaller values it is faster to use the allocated version
/// For example in a 4k f32 image this is about 50% faster than the allocated one
pub trait NdCvGaussianBlurInPlace<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>
{
    fn gaussian_blur_inplace(
        &mut self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<&mut Self, GaussianBlurError>;
    /// See [`NdCvGaussianBlur::gaussian_blur_def`].
    fn gaussian_blur_def_inplace(
        &mut self,
        kernel: impl Into<U16Vec2>,
        sigma: f64,
    ) -> Result<&mut Self, GaussianBlurError> {
        self.gaussian_blur_inplace(kernel, (sigma, sigma), BorderType::BorderDefault)
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::DataMut<Elem = T>,
    D: ndarray::Dimension,
> NdCvGaussianBlurInPlace<T, D> for ArrayBase<S, D>
where
    Self: crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>,
{
    fn gaussian_blur_inplace(
        &mut self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<&mut Self, GaussianBlurError> {
        let kernel_size = kernel_size.into();
        let sigma = sigma.into();
        let mut cv_self = self.as_image_mat_mut()?;

        unsafe {
            crate::inplace::op_inplace(&mut cv_self, |this, out| {
                opencv::imgproc::gaussian_blur(
                    this,
                    out,
                    opencv::core::Size {
                        width: kernel_size.x as i32,
                        height: kernel_size.y as i32,
                    },
                    sigma.x,
                    sigma.y,
                    border_type as i32,
                    OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
                )
            })
        }?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_gaussian_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let kernel_size = (3, 3);
        let sigma_x = 0.0;
        let sigma_y = 0.0;
        let border_type = BorderType::BorderConstant;
        let res = arr
            .gaussian_blur(kernel_size, (sigma_x, sigma_y), border_type)
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_gaussian_edge_preservation() {
        // Create an image with a sharp edge
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![..5, .., ..]).fill(255); // Top half white, bottom half black

        let res = arr
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();

        // Check that the middle row (edge) has intermediate values
        let middle_row = res.slice(s![4..6, 5, 0]);
        assert!(middle_row.iter().all(|&x| x > 0 && x < 255));
    }

    #[test]
    fn test_gaussian_different_kernel_sizes() {
        let arr = Array3::<u8>::ones((20, 20, 3));

        // Test different kernel sizes
        let kernel_sizes = [(3, 3), (5, 5), (7, 7)];
        for &kernel_size in &kernel_sizes {
            let res = arr
                .gaussian_blur(kernel_size, (1.0, 1.0), BorderType::BorderConstant)
                .unwrap();
            assert_eq!(res.shape(), &[20, 20, 3]);
        }
    }

    #[test]
    fn test_gaussian_different_border_types() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![4..7, 4..7, ..]).fill(255); // White square in center

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for border_type in border_types {
            let _res = arr.gaussian_blur((3, 3), (1.0, 1.0), border_type).unwrap();
            let res = arr
                .gaussian_blur_inplace((3, 3), (1.0, 1.0), border_type)
                .unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_gaussian_different_types() {
        // Test with different numeric types
        let arr_u8 = Array3::<u8>::ones((10, 10, 3));
        let arr_f32 = Array3::<f32>::ones((10, 10, 3));

        let res_u8 = arr_u8
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();
        let res_f32 = arr_f32
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();

        assert_eq!(res_u8.shape(), &[10, 10, 3]);
        assert_eq!(res_f32.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_gaussian_blur_def() {
        // Exercises the default helper directly; `Ok(Default::default())` would
        // yield an empty (0,0,0) array here.
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr.gaussian_blur_def((3, 3), 1.0).unwrap();
        assert_eq!(res.dim(), (10, 10, 3));
    }

    #[test]
    fn test_border_default_is_reflect_101() {
        // cv2.BORDER_DEFAULT == cv2.BORDER_REFLECT_101 == 4. The alias exists
        // so a `_def` helper can say which cv2 default it means; if it ever
        // stopped resolving to 4 every one of them would silently change.
        assert_eq!(BorderType::BorderDefault as i32, 4);
        assert_eq!(
            BorderType::BorderDefault as i32,
            BorderType::BorderReflect101 as i32
        );
    }

    #[test]
    fn test_gaussian_blur_def_uses_the_cv2_default_border() {
        // `_def` means the cv2 call, and cv2's borderType default is
        // BORDER_DEFAULT == BORDER_REFLECT_101. Reflecting a flat image keeps
        // it flat; a constant border is a black neighbour that drags the
        // corner down by well over half. Every Python slider this crate is
        // ported against calls GaussianBlur without a borderType, so the
        // difference is a parity bug, not a preference.
        let arr = Array2::<u8>::from_elem((40, 40), 255);

        let def = arr.gaussian_blur_def((9, 9), 0.0).unwrap();
        let reflect = arr
            .gaussian_blur((9, 9), (0.0, 0.0), BorderType::BorderReflect101)
            .unwrap();
        let constant = arr
            .gaussian_blur((9, 9), (0.0, 0.0), BorderType::BorderConstant)
            .unwrap();

        assert_eq!(def, reflect);
        // cv2 gives 255 and 97 for these two corners.
        assert_eq!(def[[0, 0]], 255);
        assert_eq!(constant[[0, 0]], 97);
        // Away from the border the two agree, which is why this went unnoticed.
        assert_eq!(def[[20, 20]], constant[[20, 20]]);
    }

    #[test]
    fn test_gaussian_blur_def_inplace_matches_gaussian_blur_def() {
        let arr = Array2::<u8>::from_elem((40, 40), 255);
        let mut inplace = arr.clone();
        inplace.gaussian_blur_def_inplace((9, 9), 0.0).unwrap();
        assert_eq!(inplace, arr.gaussian_blur_def((9, 9), 0.0).unwrap());
    }

    #[test]
    fn test_gaussian_invalid_kernel_size() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        // Even kernel sizes should fail; OpenCV requires odd ksize
        let err = arr
            .gaussian_blur((2, 2), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap_err();
        assert!(
            err.to_string().contains("ksize"),
            "expected the odd-ksize assertion, got: {err}"
        );
    }

    /// Deterministic noise over the type's whole range: a blur of noise mixes every row into its
    /// neighbours, so a strip that read one row too few would show.
    fn noise<T: num::NumCast + Copy, D: Dimension>(
        shape: impl IntoDimension<Dim = D>,
        max: f64,
    ) -> Array<T, D> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        Array::from_shape_simple_fn(shape, || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            num::cast(unit * max).unwrap()
        })
    }

    /// Runs `f` on a pool of `threads`, so a small image is still cut into many strips.
    fn on_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(f)
    }

    const BORDERS: [BorderType; 6] = [
        BorderType::BorderConstant,
        BorderType::BorderReplicate,
        BorderType::BorderReflect,
        BorderType::BorderReflect101,
        BorderType::BorderWrap,
        BorderType::BorderIsolated,
    ];

    /// `(kernel, sigma)`: explicit sizes, sizes from sigma, a sigma_y of 0 taking sigma_x, and
    /// different sizes on the two axes.
    const KERNELS: [((u16, u16), (f64, f64)); 8] = [
        ((3, 3), (0.0, 0.0)),
        ((5, 5), (0.0, 0.0)),
        ((1, 9), (2.0, 2.0)),
        ((15, 1), (3.0, 3.0)),
        ((0, 0), (1.0, 1.0)),
        ((0, 0), (3.7, 0.0)),
        ((0, 0), (1.5, 6.0)),
        ((0, 0), (10.0, 10.0)),
    ];

    fn assert_par_matches<T, D>(x: &Array<T, D>, threads: usize)
    where
        T: bytemuck::Pod + num::Zero + seal::OneThreadInOpenCv + crate::types::CvType + Send + Sync,
        T: PartialEq + std::fmt::Debug,
        D: Dimension,
        Array<T, D>: NdCvParGaussianBlur<T, D>,
    {
        for (kernel, sigma) in KERNELS {
            for border in BORDERS {
                // OpenCV rejects BorderWrap for most of these kernels; the fallback fails the
                // same way.
                let one = x.gaussian_blur(kernel, sigma, border).ok();
                let par = on_threads(threads, || x.par_gaussian_blur(kernel, sigma, border).ok());
                assert!(one.is_some() || matches!(border, BorderType::BorderWrap));
                assert!(
                    one == par,
                    "{:?} {} threads, kernel {kernel:?} sigma {sigma:?} {border:?}",
                    x.shape(),
                    threads,
                );
            }
        }
    }

    #[test]
    fn test_par_gaussian_blur_is_bit_identical_for_every_supported_type() {
        assert_par_matches(&noise::<i16, _>((301, 23, 3), 32767.0), 8);
        assert_par_matches(&noise::<f32, _>((301, 23, 3), 1.0), 8);
        assert_par_matches(&noise::<f64, _>((301, 23, 3), 1.0), 8);
    }

    #[test]
    fn test_par_gaussian_blur_is_bit_identical_across_shapes_and_strips() {
        // 2-D and 1, 3 and 4 channels; heights that split evenly, unevenly, into a last strip
        // shorter than the halo, and just past the point where a split starts.
        for threads in [2, 3, 7, 16, 64] {
            for rows in [33, 64, 97, 250, 1001] {
                assert_par_matches(&noise::<f32, _>((rows, 17), 1.0), threads);
                assert_par_matches(&noise::<f32, _>((rows, 2), 1.0), threads);
                assert_par_matches(&noise::<f32, _>((rows, 3, 3), 1.0), threads);
                assert_par_matches(&noise::<f32, _>((rows, 9, 1), 1.0), threads);
                assert_par_matches(&noise::<i16, _>((rows, 13, 3), 32767.0), threads);
                assert_par_matches(&noise::<f64, _>((rows, 5, 4), 1.0), threads);
            }
        }
    }

    #[test]
    fn test_par_gaussian_blur_def_is_bit_identical() {
        let x = noise::<f32, _>((500, 40, 3), 1.0);
        let par = on_threads(8, || x.par_gaussian_blur_def((0, 0), 2.5).unwrap());
        assert_eq!(par, x.gaussian_blur_def((0, 0), 2.5).unwrap());
    }

    #[test]
    fn test_par_gaussian_blur_split_plan() {
        // The comparisons above would also pass if every call fell back to one OpenCV call, so
        // pin down when the image is split.
        let s = DVec2::new(2.0, 2.0);
        let d = BorderType::BorderDefault;
        // 8 threads over 800 rows: 8 strips of 100. sigma 2 reads 9 rows past each edge.
        assert_eq!(strip_rows(800, 40, 0, s, d, 8), Some(100));
        // A strip is never shorter than four of those or 16 rows ...
        assert_eq!(strip_rows(800, 40, 0, s, d, 64), Some(36));
        assert_eq!(strip_rows(800, 40, 3, s, d, 64), Some(16));
        // ... so a short image, or a single thread, is one call.
        assert_eq!(strip_rows(36, 40, 0, s, d, 8), None);
        assert_eq!(strip_rows(800, 40, 0, s, d, 1), None);
        assert_eq!(strip_rows(800, 40, 0, DVec2::new(1e300, 1e300), d, 8), None);
        assert_eq!(strip_rows(0, 40, 3, s, d, 8), None);
        // One column is one call; two are split.
        assert_eq!(strip_rows(800, 1, 0, s, d, 8), None);
        assert_eq!(strip_rows(800, 2, 0, s, d, 8), Some(100));
        for border in [
            BorderType::BorderWrap,
            BorderType::BorderIsolated,
            BorderType::BorderTransparent,
        ] {
            assert_eq!(strip_rows(800, 40, 0, s, border, 8), None, "{border:?}");
        }

        assert_eq!(halo_rows(0, DVec2::new(2.0, 2.0)), Some(9));
        assert_eq!(halo_rows(7, DVec2::new(100.0, 100.0)), Some(3));
        // sigma_y of 0 takes sigma_x; a size of 0 with no positive sigma is left to OpenCV.
        assert_eq!(
            halo_rows(0, DVec2::new(1.0, 0.0)),
            halo_rows(0, DVec2::new(1.0, 1.0))
        );
        assert_eq!(halo_rows(0, DVec2::new(0.0, 0.0)), None);
        assert_eq!(halo_rows(0, DVec2::new(f64::NAN, 0.0)), None);
    }

    #[test]
    fn test_par_gaussian_blur_huge_sigma_falls_back() {
        let x = noise::<f32, _>((200, 20), 1.0);
        for sigma in [1e3, 1e300, f64::INFINITY] {
            let par = on_threads(8, || {
                x.par_gaussian_blur((3, 0), (1.0, sigma), BorderType::BorderDefault)
            });
            let one = x.gaussian_blur((3, 0), (1.0, sigma), BorderType::BorderDefault);
            assert_eq!(par.is_ok(), one.is_ok(), "sigma {sigma}");
            if let (Ok(par), Ok(one)) = (par, one) {
                assert_eq!(par, one, "sigma {sigma}");
            }
        }
    }

    #[test]
    fn test_par_gaussian_blur_non_standard_layouts_match_a_copy() {
        let x = noise::<f32, _>((300, 40, 3), 1.0);
        let views = [
            x.slice(s![..;2, .., ..]),
            x.slice(s![.., ..;-1, ..]),
            x.slice(s![10..290, 3..37, 1..]),
        ];
        for v in views {
            let par = on_threads(8, || {
                v.par_gaussian_blur((0, 0), (2.0, 2.0), BorderType::BorderDefault)
                    .unwrap()
            });
            assert_eq!(
                par,
                v.as_standard_layout()
                    .gaussian_blur((0, 0), (2.0, 2.0), BorderType::BorderDefault)
                    .unwrap()
            );
        }
        let t = noise::<f32, _>((40, 300), 1.0);
        let t = t.t();
        let par = on_threads(8, || {
            t.par_gaussian_blur((5, 5), (0.0, 0.0), BorderType::BorderDefault)
                .unwrap()
        });
        assert_eq!(
            par,
            t.as_standard_layout()
                .gaussian_blur((5, 5), (0.0, 0.0), BorderType::BorderDefault)
                .unwrap()
        );
    }

    #[test]
    fn test_par_gaussian_blur_unit_length_axes() {
        let x = noise::<f32, _>((400, 30, 3), 1.0);
        let column = x.slice(s![.., 7..8, ..]);
        let one_channel = x.slice(s![.., .., 1..2]);
        for v in [column, one_channel] {
            let par = on_threads(8, || {
                v.par_gaussian_blur((0, 0), (1.5, 1.5), BorderType::BorderDefault)
                    .unwrap()
            });
            assert_eq!(
                par,
                v.as_standard_layout()
                    .gaussian_blur((0, 0), (1.5, 1.5), BorderType::BorderDefault)
                    .unwrap()
            );
        }
    }

    #[test]
    fn test_par_gaussian_blur_errors_match_the_single_call() {
        let x = noise::<f32, _>((400, 30), 1.0);
        for (kernel, sigma) in [
            ((2, 2), (1.0, 1.0)),
            ((0, 0), (0.0, 0.0)),
            ((3, 4), (1.0, 1.0)),
        ] {
            let par = on_threads(8, || {
                x.par_gaussian_blur(kernel, sigma, BorderType::BorderDefault)
            });
            let one = x.gaussian_blur(kernel, sigma, BorderType::BorderDefault);
            assert!(
                par.is_err() && one.is_err(),
                "kernel {kernel:?} sigma {sigma:?}"
            );
        }
    }

    #[test]
    fn test_par_gaussian_blur_empty_input() {
        for x in [Array2::<f32>::zeros((0, 10)), Array2::<f32>::zeros((10, 0))] {
            let par = on_threads(8, || {
                x.par_gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderDefault)
            });
            let one = x.gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderDefault);
            assert_eq!(par.is_ok(), one.is_ok(), "{:?}", x.shape());
        }
    }
}
