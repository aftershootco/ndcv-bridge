use crate::{BorderType, Interpolation, MatAsNd, NdAsImage, NdAsImageMut, NdImage};

#[derive(Debug, thiserror::Error)]
pub enum AffineError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("could not estimate a transformation from the given correspondences")]
    EstimationFailed,
}

pub trait NdCvWarpAffine<T: bytemuck::Pod + num::Zero + crate::types::CvType, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn warp_affine(
        &self,
        transformation: ndarray::ArrayView2<f32>,
        output_size: impl Into<glam::USizeVec2>,
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: impl Into<glam::DVec4>,
    ) -> Result<ndarray::Array<T, D>, AffineError>;
}

pub trait NdCvInvertWarpAffine<
    T: bytemuck::Pod + crate::types::CvType + num::Float,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn invert_warp_affine(&self) -> Result<ndarray::Array<T, D>, AffineError>;
}

impl<T: bytemuck::Pod + crate::types::CvType + num::Float, S: ndarray::Data<Elem = T>>
    NdCvInvertWarpAffine<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn invert_warp_affine(&self) -> Result<ndarray::Array<T, ndarray::Ix2>, AffineError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array2::zeros((self.shape()[0], self.shape()[1]));

        opencv::imgproc::invert_affine_transform(mat.as_ref(), dest.as_image_mat_mut()?.as_mut())?;

        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvWarpAffine<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn warp_affine(
        &self,
        transformation: ndarray::ArrayView2<f32>,
        output_size: impl Into<glam::USizeVec2>,
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: impl Into<glam::DVec4>,
    ) -> Result<ndarray::Array<T, ndarray::Ix2>, AffineError> {
        let mat = self.as_image_mat()?;
        let transformation = transformation.as_image_mat()?;
        let output_size = output_size.into();
        let mut dest = ndarray::Array2::zeros((output_size.y, output_size.x));
        let mut dest_mat = dest.as_image_mat_mut()?;
        let border_value = border_value.into();

        opencv::imgproc::warp_affine(
            mat.as_ref(),
            dest_mat.as_mut(),
            transformation.as_ref(),
            opencv::core::Size::new(output_size.x as i32, output_size.y as i32),
            interpolation as i32,
            border_type as i32,
            opencv::core::VecN([
                border_value.x,
                border_value.y,
                border_value.z,
                border_value.w,
            ]),
        )?;

        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvWarpAffine<T, ndarray::Ix3> for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn warp_affine(
        &self,
        transformation: ndarray::ArrayView2<f32>,
        output_size: impl Into<glam::USizeVec2>,
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: impl Into<glam::DVec4>,
    ) -> Result<ndarray::Array<T, ndarray::Ix3>, AffineError> {
        let mat = self.as_image_mat()?;
        let transformation = transformation.as_image_mat()?;
        let output_size = output_size.into();
        let mut dest = ndarray::Array3::zeros((output_size.y, output_size.x, self.channels()));
        let mut dest_mat = dest.as_image_mat_mut()?;
        let border_value = border_value.into();

        opencv::imgproc::warp_affine(
            mat.as_ref(),
            dest_mat.as_mut(),
            transformation.as_ref(),
            opencv::core::Size::new(output_size.x as i32, output_size.y as i32),
            interpolation as i32,
            border_type as i32,
            opencv::core::VecN([
                border_value.x,
                border_value.y,
                border_value.z,
                border_value.w,
            ]),
        )?;

        Ok(dest)
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum EstimateAffineMethod {
    Lmeds = opencv::calib3d::LMEDS,
    Ransac = opencv::calib3d::RANSAC,
}

pub struct EstimateAffineResult {
    pub inliers: ndarray::Array2<u8>,
    pub transformation: ndarray::Array2<f64>,
}

pub trait NdCvEstimateAffinePartial2D<
    T: bytemuck::Pod + num::Zero + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn estimate_affine_partial_2d(
        &self,
        reference: ndarray::Array<T, D>,
        method: EstimateAffineMethod,
        ransac_reproj_threshold: f64,
        max_iters: usize,
        confidence: f64,
        refine_iters: usize,
    ) -> Result<EstimateAffineResult, AffineError>;
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvEstimateAffinePartial2D<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn estimate_affine_partial_2d(
        &self,
        reference: ndarray::Array<T, ndarray::Ix2>,
        method: EstimateAffineMethod,
        ransac_reproj_threshold: f64,
        max_iters: usize,
        confidence: f64,
        refine_iters: usize,
    ) -> Result<EstimateAffineResult, AffineError> {
        let input_mat = self.as_image_mat()?;
        let reference_mat = reference.as_image_mat()?;
        let mut inliers_mat = opencv::core::Mat::default();

        let transformation_mat = opencv::calib3d::estimate_affine_partial_2d(
            input_mat.as_ref(),
            reference_mat.as_ref(),
            &mut inliers_mat,
            method as i32,
            ransac_reproj_threshold,
            max_iters,
            confidence,
            refine_iters,
        )?;

        // estimateAffinePartial2D signals "no model found" with an empty Mat, not an error;
        // converting that default CV_8U Mat would surface as a misleading TypeMismatch
        use opencv::prelude::MatTraitConst;
        if transformation_mat.empty() {
            return Err(AffineError::EstimationFailed);
        }

        let inliers = inliers_mat.as_ndarray()?.to_owned();

        let transformation: ndarray::Array2<f64> = transformation_mat.as_ndarray()?.to_owned();

        // the exact 2-point kernel divides by zero on coincident points and "succeeds"
        // with an all-NaN matrix; never hand that to the caller as Ok
        if !transformation.iter().all(|v| v.is_finite()) {
            return Err(AffineError::EstimationFailed);
        }

        Ok(EstimateAffineResult {
            inliers,
            transformation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec4, USizeVec2};
    use ndarray::{Array2, Array3, array, s};

    fn assert_close(actual: &Array2<f64>, expected: &Array2<f64>) {
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn test_warp_affine_identity() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr[[2, 3, 0]] = 255;
        let identity = array![[1.0f32, 0., 0.], [0., 1., 0.]];
        let res = arr
            .warp_affine(
                identity.view(),
                USizeVec2::new(10, 10),
                Interpolation::Nearest,
                BorderType::BorderConstant,
                DVec4::ZERO,
            )
            .unwrap();
        assert_eq!(res, arr);
    }

    #[test]
    fn test_warp_affine_translation() {
        let mut arr = Array2::<u8>::zeros((10, 10));
        arr[[2, 2]] = 255;
        let translation = array![[1.0f32, 0., 3.], [0., 1., 1.]];
        let res = arr
            .warp_affine(
                translation.view(),
                USizeVec2::new(10, 10),
                Interpolation::Nearest,
                BorderType::BorderConstant,
                DVec4::ZERO,
            )
            .unwrap();
        // x maps to columns, y to rows: (col 2, row 2) -> (col 5, row 3)
        assert_eq!(res[[3, 5]], 255);
        assert_eq!(res[[2, 2]], 0);
    }

    #[test]
    fn test_warp_affine_output_size() {
        let arr = Array3::<u8>::from_elem((10, 10, 3), 7);
        let identity = array![[1.0f32, 0., 0.], [0., 1., 0.]];
        let res = arr
            .warp_affine(
                identity.view(),
                USizeVec2::new(5, 5),
                Interpolation::Nearest,
                BorderType::BorderConstant,
                DVec4::ZERO,
            )
            .unwrap();
        assert_eq!(res.shape(), &[5, 5, 3]);
        assert!(res.iter().all(|&v| v == 7));
    }

    #[test]
    fn test_warp_affine_non_square_output() {
        let arr = Array3::<u8>::from_elem((10, 10, 3), 7);
        let identity = array![[1.0f32, 0., 0.], [0., 1., 0.]];
        let res = arr
            .warp_affine(
                identity.view(),
                USizeVec2::new(20, 10),
                Interpolation::Nearest,
                BorderType::BorderConstant,
                DVec4::ZERO,
            )
            .unwrap();
        // output_size (20, 10) is (width, height) -> shape (rows 10, cols 20)
        assert_eq!(res.shape(), &[10, 20, 3]);
        // source fills the left 10 columns; the rest is border fill
        assert!(res.slice(s![.., ..10, ..]).iter().all(|&v| v == 7));
        assert!(res.slice(s![.., 10.., ..]).iter().all(|&v| v == 0));
    }

    #[test]
    fn test_warp_affine_translation_non_square() {
        let mut arr = Array2::<u8>::zeros((10, 10));
        arr[[2, 2]] = 255;
        let translation = array![[1.0f32, 0., 3.], [0., 1., 1.]];
        let res = arr
            .warp_affine(
                translation.view(),
                USizeVec2::new(20, 10),
                Interpolation::Nearest,
                BorderType::BorderConstant,
                DVec4::ZERO,
            )
            .unwrap();
        assert_eq!(res.shape(), &[10, 20]);
        // x maps to columns, y to rows: (col 2, row 2) -> (col 5, row 3)
        assert_eq!(res[[3, 5]], 255);
        assert_eq!(res[[2, 2]], 0);
    }

    #[test]
    fn test_invert_warp_affine_identity() {
        let identity = array![[1.0f32, 0., 0.], [0., 1., 0.]];
        let res = identity.invert_warp_affine().unwrap();
        assert_eq!(res, identity);
    }

    #[test]
    fn test_invert_warp_affine_translation() {
        let transform = array![[1.0f32, 0., 5.], [0., 1., 3.]];
        let res = transform.invert_warp_affine().unwrap();
        assert_eq!(res, array![[1.0f32, 0., -5.], [0., 1., -3.]]);
    }

    #[test]
    fn test_invert_warp_affine_scale() {
        let transform = array![[2.0f32, 0., 0.], [0., 2., 0.]];
        let res = transform.invert_warp_affine().unwrap();
        assert_eq!(res, array![[0.5f32, 0., 0.], [0., 0.5, 0.]]);
    }

    #[test]
    fn test_estimate_affine_partial_2d_identity() {
        let points = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.]];
        let res = points
            .estimate_affine_partial_2d(
                points.clone(),
                EstimateAffineMethod::Ransac,
                3.0,
                2000,
                0.99,
                10,
            )
            .unwrap();
        assert_close(&res.transformation, &array![[1.0f64, 0., 0.], [0., 1., 0.]]);
    }

    #[test]
    fn test_estimate_affine_partial_2d_translation() {
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.]];
        let dst = array![[5.0f64, -3.], [15., -3.], [15., 7.], [5., 7.]];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        assert_close(
            &res.transformation,
            &array![[1.0f64, 0., 5.], [0., 1., -3.]],
        );
    }

    #[test]
    fn test_estimate_affine_partial_2d_rotation() {
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.]];
        // 90 degree rotation: (x, y) -> (-y, x)
        let dst = array![[0.0f64, 0.], [0., 10.], [-10., 10.], [-10., 0.]];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Lmeds, 3.0, 2000, 0.99, 10)
            .unwrap();
        assert_close(
            &res.transformation,
            &array![[0.0f64, -1., 0.], [1., 0., 0.]],
        );
    }

    // ---- Regression tests for issues found reviewing PR #13 ----

    // Issue 1 (affine.rs, estimate_affine_partial_2d): the inlier mask is never written back.
    // `inliers` is allocated as an (N, 2) T-typed array over external ndarray memory, but
    // OpenCV writes an (N, 1) CV_8U mask into a *reallocated* buffer, leaving our array
    // all-zeros. Asserts the mask is populated -> FAILS on current code.
    #[test]
    fn test_estimate_affine_partial_2d_inliers_written_back() {
        // Four exact correspondences plus one gross outlier RANSAC should reject.
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.], [5., 5.]];
        let dst = array![
            [0.0f64, 0.],
            [10., 0.],
            [10., 10.],
            [0., 10.],
            [900., -900.]
        ];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        let marked = res.inliers.iter().filter(|&&v| v != 0).count();
        assert!(
            marked >= 4,
            "expected the 4 exact correspondences to be marked as inliers, \
             got {marked} non-zero entries; inliers = {:?}",
            res.inliers
        );
    }

    #[test]
    fn test_estimate_affine_partial_2d_inliers_shape_is_one_per_point() {
        // The mask has one entry per point pair, not per coordinate.
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.], [5., 5.]];
        let res = src
            .estimate_affine_partial_2d(
                src.clone(),
                EstimateAffineMethod::Ransac,
                3.0,
                2000,
                0.99,
                10,
            )
            .unwrap();
        assert_eq!(res.inliers.shape(), &[5, 1], "inliers = {:?}", res.inliers);
    }

    #[test]
    fn test_estimate_affine_partial_2d_inliers_all_marked_for_exact_fit() {
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.]];
        let dst = array![[5.0f64, -3.], [15., -3.], [15., 7.], [5., 7.]];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        assert!(
            res.inliers.iter().all(|&v| v != 0),
            "every exact correspondence should be an inlier; inliers = {:?}",
            res.inliers
        );
    }

    #[test]
    fn test_estimate_affine_partial_2d_inliers_flags_outlier_rows() {
        // Six correspondences under a pure translation, with rows 2 and 5 grossly
        // corrupted. The mask must identify exactly the corrupted rows.
        let src = array![
            [0.0f64, 0.],
            [10., 0.],
            [10., 10.],
            [0., 10.],
            [5., 5.],
            [20., 20.]
        ];
        let dst = array![
            [5.0f64, -3.],
            [15., -3.],
            [-800., 600.], // outlier
            [5., 7.],
            [10., 2.],
            [900., -900.] // outlier
        ];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        for (i, expected_inlier) in [true, true, false, true, true, false].iter().enumerate() {
            assert_eq!(
                res.inliers[[i, 0]] != 0,
                *expected_inlier,
                "point {i} misclassified; inliers = {:?}",
                res.inliers
            );
        }
        // The outliers must not have dragged the estimate off the exact translation.
        assert_close(
            &res.transformation,
            &array![[1.0f64, 0., 5.], [0., 1., -3.]],
        );
    }

    #[test]
    fn test_estimate_affine_partial_2d_inliers_written_back_lmeds() {
        // LMEDS fills the inlier mask too, not just RANSAC.
        let src = array![[0.0f64, 0.], [10., 0.], [10., 10.], [0., 10.], [5., 5.]];
        let dst = array![
            [0.0f64, 0.],
            [10., 0.],
            [10., 10.],
            [0., 10.],
            [900., -900.] // outlier
        ];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Lmeds, 3.0, 2000, 0.99, 10)
            .unwrap();
        let marked = res.inliers.iter().filter(|&&v| v != 0).count();
        assert!(
            marked >= 4,
            "LMEDS should mark the 4 exact correspondences as inliers; inliers = {:?}",
            res.inliers
        );
        assert_eq!(
            res.inliers[[4, 0]],
            0,
            "LMEDS should reject the gross outlier; inliers = {:?}",
            res.inliers
        );
    }

    #[test]
    fn test_estimate_affine_partial_2d_inliers_written_back_f32() {
        // The mask stays u8 and is populated for f32 point arrays as well.
        let src = array![[0.0f32, 0.], [10., 0.], [10., 10.], [0., 10.], [5., 5.]];
        let dst = array![
            [0.0f32, 0.],
            [10., 0.],
            [10., 10.],
            [0., 10.],
            [900., -900.] // outlier
        ];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        assert_eq!(res.inliers.shape(), &[5, 1]);
        let marked = res.inliers.iter().filter(|&&v| v != 0).count();
        assert!(
            marked >= 4,
            "expected the 4 exact f32 correspondences to be marked as inliers; inliers = {:?}",
            res.inliers
        );
        assert_eq!(
            res.inliers[[4, 0]],
            0,
            "the gross outlier should not be an inlier; inliers = {:?}",
            res.inliers
        );
    }

    // Issue 3a (affine.rs:193): the estimated transform is always CV_64F, so any T other than
    // f64 fails to convert the result (TypeMismatch). Asserts f32 works -> FAILS on current code.
    #[test]
    fn test_estimate_affine_partial_2d_accepts_f32() {
        let src = array![[0.0f32, 0.], [10., 0.], [10., 10.], [0., 10.]];
        let dst = src.clone();
        let res =
            src.estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10);
        assert!(
            res.is_ok(),
            "f32 estimate_affine_partial_2d should succeed, but errored: {:?}",
            res.err()
        );
    }

    // Unlike invertAffineTransform above, estimateAffinePartial2D converts integer point sets
    // to float internally, so integer T works end to end. Characterizes that integer input
    // succeeds and still yields the exact f64 transform and inlier mask.
    #[test]
    fn test_estimate_affine_partial_2d_accepts_integer_type() {
        let src = array![[0i16, 0], [10, 0], [10, 10], [0, 10]];
        let dst = array![[5i16, -3], [15, -3], [15, 7], [5, 7]];
        let res = src
            .estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
            .unwrap();
        assert_close(
            &res.transformation,
            &array![[1.0f64, 0., 5.], [0., 1., -3.]],
        );
        assert!(
            res.inliers.iter().all(|&v| v != 0),
            "all exact correspondences should be inliers; inliers = {:?}",
            res.inliers
        );
    }

    // ---- Regression tests: degenerate inputs must fail loudly ----

    // Exactly 2 coincident correspondences drive OpenCV's exact 2-point kernel into a
    // division by zero; it used to "succeed" and return Ok with an all-NaN transformation.
    #[test]
    fn test_estimate_affine_partial_2d_two_coincident_points_fails() {
        let src = array![[5.0f64, 5.], [5., 5.]];
        let dst = src.clone();
        match src.estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
        {
            Err(AffineError::EstimationFailed) => {}
            Err(other) => panic!("expected EstimationFailed, got: {other:?}"),
            Ok(res) => panic!(
                "degenerate 2-point input unexpectedly succeeded: {:?}",
                res.transformation
            ),
        }
    }

    // With >= 3 degenerate points RANSAC finds no model and OpenCV returns an empty Mat;
    // this used to surface as a misleading conversion TypeMismatch (u8 vs f64).
    #[test]
    fn test_estimate_affine_partial_2d_degenerate_points_estimation_failed() {
        let src = array![[5.0f64, 5.], [5., 5.], [5., 5.]];
        let dst = src.clone();
        match src.estimate_affine_partial_2d(dst, EstimateAffineMethod::Ransac, 3.0, 2000, 0.99, 10)
        {
            Err(AffineError::EstimationFailed) => {}
            Err(other) => panic!("expected EstimationFailed, got: {other:?}"),
            Ok(_) => panic!("3 coincident points unexpectedly produced a transformation"),
        }
    }
}
