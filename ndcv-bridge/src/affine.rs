use crate::{BorderType, Interpolation, MatAsNd, NdAsImage, NdAsImageMut, NdImage};

#[derive(Debug, thiserror::Error)]
pub enum AffineError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
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
    T: bytemuck::Pod + num::Zero + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn invert_warp_affine(&self) -> Result<ndarray::Array<T, D>, AffineError>;
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
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
        let mut dest = ndarray::Array2::zeros((output_size.x, output_size.y));
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
        let mut dest = ndarray::Array3::zeros((output_size.x, output_size.y, self.channels()));
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

pub struct EstimateAffineResult<T, D> {
    pub inliers: ndarray::Array<T, D>,
    pub transformation: ndarray::Array<T, D>,
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
    ) -> Result<EstimateAffineResult<T, D>, AffineError>;
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
    ) -> Result<EstimateAffineResult<T, ndarray::Ix2>, AffineError> {
        let input_mat = self.as_image_mat()?;
        let reference_mat = reference.as_image_mat()?;

        let mut inliers = ndarray::Array2::<T>::zeros(reference.dim());

        let transformation_mat = opencv::calib3d::estimate_affine_partial_2d(
            input_mat.as_ref(),
            reference_mat.as_ref(),
            inliers.as_image_mat_mut()?.as_mut(),
            method as i32,
            ransac_reproj_threshold,
            max_iters,
            confidence,
            refine_iters,
        )?
        .as_ndarray()?
        .to_owned();

        Ok(EstimateAffineResult {
            inliers,
            transformation: transformation_mat,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec4, USizeVec2};
    use ndarray::{Array2, Array3, array};

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
}
