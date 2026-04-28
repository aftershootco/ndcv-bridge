use crate::{BorderType, Interpolation, MatAsNd, NdAsImage, NdAsImageMut, NdImage, prelude_::*};
use error_stack::ResultExt;

pub trait NdCvWarpAffine<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn warp_affine(
        &self,
        transformation: &ndarray::Array2<f32>,
        output_size: (usize, usize),
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: (f64, f64, f64, f64),
    ) -> Result<ndarray::Array<T, D>, NdCvError>;
}

pub trait NdCvInvertWarpAffine<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn invert_warp_affine(&self) -> Result<ndarray::Array<T, D>, NdCvError>;
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvInvertWarpAffine<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn invert_warp_affine(&self) -> Result<ndarray::Array<T, ndarray::Ix2>, NdCvError> {
        let mat = self.as_image_mat().change_context(NdCvError)?;
        let mut dest = ndarray::Array2::zeros((self.shape()[0], self.shape()[1]));

        opencv::imgproc::invert_affine_transform(
            mat.as_ref(),
            dest.as_image_mat_mut().change_context(NdCvError)?.as_mut(),
        )
        .change_context(NdCvError)?;

        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvWarpAffine<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn warp_affine(
        &self,
        transformation: &ndarray::Array2<f32>,
        output_size: (usize, usize),
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: (f64, f64, f64, f64),
    ) -> error_stack::Result<ndarray::Array<T, ndarray::Ix2>, NdCvError> {
        let mat = self.as_image_mat().change_context(NdCvError)?;
        let transformation = transformation.as_image_mat().change_context(NdCvError)?;
        let mut dest = ndarray::Array2::zeros(output_size);
        let mut dest_mat = dest.as_image_mat_mut().change_context(NdCvError)?;

        opencv::imgproc::warp_affine(
            mat.as_ref(),
            dest_mat.as_mut(),
            transformation.as_ref(),
            opencv::core::Size::new(output_size.0 as i32, output_size.1 as i32),
            interpolation as i32,
            border_type as i32,
            opencv::core::Scalar::from(border_value),
        )
        .change_context(NdCvError)?;

        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvWarpAffine<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn warp_affine(
        &self,
        transformation: &ndarray::Array2<f32>,
        output_size: (usize, usize),
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: (f64, f64, f64, f64),
    ) -> error_stack::Result<ndarray::Array<T, ndarray::Ix3>, NdCvError> {
        let mat = self.as_image_mat().change_context(NdCvError)?;
        let transformation = transformation.as_image_mat().change_context(NdCvError)?;
        let mut dest = ndarray::Array3::zeros((output_size.0, output_size.1, self.channels()));
        let mut dest_mat = dest.as_image_mat_mut().change_context(NdCvError)?;

        opencv::imgproc::warp_affine(
            mat.as_ref(),
            dest_mat.as_mut(),
            transformation.as_ref(),
            opencv::core::Size::new(output_size.0 as i32, output_size.1 as i32),
            interpolation as i32,
            border_type as i32,
            opencv::core::Scalar::from(border_value),
        )
        .change_context(NdCvError)?;

        Ok(dest)
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum EstimateAffineMethod {
    Lmeds = opencv::calib3d::LMEDS,
}

pub struct EstimateAffineResult<T, D> {
    pub inliers: ndarray::Array<T, D>,
    pub transformation: ndarray::Array<T, D>,
}

pub trait NdCvEstimateAffinePartial2D<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn estimate_affine_partial_2d(
        &self,
        reference: ndarray::Array<T, D>,
        method: EstimateAffineMethod,
        ransac_reproj_threshold: f64,
        max_iters: usize,
        confidence: f64,
        refine_iters: usize,
    ) -> Result<EstimateAffineResult<T, D>, NdCvError>;
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>>
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
    ) -> Result<EstimateAffineResult<T, ndarray::Ix2>, NdCvError> {
        let input_mat = self.as_image_mat().change_context(NdCvError)?;
        let reference_mat = reference.as_image_mat().change_context(NdCvError)?;

        let mut inliers = ndarray::Array2::<T>::zeros(reference.dim());

        let transformation_mat = opencv::calib3d::estimate_affine_partial_2d(
            input_mat.as_ref(),
            reference_mat.as_ref(),
            inliers
                .as_image_mat_mut()
                .change_context(NdCvError)?
                .as_mut(),
            method as i32,
            ransac_reproj_threshold,
            max_iters,
            confidence,
            refine_iters,
        )
        .change_context(NdCvError)?
        .as_ndarray()
        .change_context(NdCvError)?
        .to_owned();

        Ok(EstimateAffineResult {
            inliers,
            transformation: transformation_mat,
        })
    }
}
