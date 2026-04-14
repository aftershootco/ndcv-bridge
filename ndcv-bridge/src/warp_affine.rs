use crate::{BorderType, Interpolation, NdAsImage, NdAsImageMut, prelude_::*};
use error_stack::ResultExt;

pub trait NdCvWarpAffine<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn warp_affine(
        &self,
        transformation: ndarray::Array2<f32>,
        output_size: (usize, usize),
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: T,
    ) -> Result<ndarray::Array<T, D>, NdCvError>;
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvWarpAffine<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
where
    opencv::core::VecN<f64, 4>: From<T>,
{
    fn warp_affine(
        &self,
        transformation: ndarray::Array2<f32>,
        output_size: (usize, usize),
        interpolation: Interpolation,
        border_type: BorderType,
        border_value: T,
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
