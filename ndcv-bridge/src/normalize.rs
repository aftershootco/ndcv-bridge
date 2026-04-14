use crate::{NdAsImage, NdAsImageMut, prelude_::*};
use error_stack::ResultExt;

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum NormType {
    Hamming = opencv::core::NORM_HAMMING,
    Hamming2 = opencv::core::NORM_HAMMING2,
    Inf = opencv::core::NORM_INF,
    L1 = opencv::core::NORM_L1,
    L2 = opencv::core::NORM_L2,
    L2SQR = opencv::core::NORM_L2SQR,
    MinMax = opencv::core::NORM_MINMAX,
    Relative = opencv::core::NORM_RELATIVE,
}

pub trait NdCvNormalize<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn normalize(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        dtype: i32,
        mask: &ndarray::Array2<T>,
    ) -> Result<ndarray::Array<T, D>, NdCvError>;

    fn normalize_def(&self) -> Result<ndarray::Array<T, D>, NdCvError> {
        self.normalize(
            -1.,
            1.,
            NormType::MinMax,
            -1,
            &ndarray::Array2::<T>::zeros((0, 0)),
        )
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvNormalize<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn normalize(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        dtype: i32,
        mask: &ndarray::Array2<T>,
    ) -> error_stack::Result<ndarray::Array<T, ndarray::Ix3>, NdCvError> {
        let mat = self.as_image_mat().change_context(NdCvError)?;
        let mask = mask
            .as_image_mat()
            .change_context(NdCvError)?
            .clone_pointee();
        let mut dest = ndarray::Array3::zeros((self.shape()[0], self.shape()[1], self.shape()[2]));
        let mut dest_mat = dest.as_image_mat_mut().change_context(NdCvError)?;
        opencv::core::normalize(
            mat.as_ref(),
            dest_mat.as_mut(),
            alpha,
            beta,
            norm_type as i32,
            dtype,
            &mask,
        )
        .change_context(NdCvError)?;

        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvNormalize<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn normalize(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        dtype: i32,
        mask: &ndarray::Array2<T>,
    ) -> error_stack::Result<ndarray::Array<T, ndarray::Ix2>, NdCvError> {
        let mat = self.as_image_mat().change_context(NdCvError)?;
        let mask = mask.as_image_mat().change_context(NdCvError)?;
        let mut dest = ndarray::Array2::zeros((self.shape()[0], self.shape()[1]));
        let mut dest_mat = dest.as_image_mat_mut().change_context(NdCvError)?;
        opencv::core::normalize(
            mat.as_ref(),
            dest_mat.as_mut(),
            alpha,
            beta,
            norm_type as i32,
            dtype,
            mask.as_ref(),
        )
        .change_context(NdCvError)?;

        Ok(dest)
    }
}
