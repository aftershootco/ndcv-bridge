use crate::{NdAsImage, NdAsImageMut};

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

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
        mask: &Option<ndarray::Array2<T>>,
    ) -> Result<ndarray::Array<T, D>, NormalizeError>;

    fn normalize_def(&self) -> Result<ndarray::Array<T, D>, NormalizeError> {
        self.normalize(-1., 1., NormType::MinMax, -1, &None)
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
        mask: &Option<ndarray::Array2<T>>,
    ) -> Result<ndarray::Array<T, ndarray::Ix3>, NormalizeError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array3::zeros(self.dim());
        let mut dest_mat = dest.as_image_mat_mut()?;

        match mask {
            Some(mask) => {
                let mask = mask.as_image_mat()?;

                opencv::core::normalize(
                    mat.as_ref(),
                    dest_mat.as_mut(),
                    alpha,
                    beta,
                    norm_type as i32,
                    dtype,
                    mask.as_ref(),
                )?;
            }

            None => {
                let mask = opencv::core::no_array();
                opencv::core::normalize(
                    mat.as_ref(),
                    dest_mat.as_mut(),
                    alpha,
                    beta,
                    norm_type as i32,
                    dtype,
                    &mask,
                )?;
            }
        };

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
        mask: &Option<ndarray::Array2<T>>,
    ) -> Result<ndarray::Array<T, ndarray::Ix2>, NormalizeError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array2::zeros((self.shape()[0], self.shape()[1]));
        let mut dest_mat = dest.as_image_mat_mut()?;

        match mask {
            Some(mask) => {
                let mask = mask.as_image_mat()?;

                opencv::core::normalize(
                    mat.as_ref(),
                    dest_mat.as_mut(),
                    alpha,
                    beta,
                    norm_type as i32,
                    dtype,
                    mask.as_ref(),
                )?;
            }

            None => {
                let mask = opencv::core::no_array();

                opencv::core::normalize(
                    mat.as_ref(),
                    dest_mat.as_mut(),
                    alpha,
                    beta,
                    norm_type as i32,
                    dtype,
                    &mask,
                )?;
            }
        };

        Ok(dest)
    }
}
