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

pub trait NdCvNormalize<T: bytemuck::Pod + num::Zero + crate::types::CvType, D: ndarray::Dimension>:
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

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvNormalize<T, ndarray::Ix3> for ndarray::ArrayBase<S, ndarray::Ix3>
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

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvNormalize<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_normalize_minmax_range() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 50);
        arr[[0, 0]] = 10;
        arr[[9, 9]] = 90;
        let res = arr
            .normalize(0., 255., NormType::MinMax, -1, &None)
            .unwrap();
        assert_eq!(res[[0, 0]], 0);
        assert_eq!(res[[9, 9]], 255);
    }

    #[test]
    fn test_normalize_l2() {
        let arr = Array3::<f32>::ones((4, 4, 1));
        let res = arr.normalize(1., 0., NormType::L2, -1, &None).unwrap();
        // L2 norm of 16 ones is 4, so every element becomes 1/4
        assert!(res.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_def() {
        let mut arr = Array2::<f32>::zeros((10, 10));
        arr[[0, 0]] = 20.;
        arr[[5, 5]] = 10.;
        let res = arr.normalize_def().unwrap();
        // MinMax into [-1, 1]: max -> 1, midpoint -> 0, min -> -1
        assert!((res[[0, 0]] - 1.).abs() < 1e-6);
        assert!(res[[5, 5]].abs() < 1e-6);
        assert!((res[[1, 1]] + 1.).abs() < 1e-6);
    }
}
