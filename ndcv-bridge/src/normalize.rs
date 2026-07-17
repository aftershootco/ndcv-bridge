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
    fn normalize<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        mask: Option<ndarray::ArrayView2<u8>>,
    ) -> Result<ndarray::Array<U, D>, NormalizeError>;

    fn normalize_def(&self) -> Result<ndarray::Array<T, D>, NormalizeError>
    where
        T: num::Float,
    {
        self.normalize::<T>(-1., 1., NormType::MinMax, None)
    }
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvNormalize<T, ndarray::Ix3> for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn normalize<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        mask: Option<ndarray::ArrayView2<u8>>,
    ) -> Result<ndarray::Array<U, ndarray::Ix3>, NormalizeError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array3::zeros(self.dim());
        let mut dest_mat = dest.as_image_mat_mut()?;

        let dtype = if U::cv_depth() == T::cv_depth() {
            -1
        } else {
            U::cv_depth()
        };

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
    fn normalize<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        alpha: f64,
        beta: f64,
        norm_type: NormType,
        mask: Option<ndarray::ArrayView2<u8>>,
    ) -> Result<ndarray::Array<U, ndarray::Ix2>, NormalizeError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array2::zeros((self.shape()[0], self.shape()[1]));
        let mut dest_mat = dest.as_image_mat_mut()?;

        let dtype = if U::cv_depth() == T::cv_depth() {
            -1
        } else {
            U::cv_depth()
        };

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
        let res: Array2<u8> = arr.normalize(0., 255., NormType::MinMax, None).unwrap();
        assert_eq!(res[[0, 0]], 0);
        assert_eq!(res[[9, 9]], 255);
    }

    #[test]
    fn test_normalize_l2() {
        let arr = Array3::<f32>::ones((4, 4, 1));
        let res: Array3<f32> = arr.normalize(1., 0., NormType::L2, None).unwrap();
        // L2 norm of 16 ones is 4, so every element becomes 1/4
        assert!(res.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_def() {
        let mut arr = Array2::<f32>::zeros((10, 10));
        arr[[0, 0]] = 20.;
        arr[[5, 5]] = 10.;
        let res: Array2<f32> = arr.normalize_def().unwrap();
        // MinMax into [-1, 1]: max -> 1, midpoint -> 0, min -> -1
        assert!((res[[0, 0]] - 1.).abs() < 1e-6);
        assert!(res[[5, 5]].abs() < 1e-6);
        assert!((res[[1, 1]] + 1.).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_masked_minmax() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 50);
        arr[[0, 0]] = 10;
        arr[[9, 9]] = 90;
        arr[[5, 5]] = 255; // masked out; would dominate the max if the mask were ignored
        let mut mask = Array2::<u8>::from_elem((10, 10), 255);
        mask[[5, 5]] = 0;
        let res: Array2<u8> = arr
            .normalize(0., 255., NormType::MinMax, Some(mask.view()))
            .unwrap();
        assert_eq!(res[[0, 0]], 0);
        assert_eq!(res[[9, 9]], 255);
        // masked-out pixels are not written; dest stays zero-initialized there
        assert_eq!(res[[5, 5]], 0);
    }

    #[test]
    fn test_normalize_masked_u8_to_f32_output() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 50);
        arr[[0, 0]] = 10;
        arr[[9, 9]] = 90;
        arr[[5, 5]] = 255;
        let mut mask = Array2::<u8>::from_elem((10, 10), 255);
        mask[[5, 5]] = 0;
        let res: Array2<f32> = arr
            .normalize(0., 1., NormType::MinMax, Some(mask.view()))
            .unwrap();
        assert!(res[[0, 0]].abs() < 1e-6);
        assert!((res[[9, 9]] - 1.).abs() < 1e-6);
        // 50 -> (50 - 10) / 80 = 0.5 using the masked min/max
        assert!((res[[1, 1]] - 0.5).abs() < 1e-6);
        assert!(res[[5, 5]].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_masked_u16() {
        let mut arr = Array2::<u16>::from_elem((10, 10), 50);
        arr[[0, 0]] = 10;
        arr[[9, 9]] = 90;
        let mask = Array2::<u8>::ones((10, 10));
        let res: Array2<u16> = arr
            .normalize(0., 255., NormType::MinMax, Some(mask.view()))
            .unwrap();
        assert_eq!(res[[0, 0]], 0);
        assert_eq!(res[[9, 9]], 255);
    }

    #[test]
    fn test_normalize_masked_f32() {
        let arr = Array2::<f32>::ones((4, 4));
        let mask = Array2::<u8>::ones((4, 4));
        let res: Array2<f32> = arr
            .normalize(1., 0., NormType::L2, Some(mask.view()))
            .unwrap();
        // L2 norm of 16 ones is 4, so every element becomes 1/4
        assert!(res.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_u8_to_f32_output() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 50);
        arr[[0, 0]] = 10;
        arr[[9, 9]] = 90;
        let res: Array2<f32> = arr.normalize(0., 1., NormType::MinMax, None).unwrap();
        assert!(res[[0, 0]].abs() < 1e-6);
        assert!((res[[9, 9]] - 1.).abs() < 1e-6);
        // 50 -> (50 - 10) / 80 = 0.5
        assert!((res[[1, 1]] - 0.5).abs() < 1e-6);
    }

    // ---- Coverage added while reviewing PR #13 ----

    // Issue 2 was RAISED in review as "NORM_MINMAX errors on multi-channel images because
    // minMaxIdx asserts channels()==1". These two tests REFUTE that: on opencv 0.95 here,
    // min-max on a consolidated 3-channel Mat (via normalize_def's default path and an
    // explicit call) succeeds. Kept as regression coverage for the previously-untested
    // multi-channel path so a future opencv bump that breaks it is caught.
    #[test]
    fn test_normalize_def_multichannel() {
        let mut arr = Array3::<f32>::zeros((10, 10, 3));
        arr[[0, 0, 0]] = 20.;
        arr[[5, 5, 0]] = 10.;
        let res: Result<Array3<f32>, _> = arr.normalize_def();
        assert!(
            res.is_ok(),
            "normalize_def on a 3-channel image should succeed, but errored: {:?}",
            res.err()
        );
    }

    #[test]
    fn test_normalize_minmax_multichannel() {
        let arr = Array3::<u8>::from_elem((8, 8, 3), 50);
        let res: Result<Array3<u8>, _> = arr.normalize(0., 255., NormType::MinMax, None);
        assert!(
            res.is_ok(),
            "min-max normalize on a 3-channel image should succeed, but errored: {:?}",
            res.err()
        );
    }
}
