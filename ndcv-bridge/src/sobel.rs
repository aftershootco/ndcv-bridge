use crate::{NdAsImage, NdAsImageMut, conversions::ConversionError, gaussian::BorderType};
use glam::IVec2;
use ndarray::{Array, Array2, Array3, ArrayBase, Data, Ix2, Ix3};

#[derive(Debug, thiserror::Error)]
pub enum SobelError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvSobel<T: bytemuck::Pod + num::Zero + crate::types::CvType, D: ndarray::Dimension>:
    crate::image::NdImage + NdAsImage<T, D>
{
    fn sobel<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        derivatives: impl Into<IVec2>,
        kernel_size: i32,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<Array<U, D>, SobelError>;
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: Data<Elem = T>> NdCvSobel<T, Ix2>
    for ArrayBase<S, Ix2>
{
    fn sobel<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        derivatives: impl Into<IVec2>,
        kernel_size: i32,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<Array2<U>, SobelError> {
        let derivatives = derivatives.into();
        let mut dst = Array2::zeros(self.dim());
        let src = self.as_image_mat()?;
        let mut dst_mat = dst.as_image_mat_mut()?;
        opencv::imgproc::sobel(
            &*src,
            &mut *dst_mat,
            U::cv_depth(),
            derivatives.x,
            derivatives.y,
            kernel_size,
            scale,
            delta,
            border_type as i32,
        )?;
        Ok(dst)
    }
}

impl<T: bytemuck::Pod + num::Zero + crate::types::CvType, S: Data<Elem = T>> NdCvSobel<T, Ix3>
    for ArrayBase<S, Ix3>
{
    fn sobel<U: bytemuck::Pod + num::Zero + crate::types::CvType>(
        &self,
        derivatives: impl Into<IVec2>,
        kernel_size: i32,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<Array3<U>, SobelError> {
        let derivatives = derivatives.into();
        let mut dst = Array3::zeros(self.dim());
        let src = self.as_image_mat()?;
        let mut dst_mat = dst.as_image_mat_mut()?;
        opencv::imgproc::sobel(
            &*src,
            &mut *dst_mat,
            U::cv_depth(),
            derivatives.x,
            derivatives.y,
            kernel_size,
            scale,
            delta,
            border_type as i32,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobel_gradients_on_horizontal_ramp() {
        let gray = Array2::from_shape_fn((7, 9), |(_, x)| x as u8);
        let gx: Array2<f32> = gray
            .sobel((1, 0), 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();
        let gy: Array2<f32> = gray
            .sobel((0, 1), 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();

        assert_eq!(gx.dim(), gray.dim());
        assert_eq!(gx[[3, 4]], 8.0);
        assert_eq!(gx[[3, 0]], 0.0);
        assert_eq!(gy[[3, 4]], 0.0);
    }

    #[test]
    fn sobel_preserves_negative_gradients_in_signed_output() {
        let gray = Array2::from_shape_fn((7, 9), |(_, x)| (8 - x) as u8);
        let gx: Array2<i16> = gray
            .sobel((1, 0), 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();

        assert_eq!(gx[[3, 4]], -8);
    }

    #[test]
    fn sobel_handles_multichannel_images_and_float_output() {
        let image = Array3::from_shape_fn((7, 9, 2), |(_, x, c)| (x * (c + 1)) as f32);
        let gx: Array3<f64> = image
            .sobel((1, 0), 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();

        assert_eq!(gx.dim(), image.dim());
        assert_eq!(gx[[3, 4, 0]], 8.0);
        assert_eq!(gx[[3, 4, 1]], 16.0);
    }

    #[test]
    fn invalid_derivatives_return_error() {
        let gray = Array2::<u8>::zeros((4, 4));
        let result: Result<Array2<f32>, _> =
            gray.sobel((0, 0), 3, 1.0, 0.0, BorderType::BorderReflect101);
        assert!(result.is_err());
    }
}
