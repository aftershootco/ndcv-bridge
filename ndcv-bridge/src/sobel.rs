use crate::{NdAsImage, NdAsImageMut, conversions::ConversionError, gaussian::BorderType};
use ndarray::{Array2, ArrayBase, Data, Ix2};

#[derive(Debug, thiserror::Error)]
pub enum SobelError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvSobel: NdAsImage<u8, Ix2> {
    fn sobel(
        &self,
        dx: i32,
        dy: i32,
        kernel_size: i32,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<Array2<f32>, SobelError>;
}

impl<S: Data<Elem = u8>> NdCvSobel for ArrayBase<S, Ix2> {
    fn sobel(
        &self,
        dx: i32,
        dy: i32,
        kernel_size: i32,
        scale: f64,
        delta: f64,
        border_type: BorderType,
    ) -> Result<Array2<f32>, SobelError> {
        let mut dst = Array2::<f32>::zeros(self.dim());
        let src = self.as_image_mat()?;
        let mut dst_mat = dst.as_image_mat_mut()?;
        opencv::imgproc::sobel(
            &*src,
            &mut *dst_mat,
            opencv::core::CV_32F,
            dx,
            dy,
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
        let gx = gray
            .sobel(1, 0, 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();
        let gy = gray
            .sobel(0, 1, 3, 1.0, 0.0, BorderType::BorderReflect101)
            .unwrap();

        assert_eq!(gx.dim(), gray.dim());
        assert_eq!(gx[[3, 4]], 8.0);
        assert_eq!(gx[[3, 0]], 0.0);
        assert_eq!(gy[[3, 4]], 0.0);
    }

    #[test]
    fn invalid_derivatives_return_error() {
        let gray = Array2::<u8>::zeros((4, 4));
        assert!(
            gray.sobel(0, 0, 3, 1.0, 0.0, BorderType::BorderReflect101)
                .is_err()
        );
    }
}
