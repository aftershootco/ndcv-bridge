use crate::{NdAsImage, NdAsImageMut, conversions::ConversionError};
use glam::DVec2;
use ndarray::{Array2, ArrayBase, Data, Ix2};

#[derive(Debug, thiserror::Error)]
pub enum CannyError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvCanny: crate::image::NdImage + NdAsImage<u8, Ix2> {
    fn canny(
        &self,
        thresholds: impl Into<DVec2>,
        aperture_size: i32,
        l2_gradient: bool,
    ) -> Result<Array2<u8>, CannyError>;
}

impl<S: Data<Elem = u8>> NdCvCanny for ArrayBase<S, Ix2> {
    fn canny(
        &self,
        thresholds: impl Into<DVec2>,
        aperture_size: i32,
        l2_gradient: bool,
    ) -> Result<Array2<u8>, CannyError> {
        let thresholds = thresholds.into();
        let mut dst = Array2::zeros(self.dim());
        let src = self.as_image_mat()?;
        let mut dst_mat = dst.as_image_mat_mut()?;
        opencv::imgproc::canny(
            &*src,
            &mut *dst_mat,
            thresholds.x,
            thresholds.y,
            aperture_size,
            l2_gradient,
        )?;
        Ok(dst)
    }
}
