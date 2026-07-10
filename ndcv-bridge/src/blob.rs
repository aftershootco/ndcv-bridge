use crate::{MatAsNd, NdAsImage};
use nalgebra::{Vector2, Vector4};
use opencv::core::Size_;

#[derive(Debug, thiserror::Error)]
pub enum BlobError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvBlobFromImage<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: Vector2<usize>,
        mean: Vector4<f64>,
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<f32>, BlobError>;
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvBlobFromImage<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: Vector2<usize>,
        mean: Vector4<f64>,
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<f32>, BlobError> {
        let dest = opencv::dnn::blob_from_image(
            self.as_image_mat()?.as_ref(),
            scalefactor,
            Size_ {
                width: size.x as i32,
                height: size.y as i32,
            },
            opencv::core::VecN([mean.x, mean.y, mean.z, mean.w]),
            swap_rb,
            crop,
            opencv::core::CV_32F,
        )?
        .as_ndarray()?
        .to_owned();

        Ok(dest)
    }
}
