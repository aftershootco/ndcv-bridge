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

pub trait NdCvBlobFromImage<
    T: bytemuck::Pod + num::Zero,
    D: ndarray::Dimension,
    U: bytemuck::Pod + seal::Sealed,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: Vector2<usize>,
        mean: Vector4<f64>,
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<U>, BlobError>;
}

mod seal {
    pub trait Sealed {
        fn dtype() -> i32;
    }

    impl Sealed for u8 {
        fn dtype() -> i32 {
            opencv::core::CV_8U
        }
    }

    impl Sealed for f32 {
        fn dtype() -> i32 {
            opencv::core::CV_32F
        }
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>, U: bytemuck::Pod + seal::Sealed>
    NdCvBlobFromImage<T, ndarray::Ix3, U> for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: Vector2<usize>,
        mean: Vector4<f64>,
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<U>, BlobError> {
        let dtype = U::dtype();

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
            dtype,
        )?
        .as_ndarray()?
        .to_owned();

        Ok(dest)
    }
}
