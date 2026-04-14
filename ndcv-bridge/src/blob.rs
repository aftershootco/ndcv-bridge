use crate::{MatAsNd, NdAsImage, prelude_::*};
use error_stack::ResultExt;
use opencv::core::{Scalar, Size_};

pub trait NdCvBlobFromImage<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: (u16, u16),
        mean: (f64, f64, f64, f64),
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<f32>, NdCvError>;
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvBlobFromImage<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn blob_from_image(
        &self,
        scalefactor: f64,
        size: (u16, u16),
        mean: (f64, f64, f64, f64),
        swap_rb: bool,
        crop: bool,
    ) -> Result<ndarray::Array4<f32>, NdCvError> {
        let dest = opencv::dnn::blob_from_image(
            self.as_image_mat().change_context(NdCvError)?.as_ref(),
            scalefactor,
            Size_ {
                width: size.0 as i32,
                height: size.1 as i32,
            },
            Scalar::from(mean),
            swap_rb,
            crop,
            opencv::core::CV_32F,
        )
        .change_context(NdCvError)?
        .as_ndarray()
        .change_context(NdCvError)?
        .to_owned();

        Ok(dest)
    }
}
