use error_stack::ResultExt;
use nalgebra::{Point2, Vector4};

use crate::{BorderType, NdAsImage, NdAsImageMut, NdCvError};

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum MorphType {
    Close = opencv::imgproc::MORPH_CLOSE,
    Open = opencv::imgproc::MORPH_OPEN,
}

pub trait NdCvMorphologyEx<T: bytemuck::Pod + num::Zero, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn morpohology_ex(
        &self,
        morph_type: MorphType,
        kernel: &ndarray::Array2<u8>,
        iterations: usize,
        anchor: Point2<i32>,
        border: BorderType,
        border_value: Vector4<f64>,
    ) -> error_stack::Result<ndarray::Array<T, D>, NdCvError>;

    fn morpohology_ex_def(
        &self,
        morph_type: MorphType,
        kernel: &ndarray::Array2<u8>,
    ) -> error_stack::Result<ndarray::Array<T, D>, NdCvError> {
        let bv = opencv::imgproc::morphology_default_border_value()
            .change_context(NdCvError)?
            .0;

        let border_value = bytemuck::cast::<[f64; 4], Vector4<f64>>(bv);

        self.morpohology_ex(
            morph_type,
            kernel,
            1,
            Point2::new(-1, -1),
            BorderType::BorderConstant,
            border_value,
        )
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvMorphologyEx<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn morpohology_ex(
        &self,
        morph_type: MorphType,
        kernel: &ndarray::Array2<u8>,
        iterations: usize,
        anchor: Point2<i32>,
        border_type: BorderType,
        border_value: Vector4<f64>,
    ) -> error_stack::Result<ndarray::Array<T, ndarray::Ix3>, NdCvError> {
        let img_mat = self.as_image_mat().change_context(NdCvError)?;
        let mut dst = ndarray::Array::zeros(self.dim());

        opencv::imgproc::morphology_ex(
            img_mat.as_ref(),
            dst.as_image_mat_mut().change_context(NdCvError)?.as_mut(),
            morph_type as i32,
            kernel.as_image_mat().change_context(NdCvError)?.as_ref(),
            opencv::core::Point::new(anchor.x, anchor.y),
            iterations as i32,
            border_type as i32,
            opencv::core::VecN([
                border_value.x,
                border_value.y,
                border_value.z,
                border_value.w,
            ]),
        )
        .change_context(NdCvError)?;

        Ok(dst)
    }
}
