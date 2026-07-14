use nalgebra::Vector4;

use crate::{BorderType, NdAsImage, NdAsImageMut};

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum MorphType {
    Close = opencv::imgproc::MORPH_CLOSE,
    Open = opencv::imgproc::MORPH_OPEN,
}

#[derive(Debug, thiserror::Error)]
pub enum MorphError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

mod seal {
    pub trait Sealed {}
    // src: input image; the number of channels can be arbitrary, but the depth should be
    // CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub trait NdCvMorphologyEx<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn morpohology_ex(
        &self,
        morph_type: MorphType,
        kernel: ndarray::ArrayView2<u8>,
        iterations: usize,
        anchor: impl Into<glam::ISizeVec2>,
        border: BorderType,
        border_value: Vector4<f64>,
    ) -> Result<ndarray::Array<T, D>, MorphError>;

    fn morpohology_ex_def(
        &self,
        morph_type: MorphType,
        kernel: ndarray::ArrayView2<u8>,
    ) -> Result<ndarray::Array<T, D>, MorphError> {
        let bv = opencv::imgproc::morphology_default_border_value()?.0;

        let border_value = bytemuck::cast::<[f64; 4], Vector4<f64>>(bv);

        self.morpohology_ex(
            morph_type,
            kernel,
            1,
            glam::ISizeVec2::new(-1, -1),
            BorderType::BorderConstant,
            border_value,
        )
    }
}

impl<T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvMorphologyEx<T, ndarray::Ix3> for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn morpohology_ex(
        &self,
        morph_type: MorphType,
        kernel: ndarray::ArrayView2<u8>,
        iterations: usize,
        anchor: impl Into<glam::ISizeVec2>,
        border_type: BorderType,
        border_value: Vector4<f64>,
    ) -> Result<ndarray::Array<T, ndarray::Ix3>, MorphError> {
        let img_mat = self.as_image_mat()?;
        let mut dst = ndarray::Array::zeros(self.dim());
        let anchor = anchor.into();

        opencv::imgproc::morphology_ex(
            img_mat.as_ref(),
            dst.as_image_mat_mut()?.as_mut(),
            morph_type as i32,
            kernel.as_image_mat()?.as_ref(),
            opencv::core::Point::new(anchor.x as i32, anchor.y as i32),
            iterations as i32,
            border_type as i32,
            opencv::core::VecN([
                border_value.x,
                border_value.y,
                border_value.z,
                border_value.w,
            ]),
        )?;

        Ok(dst)
    }
}

impl<T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType, S: ndarray::Data<Elem = T>>
    NdCvMorphologyEx<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn morpohology_ex(
        &self,
        morph_type: MorphType,
        kernel: ndarray::ArrayView2<u8>,
        iterations: usize,
        anchor: impl Into<glam::ISizeVec2>,
        border_type: BorderType,
        border_value: Vector4<f64>,
    ) -> Result<ndarray::Array<T, ndarray::Ix2>, MorphError> {
        let img_mat = self.as_image_mat()?;
        let mut dst = ndarray::Array::zeros(self.dim());
        let anchor = anchor.into();

        opencv::imgproc::morphology_ex(
            img_mat.as_ref(),
            dst.as_image_mat_mut()?.as_mut(),
            morph_type as i32,
            kernel.as_image_mat()?.as_ref(),
            opencv::core::Point::new(anchor.x as i32, anchor.y as i32),
            iterations as i32,
            border_type as i32,
            opencv::core::VecN([
                border_value.x,
                border_value.y,
                border_value.z,
                border_value.w,
            ]),
        )?;

        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    fn rect_kernel(size: usize) -> Array2<u8> {
        Array2::ones((size, size))
    }

    #[test]
    fn test_morphology_open_removes_speck() {
        let mut arr = Array2::<u8>::zeros((10, 10));
        arr[[5, 5]] = 255;
        let res = arr
            .morpohology_ex_def(MorphType::Open, rect_kernel(3).view())
            .unwrap();
        assert!(res.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_morphology_close_fills_hole() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 255);
        arr[[5, 5]] = 0;
        let res = arr
            .morpohology_ex_def(MorphType::Close, rect_kernel(3).view())
            .unwrap();
        assert!(res.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_morphology_full_params_array3() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let bv = opencv::imgproc::morphology_default_border_value()
            .unwrap()
            .0;
        let border_value = bytemuck::cast::<[f64; 4], Vector4<f64>>(bv);
        let res = arr
            .morpohology_ex(
                MorphType::Open,
                rect_kernel(3).view(),
                1,
                glam::ISizeVec2::new(-1, -1),
                BorderType::BorderConstant,
                border_value,
            )
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
        assert!(res.iter().all(|&v| v == 1));
    }
}
