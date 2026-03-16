//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.blur.html>
use crate::{conversions::*, types::CvType};
use glam::{IVec2, U16Vec2};
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum BlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}
pub trait BlurAllowedDepth {
    crate::seal!();
}
crate::seal!(impl, BlurAllowedDepth, u8, u16, i16, f32, f64);

pub trait NdCvBlur<T, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
where
    T: bytemuck::Pod + CvType,
    <T as CvType>::Depth: BlurAllowedDepth,
    D: ndarray::Dimension,
{
    fn blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        anchor: impl Into<IVec2>,
        border_type: crate::BorderType,
    ) -> Result<ndarray::Array<T, D>, BlurError>;
    fn blur_def(&self, kernel_size: impl Into<U16Vec2>) -> Result<ndarray::Array<T, D>, BlurError> {
        self.blur(kernel_size, (-1, -1), crate::BorderType::BorderConstant)
    }
}

impl<T, S, D> NdCvBlur<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
    T: bytemuck::Pod + num::Zero + CvType,
    <T as CvType>::Depth: BlurAllowedDepth,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    fn blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        anchor: impl Into<IVec2>,
        border_type: crate::BorderType,
    ) -> Result<ndarray::Array<T, D>, BlurError> {
        let kernel_size = kernel_size.into();
        let anchor = anchor.into();
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::blur(
            &*cv_self,
            &mut *cv_dst,
            opencv::core::Size::new(kernel_size.x.into(), kernel_size.y.into()),
            opencv::core::Point::new(anchor.x, anchor.y),
            border_type as i32,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BorderType;
    use ndarray::Array3;

    #[test]
    fn test_blur_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr
            .blur((3, 3), (-1, -1), BorderType::BorderConstant)
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_blur_edge_smoothing() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![..5, .., ..]).fill(255);

        let res = arr
            .blur((3, 3), (-1, -1), BorderType::BorderConstant)
            .unwrap();

        let middle_row = res.slice(s![4..6, 5, 0]);
        assert!(middle_row.iter().all(|&x| x > 0 && x < 255));
    }

    #[test]
    fn test_blur_different_kernel_sizes() {
        let arr = Array3::<u8>::ones((20, 20, 3));

        let kernel_sizes = [(3, 3), (5, 5), (7, 7)];
        for &kernel_size in &kernel_sizes {
            let res = arr
                .blur(kernel_size, (-1, -1), BorderType::BorderConstant)
                .unwrap();
            assert_eq!(res.shape(), &[20, 20, 3]);
        }
    }

    #[test]
    fn test_blur_different_border_types() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![4..7, 4..7, ..]).fill(255);

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for border_type in border_types {
            let res = arr.blur((3, 3), (-1, -1), border_type).unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_blur_different_types() {
        let arr_u8 = Array3::<u8>::ones((10, 10, 3));
        let arr_f32 = Array3::<f32>::ones((10, 10, 3));

        let res_u8 = arr_u8
            .blur((3, 3), (-1, -1), BorderType::BorderConstant)
            .unwrap();
        let res_f32 = arr_f32
            .blur((3, 3), (-1, -1), BorderType::BorderConstant)
            .unwrap();

        assert_eq!(res_u8.shape(), &[10, 10, 3]);
        assert_eq!(res_f32.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_blur_custom_anchor() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr
            .blur((3, 3), (0, 0), BorderType::BorderConstant)
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_blur_def() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr.blur_def((3, 3)).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }
}
