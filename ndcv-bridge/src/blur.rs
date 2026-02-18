//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.blur.html>
use crate::conversions::*;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum BlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

mod seal {
    pub trait Sealed {}
    // src: input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

pub trait NdCvBlur<T: bytemuck::Pod + seal::Sealed, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn blur(
        &self,
        kernel_size: (u16, u16),
        anchor: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<ndarray::Array<T, D>, BlurError>;
    fn blur_def(&self, kernel_size: (u16, u16)) -> Result<ndarray::Array<T, D>, BlurError> {
        self.blur(
            kernel_size,
            (-1, -1),
            crate::gaussian::BorderType::BorderConstant,
        )
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvBlur<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn blur(
        &self,
        kernel_size: (u16, u16),
        anchor: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<ndarray::Array<T, D>, BlurError> {
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::blur(
            &*cv_self,
            &mut *cv_dst,
            opencv::core::Size::new(kernel_size.0 as i32, kernel_size.1 as i32),
            opencv::core::Point::new(anchor.0, anchor.1),
            border_type as i32,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::BorderType;
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
