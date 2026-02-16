//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.blur.html>
use crate::conversions::*;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum BoxBlurError {
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

pub trait NdCvBoxBlur<T: bytemuck::Pod + seal::Sealed, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    fn box_blur(
        &self,
        kernel_size: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<ndarray::Array<T, D>, BoxBlurError>;
    fn box_blur_def(
        &self,
        kernel_size: (i32, i32),
    ) -> Result<ndarray::Array<T, D>, BoxBlurError> {
        self.box_blur(kernel_size, crate::gaussian::BorderType::BorderConstant)
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvBoxBlur<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn box_blur(
        &self,
        kernel_size: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<ndarray::Array<T, D>, BoxBlurError> {
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::blur(
            &*cv_self,
            &mut *cv_dst,
            opencv::core::Size::new(kernel_size.0, kernel_size.1),
            opencv::core::Point::new(-1, -1),
            border_type as i32,
        )?;
        Ok(dst)
    }
}

pub trait NdCvBoxBlurInPlace<T: bytemuck::Pod + seal::Sealed, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>
{
    fn box_blur_inplace(
        &mut self,
        kernel_size: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<&mut Self, BoxBlurError>;
    fn box_blur_def_inplace(
        &mut self,
        kernel_size: (i32, i32),
    ) -> Result<&mut Self, BoxBlurError> {
        self.box_blur_inplace(kernel_size, crate::gaussian::BorderType::BorderConstant)
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed,
    S: ndarray::RawData + ndarray::DataMut<Elem = T>,
    D: ndarray::Dimension,
> NdCvBoxBlurInPlace<T, D> for ArrayBase<S, D>
where
    Self: crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>,
{
    fn box_blur_inplace(
        &mut self,
        kernel_size: (i32, i32),
        border_type: crate::gaussian::BorderType,
    ) -> Result<&mut Self, BoxBlurError> {
        let mut cv_self = self.as_image_mat_mut()?;

        unsafe {
            crate::inplace::op_inplace(&mut cv_self, |this, out| {
                opencv::imgproc::blur(
                    this,
                    out,
                    opencv::core::Size::new(kernel_size.0, kernel_size.1),
                    opencv::core::Point::new(-1, -1),
                    border_type as i32,
                )
            })
        }?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::BorderType;
    use ndarray::Array3;

    #[test]
    fn test_box_blur_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr.box_blur((3, 3), BorderType::BorderConstant).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_box_blur_edge_smoothing() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![..5, .., ..]).fill(255);

        let res = arr.box_blur((3, 3), BorderType::BorderConstant).unwrap();

        let middle_row = res.slice(s![4..6, 5, 0]);
        assert!(middle_row.iter().all(|&x| x > 0 && x < 255));
    }

    #[test]
    fn test_box_blur_different_kernel_sizes() {
        let arr = Array3::<u8>::ones((20, 20, 3));

        let kernel_sizes = [(3, 3), (5, 5), (7, 7)];
        for &kernel_size in &kernel_sizes {
            let res = arr.box_blur(kernel_size, BorderType::BorderConstant).unwrap();
            assert_eq!(res.shape(), &[20, 20, 3]);
        }
    }

    #[test]
    fn test_box_blur_different_border_types() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![4..7, 4..7, ..]).fill(255);

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for border_type in border_types {
            let _res = arr.box_blur((3, 3), border_type).unwrap();
            let res = arr.box_blur_inplace((3, 3), border_type).unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_box_blur_different_types() {
        let arr_u8 = Array3::<u8>::ones((10, 10, 3));
        let arr_f32 = Array3::<f32>::ones((10, 10, 3));

        let res_u8 = arr_u8.box_blur((3, 3), BorderType::BorderConstant).unwrap();
        let res_f32 = arr_f32.box_blur((3, 3), BorderType::BorderConstant).unwrap();

        assert_eq!(res_u8.shape(), &[10, 10, 3]);
        assert_eq!(res_f32.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_box_blur_def() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr.box_blur_def((3, 3)).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }
}
