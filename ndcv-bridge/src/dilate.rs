//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.dilate.html>
//! <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c>
use crate::conversions::*;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum DilateError {
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

pub trait NdCvDilate<T: bytemuck::Pod + seal::Sealed, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// Dilates an image using a structuring element with all parameters exposed.
    ///
    /// - `kernel`: 2-D structuring element
    /// - `anchor`: position of the anchor within the element; `(-1, -1)` means the center
    /// - `iterations`: number of times dilation is applied
    /// - `border_type`: pixel extrapolation method
    /// - `border_value`: border value used when `border_type` is `BorderConstant`, as `[v0, v1, v2, v3]`
    fn dilate(
        &self,
        kernel: ndarray::ArrayView2<u8>,
        anchor: (i32, i32),
        iterations: i32,
        border_type: crate::gaussian::BorderType,
        border_value: [f64; 4],
    ) -> Result<ndarray::Array<T, D>, DilateError>;

    /// Dilates an image with default parameters: anchor at center, `BORDER_CONSTANT` border,
    /// and the morphology default border value (`f64::MAX` for all channels).
    fn dilate_def(
        &self,
        kernel: ndarray::ArrayView2<u8>,
        iterations: i32,
    ) -> Result<ndarray::Array<T, D>, DilateError> {
        let border_value = opencv::imgproc::morphology_default_border_value()?.0;
        self.dilate(
            kernel,
            (-1, -1),
            iterations,
            crate::gaussian::BorderType::BorderConstant,
            border_value,
        )
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvDilate<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn dilate(
        &self,
        kernel: ndarray::ArrayView2<u8>,
        anchor: (i32, i32),
        iterations: i32,
        border_type: crate::gaussian::BorderType,
        border_value: [f64; 4],
    ) -> Result<ndarray::Array<T, D>, DilateError> {
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        let cv_kernel = kernel.as_image_mat()?;
        opencv::imgproc::dilate(
            &*cv_self,
            &mut *cv_dst,
            &*cv_kernel,
            opencv::core::Point::new(anchor.0, anchor.1),
            iterations,
            border_type as i32,
            opencv::core::VecN(border_value),
        )?;
        Ok(dst)
    }
}

/// In-place variant of dilation.
pub trait NdCvDilateInPlace<T: bytemuck::Pod + seal::Sealed, D: ndarray::Dimension>:
    crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>
{
    fn dilate_inplace(
        &mut self,
        kernel: ndarray::ArrayView2<u8>,
        anchor: (i32, i32),
        iterations: i32,
        border_type: crate::gaussian::BorderType,
        border_value: [f64; 4],
    ) -> Result<&mut Self, DilateError>;

    fn dilate_def_inplace(
        &mut self,
        kernel: ndarray::ArrayView2<u8>,
        iterations: i32,
    ) -> Result<&mut Self, DilateError> {
        let border_value = opencv::imgproc::morphology_default_border_value()?.0;
        self.dilate_inplace(
            kernel,
            (-1, -1),
            iterations,
            crate::gaussian::BorderType::BorderConstant,
            border_value,
        )
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed,
    S: ndarray::RawData + ndarray::DataMut<Elem = T>,
    D: ndarray::Dimension,
> NdCvDilateInPlace<T, D> for ArrayBase<S, D>
where
    Self: crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>,
{
    fn dilate_inplace(
        &mut self,
        kernel: ndarray::ArrayView2<u8>,
        anchor: (i32, i32),
        iterations: i32,
        border_type: crate::gaussian::BorderType,
        border_value: [f64; 4],
    ) -> Result<&mut Self, DilateError> {
        let cv_kernel = kernel.as_image_mat()?;
        let mut cv_self = self.as_image_mat_mut()?;
        unsafe {
            crate::inplace::op_inplace(&mut cv_self, |this, out| {
                opencv::imgproc::dilate(
                    this,
                    out,
                    &*cv_kernel,
                    opencv::core::Point::new(anchor.0, anchor.1),
                    iterations,
                    border_type as i32,
                    opencv::core::VecN(border_value),
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
    use ndarray::{Array2, Array3};

    fn rect_kernel(size: usize) -> Array2<u8> {
        Array2::ones((size, size))
    }

    #[test]
    fn test_dilate_def_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr.dilate_def(rect_kernel(3).view(), 1).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_dilate_full_params_cv2_default_border_value() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let border_value = opencv::imgproc::morphology_default_border_value()
            .unwrap()
            .0;
        let res = arr
            .dilate(
                rect_kernel(3).view(),
                (-1, -1),
                1,
                BorderType::BorderConstant,
                border_value,
            )
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_dilate_border_value_zero() {
        // [0.0; 4] pads with black — dilating a white image should stay fully white
        let arr = Array3::<u8>::from_elem((10, 10, 3), 255);
        let res = arr
            .dilate(
                rect_kernel(3).view(),
                (-1, -1),
                1,
                BorderType::BorderConstant,
                [0.0; 4],
            )
            .unwrap();
        assert!(res.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_dilate_border_value_max_u8() {
        // [255.0; 4] pads with white — corner pixels of a black image get pulled up to 255
        let arr = Array3::<u8>::zeros((10, 10, 3));
        let res = arr
            .dilate(
                rect_kernel(3).view(),
                (-1, -1),
                1,
                BorderType::BorderConstant,
                [255.0; 4],
            )
            .unwrap();
        // Border pixels see the 255 padding through the kernel, so they must be 255
        assert_eq!(res[[0, 0, 0]], 255);
        assert_eq!(res[[9, 9, 0]], 255);
        // Interior pixels far from any border remain 0
        assert_eq!(res[[5, 5, 0]], 0);
    }

    #[test]
    fn test_dilate_border_value_f32() {
        // Same semantics but with an f32 image
        let arr = Array3::<f32>::zeros((10, 10, 3));
        let res = arr
            .dilate(
                rect_kernel(3).view(),
                (-1, -1),
                1,
                BorderType::BorderConstant,
                [1.0; 4],
            )
            .unwrap();
        assert_eq!(res[[0, 0, 0]], 1.0);
        assert_eq!(res[[5, 5, 0]], 0.0);
    }

    #[test]
    fn test_dilate_border_value_per_channel() {
        // Each channel gets its own constant: [128.0, 64.0, 32.0, 0.0]
        let arr = Array3::<u8>::zeros((10, 10, 3));
        let res = arr
            .dilate(
                rect_kernel(3).view(),
                (-1, -1),
                1,
                BorderType::BorderConstant,
                [128.0, 64.0, 32.0, 0.0],
            )
            .unwrap();
        assert_eq!(res[[0, 0, 0]], 128);
        assert_eq!(res[[0, 0, 1]], 64);
        assert_eq!(res[[0, 0, 2]], 32);
    }

    #[test]
    fn test_dilate_expands_region() {
        let mut arr = Array3::<u8>::zeros((10, 10, 1));
        arr[[5, 5, 0]] = 255;
        let res = arr.dilate_def(rect_kernel(3).view(), 1).unwrap();
        // The single lit pixel should have expanded to its 3x3 neighbourhood
        assert!(res[[4, 4, 0]] > 0);
        assert!(res[[4, 5, 0]] > 0);
        assert!(res[[5, 4, 0]] > 0);
        assert!(res[[6, 6, 0]] > 0);
    }

    #[test]
    fn test_dilate_multiple_iterations() {
        let mut arr = Array3::<u8>::zeros((20, 20, 1));
        arr[[10, 10, 0]] = 255;
        let res1 = arr.dilate_def(rect_kernel(3).view(), 1).unwrap();
        let res2 = arr.dilate_def(rect_kernel(3).view(), 2).unwrap();
        // More iterations → more non-zero pixels
        let count1 = res1.iter().filter(|&&v| v > 0).count();
        let count2 = res2.iter().filter(|&&v| v > 0).count();
        assert!(count2 > count1);
    }

    #[test]
    fn test_dilate_different_border_types() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let border_value = opencv::imgproc::morphology_default_border_value()
            .unwrap()
            .0;
        for border_type in [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ] {
            let res = arr
                .dilate(
                    rect_kernel(3).view(),
                    (-1, -1),
                    1,
                    border_type,
                    border_value,
                )
                .unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_dilate_f32() {
        let arr = Array3::<f32>::ones((10, 10, 3));
        let res = arr.dilate_def(rect_kernel(3).view(), 1).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_dilate_inplace() {
        let mut arr = Array3::<u8>::ones((10, 10, 3));
        arr.dilate_def_inplace(rect_kernel(3).view(), 1).unwrap();
        assert_eq!(arr.shape(), &[10, 10, 3]);
    }
}
