//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.gaussian_blur.html>
use crate::*;
use crate::{conversions::*, types::CvType};
use glam::{DVec2, U16Vec2};
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum GaussianBlurError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

/// Allowed type depth for GaussianBlur.
/// src: input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
/// Marker type to ensure only supported type depths are used
pub trait GaussianBlurAllowedDepth {
    crate::seal!();
}
crate::seal!(impl, GaussianBlurAllowedDepth, u8, u16, i16, f32, f64);

pub trait NdCvGaussianBlur<T, D>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, D>
where
    T: CvType,
    <T as CvType>::Depth: GaussianBlurAllowedDepth,
    D: ndarray::Dimension,
{
    fn gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError>;
    fn gaussian_blur_def(
        &self,
        kernel: impl Into<U16Vec2>,
        sigma: f64,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        self.gaussian_blur(kernel, (sigma, sigma), BorderType::BorderConstant)
    }
}

impl<T, S, D> NdCvGaussianBlur<T, D> for ArrayBase<S, D>
where
    T: CvType + num::Zero,
    <T as CvType>::Depth: GaussianBlurAllowedDepth,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
{
    fn gaussian_blur(
        &self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, GaussianBlurError> {
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        let k_size = kernel_size.into();
        let sigma = sigma.into();
        opencv::imgproc::gaussian_blur(
            &*cv_self,
            &mut *cv_dst,
            opencv::core::Size {
                width: k_size.x as i32,
                height: k_size.y as i32,
            },
            sigma.x,
            sigma.y,
            border_type as i32,
            OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(dst)
    }
}

/// For smaller values it is faster to use the allocated version
/// For example in a 4k f32 image this is about 50% faster than the allocated one
pub trait NdCvGaussianBlurInPlace<T, D>:
    crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>
where
    T: CvType,
    <T as CvType>::Depth: GaussianBlurAllowedDepth,
    D: ndarray::Dimension,
{
    fn gaussian_blur_inplace(
        &mut self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<&mut Self, GaussianBlurError>;
    fn gaussian_blur_def_inplace(
        &mut self,
        kernel: impl Into<U16Vec2>,
        sigma: f64,
    ) -> Result<&mut Self, GaussianBlurError> {
        self.gaussian_blur_inplace(kernel, (sigma, sigma), BorderType::BorderConstant)
    }
}

impl<T, S, D> NdCvGaussianBlurInPlace<T, D> for ArrayBase<S, D>
where
    Self: crate::image::NdImage + crate::conversions::NdAsImageMut<T, D>,
    T: CvType + num::Zero,
    <T as CvType>::Depth: GaussianBlurAllowedDepth,
    S: ndarray::RawData + ndarray::DataMut<Elem = T>,
    D: ndarray::Dimension,
{
    fn gaussian_blur_inplace(
        &mut self,
        kernel_size: impl Into<U16Vec2>,
        sigma: impl Into<DVec2>,
        border_type: BorderType,
    ) -> Result<&mut Self, GaussianBlurError> {
        let mut cv_self = self.as_image_mat_mut()?;
        let sigma = sigma.into();
        let kernel_size = kernel_size.into();

        unsafe {
            crate::inplace::op_inplace(&mut cv_self, |this, out| {
                opencv::imgproc::gaussian_blur(
                    this,
                    out,
                    opencv::core::Size {
                        width: kernel_size.x as i32,
                        height: kernel_size.y as i32,
                    },
                    sigma.x,
                    sigma.y,
                    border_type as i32,
                    OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
                )
            })
        }?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_gaussian_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let kernel_size = (3, 3);
        let sigma_x = 0.0;
        let sigma_y = 0.0;
        let border_type = BorderType::BorderConstant;
        let res = arr
            .gaussian_blur(kernel_size, (sigma_x, sigma_y), border_type)
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_gaussian_edge_preservation() {
        // Create an image with a sharp edge
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![..5, .., ..]).fill(255); // Top half white, bottom half black

        let res = arr
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();

        // Check that the middle row (edge) has intermediate values
        let middle_row = res.slice(s![4..6, 5, 0]);
        assert!(middle_row.iter().all(|&x| x > 0 && x < 255));
    }

    #[test]
    fn test_gaussian_different_kernel_sizes() {
        let arr = Array3::<u8>::ones((20, 20, 3));

        // Test different kernel sizes
        let kernel_sizes = [(3, 3), (5, 5), (7, 7)];
        for &kernel_size in &kernel_sizes {
            let res = arr
                .gaussian_blur(kernel_size, (1.0, 1.0), BorderType::BorderConstant)
                .unwrap();
            assert_eq!(res.shape(), &[20, 20, 3]);
        }
    }

    #[test]
    fn test_gaussian_different_border_types() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![4..7, 4..7, ..]).fill(255); // White square in center

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for border_type in border_types {
            let _res = arr.gaussian_blur((3, 3), (1.0, 1.0), border_type).unwrap();
            let res = arr
                .gaussian_blur_inplace((3, 3), (1.0, 1.0), border_type)
                .unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_gaussian_different_types() {
        // Test with different numeric types
        let arr_u8 = Array3::<u8>::ones((10, 10, 3));
        let arr_f32 = Array3::<f32>::ones((10, 10, 3));

        let res_u8 = arr_u8
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();
        let res_f32 = arr_f32
            .gaussian_blur((3, 3), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();

        assert_eq!(res_u8.shape(), &[10, 10, 3]);
        assert_eq!(res_f32.shape(), &[10, 10, 3]);
    }

    #[test]
    #[should_panic]
    fn test_gaussian_invalid_kernel_size() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        // Even kernel sizes should fail
        let _ = arr
            .gaussian_blur((2, 2), (1.0, 1.0), BorderType::BorderConstant)
            .unwrap();
    }
}
