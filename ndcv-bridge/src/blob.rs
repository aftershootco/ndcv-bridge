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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4, s};

    #[test]
    fn test_blob_from_image_u8_nchw_shape() {
        let arr = Array3::<u8>::ones((8, 10, 3));
        let blob: Array4<u8> = arr
            .blob_from_image(1.0, Vector2::new(10, 8), Vector4::zeros(), false, false)
            .unwrap();
        assert_eq!(blob.shape(), &[1, 3, 8, 10]);
        assert!(blob.iter().all(|&v| v == 1));
    }

    #[test]
    fn test_blob_from_image_f32_scalefactor() {
        let arr = Array3::<u8>::from_elem((8, 10, 3), 200);
        let blob: Array4<f32> = arr
            .blob_from_image(0.5, Vector2::new(10, 8), Vector4::zeros(), false, false)
            .unwrap();
        assert_eq!(blob.shape(), &[1, 3, 8, 10]);
        assert!(blob.iter().all(|&v| (v - 100.0).abs() < 1e-6));
    }

    #[test]
    fn test_blob_from_image_f32_input() {
        let arr = Array3::<f32>::from_elem((8, 10, 3), 0.75);
        let blob: Array4<f32> = arr
            .blob_from_image(
                2.0,
                Vector2::new(10, 8),
                Vector4::repeat(0.25),
                false,
                false,
            )
            .unwrap();
        assert_eq!(blob.shape(), &[1, 3, 8, 10]);
        // mean is subtracted before scaling: (0.75 - 0.25) * 2 = 1
        assert!(blob.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_blob_from_image_f32_input_u8_output() {
        let mut arr = Array3::<f32>::zeros((8, 10, 3));
        arr.slice_mut(s![.., .., 0]).fill(200.0);
        arr.slice_mut(s![.., .., 1]).fill(100.0);
        arr.slice_mut(s![.., .., 2]).fill(5.0);
        let blob: Array4<u8> = arr
            .blob_from_image(1.0, Vector2::new(10, 8), Vector4::zeros(), false, false)
            .unwrap();
        assert_eq!(blob.shape(), &[1, 3, 8, 10]);
        assert_eq!(blob[[0, 0, 0, 0]], 200);
        assert_eq!(blob[[0, 1, 0, 0]], 100);
        assert_eq!(blob[[0, 2, 0, 0]], 5);
    }

    #[test]
    fn test_blob_from_image_swap_rb() {
        let mut arr = Array3::<u8>::zeros((8, 10, 3));
        arr.slice_mut(s![.., .., 0]).fill(10);
        arr.slice_mut(s![.., .., 1]).fill(20);
        arr.slice_mut(s![.., .., 2]).fill(30);
        let blob: Array4<u8> = arr
            .blob_from_image(1.0, Vector2::new(10, 8), Vector4::zeros(), true, false)
            .unwrap();
        assert_eq!(blob[[0, 0, 0, 0]], 30);
        assert_eq!(blob[[0, 1, 0, 0]], 20);
        assert_eq!(blob[[0, 2, 0, 0]], 10);
    }
}
