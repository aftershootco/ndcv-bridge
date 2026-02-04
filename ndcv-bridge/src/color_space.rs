//! Colorspace conversion functions
//! ## Example
//! ```rust
//! use ndarray::Array3;
//! use ndcv_bridge::color_space::{Rgb, Rgba, Lab, ConvertColor};
//!
//! let arr = Array3::<u8>::ones((100, 100, 4));
//! let out = arr.cvt::<Rgba<u8>, Rgb<u8>>();
//!
//! let arr = Array3::<u8>::ones((100, 100, 3));
//! let out = arr.cvt::<Rgb<u8>, Lab<i8>>();
//! ```
use crate::{NdAsImage, NdAsImageMut, conversions::ConversionError};
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum ColorConversionError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error during color conversion: {0}")]
    OpencvError(#[from] opencv::Error),
    #[error(
        "Channel mismatch:
        expected: {expected} channels in the array ({src_type}), found: {got}, (shape: {size:?})"
    )]
    ChannelMismatch {
        expected: usize,
        src_type: &'static str,
        got: usize,
        size: Vec<usize>,
    },
}

pub trait ColorSpace<Elem> {
    type Dim: ndarray::Dimension;
    const CHANNELS: usize;
}

mod seal {
    pub trait Sealed: bytemuck::Pod {}
    impl Sealed for u8 {} // 0 to 255
    impl Sealed for i8 {} // -128 to 128
    impl Sealed for u16 {} // 0 to 65535
    impl Sealed for f32 {} // 0 to 1
}

macro_rules! define_color_space {
    ($name:ident, $channels:expr, $depth:ty) => {
        pub struct $name<T: seal::Sealed> {
            __phantom: core::marker::PhantomData<T>,
        }
        impl<T: seal::Sealed> ColorSpace<T> for $name<T> {
            type Dim = $depth;
            const CHANNELS: usize = $channels;
        }
    };
}

define_color_space!(Rgb, 3, Ix3);
define_color_space!(Bgr, 3, Ix3);
define_color_space!(Rgba, 4, Ix3);
define_color_space!(Lab, 3, Ix3);

pub trait ToColorSpace<SrcT, DstT, Dst>: ColorSpace<SrcT>
where
    SrcT: seal::Sealed,
    Dst: ColorSpace<DstT>,
{
    fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes;
}

macro_rules! impl_color_converter {
    ($src:tt, $dst:tt, $code:expr) => {
        impl<T: seal::Sealed> ToColorSpace<T, T, $dst<T>> for $src<T> {
            fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
                $code
            }
        }
    };
}

impl_color_converter!(
    Rgb,
    Bgr,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2RGB
);
impl_color_converter!(
    Bgr,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2RGB
);
impl_color_converter!(
    Rgba,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_BGRA2BGR
);
impl_color_converter!(
    Rgb,
    Rgba,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2BGRA
);

impl ToColorSpace<u8, i8, Lab<i8>> for Rgb<u8> {
    fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
        opencv::imgproc::ColorConversionCodes::COLOR_RGB2Lab
    }
}

// impl ToColorSpace<u8, u8, Rgb<u8>> for Lab<u8> {
//     fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
//         opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB
//     }
// }

impl_color_converter!(
    Lab,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB
);

pub trait ConvertColor<T, T2, S>
where
    T: seal::Sealed,
    T2: seal::Sealed,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(
        &self,
    ) -> Result<ArrayBase<CowRepr<'_, T2>, Dst::Dim>, ColorConversionError>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>;
    fn cvt<Src, Dst>(&self) -> ArrayBase<CowRepr<'_, T2>, Dst::Dim>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>,
    {
        self.try_cvt::<Src, Dst>().expect("Color conversion failed")
    }
}

impl<T, S, T2> ConvertColor<T, T2, S> for ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>
where
    T: seal::Sealed + num::Zero,
    T2: seal::Sealed + num::Zero,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(
        &self,
    ) -> Result<ArrayBase<CowRepr<'_, T2>, Dst::Dim>, ColorConversionError>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>,
        ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>: NdAsImage<T, <Rgb<T> as ColorSpace<T>>::Dim>,
    {
        let size = self.shape();
        let mut new_size = Dst::Dim::zeros(Dst::Dim::NDIM.unwrap_or(self.ndim()));
        let src_channels = size[size.len() - 1];
        if src_channels != Src::CHANNELS {
            return Err(ColorConversionError::ChannelMismatch {
                expected: Src::CHANNELS,
                src_type: std::any::type_name::<Src>().rsplit_once("::").unwrap().1,
                got: src_channels,
                size: size.to_vec(),
            });
        }
        size.iter()
            .cloned()
            .take(size.len() - 1)
            .chain(std::iter::once(Dst::CHANNELS))
            .enumerate()
            .for_each(|(idx, val)| {
                new_size[idx] = val;
            });
        let mut dst_ndarray = ArrayBase::<ndarray::OwnedRepr<T2>, Dst::Dim>::zeros(new_size);
        let mat = self.as_image_mat()?;
        let mut dst_mat = dst_ndarray.as_image_mat_mut()?;
        opencv::imgproc::cvt_color(
            &*mat,
            &mut *dst_mat,
            <Src as ToColorSpace<T, T2, Dst>>::cv_colorspace_code().into(),
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(dst_ndarray.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, CowArray};

    #[test]
    fn test_usage() {
        let arr = Array3::<u8>::ones((100, 100, 3));
        let out: CowArray<i8, Ix3> = arr.cvt::<Rgb<u8>, Lab<i8>>();
        let expected = Array3::<i8>::zeros((100, 100, 3));
        assert_eq!(out, expected);
    }

    #[test]
    fn test_rgb_to_bgr_conversion() {
        let rgb_data = Array3::<u8>::from_shape_fn((10, 10, 3), |(_y, _x, c)| match c {
            0 => 255, // Red
            1 => 128, // Green
            2 => 64,  // Blue
            _ => 0,
        });

        let bgr_result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Bgr<u8>>();

        assert_eq!(bgr_result.shape(), [10, 10, 3]);
        // Verify channel swapping: RGB -> BGR
        assert_eq!(bgr_result[[5, 5, 0]], 64); // Blue becomes first
        assert_eq!(bgr_result[[5, 5, 1]], 128); // Green stays middle
        assert_eq!(bgr_result[[5, 5, 2]], 255); // Red becomes last
    }

    #[test]
    fn test_bgr_to_rgb_conversion() {
        let bgr_data = Array3::<u8>::from_shape_fn((10, 10, 3), |(_y, _x, c)| match c {
            0 => 64,  // Blue
            1 => 128, // Green
            2 => 255, // Red
            _ => 0,
        });

        let rgb_result: CowArray<u8, Ix3> = bgr_data.cvt::<Bgr<u8>, Rgb<u8>>();

        assert_eq!(rgb_result.shape(), [10, 10, 3]);
        assert_eq!(rgb_result[[5, 5, 0]], 255); // Red becomes first
        assert_eq!(rgb_result[[5, 5, 1]], 128); // Green stays middle
        assert_eq!(rgb_result[[5, 5, 2]], 64); // Blue becomes last
    }

    #[test]
    fn test_rgb_to_rgba_conversion() {
        let rgb_data = Array3::<u8>::from_shape_fn((5, 5, 3), |(_, _, c)| match c {
            0 => 100,
            1 => 150,
            2 => 200,
            _ => 0,
        });

        let rgba_result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Rgba<u8>>();

        assert_eq!(rgba_result.shape(), [5, 5, 4]);
        assert_eq!(rgba_result[[2, 2, 0]], 100);
        assert_eq!(rgba_result[[2, 2, 1]], 150);
        assert_eq!(rgba_result[[2, 2, 2]], 200);
        assert_eq!(rgba_result[[2, 2, 3]], 255); // Alpha should be 255 (opaque)
    }

    #[test]
    fn test_rgba_to_rgb_conversion() {
        let rgba_data = Array3::<u8>::from_shape_fn((5, 5, 4), |(_, _, c)| match c {
            0 => 100,
            1 => 150,
            2 => 200,
            3 => 128, // Alpha
            _ => 0,
        });

        let rgb_result: CowArray<u8, Ix3> = rgba_data.cvt::<Rgba<u8>, Rgb<u8>>();

        assert_eq!(rgb_result.shape(), [5, 5, 3]);
        assert_eq!(rgb_result[[2, 2, 0]], 100);
        assert_eq!(rgb_result[[2, 2, 1]], 150);
        assert_eq!(rgb_result[[2, 2, 2]], 200);
    }

    #[test]
    fn test_rgb_to_lab_conversion() {
        let rgb_data = Array3::<u8>::from_shape_fn((8, 8, 3), |(_, _, c)| match c {
            0 => 255, // Pure red
            1 => 0,
            2 => 0,
            _ => 0,
        });

        let lab_result: CowArray<i8, Ix3> = rgb_data.cvt::<Rgb<u8>, Lab<i8>>();

        assert_eq!(lab_result.shape(), [8, 8, 3]);
        // Lab conversion with u8->i8 is supported but values may not be intuitive
        // Just verify the conversion completes and maintains correct shape
        assert_eq!(lab_result.ndim(), 3);
    }

    // #[test]
    // fn test_lab_color_space_definition() {
    //     // Test that Lab color space is properly defined
    //     assert_eq!(Lab::<i8>::CHANNELS, 3);
    //
    //     // Verify Lab to RGB conversion is available (but we can't test the full conversion
    //     // due to OpenCV limitations with i8 depth)
    //     let _code = <Lab<i8> as ToColorSpace<i8, u8, Rgb<u8>>>::cv_colorspace_code();
    // }

    #[test]
    fn test_different_data_types() {
        // Test with u16
        let rgb_u16 = Array3::<u16>::from_shape_fn((4, 4, 3), |(_, _, c)| match c {
            0 => 65535, // Max u16
            1 => 32768,
            2 => 0,
            _ => 0,
        });

        let bgr_u16: CowArray<u16, Ix3> = rgb_u16.cvt::<Rgb<u16>, Bgr<u16>>();
        assert_eq!(bgr_u16.shape(), [4, 4, 3]);
        assert_eq!(bgr_u16[[2, 2, 2]], 65535); // Red moved to last channel

        // Test with f32
        let rgb_f32 = Array3::<f32>::from_shape_fn((4, 4, 3), |(_, _, c)| match c {
            0 => 1.0,
            1 => 0.5,
            2 => 0.0,
            _ => 0.0,
        });

        let bgr_f32: CowArray<f32, Ix3> = rgb_f32.cvt::<Rgb<f32>, Bgr<f32>>();
        assert_eq!(bgr_f32.shape(), [4, 4, 3]);
        assert!((bgr_f32[[2, 2, 2]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        // Test with minimum size
        let min_rgb = Array3::<u8>::from_shape_fn((1, 1, 3), |(_, _, c)| (c * 85) as u8);
        let bgr_result: CowArray<u8, Ix3> = min_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [1, 1, 3]);

        // Test with maximum reasonable size
        let large_rgb = Array3::<u8>::zeros((100, 100, 3));
        let bgr_result: CowArray<u8, Ix3> = large_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [100, 100, 3]);

        // Test with all zeros
        let zero_rgb = Array3::<u8>::zeros((10, 10, 3));
        let bgr_result: CowArray<u8, Ix3> = zero_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert!(bgr_result.iter().all(|&v| v == 0));

        // Test with all maximum values
        let max_rgb = Array3::<u8>::from_elem((10, 10, 3), 255);
        let bgr_result: CowArray<u8, Ix3> = max_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert!(bgr_result.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_round_trip_conversions() {
        // RGB -> BGR -> RGB should return to original
        let original_rgb = Array3::<u8>::from_shape_fn((6, 6, 3), |(_, _, c)| match c {
            0 => 12,
            1 => 34,
            2 => 56,
            _ => 0,
        });

        let bgr_intermediate: CowArray<u8, Ix3> = original_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        let final_rgb: CowArray<u8, Ix3> = bgr_intermediate.cvt::<Bgr<u8>, Rgb<u8>>();

        assert_eq!(original_rgb.shape(), final_rgb.shape());
        assert_eq!(original_rgb, final_rgb.into_owned());

        // RGBA -> RGB -> RGBA round trip
        let original_rgba = Array3::<u8>::from_shape_fn((4, 4, 4), |(_, _, c)| match c {
            0 => 10,
            1 => 20,
            2 => 30,
            3 => 255,
            _ => 0,
        });

        let rgb_intermediate: CowArray<u8, Ix3> = original_rgba.cvt::<Rgba<u8>, Rgb<u8>>();
        let final_rgba: CowArray<u8, Ix3> = rgb_intermediate.cvt::<Rgb<u8>, Rgba<u8>>();

        assert_eq!(original_rgba.shape(), final_rgba.shape());
        // Alpha should be preserved
        assert_eq!(final_rgba[[2, 2, 3]], 255);
    }

    #[test]
    fn test_color_space_constants() {
        assert_eq!(Rgb::<u8>::CHANNELS, 3);
        assert_eq!(Bgr::<u8>::CHANNELS, 3);
        assert_eq!(Rgba::<u8>::CHANNELS, 4);
        assert_eq!(Lab::<i8>::CHANNELS, 3);
    }

    #[test]
    fn test_output_dimensions_preserved() {
        // Test that spatial dimensions are preserved during conversion
        let rgb_data = Array3::<u8>::zeros((15, 20, 3));

        let bgr_result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [15, 20, 3]);

        let rgba_result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Rgba<u8>>();
        assert_eq!(rgba_result.shape(), [15, 20, 4]);

        let lab_result: CowArray<i8, Ix3> = rgb_data.cvt::<Rgb<u8>, Lab<i8>>();
        assert_eq!(lab_result.shape(), [15, 20, 3]);
    }

    #[test]
    fn test_try_cvt_error_handling() {
        let rgb_data = Array3::<u8>::zeros((5, 5, 3));

        // This should work
        let result = rgb_data.try_cvt::<Rgb<u8>, Bgr<u8>>();
        assert!(result.is_ok());

        // Test cvt() method which calls try_cvt internally
        let _result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Bgr<u8>>();
    }
}
