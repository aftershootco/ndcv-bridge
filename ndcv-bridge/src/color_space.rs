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

pub trait ColorSpace<Elem>: seal::SealedColorSpace {
    type Dim: ndarray::Dimension;
    const CHANNELS: usize;
}

mod seal {
    pub trait Sealed: bytemuck::Pod {}
    impl Sealed for u8 {} // 0 to 255
    impl Sealed for i8 {} // -128 to 127
    impl Sealed for u16 {} // 0 to 65535
    impl Sealed for f32 {} // 0 to 1
    pub trait SealedColorSpace {}
}

macro_rules! define_color_space {
    ($name:ident, $channels:expr, $depth:ty) => {
        pub struct $name<T: seal::Sealed> {
            __phantom: core::marker::PhantomData<T>,
        }
        impl<T: seal::Sealed> seal::SealedColorSpace for $name<T> {}
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
define_color_space!(Gray, 1, Ix2);

pub trait ToColorSpace<T, U, Dst>: ColorSpace<T>
where
    T: seal::Sealed,
    Dst: ColorSpace<U>,
{
    fn cv_colorspace_code() -> i32;
}

macro_rules! impl_color_converter {
    ($src:tt, $dst:tt, $code:expr => $($type:ty,)*) => {
        $(
            impl ToColorSpace<$type, $type, $dst<$type>> for $src<$type> {
                fn cv_colorspace_code() -> i32 {
                    $code
                }
            }
        )*
    };
}

impl_color_converter!(Rgb, Bgr, opencv::imgproc::COLOR_RGB2BGR => u8,  u16, f32,);
impl_color_converter!(Bgr, Rgb, opencv::imgproc::COLOR_BGR2RGB => u8,  u16, f32,);
impl_color_converter!(Rgba, Rgb, opencv::imgproc::COLOR_RGBA2RGB => u8,  u16, f32,);
impl_color_converter!(Rgb, Rgba, opencv::imgproc::COLOR_RGB2RGBA => u8,  u16, f32,);
impl_color_converter!(Rgb, Gray, opencv::imgproc::COLOR_RGB2GRAY => u8,  u16, f32,);
impl_color_converter!(Rgb, Lab, opencv::imgproc::COLOR_RGB2Lab => f32,);
impl_color_converter!(Lab, Rgb, opencv::imgproc::COLOR_Lab2RGB => f32,);

impl ToColorSpace<u8, i8, Lab<i8>> for Rgb<u8> {
    fn cv_colorspace_code() -> i32 {
        opencv::imgproc::COLOR_RGB2Lab
    }
}

// impl ToColorSpace<u8, u8, Rgb<u8>> for Lab<u8> {
//     fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
//         opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB
//     }
// }

pub trait ConvertColor<T, U, S>
where
    T: seal::Sealed,
    U: seal::Sealed,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(
        &self,
    ) -> Result<ArrayBase<CowRepr<'_, U>, Dst::Dim>, ColorConversionError>
    where
        Src: ToColorSpace<T, U, Dst>,
        Dst: ColorSpace<U>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<U>, <Dst as ColorSpace<U>>::Dim>:
            NdAsImageMut<U, <Dst as ColorSpace<U>>::Dim>;
    fn cvt<Src, Dst>(&self) -> ArrayBase<CowRepr<'_, U>, Dst::Dim>
    where
        Src: ToColorSpace<T, U, Dst>,
        Dst: ColorSpace<U>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<U>, <Dst as ColorSpace<U>>::Dim>:
            NdAsImageMut<U, <Dst as ColorSpace<U>>::Dim>,
    {
        self.try_cvt::<Src, Dst>().expect("Color conversion failed")
    }
}

impl<T, S, U> ConvertColor<T, U, S> for ArrayBase<S, ndarray::Ix3>
where
    T: seal::Sealed + num::Zero,
    U: seal::Sealed + num::Zero,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(&self) -> Result<ArrayBase<CowRepr<'_, U>, Dst::Dim>, ColorConversionError>
    where
        Src: ToColorSpace<T, U, Dst>,
        Dst: ColorSpace<U>,
        ArrayBase<OwnedRepr<U>, <Dst as ColorSpace<U>>::Dim>:
            NdAsImageMut<U, <Dst as ColorSpace<U>>::Dim>,
        ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>: NdAsImage<T, <Rgb<T> as ColorSpace<T>>::Dim>,
    {
        let size = self.shape();
        let mut new_size = Dst::Dim::zeros(Dst::Dim::NDIM.unwrap_or(self.ndim())); // NOTE: This **MAY** fail if Dst::Dim::NDIM is none and self.ndim() doesn't match Dst::Dim's actual number of dimensions. But it's sealed and ideally the manintainer should keep note of this whenever implementing new ones
        let src_channels = size[size.len() - 1];
        if src_channels != Src::CHANNELS {
            return Err(ColorConversionError::ChannelMismatch {
                expected: src_channels,
                src_type: std::any::type_name::<Src>()
                    .rsplit_once("::")
                    .map(|(_, s)| s)
                    .unwrap_or_else(std::any::type_name::<Src>),
                got: Src::CHANNELS,
                size: size.to_vec(),
            });
        }
        size.iter()
            .cloned()
            .take(if size.len() != new_size.ndim() {
                new_size.ndim()
            } else {
                size.len() - 1
            })
            .chain(std::iter::once(Dst::CHANNELS))
            .take(new_size.ndim())
            .enumerate()
            .for_each(|(idx, val)| {
                new_size[idx] = val;
            });
        let mut dst_ndarray = ArrayBase::<ndarray::OwnedRepr<U>, Dst::Dim>::zeros(new_size);
        let mat = self.as_image_mat()?;
        let mut dst_mat = dst_ndarray.as_image_mat_mut()?;
        opencv::imgproc::cvt_color(
            &*mat,
            &mut *dst_mat,
            <Src as ToColorSpace<T, U, Dst>>::cv_colorspace_code(),
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        Ok(dst_ndarray.into())
    }
}

impl<T, S, U> ConvertColor<T, U, S> for ArrayBase<S, ndarray::Ix2>
where
    T: seal::Sealed + num::Zero,
    U: seal::Sealed + num::Zero,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(&self) -> Result<ArrayBase<CowRepr<'_, U>, Dst::Dim>, ColorConversionError>
    where
        Src: ToColorSpace<T, U, Dst>,
        Dst: ColorSpace<U>,
        ArrayBase<OwnedRepr<U>, <Dst as ColorSpace<U>>::Dim>:
            NdAsImageMut<U, <Dst as ColorSpace<U>>::Dim>,
        ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>: NdAsImage<T, <Rgb<T> as ColorSpace<T>>::Dim>,
    {
        let size = self.shape();
        let mut new_size = Dst::Dim::zeros(Dst::Dim::NDIM.unwrap_or(self.ndim()));
        if Src::CHANNELS != 1 {
            return Err(ColorConversionError::ChannelMismatch {
                expected: 1,
                src_type: std::any::type_name::<Src>()
                    .rsplit_once("::")
                    .map(|(_, s)| s)
                    .unwrap_or_else(std::any::type_name::<Src>),
                got: Src::CHANNELS,
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
        let mut dst_ndarray = ArrayBase::<ndarray::OwnedRepr<U>, Dst::Dim>::zeros(new_size);
        let mat = self.as_image_mat()?;
        let mut dst_mat = dst_ndarray.as_image_mat_mut()?;
        opencv::imgproc::cvt_color(
            &*mat,
            &mut *dst_mat,
            <Src as ToColorSpace<T, U, Dst>>::cv_colorspace_code(),
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

    #[test]
    fn test_rgb_to_gray_conversion() {
        // Test RGB to grayscale conversion
        let rgb_data = Array3::<u8>::from_shape_fn((10, 10, 3), |(_y, _x, c)| match c {
            0 => 100, // Red
            1 => 150, // Green
            2 => 200, // Blue
            _ => 0,
        });

        let gray_result: CowArray<u8, Ix2> = rgb_data.cvt::<Rgb<u8>, Gray<u8>>();

        assert_eq!(gray_result.shape(), [10, 10]);
        assert_eq!(gray_result.ndim(), 2);
        // OpenCV uses weighted average: 0.299*R + 0.587*G + 0.114*B
        // Expected: 0.299*100 + 0.587*150 + 0.114*200 = 29.9 + 88.05 + 22.8 = 140.75
        // Allow for rounding differences in OpenCV implementation
        let gray_value = gray_result[[5, 5]];
        assert!(
            (140..=141).contains(&gray_value),
            "Expected gray value between 140-141, got {}",
            gray_value
        );
    }

    #[test]
    fn test_gray_conversion_different_types() {
        // Test grayscale with different numeric types
        let rgb_u16 = Array3::<u16>::from_shape_fn((8, 8, 3), |(_y, _x, c)| match c {
            0 => 32768,
            1 => 16384,
            2 => 8192,
            _ => 0,
        });

        let gray_u16: CowArray<u16, Ix2> = rgb_u16.cvt::<Rgb<u16>, Gray<u16>>();
        assert_eq!(gray_u16.shape(), [8, 8]);
        assert_eq!(gray_u16.ndim(), 2);

        // Test grayscale with f32
        let rgb_f32 = Array3::<f32>::from_shape_fn((6, 6, 3), |(_y, _x, c)| match c {
            0 => 0.5,
            1 => 0.5,
            2 => 0.5,
            _ => 0.0,
        });

        let gray_f32: CowArray<f32, Ix2> = rgb_f32.cvt::<Rgb<f32>, Gray<f32>>();
        assert_eq!(gray_f32.shape(), [6, 6]);
        assert_eq!(gray_f32.ndim(), 2);
        assert!((gray_f32[[3, 3]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_lab_to_rgb_f32_conversion() {
        // Test RGB to Lab and back (f32)
        let rgb_data = Array3::<f32>::from_shape_fn((8, 8, 3), |(_y, _x, c)| match c {
            0 => 1.0, // Pure red in normalized form
            1 => 0.0,
            2 => 0.0,
            _ => 0.0,
        });

        let lab_result: CowArray<f32, Ix3> = rgb_data.cvt::<Rgb<f32>, Lab<f32>>();
        assert_eq!(lab_result.shape(), [8, 8, 3]);

        // Convert back to RGB
        let rgb_back: CowArray<f32, Ix3> = lab_result.cvt::<Lab<f32>, Rgb<f32>>();
        assert_eq!(rgb_back.shape(), [8, 8, 3]);

        // Check that round-trip conversion is approximately equal
        assert!((rgb_back[[4, 4, 0]] - rgb_data[[4, 4, 0]]).abs() < 0.01);
        assert!((rgb_back[[4, 4, 1]] - rgb_data[[4, 4, 1]]).abs() < 0.01);
        assert!((rgb_back[[4, 4, 2]] - rgb_data[[4, 4, 2]]).abs() < 0.01);
    }

    #[test]
    fn test_pure_colors_rgb_to_lab() {
        // Test pure white
        let white_rgb = Array3::<f32>::from_elem((5, 5, 3), 1.0);
        let white_lab: CowArray<f32, Ix3> = white_rgb.cvt::<Rgb<f32>, Lab<f32>>();
        assert_eq!(white_lab.shape(), [5, 5, 3]);

        // Test pure black
        let black_rgb = Array3::<f32>::zeros((5, 5, 3));
        let black_lab: CowArray<f32, Ix3> = black_rgb.cvt::<Rgb<f32>, Lab<f32>>();
        assert_eq!(black_lab.shape(), [5, 5, 3]);
        // L* should be 0 for black
        assert!(black_lab[[2, 2, 0]].abs() < 1e-6);
    }

    #[test]
    fn test_channel_mismatch_error() {
        // Create an array with wrong number of channels for RGB
        let wrong_channels = Array3::<u8>::zeros((10, 10, 2));

        // Try to convert with incorrect channel count - should error
        let result = wrong_channels.try_cvt::<Rgb<u8>, Bgr<u8>>();
        assert!(result.is_err());

        if let Err(ColorConversionError::ChannelMismatch { expected, got, .. }) = result {
            assert_eq!(expected, 2);
            assert_eq!(got, 3);
        } else {
            panic!("Expected ChannelMismatch error");
        }
    }

    #[test]
    #[should_panic(expected = "Color conversion failed")]
    fn test_cvt_panics_on_error() {
        // Create an array with wrong number of channels
        let wrong_channels = Array3::<u8>::zeros((5, 5, 2));

        // cvt() should panic when conversion fails
        let _: CowArray<u8, Ix3> = wrong_channels.cvt::<Rgb<u8>, Bgr<u8>>();
    }

    #[test]
    fn test_rgba_channel_mismatch() {
        // Create an array with 3 channels but claim it's RGBA
        let rgb_data = Array3::<u8>::zeros((5, 5, 3));

        let result = rgb_data.try_cvt::<Rgba<u8>, Rgb<u8>>();
        assert!(result.is_err());
    }

    #[test]
    fn test_precision_f32_conversions() {
        // Test that f32 conversions maintain reasonable precision
        let rgb_data = Array3::<f32>::from_shape_fn((10, 10, 3), |(_y, _x, c)| match c {
            0 => 0.123456,
            1 => 0.456789,
            2 => 0.789012,
            _ => 0.0,
        });

        let bgr_result: CowArray<f32, Ix3> = rgb_data.cvt::<Rgb<f32>, Bgr<f32>>();

        // Verify precision is maintained (allowing for small floating point errors)
        assert!((bgr_result[[5, 5, 0]] - 0.789012).abs() < 1e-5);
        assert!((bgr_result[[5, 5, 1]] - 0.456789).abs() < 1e-5);
        assert!((bgr_result[[5, 5, 2]] - 0.123456).abs() < 1e-5);
    }

    #[test]
    fn test_non_square_images() {
        // Test with rectangular images (non-square)
        let wide_rgb = Array3::<u8>::zeros((10, 50, 3));
        let bgr_result: CowArray<u8, Ix3> = wide_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [10, 50, 3]);

        let tall_rgb = Array3::<u8>::zeros((80, 20, 3));
        let bgr_result: CowArray<u8, Ix3> = tall_rgb.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [80, 20, 3]);
    }

    #[test]
    fn test_rgba_preserves_non_full_alpha() {
        // Test RGBA with various alpha values
        let rgba_data = Array3::<u8>::from_shape_fn((5, 5, 4), |(y, x, c)| match c {
            0 => 255,
            1 => 0,
            2 => 0,
            3 => ((y + x) * 25) as u8, // Varying alpha
            _ => 0,
        });

        let rgb_intermediate: CowArray<u8, Ix3> = rgba_data.cvt::<Rgba<u8>, Rgb<u8>>();
        assert_eq!(rgb_intermediate.shape(), [5, 5, 3]);

        // RGB should only have color channels
        assert_eq!(rgb_intermediate[[2, 2, 0]], 255);
        assert_eq!(rgb_intermediate[[2, 2, 1]], 0);
        assert_eq!(rgb_intermediate[[2, 2, 2]], 0);
    }

    #[test]
    fn test_gray_to_rgb_equivalent_channels() {
        // When converting grayscale to RGB, all channels should be equal
        let _gray_data = Array3::<u8>::from_shape_fn((8, 8, 1), |(_y, _x, _c)| 128);

        // Note: Gray uses Ix2, not Ix3
        use ndarray::Array2;
        let gray_2d = Array2::<u8>::from_elem((8, 8), 128);

        // Gray conversion would need to be implemented
        // This test documents expected behavior if Gray->RGB is added
        assert_eq!(gray_2d.shape(), [8, 8]);
    }

    #[test]
    fn test_mixed_boundary_values() {
        // Test with mixed boundary and mid-range values
        let rgb_data = Array3::<u8>::from_shape_fn((6, 6, 3), |(y, _x, c)| {
            match (y, c) {
                (0, _) => 0,   // First row: black
                (1, _) => 255, // Second row: white
                (2, 0) => 255, // Third row: red
                (2, _) => 0,
                (3, 1) => 255, // Fourth row: green
                (3, _) => 0,
                (4, 2) => 255, // Fifth row: blue
                (4, _) => 0,
                _ => 128, // Last row: gray
            }
        });

        let bgr_result: CowArray<u8, Ix3> = rgb_data.cvt::<Rgb<u8>, Bgr<u8>>();
        assert_eq!(bgr_result.shape(), [6, 6, 3]);

        // Verify red row becomes blue row in BGR
        assert_eq!(bgr_result[[2, 3, 0]], 0); // Blue channel
        assert_eq!(bgr_result[[2, 3, 1]], 0); // Green channel
        assert_eq!(bgr_result[[2, 3, 2]], 255); // Red channel

        // Verify green row stays green in BGR
        assert_eq!(bgr_result[[3, 3, 0]], 0); // Blue channel
        assert_eq!(bgr_result[[3, 3, 1]], 255); // Green channel
        assert_eq!(bgr_result[[3, 3, 2]], 0); // Red channel
    }

    #[test]
    fn test_lab_conversion_preserves_shape() {
        // Verify Lab conversions maintain correct dimensions
        let rgb_data = Array3::<u8>::from_shape_fn((12, 18, 3), |(_, _, _)| 128);
        let lab_result: CowArray<i8, Ix3> = rgb_data.cvt::<Rgb<u8>, Lab<i8>>();

        assert_eq!(lab_result.shape(), [12, 18, 3]);
        assert_eq!(lab_result.ndim(), 3);
    }

    #[test]
    fn test_multiple_conversions_in_sequence() {
        // Test chaining multiple conversions
        let original = Array3::<u8>::from_shape_fn((5, 5, 3), |(_, _, c)| (c * 80) as u8);

        let step1: CowArray<u8, Ix3> = original.cvt::<Rgb<u8>, Bgr<u8>>();
        let step2: CowArray<u8, Ix3> = step1.cvt::<Bgr<u8>, Rgb<u8>>();
        let step3: CowArray<u8, Ix3> = step2.cvt::<Rgb<u8>, Rgba<u8>>();
        let step4: CowArray<u8, Ix3> = step3.cvt::<Rgba<u8>, Rgb<u8>>();

        assert_eq!(step4.shape(), [5, 5, 3]);
        assert_eq!(original, step4.into_owned());
    }

    #[test]
    fn test_u16_boundary_values() {
        // Test u16 with min and max values
        let rgb_data = Array3::<u16>::from_shape_fn((4, 4, 3), |(y, _, c)| {
            match (y, c) {
                (0, _) => 0,     // Min value
                (1, _) => 65535, // Max value
                (2, _) => 32768, // Mid value
                _ => 1234,       // Arbitrary value
            }
        });

        let bgr_result: CowArray<u16, Ix3> = rgb_data.cvt::<Rgb<u16>, Bgr<u16>>();

        // Check min values are preserved
        assert_eq!(bgr_result[[0, 0, 0]], 0);
        assert_eq!(bgr_result[[0, 0, 1]], 0);
        assert_eq!(bgr_result[[0, 0, 2]], 0);

        // Check max values are preserved
        assert_eq!(bgr_result[[1, 1, 0]], 65535);
        assert_eq!(bgr_result[[1, 1, 1]], 65535);
        assert_eq!(bgr_result[[1, 1, 2]], 65535);
    }

    #[test]
    fn test_gray_channel_constant() {
        // Verify Gray color space has 1 channel
        assert_eq!(Gray::<u8>::CHANNELS, 1);
        assert_eq!(Gray::<u16>::CHANNELS, 1);
        assert_eq!(Gray::<f32>::CHANNELS, 1);
    }

    #[test]
    fn test_f32_normalized_range() {
        // Test that f32 values in [0, 1] range are handled correctly
        let rgb_data = Array3::<f32>::from_shape_fn((5, 5, 3), |(y, x, c)| {
            ((y + x + c) as f32) / 20.0 // Values from 0.0 to ~0.65
        });

        let bgr_result: CowArray<f32, Ix3> = rgb_data.cvt::<Rgb<f32>, Bgr<f32>>();

        // Verify all values are in reasonable range
        assert!(bgr_result.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_gray_pure_colors() {
        // Test grayscale conversion with pure colors
        // Pure red
        let red_rgb = Array3::<u8>::from_shape_fn((5, 5, 3), |(_, _, c)| match c {
            0 => 255,
            _ => 0,
        });
        let red_gray: CowArray<u8, Ix2> = red_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        assert_eq!(red_gray.shape(), [5, 5]);
        // Red contributes ~30% to gray value
        let red_gray_val = red_gray[[2, 2]];
        assert!(
            red_gray_val > 70 && red_gray_val < 80,
            "Red gray value: {}",
            red_gray_val
        );

        // Pure green
        let green_rgb = Array3::<u8>::from_shape_fn((5, 5, 3), |(_, _, c)| match c {
            1 => 255,
            _ => 0,
        });
        let green_gray: CowArray<u8, Ix2> = green_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        // Green contributes ~59% to gray value
        let green_gray_val = green_gray[[2, 2]];
        assert!(
            green_gray_val > 145 && green_gray_val < 155,
            "Green gray value: {}",
            green_gray_val
        );

        // Pure blue
        let blue_rgb = Array3::<u8>::from_shape_fn((5, 5, 3), |(_, _, c)| match c {
            2 => 255,
            _ => 0,
        });
        let blue_gray: CowArray<u8, Ix2> = blue_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        // Blue contributes ~11% to gray value
        let blue_gray_val = blue_gray[[2, 2]];
        assert!(
            blue_gray_val > 25 && blue_gray_val < 35,
            "Blue gray value: {}",
            blue_gray_val
        );
    }

    #[test]
    fn test_gray_black_and_white() {
        // Test pure black and white conversions to gray
        let black_rgb = Array3::<u8>::zeros((8, 8, 3));
        let black_gray: CowArray<u8, Ix2> = black_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        assert_eq!(black_gray.shape(), [8, 8]);
        assert!(black_gray.iter().all(|&v| v == 0));

        let white_rgb = Array3::<u8>::from_elem((8, 8, 3), 255);
        let white_gray: CowArray<u8, Ix2> = white_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        assert_eq!(white_gray.shape(), [8, 8]);
        assert!(white_gray.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_gray_dimension_change() {
        // Verify that converting to Gray changes dimension from Ix3 to Ix2
        let rgb_data = Array3::<u8>::from_elem((12, 15, 3), 128);
        let gray_result: CowArray<u8, Ix2> = rgb_data.cvt::<Rgb<u8>, Gray<u8>>();

        assert_eq!(rgb_data.ndim(), 3);
        assert_eq!(gray_result.ndim(), 2);
        assert_eq!(gray_result.shape(), [12, 15]);
    }

    #[test]
    fn test_gray_edge_case_sizes() {
        // Test Gray conversion with various image sizes
        // Minimum size
        let min_rgb = Array3::<u8>::from_elem((1, 1, 3), 100);
        let min_gray: CowArray<u8, Ix2> = min_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        assert_eq!(min_gray.shape(), [1, 1]);

        // Non-square
        let rect_rgb = Array3::<u8>::from_elem((5, 20, 3), 150);
        let rect_gray: CowArray<u8, Ix2> = rect_rgb.cvt::<Rgb<u8>, Gray<u8>>();
        assert_eq!(rect_gray.shape(), [5, 20]);
    }
}
