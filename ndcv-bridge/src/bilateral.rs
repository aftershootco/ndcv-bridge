//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.bilateral_filter.html>
use crate::conversions::*;
use crate::gaussian::BorderType;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum BilateralFilterError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

mod seal {
    pub trait Sealed {}
    // src: 8-bit or floating-point, 1-channel or 3-channel image. Unlike the
    // box and gaussian filters, this one is not separable and does not treat
    // channels independently: the colour term measures distance across all of
    // them at once, so OpenCV accepts only these two depths.
    impl Sealed for u8 {}
    impl Sealed for f32 {}
}

/// Edge-preserving smoothing: each output pixel is a weighted mean of its
/// neighbourhood, where a neighbour's weight falls off both with distance
/// (`sigma_space`) and with how different its colour is (`sigma_color`).
///
/// There is no in-place variant. OpenCV rejects a call whose source and
/// destination are the same buffer, because the filter reads a whole
/// neighbourhood per output pixel and would be reading its own partial output.
pub trait NdCvBilateralFilter<
    T: bytemuck::Pod + seal::Sealed + crate::types::CvType,
    D: ndarray::Dimension,
>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
{
    /// `diameter` is the neighbourhood width in pixels; a value <= 0 makes
    /// OpenCV derive it from `sigma_space`. A 1- or 3-channel image of `u8` or
    /// `f32` is the whole of what OpenCV supports here — anything else comes
    /// back as a [`BilateralFilterError::OpenCvError`] rather than panicking.
    fn bilateral_filter(
        &self,
        diameter: i32,
        sigma_color: f64,
        sigma_space: f64,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, BilateralFilterError>;

    /// `cv2.bilateralFilter` defaults, whose `borderType` is
    /// [`BorderType::BorderDefault`].
    fn bilateral_filter_def(
        &self,
        diameter: i32,
        sigma_color: f64,
        sigma_space: f64,
    ) -> Result<ndarray::Array<T, D>, BilateralFilterError> {
        self.bilateral_filter(
            diameter,
            sigma_color,
            sigma_space,
            BorderType::BorderDefault,
        )
    }
}

impl<
    T: bytemuck::Pod + num::Zero + seal::Sealed + crate::types::CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
> NdCvBilateralFilter<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
{
    fn bilateral_filter(
        &self,
        diameter: i32,
        sigma_color: f64,
        sigma_space: f64,
        border_type: BorderType,
    ) -> Result<ndarray::Array<T, D>, BilateralFilterError> {
        let mut dst = ndarray::Array::zeros(self.dim());
        let cv_self = self.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::bilateral_filter(
            &*cv_self,
            &mut *cv_dst,
            diameter,
            sigma_color,
            sigma_space,
            border_type as i32,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_bilateral_basic() {
        let arr = Array3::<u8>::ones((10, 10, 3));
        let res = arr
            .bilateral_filter(5, 25.0, 25.0, BorderType::BorderReflect101)
            .unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_bilateral_preserves_a_step_edge() {
        // The point of the filter: a step far larger than sigma_color is left
        // alone, where `blur`/`gaussian_blur` would smear it into a ramp.
        let mut arr = Array3::<u8>::zeros((20, 20, 3));
        arr.slice_mut(s![..10, .., ..]).fill(255);

        let res = arr
            .bilateral_filter(9, 10.0, 20.0, BorderType::BorderReflect101)
            .unwrap();

        assert_eq!(res[[9, 10, 0]], 255, "last row above the edge");
        assert_eq!(res[[10, 10, 0]], 0, "first row below the edge");
    }

    #[test]
    fn test_bilateral_smooths_within_sigma_color() {
        // A step well inside sigma_color is not an edge to this filter, so the
        // same geometry does get averaged across.
        let mut arr = Array3::<u8>::from_elem((20, 20, 3), 100);
        arr.slice_mut(s![..10, .., ..]).fill(120);

        let res = arr
            .bilateral_filter(9, 200.0, 20.0, BorderType::BorderReflect101)
            .unwrap();

        assert!(
            (res[[9, 10, 0]] as i32) < 120 && (res[[10, 10, 0]] as i32) > 100,
            "expected the two sides to pull toward each other, got {} and {}",
            res[[9, 10, 0]],
            res[[10, 10, 0]]
        );
    }

    #[test]
    fn test_bilateral_couples_the_channels() {
        // The one failure mode unique to this filter: it measures colour
        // distance across all channels at once, so a wrong HWC -> Mat
        // interleave, or a per-channel stand-in, changes the output rather
        // than erroring. Channel 0 steps by 10, well inside sigma_color, but
        // the other two step by 230 at the same row -- so a coupled filter
        // refuses to mix across it and channel 0 keeps its small step, where
        // a per-channel one would smooth it into a ramp.
        let mut arr = Array3::<u8>::zeros((20, 20, 3));
        arr.slice_mut(s![..10, .., 0]).fill(100);
        arr.slice_mut(s![..10, .., 1..]).fill(10);
        arr.slice_mut(s![10.., .., 0]).fill(110);
        arr.slice_mut(s![10.., .., 1..]).fill(240);

        let res = arr
            .bilateral_filter(9, 30.0, 20.0, BorderType::BorderReflect101)
            .unwrap();

        // cv2 gives [100, 100, 110, 110] here; filtering channel 0 on its own
        // gives [103, 104, 106, 107].
        assert_eq!(
            res.slice(s![8..12, 10, 0]).to_vec(),
            vec![100, 100, 110, 110]
        );
    }

    #[test]
    fn test_bilateral_accepts_every_border_type() {
        let mut arr = Array3::<u8>::zeros((10, 10, 3));
        arr.slice_mut(s![4..7, 4..7, ..]).fill(255);

        for border_type in [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ] {
            let res = arr.bilateral_filter(5, 25.0, 10.0, border_type).unwrap();
            assert_eq!(res.shape(), &[10, 10, 3]);
        }
    }

    #[test]
    fn test_bilateral_leaves_a_flat_image_flat() {
        let arr = Array3::<u8>::from_elem((16, 16, 3), 77);
        let res = arr
            .bilateral_filter(7, 30.0, 14.0, BorderType::BorderReflect101)
            .unwrap();
        assert!(res.iter().all(|&v| v == 77));
    }

    #[test]
    fn test_bilateral_single_channel_and_f32() {
        let arr_u8 = Array2::<u8>::ones((10, 10));
        let arr_f32 = Array3::<f32>::ones((10, 10, 3));

        let res_u8 = arr_u8.bilateral_filter_def(5, 25.0, 25.0).unwrap();
        let res_f32 = arr_f32.bilateral_filter_def(5, 25.0, 25.0).unwrap();

        assert_eq!(res_u8.shape(), &[10, 10]);
        assert_eq!(res_f32.shape(), &[10, 10, 3]);
    }

    #[test]
    fn test_bilateral_def_reflects_the_border() {
        // `_def` has to mean cv2's own default: a constant border is a black
        // neighbour that drags the edge of a flat mid-grey image down by half.
        let arr = Array2::<u8>::from_elem((12, 12), 200);

        let def = arr.bilateral_filter_def(7, 200.0, 14.0).unwrap();
        let reflect = arr
            .bilateral_filter(7, 200.0, 14.0, BorderType::BorderReflect101)
            .unwrap();
        let constant = arr
            .bilateral_filter(7, 200.0, 14.0, BorderType::BorderConstant)
            .unwrap();

        assert_eq!(def, reflect);
        assert_eq!(def[[0, 0]], 200, "reflecting a flat image keeps it flat");
        assert!(
            constant[[0, 0]] < 150,
            "a constant border should pull the corner well down, got {}",
            constant[[0, 0]]
        );
    }

    #[test]
    fn test_bilateral_rejects_unsupported_channel_count() {
        // 2 channels is neither of the two shapes OpenCV accepts. It must come
        // back as an error, not a panic, since a slider runs this on user data.
        let arr = Array3::<u8>::ones((10, 10, 2));
        let err = arr.bilateral_filter_def(5, 25.0, 25.0).unwrap_err();
        assert!(
            matches!(err, BilateralFilterError::OpenCvError(_)),
            "expected an OpenCV assertion, got: {err}"
        );
    }
}
