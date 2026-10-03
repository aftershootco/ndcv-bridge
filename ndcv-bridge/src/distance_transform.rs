//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.distance_transform.html>
//! <https://docs.opencv.org/4.13.0/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042>
//!
//! Computes, for every pixel, the distance to the nearest **zero** pixel of the
//! source mask. Zero pixels map to `0.0`; non-zero pixels map to their distance.
//!
//! Unlike the morphology operators, this one is not generic over the element
//! type: OpenCV requires a single-channel 8-bit source, and writes a
//! single-channel `f32` result. The source is therefore fixed to `u8`/`Ix2` and
//! the output to `Array2<f32>`.
use crate::conversions::{ConversionError, NdAsImage, NdAsImageMut};

#[derive(Debug, thiserror::Error)]
pub enum DistanceTransformError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

impl DistanceTransformError {
    pub fn into_error(self) -> impl core::error::Error + Send + Sync + 'static {
        self
    }
}

/// Distance metric used by [`NdCvDistanceTransform`].
///
/// OpenCV's `DistanceTypes` enum also carries the M-estimator variants
/// (`DIST_L12`, `DIST_FAIR`, `DIST_WELSCH`, `DIST_HUBER`), but `distanceTransform`
/// only accepts these three, so the others are deliberately not surfaced here.
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum DistanceType {
    /// `distance = |x1 - x2| + |y1 - y2|`
    DistL1 = 1,
    /// Simple Euclidean distance.
    #[default]
    DistL2 = 2,
    /// `distance = max(|x1 - x2|, |y1 - y2|)` (chessboard).
    DistC = 3,
}

/// Size of the distance transform mask.
///
/// `Precise` is the exact Euclidean transform and is only meaningful with
/// [`DistanceType::DistL2`]; the 3x3 and 5x5 masks are cheaper approximations.
#[repr(C)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum DistanceTransformMask {
    /// `cv2.DIST_MASK_PRECISE`
    #[default]
    Precise = 0,
    /// `cv2.DIST_MASK_3`
    Mask3 = 3,
    /// `cv2.DIST_MASK_5`
    Mask5 = 5,
}

pub trait NdCvDistanceTransform {
    /// Computes the distance to the nearest zero pixel for every pixel of the mask.
    ///
    /// - `distance_type`: the distance metric
    /// - `mask_size`: the mask used to approximate the metric. OpenCV forces
    ///   `Precise` to behave as a 5x5 mask for [`DistanceType::DistL1`] and
    ///   [`DistanceType::DistC`], where the 3x3 mask is already exact.
    ///
    /// Returns an `Array2<f32>` with the same shape as `self`.
    fn distance_transform(
        &self,
        distance_type: DistanceType,
        mask_size: DistanceTransformMask,
    ) -> Result<ndarray::Array2<f32>, DistanceTransformError>;

    /// Exact Euclidean distance transform: [`DistanceType::DistL2`] with
    /// [`DistanceTransformMask::Precise`].
    ///
    /// This is the equivalent of
    /// `cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)`.
    fn distance_transform_def(
        &self,
    ) -> Result<ndarray::Array2<f32>, DistanceTransformError> {
        self.distance_transform(DistanceType::DistL2, DistanceTransformMask::Precise)
    }
}

// `NdAsImage<u8, Ix2>` / `NdAsImageMut<f32, Ix2>` are covered by the blanket
// single-channel impls in `conversions`, so no extra bounds are needed here.
impl<S: ndarray::Data<Elem = u8>> NdCvDistanceTransform for ndarray::ArrayBase<S, ndarray::Ix2> {
    fn distance_transform(
        &self,
        distance_type: DistanceType,
        mask_size: DistanceTransformMask,
    ) -> Result<ndarray::Array2<f32>, DistanceTransformError> {
        let cv_src = self.as_image_mat()?;
        let mut dst = ndarray::Array2::<f32>::zeros(self.dim());
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::distance_transform(
            &*cv_src,
            &mut *cv_dst,
            distance_type as i32,
            mask_size as i32,
            opencv::core::CV_32F,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_distance_transform_all_zero_is_all_zero() {
        // No non-zero pixels: nothing has a distance to travel.
        let arr = Array2::<u8>::zeros((8, 8));
        let res = arr.distance_transform_def().unwrap();
        assert_eq!(res.shape(), &[8, 8]);
        assert!(res.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_distance_transform_zero_pixels_stay_zero() {
        let mut arr = Array2::<u8>::from_elem((9, 9), 255);
        arr[[4, 4]] = 0;
        let res = arr.distance_transform_def().unwrap();
        assert_eq!(res[[4, 4]], 0.0);
        // 4-neighbours of the single zero are exactly 1 away.
        assert!((res[[3, 4]] - 1.0).abs() < 1e-5);
        assert!((res[[5, 4]] - 1.0).abs() < 1e-5);
        assert!((res[[4, 3]] - 1.0).abs() < 1e-5);
        assert!((res[[4, 5]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_distance_transform_l2_diagonal_is_euclidean() {
        // With the precise L2 transform the diagonal neighbour is sqrt(2), not 2.
        let mut arr = Array2::<u8>::from_elem((9, 9), 255);
        arr[[4, 4]] = 0;
        let res = arr.distance_transform_def().unwrap();
        assert!((res[[3, 3]] - std::f32::consts::SQRT_2).abs() < 1e-4);
    }

    #[test]
    fn test_distance_transform_c_diagonal_is_chessboard() {
        // DIST_C makes the diagonal neighbour 1 away, unlike L2.
        let mut arr = Array2::<u8>::from_elem((9, 9), 255);
        arr[[4, 4]] = 0;
        let res = arr
            .distance_transform(DistanceType::DistC, DistanceTransformMask::Mask3)
            .unwrap();
        assert!((res[[3, 3]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_distance_transform_l1_diagonal_is_manhattan() {
        let mut arr = Array2::<u8>::from_elem((9, 9), 255);
        arr[[4, 4]] = 0;
        let res = arr
            .distance_transform(DistanceType::DistL1, DistanceTransformMask::Mask3)
            .unwrap();
        assert!((res[[3, 3]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_distance_transform_ramps_away_from_edge() {
        // Left column is the only zero; distance should grow by 1 per column.
        let mut arr = Array2::<u8>::from_elem((5, 6), 255);
        for r in 0..5 {
            arr[[r, 0]] = 0;
        }
        let res = arr.distance_transform_def().unwrap();
        for c in 0..6 {
            assert!(
                (res[[2, c]] - c as f32).abs() < 1e-4,
                "column {c} expected {c}, got {}",
                res[[2, c]]
            );
        }
    }

    #[test]
    fn test_distance_transform_treats_any_nonzero_as_foreground() {
        // OpenCV thresholds on != 0, so a mask of 1s behaves like a mask of 255s.
        let mut ones = Array2::<u8>::ones((7, 7));
        let mut full = Array2::<u8>::from_elem((7, 7), 255);
        ones[[3, 3]] = 0;
        full[[3, 3]] = 0;
        let a = ones.distance_transform_def().unwrap();
        let b = full.distance_transform_def().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_distance_transform_all_mask_sizes_run() {
        let mut arr = Array2::<u8>::from_elem((10, 10), 255);
        arr[[5, 5]] = 0;
        for mask_size in [
            DistanceTransformMask::Precise,
            DistanceTransformMask::Mask3,
            DistanceTransformMask::Mask5,
        ] {
            let res = arr
                .distance_transform(DistanceType::DistL2, mask_size)
                .unwrap();
            assert_eq!(res.shape(), &[10, 10]);
        }
    }

    #[test]
    fn test_distance_transform_non_square_keeps_row_col_order() {
        // Guards the ndarray -> Mat mapping: a transposed bridge would either
        // change the shape or put the column ramp on the row axis.
        let mut arr = Array2::<u8>::from_elem((4, 9), 255);
        arr[[0, 0]] = 0;
        let res = arr.distance_transform_def().unwrap();
        assert_eq!(res.shape(), &[4, 9]);
        assert!((res[[3, 0]] - 3.0).abs() < 1e-4, "got {}", res[[3, 0]]);
        assert!((res[[0, 8]] - 8.0).abs() < 1e-4, "got {}", res[[0, 8]]);
    }
}
