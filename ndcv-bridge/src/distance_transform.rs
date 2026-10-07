//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.distance_transform.html>
//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.distance_transform_with_labels.html>
use crate::conversions::*;
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum DistanceTransformError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("u8 output is only supported with DistanceType::L1, got {0:?}")]
    U8OutputRequiresL1(DistanceType),
}

/// The metric the distance is measured in. Only these three are accepted by
/// `distanceTransform`; the other `DIST_*` constants belong to `fitLine`.
///
/// No `Default`: cv2 makes `distanceType` a required argument.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DistanceType {
    /// `|dx| + |dy|`
    L1 = opencv::imgproc::DIST_L1,
    /// Euclidean.
    L2 = opencv::imgproc::DIST_L2,
    /// `max(|dx|, |dy|)`
    C = opencv::imgproc::DIST_C,
}

/// Size of the mask the approximate algorithm propagates distances with.
///
/// Only matters for [`DistanceType::L2`]: for `L1` and `C` OpenCV forces it to
/// 3, which is already exact for those metrics. No `Default`, since cv2 makes
/// `maskSize` a required argument.
///
/// Ignored by [`NdCvDistanceTransform::distance_transform_with_labels`]:
/// OpenCV runs every mask there as [`DistanceTransformMask::Mask5`], for every
/// metric.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DistanceTransformMask {
    /// Coarse `L2`: a few percent of error off the axes.
    Mask3 = opencv::imgproc::DIST_MASK_3,
    /// Closer `L2`, adding knight's moves.
    Mask5 = opencv::imgproc::DIST_MASK_5,
    /// Exact `L2`.
    Precise = opencv::imgproc::DIST_MASK_PRECISE,
}

/// What [`NdCvDistanceTransform::distance_transform_with_labels`] labels each
/// pixel with.
#[repr(i32)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum DistanceTransformLabelType {
    /// The connected component of zero pixels that is nearest.
    #[default]
    ConnectedComponent = opencv::imgproc::DIST_LABEL_CCOMP,
    /// The single zero pixel that is nearest.
    Pixel = opencv::imgproc::DIST_LABEL_PIXEL,
}

#[derive(Debug, Clone)]
pub struct DistanceTransformWithLabels {
    pub distances: ndarray::Array2<f32>,
    /// The discrete Voronoi diagram: per pixel, the label of the nearest zero
    /// component or zero pixel. Labels start at 1.
    pub labels: ndarray::Array2<i32>,
}

pub(crate) mod seal {
    // dst: 8-bit or 32-bit floating-point, single-channel image.
    pub trait DistanceTransformOutput:
        Sized + Copy + bytemuck::Pod + num::Zero + crate::types::CvType
    {
        fn as_cv_depth() -> i32 {
            <Self as crate::types::CvType>::cv_depth()
        }
    }
    impl DistanceTransformOutput for u8 {}
    impl DistanceTransformOutput for f32 {}
}

/// For every pixel of a single-channel `u8` image, the distance to the nearest
/// zero pixel. Zero pixels map to zero; any non-zero value counts as
/// foreground, so the input does not need to be strictly 0/1 or 0/255.
///
/// Any other input depth or channel count comes back as a
/// [`DistanceTransformError::OpenCvError`] rather than panicking.
pub trait NdCvDistanceTransform<T: crate::types::CvType>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, ndarray::Ix2>
{
    /// `O` is the output depth: `f32`, or `u8`, which OpenCV supports only
    /// together with [`DistanceType::L1`] and saturates at 255.
    ///
    /// `u8` with any other metric is a
    /// [`DistanceTransformError::U8OutputRequiresL1`]. OpenCV does not reject
    /// it, but writes `f32` into a buffer of its own, which would leave the
    /// returned array all zeros.
    fn distance_transform<O: seal::DistanceTransformOutput>(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
    ) -> Result<ndarray::Array2<O>, DistanceTransformError>;

    /// `mask` is ignored: OpenCV forces [`DistanceTransformMask::Mask5`]
    /// whenever labels are requested, whatever the metric. So `Mask3` and
    /// `Precise` silently give the same result as `Mask5`, and `L2` distances
    /// are the 5x5 approximation, never exact.
    fn distance_transform_with_labels(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
        label_type: DistanceTransformLabelType,
    ) -> Result<DistanceTransformWithLabels, DistanceTransformError>;

    /// `cv2.distanceTransform` defaults, whose `dstType` is `CV_32F`.
    fn distance_transform_def(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
    ) -> Result<ndarray::Array2<f32>, DistanceTransformError> {
        self.distance_transform::<f32>(distance_type, mask)
    }

    /// `cv2.distanceTransformWithLabels` defaults, whose `labelType` is
    /// [`DistanceTransformLabelType::ConnectedComponent`].
    fn distance_transform_with_labels_def(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
    ) -> Result<DistanceTransformWithLabels, DistanceTransformError> {
        self.distance_transform_with_labels(
            distance_type,
            mask,
            DistanceTransformLabelType::ConnectedComponent,
        )
    }
}

impl<T: bytemuck::Pod + crate::types::CvType, S: ndarray::Data<Elem = T>> NdCvDistanceTransform<T>
    for ArrayBase<S, Ix2>
where
    ndarray::ArrayBase<S, Ix2>: crate::image::NdImage + crate::conversions::NdAsImage<T, Ix2>,
{
    fn distance_transform<O: seal::DistanceTransformOutput>(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
    ) -> Result<ndarray::Array2<O>, DistanceTransformError> {
        if O::as_cv_depth() == opencv::core::CV_8U && distance_type != DistanceType::L1 {
            return Err(DistanceTransformError::U8OutputRequiresL1(distance_type));
        }
        let mut dst = ndarray::Array2::<O>::zeros(self.dim());
        // OpenCV writes the first pixel unconditionally, through ndarray's
        // dangling pointer for an empty array.
        if self.is_empty() {
            return Ok(dst);
        }
        // The Mat conversion keeps only the outer stride, so anything not in
        // standard layout would be read as if it were.
        let src = self.as_standard_layout();
        let cv_self = src.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::imgproc::distance_transform(
            &*cv_self,
            &mut *cv_dst,
            distance_type as i32,
            mask as i32,
            O::as_cv_depth(),
        )?;
        Ok(dst)
    }

    fn distance_transform_with_labels(
        &self,
        distance_type: DistanceType,
        mask: DistanceTransformMask,
        label_type: DistanceTransformLabelType,
    ) -> Result<DistanceTransformWithLabels, DistanceTransformError> {
        let mut distances = ndarray::Array2::<f32>::zeros(self.dim());
        let mut labels = ndarray::Array2::<i32>::zeros(self.dim());
        if self.is_empty() {
            return Ok(DistanceTransformWithLabels { distances, labels });
        }
        let src = self.as_standard_layout();
        let cv_self = src.as_image_mat()?;
        let mut cv_distances = distances.as_image_mat_mut()?;
        let mut cv_labels = labels.as_image_mat_mut()?;
        opencv::imgproc::distance_transform_with_labels(
            &*cv_self,
            &mut *cv_distances,
            &mut *cv_labels,
            distance_type as i32,
            mask as i32,
            label_type as i32,
        )?;
        Ok(DistanceTransformWithLabels { distances, labels })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// All foreground except a single zero pixel at `(y, x)`.
    fn single_seed(h: usize, w: usize, y: usize, x: usize) -> Array2<u8> {
        let mut arr = Array2::<u8>::from_elem((h, w), 255);
        arr[[y, x]] = 0;
        arr
    }

    #[test]
    fn test_distance_transform_basic() {
        let mut arr = Array2::<u8>::from_elem((8, 8), 1);
        arr.slice_mut(s![.., ..2]).fill(0);
        let res = arr
            .distance_transform_def(DistanceType::L2, DistanceTransformMask::Precise)
            .unwrap();
        assert_eq!(res.dim(), (8, 8));
        assert!(res.slice(s![.., ..2]).iter().all(|&v| v == 0.0));
        // Columns 2.. are 1, 2, ... pixels from the zero band.
        assert_eq!(res[[4, 2]], 1.0);
        assert_eq!(res[[4, 7]], 6.0);
    }

    #[test]
    fn test_distance_transform_metrics_from_a_single_seed() {
        // From a seed at (0, 0), the pixel at (3, 4) is 7 away in L1, 4 in C
        // and exactly 5 in precise L2.
        let arr = single_seed(10, 10, 0, 0);
        let run = |distance_type, mask| arr.distance_transform_def(distance_type, mask).unwrap();

        let l1 = run(DistanceType::L1, DistanceTransformMask::Mask3);
        let c = run(DistanceType::C, DistanceTransformMask::Mask3);
        let l2 = run(DistanceType::L2, DistanceTransformMask::Precise);

        assert_eq!(l1[[3, 4]], 7.0);
        assert_eq!(c[[3, 4]], 4.0);
        assert!((l2[[3, 4]] - 5.0).abs() < 1e-4, "got {}", l2[[3, 4]]);
    }

    #[test]
    fn test_distance_transform_approximate_masks_are_close_to_precise() {
        let arr = single_seed(20, 20, 10, 10);
        let precise = arr
            .distance_transform_def(DistanceType::L2, DistanceTransformMask::Precise)
            .unwrap();
        for mask in [DistanceTransformMask::Mask3, DistanceTransformMask::Mask5] {
            let approx = arr.distance_transform_def(DistanceType::L2, mask).unwrap();
            for (a, p) in approx.iter().zip(precise.iter()) {
                assert!((a - p).abs() <= 0.1 * p.max(1.0), "{mask:?}: {a} vs {p}");
            }
        }
    }

    #[test]
    fn test_distance_transform_def_is_f32() {
        let arr = single_seed(10, 10, 5, 5);
        let def = arr
            .distance_transform_def(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        let explicit = arr
            .distance_transform::<f32>(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        assert_eq!(def, explicit);
    }

    #[test]
    fn test_distance_transform_u8_output_saturates() {
        // u8 output with L1: the same distances as f32, clipped at 255.
        let arr = single_seed(1, 300, 0, 0);
        let res = arr
            .distance_transform::<u8>(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        assert_eq!(res[[0, 7]], 7);
        assert_eq!(res[[0, 299]], 255);
    }

    #[test]
    fn test_distance_transform_u8_output_rejects_l2() {
        // OpenCV supports u8 output only for L1, and for anything else
        // silently writes f32 into a fresh buffer, leaving ours all zeros.
        // That has to be an error, not a quietly blank mask.
        let arr = single_seed(10, 10, 5, 5);
        let err = arr
            .distance_transform::<u8>(DistanceType::L2, DistanceTransformMask::Mask3)
            .unwrap_err();
        assert!(
            matches!(
                err,
                DistanceTransformError::U8OutputRequiresL1(DistanceType::L2)
            ),
            "got: {err}"
        );
        assert!(
            arr.distance_transform::<u8>(DistanceType::C, DistanceTransformMask::Mask3)
                .is_err()
        );
    }

    #[test]
    fn test_distance_transform_u8_output_ignores_the_mask_for_l1() {
        // L1 is exact with any mask, so every mask has to give the same u8
        // result -- including Precise, which must not trip the guard.
        let arr = single_seed(10, 10, 5, 5);
        let expected = arr
            .distance_transform::<u8>(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        assert_eq!(expected[[0, 0]], 10);
        for mask in [DistanceTransformMask::Mask5, DistanceTransformMask::Precise] {
            let res = arr
                .distance_transform::<u8>(DistanceType::L1, mask)
                .unwrap();
            assert_eq!(res, expected, "{mask:?}");
        }
    }

    #[test]
    fn test_distance_transform_accepts_a_view() {
        let arr = single_seed(10, 10, 5, 5);
        let owned = arr
            .distance_transform_def(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        let viewed = arr
            .view()
            .distance_transform_def(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();
        assert_eq!(owned, viewed);
    }

    #[test]
    fn test_distance_transform_with_labels_splits_into_voronoi_cells() {
        // Two zero pixels at opposite ends of a row: everything left of the
        // middle belongs to one, everything right of it to the other.
        let mut arr = Array2::<u8>::from_elem((1, 11), 255);
        arr[[0, 0]] = 0;
        arr[[0, 10]] = 0;

        let res = arr
            .distance_transform_with_labels_def(DistanceType::L1, DistanceTransformMask::Mask3)
            .unwrap();

        assert_eq!(res.distances[[0, 3]], 3.0);
        assert_eq!(res.distances[[0, 8]], 2.0);
        let left = res.labels[[0, 0]];
        let right = res.labels[[0, 10]];
        assert!(left > 0 && right > 0 && left != right);
        assert!(res.labels.slice(s![0, ..5]).iter().all(|&l| l == left));
        assert!(res.labels.slice(s![0, 6..]).iter().all(|&l| l == right));
    }

    #[test]
    fn test_distance_transform_with_labels_pixel_vs_component() {
        // A 2-pixel zero blob is one component but two pixels.
        let mut arr = Array2::<u8>::from_elem((5, 5), 255);
        arr[[2, 1]] = 0;
        arr[[2, 2]] = 0;
        let run = |label_type| {
            arr.distance_transform_with_labels(
                DistanceType::L2,
                DistanceTransformMask::Mask5,
                label_type,
            )
            .unwrap()
            .labels
        };

        let ccomp = run(DistanceTransformLabelType::ConnectedComponent);
        let pixel = run(DistanceTransformLabelType::Pixel);

        assert_eq!(ccomp[[2, 1]], ccomp[[2, 2]]);
        assert_ne!(pixel[[2, 1]], pixel[[2, 2]]);
    }

    #[test]
    fn test_distance_transform_with_labels_def_labels_components() {
        let mut arr = Array2::<u8>::from_elem((5, 5), 255);
        arr[[2, 1]] = 0;
        arr[[2, 2]] = 0;
        let def = arr
            .distance_transform_with_labels_def(DistanceType::L2, DistanceTransformMask::Mask5)
            .unwrap();
        let ccomp = arr
            .distance_transform_with_labels(
                DistanceType::L2,
                DistanceTransformMask::Mask5,
                DistanceTransformLabelType::ConnectedComponent,
            )
            .unwrap();
        assert_eq!(def.labels, ccomp.labels);
        assert_eq!(def.distances, ccomp.distances);
    }

    #[test]
    fn test_distance_transform_with_labels_ignores_the_mask() {
        // OpenCV forces the 5x5 mask whenever labels are requested, for every
        // metric. Pin that, since a caller asking for exact or 3x3 L2 gets the
        // 5x5 approximation.
        let arr = single_seed(9, 9, 4, 4);
        for distance_type in [DistanceType::L1, DistanceType::L2, DistanceType::C] {
            let run = |mask| {
                arr.distance_transform_with_labels_def(distance_type, mask)
                    .unwrap()
                    .distances
            };
            let mask5 = run(DistanceTransformMask::Mask5);
            assert_eq!(
                run(DistanceTransformMask::Mask3),
                mask5,
                "{distance_type:?}"
            );
            assert_eq!(
                run(DistanceTransformMask::Precise),
                mask5,
                "{distance_type:?}"
            );
        }
        let l2 = arr
            .distance_transform_with_labels_def(DistanceType::L2, DistanceTransformMask::Mask3)
            .unwrap()
            .distances;
        // The 5x5 costs: 1 along the axis (3x3 would give 0.955) and 1.4 on the
        // diagonal (exact would give sqrt(2)).
        assert_eq!(l2[[4, 5]], 1.0);
        assert!((l2[[3, 3]] - 1.4).abs() < 1e-3, "got {}", l2[[3, 3]]);
    }

    #[test]
    fn test_distance_transform_empty_input() {
        for dim in [(0, 5), (5, 0), (0, 0)] {
            let arr = Array2::<u8>::zeros(dim);
            for mask in [DistanceTransformMask::Mask3, DistanceTransformMask::Precise] {
                for distance_type in [DistanceType::L1, DistanceType::L2] {
                    let res = arr.distance_transform_def(distance_type, mask).unwrap();
                    assert_eq!(res.dim(), dim);
                }
            }
            let res = arr
                .distance_transform::<u8>(DistanceType::L1, DistanceTransformMask::Mask3)
                .unwrap();
            assert_eq!(res.dim(), dim);
            for label_type in [
                DistanceTransformLabelType::ConnectedComponent,
                DistanceTransformLabelType::Pixel,
            ] {
                let res = arr
                    .distance_transform_with_labels(
                        DistanceType::L2,
                        DistanceTransformMask::Mask5,
                        label_type,
                    )
                    .unwrap();
                assert_eq!(res.distances.dim(), dim);
                assert_eq!(res.labels.dim(), dim);
            }
        }
    }

    #[test]
    fn test_distance_transform_non_standard_layouts_match_a_copy() {
        // Asymmetric seeds, so any transposition or flip shows up.
        let mut hwc = Array3::<u8>::from_elem((6, 9, 3), 255);
        hwc[[0, 7, 0]] = 0;
        hwc[[4, 1, 0]] = 0;
        let base = hwc.slice(s![.., .., 0]).to_owned();
        let f_order = {
            let mut a = Array2::<u8>::zeros(ShapeBuilder::f(base.dim()));
            a.assign(&base);
            a
        };
        let views = [
            ("channel slice", hwc.slice(s![.., .., 0])),
            ("transposed", base.t()),
            ("f-order", f_order.view()),
            ("columns reversed", base.slice(s![.., ..;-1])),
            ("rows reversed", base.slice(s![..;-1, ..])),
            ("every other column", base.slice(s![.., ..;2])),
        ];
        for (name, view) in views {
            let copy = view.to_owned();
            for (distance_type, mask) in [
                (DistanceType::L1, DistanceTransformMask::Mask3),
                (DistanceType::L2, DistanceTransformMask::Precise),
            ] {
                assert_eq!(
                    view.distance_transform_def(distance_type, mask).unwrap(),
                    copy.distance_transform_def(distance_type, mask).unwrap(),
                    "{name} {distance_type:?}"
                );
            }
            let labeled = view
                .distance_transform_with_labels_def(DistanceType::L2, DistanceTransformMask::Mask5)
                .unwrap();
            let expected = copy
                .distance_transform_with_labels_def(DistanceType::L2, DistanceTransformMask::Mask5)
                .unwrap();
            assert_eq!(labeled.distances, expected.distances, "{name}");
            assert_eq!(labeled.labels, expected.labels, "{name}");
        }
    }

    #[test]
    fn test_distance_transform_rejects_non_u8_negative_stride_view() {
        // Used to overflow `as usize` in the conversion and panic in debug.
        let arr = Array2::<f32>::ones((4, 4));
        let err = arr
            .slice(s![..;-1, ..])
            .distance_transform_def(DistanceType::L2, DistanceTransformMask::Mask3)
            .unwrap_err();
        assert!(
            matches!(err, DistanceTransformError::OpenCvError(_)),
            "{err}"
        );
    }

    #[test]
    fn test_distance_transform_rejects_non_u8_input() {
        // OpenCV takes only 8-bit input. Without a seal on T, that has to come
        // back as an error, not a panic.
        let arr = Array2::<f32>::from_elem((10, 10), 1.0);
        let err = arr
            .distance_transform_def(DistanceType::L2, DistanceTransformMask::Precise)
            .unwrap_err();
        assert!(
            matches!(err, DistanceTransformError::OpenCvError(_)),
            "expected an OpenCV assertion, got: {err}"
        );
        assert!(
            Array2::<u16>::ones((10, 10))
                .distance_transform_with_labels_def(DistanceType::L1, DistanceTransformMask::Mask3)
                .is_err()
        );
    }
}
