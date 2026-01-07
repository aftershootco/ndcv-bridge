use bounding_box::Aabb2;
pub use bounding_box::roi::{Roi, RoiMut};
mod seal {
    use ndarray::{Ix2, Ix3};
    pub trait Sealed {}
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> Sealed for ndarray::ArrayBase<S, Ix2> {}
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> Sealed for ndarray::ArrayBase<S, Ix3> {}
}

/// ```text
///    ┌──────────────────┐
///    │      padded      │
///    │    ┌────────┐    │
///    │    │        │    │
///    │    │original│    │
///    │    │        │    │
///    │    └────────┘    │
///    │      zeroed      │
///    └──────────────────┘
/// ```
///
/// Returns the padded bounding box and the padded segment
/// The padded is the padded bounding box
/// The original is the original bounding box
/// Returns the padded bounding box as zeros and the original bbox as the original segment
pub trait NdRoiZeroPadded<T, D: ndarray::Dimension>: seal::Sealed {
    fn roi_zero_padded(
        &self,
        original: Aabb2<usize>,
        padded: Aabb2<usize>,
    ) -> (Aabb2<usize>, ndarray::Array<T, D>);
}

impl<T: bytemuck::Pod + num::Zero + bounding_box::Num> NdRoiZeroPadded<T, ndarray::Ix2>
    for ndarray::Array2<T>
{
    fn roi_zero_padded(
        &self,
        original: Aabb2<usize>,
        padded: Aabb2<usize>,
    ) -> (Aabb2<usize>, ndarray::Array2<T>) {
        // The co-ordinates of both the original and the padded bounding boxes must be contained in
        // self or it will panic

        let self_bbox = Aabb2::from_xywh(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(&original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(&padded) {
            panic!("padded bounding box is not contained in self");
        }

        let padded_top_left = padded.min_vertex();
        let original_roi_in_padded = original.move_origin(padded_top_left);

        use bounding_box::roi::Roi;
        let original_segment = self.roi(original).expect("original roi should be valid");
        let mut padded_segment = ndarray::Array2::<T>::zeros((padded.height(), padded.width()));

        padded_segment
            .roi_mut(original_roi_in_padded)
            .expect("UNEXPECTED: original_roi_in_padded should be valid")
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

impl<T: bytemuck::Pod + num::Zero + bounding_box::Num> NdRoiZeroPadded<T, ndarray::Ix3>
    for ndarray::Array3<T>
{
    fn roi_zero_padded(
        &self,
        original: Aabb2<usize>,
        padded: Aabb2<usize>,
    ) -> (Aabb2<usize>, ndarray::Array3<T>) {
        let self_bbox = Aabb2::from_xywh(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(&original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(&padded) {
            panic!("padded bounding box is not contained in self");
        }

        let padded_top_left = padded.min_vertex();
        let original_roi_in_padded = original.move_origin(padded_top_left);

        let original_segment = self.roi(original).expect("original roi should be valid");
        // let mut padded_segment = bbox_zeros_ndarray_3d::<T>(&padded, self.len_of(ndarray::Axis(2)));
        let mut padded_segment = ndarray::Array3::<T>::zeros((
            padded.height(),
            padded.width(),
            self.len_of(ndarray::Axis(2)),
        ));
        padded_segment
            .roi_mut(original_roi_in_padded)
            .expect("original_roi_in_padded should be valid")
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

#[test]
fn test_roi_zero_padded() {
    let arr = ndarray::Array2::<u8>::ones((10, 10));
    let original = Aabb2::from_xywh(1.0, 1.0, 3.0, 3.0);
    let clamp = Aabb2::from_xywh(0.0, 0.0, 10.0, 10.0);
    let padded = original.padding(2.0).clamp(&clamp).unwrap();
    let (padded_result, padded_segment) = arr.roi_zero_padded(original.cast(), padded.cast());
    assert_eq!(padded_result, bounding_box::Aabb2::from_xywh(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6]);
}

#[test]
pub fn bbox_clamp_failure_preload() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let og = Aabb2::from_xywh(475.0, 79.625, 37.0, 282.15);
    let clamp = Aabb2::from_xywh(0.0, 0.0, 512.0, 512.0);
    use ::tap::*;
    let padded = og
        .tap(|bbox| {
            println!("Unscaled bbox: {:?}", bbox);
        })
        .scale(nalgebra::Vector2::new(1.2, 1.2))
        .tap(|bbox| {
            println!("Scaled bbox: {:?}", bbox);
        })
        .clamp(&clamp)
        .tap(|bbox| {
            println!("Clamped bbox: {:?}", bbox);
        })
        .unwrap();
    let (_bbox, _segment) = segment_mask.roi_zero_padded(og.cast(), padded.cast());
}

#[test]
pub fn bbox_clamp_failure_preload_2() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let bbox = Aabb2::from_xywh(354.25, 98.5, 116.25, 413.5);
    // let padded = bounding_box::Aabb2::from_xywh(342.625, 57.150000000000006, 139.5, 454.85);
    let clamp = Aabb2::from_xywh(0.0, 0.0, 512.0, 512.0);
    let padded = bbox
        .scale(nalgebra::Vector2::new(1.2, 1.2))
        .clamp(&clamp)
        .unwrap();
    let (_bbox, _segment) = segment_mask.roi_zero_padded(bbox.cast(), padded.cast());
}

#[test]
fn test_roi_zero_padded_3d() {
    let arr = ndarray::Array3::<u8>::ones((10, 10, 3));
    let original = Aabb2::from_xywh(1.0, 1.0, 3.0, 3.0);
    let clamp = Aabb2::from_xywh(0.0, 0.0, 10.0, 10.0);
    let padded = original.padding(2.0).clamp(&clamp).unwrap();
    let (padded_result, padded_segment) = arr.roi_zero_padded(original.cast(), padded.cast());
    assert_eq!(padded_result, bounding_box::Aabb2::from_xywh(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6, 3]);
}
