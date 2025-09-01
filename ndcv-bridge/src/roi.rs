pub trait NdRoi<T, D>: seal::Sealed {
    fn roi(&self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayView<'_, T, D>;
}

pub trait NdRoiMut<T, D>: seal::Sealed {
    fn roi_mut(&mut self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayViewMut<'_, T, D>;
}

mod seal {
    use ndarray::{Ix2, Ix3};
    pub trait Sealed {}
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> Sealed for ndarray::ArrayBase<S, Ix2> {}
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> Sealed for ndarray::ArrayBase<S, Ix3> {}
}

impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> NdRoi<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn roi(&self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayView3<'_, T> {
        let y1 = rect.y1();
        let y2 = rect.y2();
        let x1 = rect.x1();
        let x2 = rect.x2();
        self.slice(ndarray::s![y1..y2, x1..x2, ..])
    }
}

impl<T: bytemuck::Pod, S: ndarray::DataMut<Elem = T>> NdRoiMut<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn roi_mut(&mut self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayViewMut3<'_, T> {
        let y1 = rect.y1();
        let y2 = rect.y2();
        let x1 = rect.x1();
        let x2 = rect.x2();
        self.slice_mut(ndarray::s![y1..y2, x1..x2, ..])
    }
}

impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> NdRoi<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn roi(&self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayView2<'_, T> {
        let y1 = rect.y1();
        let y2 = rect.y2();
        let x1 = rect.x1();
        let x2 = rect.x2();
        self.slice(ndarray::s![y1..y2, x1..x2])
    }
}

impl<T: bytemuck::Pod, S: ndarray::DataMut<Elem = T>> NdRoiMut<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn roi_mut(&mut self, rect: bounding_box::Aabb2<usize>) -> ndarray::ArrayViewMut2<'_, T> {
        let y1 = rect.y1();
        let y2 = rect.y2();
        let x1 = rect.x1();
        let x2 = rect.x2();
        self.slice_mut(ndarray::s![y1..y2, x1..x2])
    }
}

#[test]
fn test_roi() {
    let arr = ndarray::Array3::<u8>::zeros((10, 10, 3));
    let rect = bounding_box::Aabb2::from_xywh(1, 1, 3, 3);
    let roi = arr.roi(rect);
    assert_eq!(roi.shape(), &[3, 3, 3]);
}

#[test]
fn test_roi_2d() {
    let arr = ndarray::Array2::<u8>::zeros((10, 10));
    let rect = bounding_box::Aabb2::from_xywh(1, 1, 3, 3);
    let roi = arr.roi(rect);
    assert_eq!(roi.shape(), &[3, 3]);
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
// Helper functions for missing methods from old bbox crate
fn bbox_top_left_usize(bbox: &bounding_box::Aabb2<usize>) -> (usize, usize) {
    (bbox.x1(), bbox.y1())
}

fn bbox_with_top_left_usize(
    bbox: &bounding_box::Aabb2<usize>,
    x: usize,
    y: usize,
) -> bounding_box::Aabb2<usize> {
    let width = bbox.x2() - bbox.x1();
    let height = bbox.y2() - bbox.y1();
    bounding_box::Aabb2::from_xywh(x, y, width, height)
}

fn bbox_with_origin_usize(point: (usize, usize), origin: (usize, usize)) -> (usize, usize) {
    (point.0 - origin.0, point.1 - origin.1)
}

fn bbox_zeros_ndarray_2d<T: num::Zero + Copy>(
    bbox: &bounding_box::Aabb2<usize>,
) -> ndarray::Array2<T> {
    let width = bbox.x2() - bbox.x1();
    let height = bbox.y2() - bbox.y1();
    ndarray::Array2::<T>::zeros((height, width))
}

fn bbox_zeros_ndarray_3d<T: num::Zero + Copy>(
    bbox: &bounding_box::Aabb2<usize>,
    channels: usize,
) -> ndarray::Array3<T> {
    let width = bbox.x2() - bbox.x1();
    let height = bbox.y2() - bbox.y1();
    ndarray::Array3::<T>::zeros((height, width, channels))
}

fn bbox_round_f64(bbox: &bounding_box::Aabb2<f64>) -> bounding_box::Aabb2<f64> {
    let x1 = bbox.x1().round();
    let y1 = bbox.y1().round();
    let x2 = bbox.x2().round();
    let y2 = bbox.y2().round();
    bounding_box::Aabb2::from_x1y1x2y2(x1, y1, x2, y2)
}

fn bbox_cast_f64_to_usize(bbox: &bounding_box::Aabb2<f64>) -> bounding_box::Aabb2<usize> {
    let x1 = bbox.x1() as usize;
    let y1 = bbox.y1() as usize;
    let x2 = bbox.x2() as usize;
    let y2 = bbox.y2() as usize;
    bounding_box::Aabb2::from_x1y1x2y2(x1, y1, x2, y2)
}

pub trait NdRoiZeroPadded<T, D: ndarray::Dimension>: seal::Sealed {
    fn roi_zero_padded(
        &self,
        original: bounding_box::Aabb2<usize>,
        padded: bounding_box::Aabb2<usize>,
    ) -> (bounding_box::Aabb2<usize>, ndarray::Array<T, D>);
}

impl<T: bytemuck::Pod + num::Zero> NdRoiZeroPadded<T, ndarray::Ix2> for ndarray::Array2<T> {
    fn roi_zero_padded(
        &self,
        original: bounding_box::Aabb2<usize>,
        padded: bounding_box::Aabb2<usize>,
    ) -> (bounding_box::Aabb2<usize>, ndarray::Array2<T>) {
        // The co-ordinates of both the original and the padded bounding boxes must be contained in
        // self or it will panic

        let self_bbox = bounding_box::Aabb2::from_xywh(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(&original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(&padded) {
            panic!("padded bounding box is not contained in self");
        }

        let original_top_left = bbox_top_left_usize(&original);
        let padded_top_left = bbox_top_left_usize(&padded);
        let origin_offset = bbox_with_origin_usize(original_top_left, padded_top_left);
        let original_roi_in_padded =
            bbox_with_top_left_usize(&original, origin_offset.0, origin_offset.1);

        let original_segment = self.roi(original);
        let mut padded_segment = bbox_zeros_ndarray_2d::<T>(&padded);
        padded_segment
            .roi_mut(original_roi_in_padded)
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

impl<T: bytemuck::Pod + num::Zero> NdRoiZeroPadded<T, ndarray::Ix3> for ndarray::Array3<T> {
    fn roi_zero_padded(
        &self,
        original: bounding_box::Aabb2<usize>,
        padded: bounding_box::Aabb2<usize>,
    ) -> (bounding_box::Aabb2<usize>, ndarray::Array3<T>) {
        let self_bbox = bounding_box::Aabb2::from_xywh(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(&original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(&padded) {
            panic!("padded bounding box is not contained in self");
        }

        let original_top_left = bbox_top_left_usize(&original);
        let padded_top_left = bbox_top_left_usize(&padded);
        let origin_offset = bbox_with_origin_usize(original_top_left, padded_top_left);
        let original_roi_in_padded =
            bbox_with_top_left_usize(&original, origin_offset.0, origin_offset.1);

        let original_segment = self.roi(original);
        let mut padded_segment = bbox_zeros_ndarray_3d::<T>(&padded, self.len_of(ndarray::Axis(2)));
        padded_segment
            .roi_mut(original_roi_in_padded)
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

#[test]
fn test_roi_zero_padded() {
    let arr = ndarray::Array2::<u8>::ones((10, 10));
    let original = bounding_box::Aabb2::from_xywh(1.0, 1.0, 3.0, 3.0);
    let clamp = bounding_box::Aabb2::from_xywh(0.0, 0.0, 10.0, 10.0);
    let padded = original.padding(2.0).clamp(&clamp).unwrap();
    let padded_cast = bbox_cast_f64_to_usize(&padded);
    let original_cast = bbox_cast_f64_to_usize(&original);
    let (padded_result, padded_segment) = arr.roi_zero_padded(original_cast, padded_cast);
    assert_eq!(padded_result, bounding_box::Aabb2::from_xywh(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6]);
}

#[test]
pub fn bbox_clamp_failure_preload() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let og = bounding_box::Aabb2::from_xywh(475.0, 79.625, 37.0, 282.15);
    let clamp = bounding_box::Aabb2::from_xywh(0.0, 0.0, 512.0, 512.0);
    let padded = og
        .scale(nalgebra::Vector2::new(1.2, 1.2))
        .clamp(&clamp)
        .unwrap();
    let padded = bbox_round_f64(&padded);
    let og_cast = bbox_cast_f64_to_usize(&bbox_round_f64(&og));
    let padded_cast = bbox_cast_f64_to_usize(&padded);
    let (_bbox, _segment) = segment_mask.roi_zero_padded(og_cast, padded_cast);
}

#[test]
pub fn bbox_clamp_failure_preload_2() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let bbox = bounding_box::Aabb2::from_xywh(354.25, 98.5, 116.25, 413.5);
    // let padded = bounding_box::Aabb2::from_xywh(342.625, 57.150000000000006, 139.5, 454.85);
    let clamp = bounding_box::Aabb2::from_xywh(0.0, 0.0, 512.0, 512.0);
    let padded = bbox
        .scale(nalgebra::Vector2::new(1.2, 1.2))
        .clamp(&clamp)
        .unwrap();
    let padded = bbox_round_f64(&padded);
    let bbox_cast = bbox_cast_f64_to_usize(&bbox_round_f64(&bbox));
    let padded_cast = bbox_cast_f64_to_usize(&padded);
    let (_bbox, _segment) = segment_mask.roi_zero_padded(bbox_cast, padded_cast);
}

#[test]
fn test_roi_zero_padded_3d() {
    let arr = ndarray::Array3::<u8>::ones((10, 10, 3));
    let original = bounding_box::Aabb2::from_xywh(1.0, 1.0, 3.0, 3.0);
    let clamp = bounding_box::Aabb2::from_xywh(0.0, 0.0, 10.0, 10.0);
    let padded = original.padding(2.0).clamp(&clamp).unwrap();
    let padded_cast = bbox_cast_f64_to_usize(&padded);
    let original_cast = bbox_cast_f64_to_usize(&original);
    let (padded_result, padded_segment) = arr.roi_zero_padded(original_cast, padded_cast);
    assert_eq!(padded_result, bounding_box::Aabb2::from_xywh(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6, 3]);
}
