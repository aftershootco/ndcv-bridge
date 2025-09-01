pub trait NdRoi<T, D>: seal::Sealed {
    fn roi(&self, rect: bbox::BBox<usize>) -> ndarray::ArrayView<T, D>;
}

pub trait NdRoiMut<T, D>: seal::Sealed {
    fn roi_mut(&mut self, rect: bbox::BBox<usize>) -> ndarray::ArrayViewMut<T, D>;
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
    fn roi(&self, rect: bbox::BBox<usize>) -> ndarray::ArrayView3<T> {
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
    fn roi_mut(&mut self, rect: bbox::BBox<usize>) -> ndarray::ArrayViewMut3<T> {
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
    fn roi(&self, rect: bbox::BBox<usize>) -> ndarray::ArrayView2<T> {
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
    fn roi_mut(&mut self, rect: bbox::BBox<usize>) -> ndarray::ArrayViewMut2<T> {
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
    let rect = bbox::BBox::new(1, 1, 3, 3);
    let roi = arr.roi(rect);
    assert_eq!(roi.shape(), &[3, 3, 3]);
}

#[test]
fn test_roi_2d() {
    let arr = ndarray::Array2::<u8>::zeros((10, 10));
    let rect = bbox::BBox::new(1, 1, 3, 3);
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
pub trait NdRoiZeroPadded<T, D: ndarray::Dimension>: seal::Sealed {
    fn roi_zero_padded(
        &self,
        original: bbox::BBox<usize>,
        padded: bbox::BBox<usize>,
    ) -> (bbox::BBox<usize>, ndarray::Array<T, D>);
}

impl<T: bytemuck::Pod + num::Zero> NdRoiZeroPadded<T, ndarray::Ix2> for ndarray::Array2<T> {
    fn roi_zero_padded(
        &self,
        original: bbox::BBox<usize>,
        padded: bbox::BBox<usize>,
    ) -> (bbox::BBox<usize>, ndarray::Array2<T>) {
        // The co-ordinates of both the original and the padded bounding boxes must be contained in
        // self or it will panic

        let self_bbox = bbox::BBox::new(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(padded) {
            panic!("padded bounding box is not contained in self");
        }

        let original_roi_in_padded =
            original.with_top_left(original.top_left().with_origin(padded.top_left()));
        let original_segment = self.roi(original);
        let mut padded_segment = padded.zeros_ndarray_2d::<T>();
        padded_segment
            .roi_mut(original_roi_in_padded)
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

impl<T: bytemuck::Pod + num::Zero> NdRoiZeroPadded<T, ndarray::Ix3> for ndarray::Array3<T> {
    fn roi_zero_padded(
        &self,
        original: bbox::BBox<usize>,
        padded: bbox::BBox<usize>,
    ) -> (bbox::BBox<usize>, ndarray::Array3<T>) {
        let self_bbox = bbox::BBox::new(0, 0, self.shape()[1], self.shape()[0]);
        if !self_bbox.contains_bbox(original) {
            panic!("original bounding box is not contained in self");
        }
        if !self_bbox.contains_bbox(padded) {
            panic!("padded bounding box is not contained in self");
        }

        let original_roi_in_padded =
            original.with_top_left(original.top_left().with_origin(padded.top_left()));
        let original_segment = self.roi(original);
        let mut padded_segment = padded.zeros_ndarray_3d::<T>(self.len_of(ndarray::Axis(2)));
        padded_segment
            .roi_mut(original_roi_in_padded)
            .assign(&original_segment);

        (padded, padded_segment)
    }
}

#[test]
fn test_roi_zero_padded() {
    let arr = ndarray::Array2::<u8>::ones((10, 10));
    let original = bbox::BBox::new(1, 1, 3, 3);
    let clamp = bbox::BBox::new(0, 0, 10, 10);
    let padded = original.padding(2).clamp_box(clamp);
    let (padded, padded_segment) = arr.roi_zero_padded(original.cast(), padded.cast());
    assert_eq!(padded, bbox::BBox::new(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6]);
}

#[test]
pub fn bbox_clamp_failure_preload() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let og = bbox::BBox::new(475.0, 79.625, 37.0, 282.15);
    let clamp = bbox::BBox::new(0.0, 0.0, 512.0, 512.0);
    let padded = og.scale(1.2).clamp_box(clamp);
    let padded = padded.round();
    let (_bbox, _segment) = segment_mask.roi_zero_padded(og.cast(), padded.cast());
}

#[test]
pub fn bbox_clamp_failure_preload_2() {
    let segment_mask = ndarray::Array2::<u8>::zeros((512, 512));
    let bbox = bbox::BBox::new(354.25, 98.5, 116.25, 413.5);
    // let padded = bbox::BBox::new(342.625, 57.150000000000006, 139.5, 454.85);
    let clamp = bbox::BBox::new(0.0, 0.0, 512.0, 512.0);
    let padded = bbox.scale(1.2).clamp_box(clamp);
    let padded = padded.round();
    let (_bbox, _segment) = segment_mask.roi_zero_padded(bbox.round().cast(), padded.cast());
}

#[test]
fn test_roi_zero_padded_3d() {
    let arr = ndarray::Array3::<u8>::ones((10, 10, 3));
    let original = bbox::BBox::new(1, 1, 3, 3);
    let clamp = bbox::BBox::new(0, 0, 10, 10);
    let padded = original.padding(2).clamp_box(clamp);
    let (padded, padded_segment) = arr.roi_zero_padded(original.cast(), padded.cast());
    assert_eq!(padded, bbox::BBox::new(0, 0, 6, 6));
    assert_eq!(padded_segment.shape(), &[6, 6, 3]);
}
