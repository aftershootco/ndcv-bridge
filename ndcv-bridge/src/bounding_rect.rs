//! Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image.
//! The function calculates and returns the minimal up-right bounding rectangle for the specified point set or non-zero pixels of gray-scale image.
use crate::{NdAsImage, prelude_::*};
pub trait BoundingRect: seal::SealedInternal {
    fn bounding_rect(&self) -> Result<bounding_box::Aabb2<i32>, NdCvError>;
}

mod seal {
    pub trait SealedInternal {}
    impl<T, S: ndarray::Data<Elem = T>> SealedInternal for ndarray::ArrayBase<S, ndarray::Ix2> {}
}

impl<S: ndarray::Data<Elem = u8>> BoundingRect for ndarray::ArrayBase<S, ndarray::Ix2> {
    fn bounding_rect(&self) -> Result<bounding_box::Aabb2<i32>, NdCvError> {
        let mat = self.as_image_mat()?;
        let rect = opencv::imgproc::bounding_rect(mat.as_ref()).change_context(NdCvError)?;
        Ok(bounding_box::Aabb2::from_xywh(
            rect.x,
            rect.y,
            rect.width,
            rect.height,
        ))
    }
}

#[test]
fn test_bounding_rect_empty() {
    let arr = ndarray::Array2::<u8>::zeros((10, 10));
    let rect = arr.bounding_rect().unwrap();
    assert_eq!(rect, bounding_box::Aabb2::from_xywh(0, 0, 0, 0));
}

#[test]
fn test_bounding_rect_valued() {
    let mut arr = ndarray::Array2::<u8>::zeros((10, 10));
    crate::NdRoiMut::roi_mut(&mut arr, bounding_box::Aabb2::from_xywh(1, 1, 3, 3)).fill(1);
    let rect = arr.bounding_rect().unwrap();
    assert_eq!(rect, bounding_box::Aabb2::from_xywh(1, 1, 3, 3));
}

#[test]
fn test_bounding_rect_complex() {
    let mut arr = ndarray::Array2::<u8>::zeros((10, 10));
    crate::NdRoiMut::roi_mut(&mut arr, bounding_box::Aabb2::from_xywh(1, 3, 3, 3)).fill(1);
    crate::NdRoiMut::roi_mut(&mut arr, bounding_box::Aabb2::from_xywh(2, 3, 3, 5)).fill(5);
    let rect = arr.bounding_rect().unwrap();
    assert_eq!(rect, bounding_box::Aabb2::from_xywh(1, 3, 4, 5));
}
