use crate::*;
use ndarray::{Array3, ArrayView3, ArrayViewMut3};
/// A trait that extracts a region of interest from an image
pub trait Roi<'a, Output> {
    type Error;
    fn roi(&'a self, aabb: Aabb2<usize>) -> Result<Output, Self::Error>;
}

pub trait RoiMut<'a, Output> {
    type Error;
    fn roi_mut(&'a mut self, aabb: Aabb2<usize>) -> Result<Output, Self::Error>;
}

pub trait MultiRoi<'a, Output> {
    type Error;
    fn multi_roi(&'a self, aabbs: &[Aabb2<usize>]) -> Result<Output, Self::Error>;
}

#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum RoiError {
    #[error("Region of intereset is out of bounds")]
    RoiOutOfBounds,
}

impl<'a, T: Num> Roi<'a, ArrayView3<'a, T>> for Array3<T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: Aabb2<usize>) -> Result<ArrayView3<'a, T>, Self::Error> {
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if x1 >= x2 || y1 >= y2 || x2 > self.shape()[1] || y2 > self.shape()[0] {
            return Err(RoiError::RoiOutOfBounds);
        }
        Ok(self.slice(ndarray::s![y1..y2, x1..x2, ..]))
    }
}

impl<'a, T: Num> RoiMut<'a, ArrayViewMut3<'a, T>> for Array3<T> {
    type Error = RoiError;
    fn roi_mut(&'a mut self, aabb: Aabb2<usize>) -> Result<ArrayViewMut3<'a, T>, Self::Error> {
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if x1 > x2 || y1 > y2 || x2 > self.shape()[1] || y2 > self.shape()[0] {
            return Err(RoiError::RoiOutOfBounds);
        }
        Ok(self.slice_mut(ndarray::s![y1..y2, x1..x2, ..]))
    }
}

impl<'a, 'b, T: Num> Roi<'a, ArrayView3<'b, T>> for ArrayView3<'b, T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: Aabb2<usize>) -> Result<ArrayView3<'b, T>, Self::Error> {
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if x1 >= x2 || y1 >= y2 || x2 > self.shape()[1] || y2 > self.shape()[0] {
            return Err(RoiError::RoiOutOfBounds);
        }
        Ok(self.slice_move(ndarray::s![y1..y2, x1..x2, ..]))
    }
}
// impl<'a, 'b, T: Num> Roi<'a, ArrayViewMut3<'b, T>> for ArrayViewMut3<'b, T> {
//     type Error = RoiError;
//     fn roi(&'a self, aabb: Aabb2<usize>) -> Result<ArrayViewMut3<'b, T>, Self::Error> {
//         let x1 = aabb.x1();
//         let x2 = aabb.x2();
//         let y1 = aabb.y1();
//         let y2 = aabb.y2();
//         if x1 >= x2 || y1 >= y2 || x2 > self.shape()[1] || y2 > self.shape()[0] {
//             return Err(RoiError::RoiOutOfBounds);
//         }
//         Ok(self.slice(ndarray::s![y1..y2, x1..x2, ..]))
//     }
// }

impl<'a, 'b: 'a, T: Num> RoiMut<'a, ArrayViewMut3<'a, T>> for ArrayViewMut3<'b, T> {
    type Error = RoiError;
    fn roi_mut(&'a mut self, aabb: Aabb2<usize>) -> Result<ArrayViewMut3<'a, T>, Self::Error> {
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if x1 >= x2 || y1 >= y2 || x2 > self.shape()[1] || y2 > self.shape()[0] {
            return Err(RoiError::RoiOutOfBounds);
        }
        let out: ArrayViewMut3<'a, T> = self.slice_mut(ndarray::s![y1..y2, x1..x2, ..]);
        Ok(out)
    }
}

#[test]
pub fn reborrow_test() {
    let ndarray = ndarray::Array::from_shape_vec((5, 5, 5), vec![33; 5 * 5 * 5]).unwrap();
    let aabb = Aabb2::from_x1y1x2y2(2, 3, 4, 5);
    let y = {
        let view = ndarray.view();
        view.roi(aabb).unwrap()
    };
    dbg!(y);
}

impl<'a> MultiRoi<'a, Vec<ArrayView3<'a, u8>>> for Array3<u8> {
    type Error = RoiError;
    fn multi_roi(&'a self, aabbs: &[Aabb2<usize>]) -> Result<Vec<ArrayView3<'a, u8>>, Self::Error> {
        let (height, width, _channels) = self.dim();
        let outer_aabb = Aabb2::from_x1y1x2y2(0, 0, width, height);
        aabbs
            .iter()
            .map(|aabb| {
                let slice_arg =
                    bbox_to_slice_arg(aabb.clamp(&outer_aabb).ok_or(RoiError::RoiOutOfBounds)?);
                Ok(self.slice(slice_arg))
            })
            .collect::<Result<Vec<_>, RoiError>>()
    }
}

impl<'a, 'b> MultiRoi<'a, Vec<ArrayView3<'b, u8>>> for ArrayView3<'b, u8> {
    type Error = RoiError;
    fn multi_roi(&'a self, aabbs: &[Aabb2<usize>]) -> Result<Vec<ArrayView3<'b, u8>>, Self::Error> {
        let (height, width, _channels) = self.dim();
        let outer_aabb = Aabb2::from_x1y1x2y2(0, 0, width, height);
        aabbs
            .iter()
            .map(|aabb| {
                let slice_arg =
                    bbox_to_slice_arg(aabb.clamp(&outer_aabb).ok_or(RoiError::RoiOutOfBounds)?);
                Ok(self.slice_move(slice_arg))
            })
            .collect::<Result<Vec<_>, RoiError>>()
    }
}

fn bbox_to_slice_arg(
    aabb: Aabb2<usize>,
) -> ndarray::SliceInfo<[ndarray::SliceInfoElem; 3], ndarray::Ix3, ndarray::Ix3> {
    // This function should convert the bounding box to a slice argument
    // For now, we will return a dummy value
    let x1 = aabb.x1();
    let x2 = aabb.x2();
    let y1 = aabb.y1();
    let y2 = aabb.y2();
    ndarray::s![y1..y2, x1..x2, ..]
}
