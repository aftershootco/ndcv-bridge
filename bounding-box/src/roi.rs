use crate::*;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3};
/// A trait that extracts a region of interest from an image
pub trait Roi<'a, Output> {
    type Error;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<Output, Self::Error>;
}

pub trait RoiMut<'a, Output> {
    type Error;
    fn roi_mut(&'a mut self, aabb: impl Into<Aabb2<usize>>) -> Result<Output, Self::Error>;
}

pub trait MultiRoi<'a, Output> {
    type Error;
    fn multi_roi(&'a self, aabbs: &[Aabb2<usize>]) -> Result<Output, Self::Error>;
}

#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum RoiError {
    #[error("Region of interest is out of bounds: Max possible {max:?}, Got {got:?}")]
    RoiOutOfBounds {
        max: Aabb2<usize>,
        got: Aabb2<usize>,
    },
    #[error("Invalid region of interest: {got:?}")]
    InvalidRoi { got: Aabb2<usize> },
}

impl<'a, T: Num> Roi<'a, ArrayView3<'a, T>> for Array3<T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView3<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice(ndarray::s![y1..y2, x1..x2, ..]))
    }
}

impl<'a, T: Num> Roi<'a, ArrayView2<'a, T>> for Array2<T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView2<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice(ndarray::s![y1..y2, x1..x2]))
    }
}

impl<'a, T: Num> RoiMut<'a, ArrayViewMut3<'a, T>> for Array3<T> {
    type Error = RoiError;
    fn roi_mut(
        &'a mut self,
        aabb: impl Into<Aabb2<usize>>,
    ) -> Result<ArrayViewMut3<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice_mut(ndarray::s![y1..y2, x1..x2, ..]))
    }
}

impl<'a, T: Num> RoiMut<'a, ArrayViewMut2<'a, T>> for Array2<T> {
    type Error = RoiError;
    fn roi_mut(
        &'a mut self,
        aabb: impl Into<Aabb2<usize>>,
    ) -> Result<ArrayViewMut2<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice_mut(ndarray::s![y1..y2, x1..x2]))
    }
}

impl<'a, 'b, T: Num> Roi<'a, ArrayView3<'b, T>> for ArrayView3<'b, T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView3<'b, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice_move(ndarray::s![y1..y2, x1..x2, ..]))
    }
}

impl<'a, 'b, T: Num> Roi<'a, ArrayView2<'b, T>> for ArrayView2<'b, T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView2<'b, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        Ok(self.slice_move(ndarray::s![y1..y2, x1..x2]))
    }
}

impl<'a, 'b: 'a, T: Num> RoiMut<'a, ArrayViewMut3<'a, T>> for ArrayViewMut3<'b, T> {
    type Error = RoiError;
    fn roi_mut(
        &'a mut self,
        aabb: impl Into<Aabb2<usize>>,
    ) -> Result<ArrayViewMut3<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        let out: ArrayViewMut3<'a, T> = self.slice_mut(ndarray::s![y1..y2, x1..x2, ..]);
        Ok(out)
    }
}

impl<'a, 'b: 'a, T: Num> RoiMut<'a, ArrayViewMut2<'a, T>> for ArrayViewMut2<'b, T> {
    type Error = RoiError;
    fn roi_mut(
        &'a mut self,
        aabb: impl Into<Aabb2<usize>>,
    ) -> Result<ArrayViewMut2<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        let out: ArrayViewMut2<'a, T> = self.slice_mut(ndarray::s![y1..y2, x1..x2]);
        Ok(out)
    }
}

impl<'a, 'b: 'a, T: Num> Roi<'a, ArrayView2<'a, T>> for ArrayViewMut2<'b, T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView2<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        let out: ArrayView2<'a, T> = self.slice(ndarray::s![y1..y2, x1..x2]);
        Ok(out)
    }
}

impl<'a, 'b: 'a, T: Num> Roi<'a, ArrayView3<'a, T>> for ArrayViewMut3<'b, T> {
    type Error = RoiError;
    fn roi(&'a self, aabb: impl Into<Aabb2<usize>>) -> Result<ArrayView3<'a, T>, Self::Error> {
        let aabb = aabb.into();
        let x1 = aabb.x1();
        let x2 = aabb.x2();
        let y1 = aabb.y1();
        let y2 = aabb.y2();
        if !aabb.is_positive() {
            return Err(RoiError::InvalidRoi { got: aabb });
        }
        let max_aabb = Aabb2::from_x1y1x2y2(0, 0, self.shape()[1], self.shape()[0]);
        if !max_aabb.contains_bbox(&aabb) {
            return Err(RoiError::RoiOutOfBounds {
                max: max_aabb,
                got: aabb,
            });
        }
        let out: ArrayView3<'a, T> = self.slice(ndarray::s![y1..y2, x1..x2, ..]);
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
                    bbox_to_slice_arg(aabb.clamp(outer_aabb).ok_or(RoiError::RoiOutOfBounds {
                        max: outer_aabb,
                        got: *aabb,
                    })?);
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
                    bbox_to_slice_arg(aabb.clamp(outer_aabb).ok_or(RoiError::RoiOutOfBounds {
                        max: outer_aabb,
                        got: *aabb,
                    })?);
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

#[cfg(test)]
mod roi_tests {
    use super::*;

    // aabb (x1,y1)=(1,1), (x2,y2)=(3,4) -> slice rows 1..4, cols 1..3.
    fn aabb() -> Aabb2<usize> {
        Aabb2::from_x1y1x2y2(1, 1, 3, 4)
    }

    fn arr3() -> Array3<i32> {
        Array3::from_shape_fn((5, 5, 3), |(y, x, c)| (y * 100 + x * 10 + c) as i32)
    }

    fn arr2() -> Array2<i32> {
        Array2::from_shape_fn((5, 5), |(y, x)| (y * 100 + x * 10) as i32)
    }

    // Every impl below performs the same two bounds checks; deleting the `!` on
    // either one turns a valid ROI into an error, so a single successful call
    // that asserts on the returned view is enough to kill both mutants.

    #[test]
    fn roi_array3_valid() {
        let arr = arr3();
        let view = arr.roi(aabb()).unwrap();
        assert_eq!(view.dim(), (3, 2, 3));
        assert_eq!(view[[0, 0, 0]], arr[[1, 1, 0]]);
    }

    #[test]
    fn roi_array2_valid() {
        let arr = arr2();
        let view = arr.roi(aabb()).unwrap();
        assert_eq!(view.dim(), (3, 2));
        assert_eq!(view[[0, 0]], arr[[1, 1]]);
    }

    #[test]
    fn roi_mut_array3_valid() {
        let mut arr = arr3();
        let view = arr.roi_mut(aabb()).unwrap();
        assert_eq!(view.dim(), (3, 2, 3));
    }

    #[test]
    fn roi_mut_array2_valid() {
        let mut arr = arr2();
        let view = arr.roi_mut(aabb()).unwrap();
        assert_eq!(view.dim(), (3, 2));
    }

    #[test]
    fn roi_arrayview2_valid() {
        let arr = arr2();
        let view = arr.view();
        let sub = view.roi(aabb()).unwrap();
        assert_eq!(sub.dim(), (3, 2));
    }

    #[test]
    fn roi_mut_arrayviewmut3_valid() {
        let mut arr = arr3();
        let mut view = arr.view_mut();
        let sub = view.roi_mut(aabb()).unwrap();
        assert_eq!(sub.dim(), (3, 2, 3));
    }

    #[test]
    fn roi_mut_arrayviewmut2_valid() {
        let mut arr = arr2();
        let mut view = arr.view_mut();
        let sub = view.roi_mut(aabb()).unwrap();
        assert_eq!(sub.dim(), (3, 2));
    }

    #[test]
    fn roi_view2_from_arrayviewmut2_valid() {
        let mut arr = arr2();
        let view = arr.view_mut();
        let sub: ArrayView2<i32> = view.roi(aabb()).unwrap();
        assert_eq!(sub.dim(), (3, 2));
    }

    #[test]
    fn roi_view3_from_arrayviewmut3_valid() {
        let mut arr = arr3();
        let view = arr.view_mut();
        let sub: ArrayView3<i32> = view.roi(aabb()).unwrap();
        assert_eq!(sub.dim(), (3, 2, 3));
    }

    #[test]
    fn roi_out_of_bounds_errors() {
        let arr = arr3();
        let oob = Aabb2::from_x1y1x2y2(1, 1, 10, 10);
        assert!(matches!(
            arr.roi(oob),
            Err(RoiError::RoiOutOfBounds { .. })
        ));
    }

    // multi_roi is only implemented for u8 arrays.
    fn arr3_u8() -> Array3<u8> {
        Array3::from_shape_fn((5, 5, 3), |(y, x, c)| (y * 25 + x * 5 + c) as u8)
    }

    #[test]
    fn multi_roi_array3_returns_all_views() {
        let arr = arr3_u8();
        let aabbs = [
            Aabb2::from_x1y1x2y2(0, 0, 2, 2),
            Aabb2::from_x1y1x2y2(1, 1, 3, 3),
        ];
        let rois = arr.multi_roi(&aabbs).unwrap();
        assert_eq!(rois.len(), 2);
        assert_eq!(rois[0].dim(), (2, 2, 3));
    }

    #[test]
    fn multi_roi_arrayview3_returns_all_views() {
        let arr = arr3_u8();
        let view = arr.view();
        let aabbs = [
            Aabb2::from_x1y1x2y2(0, 0, 2, 2),
            Aabb2::from_x1y1x2y2(1, 1, 3, 3),
        ];
        let rois = view.multi_roi(&aabbs).unwrap();
        assert_eq!(rois.len(), 2);
        assert_eq!(rois[0].dim(), (2, 2, 3));
    }
}
