//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.find_contours.html>

#![deny(warnings)]

use crate::conversions::*;
use crate::prelude_::*;
use nalgebra::Point2;
use ndarray::*;

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ContourRetrievalMode {
    #[default]
    External = 0, // RETR_EXTERNAL
    List = 1,      // RETR_LIST
    CComp = 2,     // RETR_CCOMP
    Tree = 3,      // RETR_TREE
    FloodFill = 4, // RETR_FLOODFILL
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ContourApproximationMethod {
    #[default]
    None = 1, // CHAIN_APPROX_NONE
    Simple = 2,   // CHAIN_APPROX_SIMPLE
    Tc89L1 = 3,   // CHAIN_APPROX_TC89_L1
    Tc89Kcos = 4, // CHAIN_APPROX_TC89_KCOS
}

#[derive(Debug, Clone)]
pub struct ContourHierarchy {
    pub next: i32,
    pub previous: i32,
    pub first_child: i32,
    pub parent: i32,
}

#[derive(Debug, Clone)]
pub struct ContourResult {
    pub contours: Vec<Vec<Point2<i32>>>,
    pub hierarchy: Vec<ContourHierarchy>,
}

mod seal {
    pub trait Sealed {}
    impl Sealed for u8 {}
}

pub trait NdCvFindContours<T: bytemuck::Pod + seal::Sealed>:
    crate::image::NdImage + crate::conversions::NdAsImage<T, ndarray::Ix2>
{
    fn find_contours(
        &self,
        mode: ContourRetrievalMode,
        method: ContourApproximationMethod,
    ) -> Result<Vec<Vec<Point2<i32>>>, NdCvError>;

    fn find_contours_with_hierarchy(
        &self,
        mode: ContourRetrievalMode,
        method: ContourApproximationMethod,
    ) -> Result<ContourResult, NdCvError>;

    fn find_contours_def(&self) -> Result<Vec<Vec<Point2<i32>>>, NdCvError> {
        self.find_contours(
            ContourRetrievalMode::External,
            ContourApproximationMethod::Simple,
        )
    }

    fn find_contours_with_hierarchy_def(&self) -> Result<ContourResult, NdCvError> {
        self.find_contours_with_hierarchy(
            ContourRetrievalMode::External,
            ContourApproximationMethod::Simple,
        )
    }
}

pub trait NdCvContourArea<T: bytemuck::Pod> {
    fn contours_area(&self, oriented: bool) -> Result<f64, NdCvError>;

    fn contours_area_def(&self) -> Result<f64, NdCvError> {
        self.contours_area(false)
    }
}

impl<T: ndarray::RawData + ndarray::Data<Elem = u8>> NdCvFindContours<u8> for ArrayBase<T, Ix2> {
    fn find_contours(
        &self,
        mode: ContourRetrievalMode,
        method: ContourApproximationMethod,
    ) -> Result<Vec<Vec<Point2<i32>>>, NdCvError> {
        let cv_self = self.as_image_mat()?;
        let mut contours = opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();

        opencv::imgproc::find_contours(
            &*cv_self,
            &mut contours,
            mode as i32,
            method as i32,
            opencv::core::Point::new(0, 0),
        )
        .change_context(NdCvError)
        .attach("Failed to find contours")?;
        let mut result: Vec<Vec<Point2<i32>>> = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i).change_context(NdCvError)?;
            let points: Vec<Point2<i32>> =
                contour.iter().map(|pt| Point2::new(pt.x, pt.y)).collect();
            result.push(points);
        }

        Ok(result)
    }

    fn find_contours_with_hierarchy(
        &self,
        mode: ContourRetrievalMode,
        method: ContourApproximationMethod,
    ) -> Result<ContourResult, NdCvError> {
        let cv_self = self.as_image_mat()?;
        let mut contours = opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();
        let mut hierarchy = opencv::core::Vector::<opencv::core::Vec4i>::new();

        opencv::imgproc::find_contours_with_hierarchy(
            &*cv_self,
            &mut contours,
            &mut hierarchy,
            mode as i32,
            method as i32,
            opencv::core::Point::new(0, 0),
        )
        .change_context(NdCvError)
        .attach("Failed to find contours with hierarchy")?;
        let mut contour_list: Vec<Vec<Point2<i32>>> = Vec::new();

        for i in 0..contours.len() {
            let contour = contours.get(i).change_context(NdCvError)?;
            let points: Vec<Point2<i32>> =
                contour.iter().map(|pt| Point2::new(pt.x, pt.y)).collect();
            contour_list.push(points);
        }

        let mut hierarchy_list = Vec::new();
        for i in 0..hierarchy.len() {
            let h = hierarchy.get(i).change_context(NdCvError)?;
            hierarchy_list.push(ContourHierarchy {
                next: h[0],
                previous: h[1],
                first_child: h[2],
                parent: h[3],
            });
        }

        Ok(ContourResult {
            contours: contour_list,
            hierarchy: hierarchy_list,
        })
    }
}

impl<T> NdCvContourArea<T> for Vec<Point2<T>>
where
    T: bytemuck::Pod + num::traits::AsPrimitive<i32> + std::cmp::PartialEq + std::fmt::Debug + Copy,
{
    fn contours_area(&self, oriented: bool) -> Result<f64, NdCvError> {
        if self.is_empty() {
            return Ok(0.0);
        }

        let mut cv_contour: opencv::core::Vector<opencv::core::Point> = opencv::core::Vector::new();
        self.iter().for_each(|point| {
            cv_contour.push(opencv::core::Point::new(
                point.coords[0].as_(),
                point.coords[1].as_(),
            ));
        });

        opencv::imgproc::contour_area(&cv_contour, oriented)
            .change_context(NdCvError)
            .attach("Failed to calculate contour area")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn simple_binary_rect_image() -> Array2<u8> {
        let mut img = Array2::<u8>::zeros((10, 10));
        for i in 2..8 {
            for j in 3..7 {
                img[(i, j)] = 255;
            }
        }
        img
    }

    #[test]
    fn test_find_contours_external_simple() {
        let img = simple_binary_rect_image();
        let contours = img
            .find_contours(
                ContourRetrievalMode::External,
                ContourApproximationMethod::Simple,
            )
            .expect("Failed to find contours");
        assert_eq!(contours.len(), 1);
        assert!(contours[0].len() >= 4);
    }

    #[test]
    fn test_find_contours_with_hierarchy() {
        let img = simple_binary_rect_image();
        let res = img
            .find_contours_with_hierarchy(
                ContourRetrievalMode::External,
                ContourApproximationMethod::Simple,
            )
            .expect("Failed to find contours with hierarchy");
        assert_eq!(res.contours.len(), 1);
        assert_eq!(res.hierarchy.len(), 1);

        let h = &res.hierarchy[0];
        assert_eq!(h.parent, -1);
        assert_eq!(h.first_child, -1);
    }

    #[test]
    fn test_default_methods() {
        let img = simple_binary_rect_image();
        let contours = img.find_contours_def().unwrap();
        let res = img.find_contours_with_hierarchy_def().unwrap();
        assert_eq!(contours.len(), 1);
        assert_eq!(res.contours.len(), 1);
    }

    #[test]
    fn test_contour_area_calculation() {
        let img = simple_binary_rect_image();
        let contours = img.find_contours_def().unwrap();
        let expected_area = 15.;
        let area = contours[0].contours_area_def().unwrap();
        assert!(
            (area - expected_area).abs() < 1.0,
            "Area mismatch: got {area}, expected {expected_area}",
        );
    }

    #[test]
    fn test_empty_input_returns_no_contours() {
        let img = Array2::<u8>::zeros((10, 10));
        let contours = img.find_contours_def().unwrap();
        assert!(contours.is_empty());

        let res = img.find_contours_with_hierarchy_def().unwrap();
        assert!(res.contours.is_empty());
        assert!(res.hierarchy.is_empty());
    }

    #[test]
    fn test_contour_area_empty_contour() {
        let contour: Vec<Point2<i32>> = vec![];
        let area = contour.contours_area_def().unwrap();
        assert_eq!(area, 0.0);
    }
}
