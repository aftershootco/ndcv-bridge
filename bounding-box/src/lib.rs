#[cfg(feature = "compat-bbox")]
pub mod bbox;
pub mod draw;
pub mod nms;
pub mod roi;

use nalgebra::{Point, Point2, SVector, Vector2};
pub trait Num:
    num::Num
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
    + core::cmp::PartialOrd
    + core::cmp::PartialEq
    + nalgebra::SimdPartialOrd
    + nalgebra::SimdValue
    + Copy
    + core::fmt::Debug
    + 'static
{
}
impl<
    T: num::Num
        + core::ops::AddAssign
        + core::ops::SubAssign
        + core::ops::MulAssign
        + core::ops::DivAssign
        + core::cmp::PartialOrd
        + core::cmp::PartialEq
        + nalgebra::SimdPartialOrd
        + nalgebra::SimdValue
        + Copy
        + core::fmt::Debug
        + 'static,
> Num for T
{
}

/// An axis aligned bounding box in `D` dimensions, defined by the minimum vertex and a size vector.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AxisAlignedBoundingBox<T: Num, const D: usize> {
    /// The point of the bounding box closest to the origin
    point: Point<T, D>,
    /// The size of the bounding box in each dimension
    size: SVector<T, D>,
}

pub type Aabb<T, const D: usize> = AxisAlignedBoundingBox<T, D>;
pub type Aabb2<T> = AxisAlignedBoundingBox<T, 2>;
pub type Aabb3<T> = AxisAlignedBoundingBox<T, 3>;
pub type BBox<T, const D: usize> = AxisAlignedBoundingBox<T, D>;
pub type BBox2<T> = AxisAlignedBoundingBox<T, 2>;
pub type BBox3<T> = AxisAlignedBoundingBox<T, 3>;

impl<T: Num, const D: usize> AxisAlignedBoundingBox<T, D> {
    // Panics if max < min
    pub fn new(min_point: Point<T, D>, max_point: Point<T, D>) -> Self {
        if max_point >= min_point {
            Self::from_min_max_vertices(min_point, max_point)
        } else {
            panic!("max_point must be greater than or equal to min_point");
        }
    }
    pub fn try_new(min_point: Point<T, D>, max_point: Point<T, D>) -> Option<Self> {
        if max_point < min_point {
            return None;
        }
        Some(Self::from_min_max_vertices(min_point, max_point))
    }

    pub fn new_point_size(point: Point<T, D>, size: SVector<T, D>) -> Self {
        Self { point, size }
    }

    pub fn from_min_max_vertices(min: Point<T, D>, max: Point<T, D>) -> Self {
        let size = max - min;
        Self::new_point_size(min, SVector::from(size))
    }

    /// Only considers the points closest and furthest from origin
    /// Points which are rotated along in the z axis (in 2d) are not considered
    pub fn from_vertices(points: [Point<T, D>; 4]) -> Option<Self>
    where
        T: core::ops::SubAssign,
        T: PartialOrd,
    {
        // find the closest and farthest points from the origin
        let min = points
            .iter()
            .reduce(|acc, p| if acc > p { p } else { acc })?;
        let max = points
            .iter()
            .reduce(|acc, p| if acc < p { p } else { acc })?;
        Some(Self::from_min_max_vertices(*min, *max))
    }

    pub fn size(&self) -> SVector<T, D> {
        self.size
    }

    pub fn center(&self) -> Point<T, D>
    where
        T: core::ops::AddAssign,
        T: core::ops::DivAssign,
    {
        self.point + self.size / (T::one() + T::one())
    }

    pub fn padding(mut self, padding: T) -> Self
    where
        T: core::ops::AddAssign,
        T: core::ops::DivAssign,
        T: core::ops::SubAssign,
    {
        self.size.iter_mut().for_each(|s| {
            *s += padding;
        });
        let two = T::one() + T::one();
        self.point
            .coords
            .iter_mut()
            .for_each(|c| *c -= padding / two);
        self
    }

    pub fn translate(mut self, translation: SVector<T, D>) -> Self
    where
        T: core::ops::AddAssign,
    {
        self.point += translation;
        self
    }

    pub fn min_vertex(&self) -> Point<T, D> {
        self.point
    }

    pub fn move_to(mut self, point: Point<T, D>) -> Self
    where
        T: core::ops::SubAssign,
    {
        self.point = point;
        self
    }

    pub fn move_origin(mut self, origin: Point<T, D>) -> Self
    where
        T: core::ops::SubAssign,
    {
        self.point -= origin.coords;
        self
    }

    pub fn max_vertex(&self) -> Point<T, D>
    where
        T: core::ops::AddAssign,
    {
        self.point + self.size
    }

    pub fn contains_point(&self, point: &Point<T, D>) -> bool
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
    {
        let min = self.min_vertex();
        let max = self.max_vertex();

        *point >= min && *point <= max
    }

    pub fn scale(self, vector: SVector<T, D>) -> Self
    where
        T: core::ops::MulAssign,
        T: core::ops::DivAssign,
        T: core::ops::SubAssign,
    {
        let two = T::one() + T::one();
        let new_size = self.size.component_mul(&vector);
        let new_point = self.point.coords - new_size / two;
        Self {
            point: Point::from(new_point),
            size: new_size,
        }
    }

    pub fn contains_bbox(&self, other: &Self) -> bool
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
    {
        let self_min = self.min_vertex();
        let self_max = self.max_vertex();
        let other_min = other.min_vertex();
        let other_max = other.max_vertex();

        other_min >= self_min && other_max <= self_max
    }

    pub fn clamp(&self, other: &Self) -> Option<Self>
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
        T: nalgebra::SimdPartialOrd,
        T: nalgebra::SimdValue,
    {
        if other.contains_bbox(self) {
            return Some(*self);
        }
        self.intersection(other)
    }

    pub fn component_clamp(&self, min: T, max: T) -> Self
    where
        T: PartialOrd,
    {
        let mut this = *self;
        this.point.iter_mut().for_each(|x| {
            *x = nalgebra::clamp(*x, min, max);
        });
        this.size.iter_mut().for_each(|x| {
            *x = nalgebra::clamp(*x, min, max);
        });
        this
    }

    pub fn merge(&self, other: &Self) -> Self
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
        T: nalgebra::SimdValue,
        T: nalgebra::SimdPartialOrd,
    {
        let min = self.min_vertex().inf(&other.min_vertex());
        let max = self.min_vertex().sup(&other.max_vertex());
        Self::new(min, max)
    }

    pub fn union(&self, other: &Self) -> T
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: core::ops::MulAssign,
        T: PartialOrd,
        T: nalgebra::SimdValue,
        T: nalgebra::SimdPartialOrd,
    {
        self.measure() + other.measure()
            - Self::intersection(self, other)
                .map(|x| x.measure())
                .unwrap_or(T::zero())
    }

    pub fn intersection(&self, other: &Self) -> Option<Self>
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
        T: nalgebra::SimdPartialOrd,
        T: nalgebra::SimdValue,
    {
        let inter_min = self.min_vertex().sup(&other.min_vertex());
        let inter_max = self.max_vertex().inf(&other.max_vertex());
        Self::try_new(inter_min, inter_max)
    }

    pub fn denormalize(&self, factor: nalgebra::SVector<T, D>) -> Self
    where
        T: core::ops::MulAssign,
        T: core::ops::AddAssign,
        // nalgebra::constraint::ShapeConstraint:
        //     nalgebra::constraint::DimEq<nalgebra::Const<D>, nalgebra::Const<D>>,
    {
        Self {
            point: (self.point.coords.component_mul(&factor)).into(),
            size: self.size.component_mul(&factor),
        }
    }

    pub fn try_cast<T2>(&self) -> Option<Aabb<T2, D>>
    where
        // T: num::NumCast,
        T2: Num + simba::scalar::SubsetOf<T>,
    {
        Some(Aabb {
            point: Point::from(self.point.coords.try_cast::<T2>()?),
            size: self.size.try_cast::<T2>()?,
        })
    }

    pub fn cast<T2>(&self) -> Aabb<T2, D>
    where
        // T: num::NumCast,
        T2: Num + simba::scalar::SubsetOf<T>,
    {
        Self::try_cast(self)
            .expect(format!("Failed to cast to Aabb<{}>", std::any::type_name::<T2>()).as_str())
    }

    // pub fn as_<T2>(&self) -> Option<Aabb<T2, D>>
    // where
    //     T2: Num + simba::scalar::SubsetOf<T>,
    // {
    //     Some(Aabb {
    //         point: Point::from(self.point.coords.as_()),
    //         size: self.size.as_(),
    //     })
    // }
    pub fn measure(&self) -> T
    where
        T: core::ops::MulAssign,
    {
        self.size.product()
    }

    pub fn iou(&self, other: &Self) -> T
    where
        T: core::ops::AddAssign,
        T: core::ops::SubAssign,
        T: PartialOrd,
        T: nalgebra::SimdPartialOrd,
        T: nalgebra::SimdValue,
        T: core::ops::MulAssign,
    {
        let lhs_min = self.min_vertex();
        let lhs_max = self.max_vertex();
        let rhs_min = other.min_vertex();
        let rhs_max = other.max_vertex();

        let inter_min = lhs_min.sup(&rhs_min);
        let inter_max = lhs_max.inf(&rhs_max);
        if inter_max >= inter_min {
            let intersection = Aabb::new(inter_min, inter_max).measure();
            intersection / (self.measure() + other.measure() - intersection)
        } else {
            T::zero()
        }
    }

    pub fn is_positive(&self) -> bool
    where
        T: PartialOrd,
        T: core::ops::AddAssign,
    {
        self.point >= Point::origin()
    }
}

#[test]
fn test_is_positive() {
    let bbox = Aabb2::from_xywh(1, 1, 2, 2);
    assert!(bbox.is_positive());
    let bbox = Aabb2::from_xywh(0, 0, 2, 2);
    assert!(bbox.is_positive());
    let bbox = Aabb2::from_xywh(-1, -1, 2, 2);
    assert!(!bbox.is_positive());
    let bbox = Aabb2::from_xywh(-1, 1, 2, 2);
    assert!(!bbox.is_positive());
    let bbox = Aabb2::from_xywh(1, -1, 2, 2);
    assert!(!bbox.is_positive());
}

impl<T: Num> Aabb2<T> {
    pub fn from_x1y1x2y2(x1: T, y1: T, x2: T, y2: T) -> Self
    where
        T: core::ops::SubAssign,
    {
        let point1 = Point2::new(x1, y1);
        let point2 = Point2::new(x2, y2);
        Self::new(point1, point2)
    }

    pub fn from_xywh(x: T, y: T, w: T, h: T) -> Self {
        let point = Point2::new(x, y);
        let size = Vector2::new(w, h);
        Self::new_point_size(point, size)
    }

    pub fn x1y1(&self) -> Point2<T> {
        self.point
    }

    pub fn x2y2(&self) -> Point2<T>
    where
        T: core::ops::AddAssign,
    {
        self.point + self.size
    }

    pub fn x2y1(&self) -> Point2<T>
    where
        T: core::ops::AddAssign,
    {
        Point2::new(self.point.x + self.size.x, self.point.y)
    }

    pub fn x1y2(&self) -> Point2<T>
    where
        T: core::ops::AddAssign,
    {
        Point2::new(self.point.x, self.point.y + self.size.y)
    }

    pub fn x1(&self) -> T {
        self.point.x
    }

    pub fn y1(&self) -> T {
        self.point.y
    }

    pub fn x2(&self) -> T
    where
        T: core::ops::AddAssign,
    {
        self.point.x + self.size.x
    }

    pub fn y2(&self) -> T
    where
        T: core::ops::AddAssign,
    {
        self.point.y + self.size.y
    }

    pub fn corners(&self) -> [Point2<T>; 4]
    where
        T: core::ops::AddAssign,
    {
        [self.x1y1(), self.x2y1(), self.x2y2(), self.x1y2()]
    }

    pub fn area(&self) -> T
    where
        T: core::ops::MulAssign,
    {
        self.measure()
    }

    pub fn width(&self) -> T {
        self.size.x
    }

    pub fn height(&self) -> T {
        self.size.y
    }
}

impl<T: Num> Aabb3<T> {
    pub fn volume(&self) -> T
    where
        T: core::ops::MulAssign,
    {
        self.measure()
    }
}

impl<T: core::fmt::Display, const D: usize> core::fmt::Display for Aabb<T, D>
where
    T: Num,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aabb(point: {}, size: {})", self.point, self.size)
    }
}

#[cfg(test)]
mod boudning_box_tests {
    use super::*;
    use nalgebra::*;

    #[test]
    fn test_bbox_new() {
        let point1 = Point2::new(1.0, 2.0);
        let point2 = Point2::new(4.0, 6.0);
        let bbox = AxisAlignedBoundingBox::new(point1, point2);

        assert_eq!(bbox.min_vertex(), point1);
        assert_eq!(bbox.size(), Vector2::new(3.0, 4.0));
        assert_eq!(bbox.center(), Point2::new(2.5, 4.0));
    }

    #[test]
    fn test_intersection_and_merge() {
        let point1 = Point2::new(1, 5);
        let point2 = Point2::new(3, 2);
        let size1 = Vector2::new(3, 4);
        let size2 = Vector2::new(1, 3);

        let this = Aabb2::new_point_size(point1, size1);
        let other = Aabb2::new_point_size(point2, size2);
        let inter = this.intersection(&other);
        let merged = this.merge(&other);
        assert_ne!(inter, Some(merged))
    }

    #[test]
    fn test_bounding_box_center_2d() {
        let point = Point2::new(1.0, 2.0);
        let size = Vector2::new(3.0, 4.0);
        let bbox = AxisAlignedBoundingBox::new_point_size(point, size);

        assert_eq!(bbox.min_vertex(), point);
        assert_eq!(bbox.size(), size);
        assert_eq!(bbox.center(), Point2::new(2.5, 4.0));
    }

    #[test]
    fn test_bounding_box_center_3d() {
        let point = Point3::new(1.0, 2.0, 3.0);
        let size = Vector3::new(4.0, 5.0, 6.0);
        let bbox = AxisAlignedBoundingBox::new_point_size(point, size);

        assert_eq!(bbox.min_vertex(), point);
        assert_eq!(bbox.size(), size);
        assert_eq!(bbox.center(), Point3::new(3.0, 4.5, 6.0));
    }

    #[test]
    fn test_bounding_box_padding_2d() {
        let point = Point2::new(1.0, 2.0);
        let size = Vector2::new(3.0, 4.0);
        let bbox = AxisAlignedBoundingBox::new_point_size(point, size);

        let padded_bbox = bbox.padding(1.0);
        assert_eq!(padded_bbox.min_vertex(), Point2::new(0.5, 1.5));
        assert_eq!(padded_bbox.size(), Vector2::new(4.0, 5.0));
    }

    #[test]
    fn test_bounding_box_scaling_2d() {
        let point = Point2::new(1.0, 1.0);
        let size = Vector2::new(3.0, 4.0);
        let bbox = AxisAlignedBoundingBox::new_point_size(point, size);

        let padded_bbox = bbox.scale(Vector2::new(2.0, 2.0));
        assert_eq!(padded_bbox.min_vertex(), Point2::new(-2.0, -3.0));
        assert_eq!(padded_bbox.size(), Vector2::new(6.0, 8.0));
    }

    #[test]
    fn test_bounding_box_contains_2d() {
        let point1 = Point2::new(1.0, 2.0);
        let point2 = Point2::new(4.0, 6.0);
        let bbox = AxisAlignedBoundingBox::new(point1, point2);

        assert!(bbox.contains_point(&Point2::new(2.0, 3.0)));
        assert!(!bbox.contains_point(&Point2::new(5.0, 7.0)));
    }

    #[test]
    fn test_bounding_box_union_2d() {
        let point1 = Point2::new(1.0, 2.0);
        let point2 = Point2::new(4.0, 6.0);
        let bbox1 = AxisAlignedBoundingBox::new(point1, point2);

        let point3 = Point2::new(3.0, 5.0);
        let point4 = Point2::new(7.0, 8.0);
        let bbox2 = AxisAlignedBoundingBox::new(point3, point4);

        let union_bbox = bbox1.merge(&bbox2);
        assert_eq!(union_bbox.min_vertex(), Point2::new(1.0, 2.0));
        assert_eq!(union_bbox.size(), Vector2::new(6.0, 6.0));
    }

    #[test]
    fn test_bounding_box_intersection_2d() {
        let point1 = Point2::new(1.0, 2.0);
        let point2 = Point2::new(4.0, 6.0);
        let bbox1 = AxisAlignedBoundingBox::new(point1, point2);

        let point3 = Point2::new(3.0, 5.0);
        let point4 = Point2::new(5.0, 7.0);
        let bbox2 = AxisAlignedBoundingBox::new(point3, point4);

        let intersection_bbox = bbox1.intersection(&bbox2).unwrap();
        assert_eq!(intersection_bbox.min_vertex(), Point2::new(3.0, 5.0));
        assert_eq!(intersection_bbox.size(), Vector2::new(1.0, 1.0));
    }

    #[test]
    fn test_bounding_box_contains_point() {
        let point1 = Point2::new(2, 3);
        let point2 = Point2::new(5, 4);
        let bbox = AxisAlignedBoundingBox::new(point1, point2);
        use itertools::Itertools;
        for (i, j) in (0..=10).cartesian_product(0..=10) {
            if bbox.contains_point(&Point2::new(i, j)) {
                if !(2..=5).contains(&i) && !(3..=4).contains(&j) {
                    panic!(
                        "Point ({}, {}) should not be contained in the bounding box",
                        i, j
                    );
                }
            } else {
                if (2..=5).contains(&i) && (3..=4).contains(&j) {
                    panic!(
                        "Point ({}, {}) should be contained in the bounding box",
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_bounding_box_clamp_box_2d() {
        let bbox1 = Aabb2::from_x1y1x2y2(1, 1, 4, 4);
        let bbox2 = Aabb2::from_x1y1x2y2(2, 2, 3, 3);
        let clamped = bbox2.clamp(&bbox1).unwrap();
        assert_eq!(bbox2, clamped);
        let clamped = bbox1.clamp(&bbox2).unwrap();
        assert_eq!(bbox2, clamped);

        let bbox1 = Aabb2::from_x1y1x2y2(4, 5, 7, 8);
        let bbox2 = Aabb2::from_x1y1x2y2(5, 4, 8, 7);
        let clamped = bbox1.clamp(&bbox2).unwrap();
        let expected = Aabb2::from_x1y1x2y2(5, 5, 7, 7);
        assert_eq!(clamped, expected)
    }

    #[test]
    fn test_iou_identical_boxes() {
        let a = Aabb2::from_x1y1x2y2(1.0, 2.0, 4.0, 6.0);
        let b = Aabb2::from_x1y1x2y2(1.0, 2.0, 4.0, 6.0);
        assert_eq!(a.iou(&b), 1.0);
    }

    #[test]
    fn test_iou_non_overlapping_boxes() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 1.0, 1.0);
        let b = Aabb2::from_x1y1x2y2(2.0, 2.0, 3.0, 3.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0);
        let b = Aabb2::from_x1y1x2y2(1.0, 1.0, 3.0, 3.0);
        // Intersection area = 1, Union area = 7
        assert!((a.iou(&b) - 1.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_iou_one_inside_another() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 4.0, 4.0);
        let b = Aabb2::from_x1y1x2y2(1.0, 1.0, 3.0, 3.0);
        // Intersection area = 4, Union area = 16
        assert!((a.iou(&b) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_iou_edge_touching() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 1.0, 1.0);
        let b = Aabb2::from_x1y1x2y2(1.0, 0.0, 2.0, 1.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_corner_touching() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 1.0, 1.0);
        let b = Aabb2::from_x1y1x2y2(1.0, 1.0, 2.0, 2.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_iou_zero_area_box() {
        let a = Aabb2::from_x1y1x2y2(0.0, 0.0, 0.0, 0.0);
        let b = Aabb2::from_x1y1x2y2(0.0, 0.0, 1.0, 1.0);
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn test_specific_values() {
        let box1 = Aabb2::from_xywh(0.69482, 0.6716774, 0.07493961, 0.14968264);
        let box2 = Aabb2::from_xywh(0.41546485, 0.70290875, 0.06197411, 0.08818436);
        assert!(box1.iou(&box2) >= 0.0);
    }

    #[test]
    fn test_move_origin() {
        let bbox = Aabb2::from_xywh(2, 3, 4, 5);
        let moved = bbox.move_origin(Point2::new(0, 0));
        assert_eq!(moved.min_vertex(), Point2::new(2, 3));
        assert_eq!(moved.size(), Vector2::new(4, 5));
        let moved = bbox.move_origin(Point2::new(2, 3));
        let expected = Aabb2::from_xywh(0, 0, 4, 5);
        assert_eq!(moved, expected);
    }
}
