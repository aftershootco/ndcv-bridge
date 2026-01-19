use super::{Aabb2, Num};
use nalgebra::Point2;
use nalgebra::Vector2;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "BBoxCompat<T>", into = "BBoxCompat<T>")]
#[repr(transparent)]
pub struct BBox<T = f32>(Aabb2<T>)
where
    T: Num;

impl<T: Num> From<BBox<T>> for Aabb2<T> {
    fn from(value: BBox<T>) -> Self {
        value.0
    }
}

impl<T: Num> From<Aabb2<T>> for BBox<T> {
    fn from(value: Aabb2<T>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BBoxCompat<T = f32> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T: Num> From<BBoxCompat<T>> for BBox<T> {
    fn from(value: BBoxCompat<T>) -> Self {
        Self(Aabb2::from_xywh(
            value.x,
            value.y,
            value.width,
            value.height,
        ))
    }
}

impl<T: Num> From<BBox<T>> for BBoxCompat<T> {
    fn from(value: BBox<T>) -> Self {
        Self {
            x: value.0.x1(),
            y: value.0.y1(),
            width: value.0.width(),
            height: value.0.height(),
        }
    }
}

impl<T: Num> core::ops::Deref for BBox<T> {
    type Target = Aabb2<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Num> BBox<T> {
    pub fn into_inner(self) -> Aabb2<T> {
        self.0
    }

    pub fn as_<T2: Num>(&self) -> BBox<T2>
    where
        T: num::cast::AsPrimitive<T2>,
    {
        BBox(self.0.as_())
    }

    pub fn new_xywh(x: T, y: T, width: T, height: T) -> Self {
        Self(Aabb2::from_xywh(x, y, width, height))
    }

    pub fn new_xyxy(x1: T, y1: T, x2: T, y2: T) -> Self {
        Self(Aabb2::from_x1y1x2y2(x1, y1, x2, y2))
    }

    pub fn overlap(&self, other: &Self) -> T
    where
        T: num::Zero,
    {
        self.0
            .intersection(other.0)
            .map_or_else(T::zero, |r| r.area())
    }

    pub fn contains(&self, point: nalgebra::Point2<T>) -> bool {
        self.0.contains_point(point)
    }
}

impl<T: Num> From<[T; 4]> for BBox<T> {
    fn from([x, y, width, height]: [T; 4]) -> Self {
        Self::new_xywh(x, y, width, height)
    }
}

impl<T: Copy + Num> BBox<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        Self::new_xywh(x, y, width, height)
    }

    /// Casts the internal values to another type using [as] keyword
    pub fn cast<T2>(self) -> BBox<T2>
    where
        T: num::cast::AsPrimitive<T2>,
        T2: Copy + 'static,
        T2: Num,
    {
        BBox(self.0.as_::<T2>())
    }

    /// Clamps all the internal values to the given min and max.
    pub fn clamp(&self, min: T, max: T) -> Self
    where
        T: std::cmp::PartialOrd,
    {
        self.0.component_clamp(min, max).into()
    }

    pub fn clamp_box(&self, bbox: BBox<T>) -> Self
    where
        T: std::cmp::PartialOrd,
        T: num::Zero,
        T: core::ops::Add<Output = T>,
        T: core::ops::Sub<Output = T>,
    {
        // let x1 = num::clamp(self.x1(), bbox.x1(), bbox.x2());
        // let y1 = num::clamp(self.y1(), bbox.y1(), bbox.y2());
        // let x2 = num::clamp(self.x2(), bbox.x1(), bbox.x2());
        // let y2 = num::clamp(self.y2(), bbox.y1(), bbox.y2());
        // Self::new_xyxy(x1, y1, x2, y2)
        self.0.clamp(bbox.0).unwrap_or(Aabb2::zero()).into()
    }

    pub fn normalize(&self, width: T, height: T) -> Self
    where
        T: core::ops::Div<Output = T> + Copy,
    {
        self.0.normalize(Vector2::new(width, height)).into()
    }

    /// Normalize after casting to float
    pub fn normalize_f64(&self, width: T, height: T) -> BBox<f64>
    where
        T: core::ops::Div<Output = T> + Copy,
        T: num::cast::AsPrimitive<f64>,
    {
        self.0
            .as_()
            .normalize(Vector2::new(width.as_(), height.as_()))
            .into()
    }

    pub fn denormalize(&self, width: T, height: T) -> Self
    where
        T: core::ops::Mul<Output = T> + Copy,
    {
        self.0.denormalize(Vector2::new(width, height)).into()
    }

    pub fn height(&self) -> T {
        self.0.height()
    }

    pub fn width(&self) -> T {
        self.0.width()
    }

    pub fn padding(&self, padding: T) -> Self
    where
        T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + Copy,
    {
        // Self {
        //     x: self.x - padding,
        //     y: self.y - padding,
        //     width: self.width + padding + padding,
        //     height: self.height + padding + padding,
        // }
        let two = T::one() + T::one();
        self.0.padding_uniform(padding * two).into()
    }

    pub fn padding_height(&self, padding: T) -> Self
    where
        T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + Copy,
    {
        // Self {
        //     x: self.x,
        //     y: self.y - padding,
        //     width: self.width,
        //     height: self.height + padding + padding,
        // }
        let two = T::one() + T::one();
        self.0
            .padding(Vector2::new(T::zero(), padding * two))
            .into()
    }

    pub fn padding_width(&self, padding: T) -> Self {
        // Self {
        //     x: self.x - padding,
        //     y: self.y,
        //     width: self.width + padding + padding,
        //     height: self.height,
        // }
        let two = T::one() + T::one();
        self.0
            .padding(Vector2::new(padding * two, T::zero()))
            .into()
    }

    // Enlarge / shrink the bounding box by a factor while
    // keeping the center point and the aspect ratio fixed
    pub fn scale(&self, factor: T) -> Self {
        self.0.scale(Vector2::from_element(factor)).into()
    }

    pub fn scale_x(&self, factor: T) -> Self {
        self.0.scale(Vector2::new(factor, num::one::<T>())).into()
    }

    pub fn scale_y(&self, factor: T) -> Self {
        self.0.scale(Vector2::new(num::one::<T>(), factor)).into()
    }

    pub fn offset(&self, offset: Vector2<T>) -> Self {
        self.0.translate(offset).into()
    }

    /// Translate the bounding box by the given offset
    /// if they are in the same scale
    pub fn translate(&self, bbox: Self) -> Self
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        self.0.translate(bbox.0.point.coords).into()
    }

    pub fn with_top_left(&self, top_left: Point2<T>) -> Self {
        self.move_to(top_left).into()
    }

    pub fn center(&self) -> Point2<T> {
        self.0.center()
    }

    pub fn area(&self) -> T {
        self.0.area()
    }

    // Corresponds to self.x1() and self.y1()
    pub fn top_left(&self) -> Point2<T> {
        self.x1y1()
    }

    // pub fn top_right(&self) -> Point<T>
    // where
    //     T: core::ops::Add<Output = T> + Copy,
    // {
    //     Point2::new(self.x + self.width, self.y)
    // }
    //
    // pub fn bottom_left(&self) -> Point<T>
    // where
    //     T: core::ops::Add<Output = T> + Copy,
    // {
    //     Point2::new(self.x, self.y + self.height)
    // }
    //
    // // Corresponds to self.x2() and self.y2()
    // pub fn bottom_right(&self) -> Point<T>
    // where
    //     T: core::ops::Add<Output = T> + Copy,
    // {
    //     // Point2::new(self.x + self.width, self.y + self.height)
    //     self.x2
    // }

    pub fn x1(&self) -> T {
        self.0.x1()
    }

    pub fn y1(&self) -> T {
        self.0.y1()
    }

    pub fn x2(&self) -> T
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        self.0.x2()
    }

    pub fn y2(&self) -> T
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        self.0.y2()
    }

    // pub fn overlap(&self, other: &Self) -> T
    // where
    //     T: std::cmp::PartialOrd
    //         + traits::min::Min
    //         + traits::max::Max
    //         + num::Zero
    //         + core::ops::Add<Output = T>
    //         + core::ops::Sub<Output = T>
    //         + core::ops::Mul<Output = T>
    //         + Copy,
    // {
    //     let x1 = self.x.max(other.x);
    //     let y1 = self.y.max(other.y);
    //     let x2 = (self.x + self.width).min(other.x + other.width);
    //     let y2 = (self.y + self.height).min(other.y + other.height);
    //     let width = (x2 - x1).max(T::zero());
    //     let height = (y2 - y1).max(T::zero());
    //     width * height
    // }

    pub fn iou(&self, other: &Self) -> T
    where
        T: std::cmp::Ord
            + num::Zero
            + traits::min::Min
            + traits::max::Max
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Mul<Output = T>
            + core::ops::Div<Output = T>
            + Copy,
    {
        self.0.iou(other.0)
    }

    // pub fn contains(&self, point: Point<T>) -> bool
    // where
    //     T: std::cmp::PartialOrd + core::ops::Add<Output = T> + Copy,
    // {
    //     point.x >= self.x
    //         && point.x <= self.x + self.width
    //         && point.y >= self.y
    //         && point.y <= self.y + self.height
    // }

    pub fn contains_bbox(&self, other: Self) -> bool
    where
        T: std::cmp::PartialOrd + Copy,
        T: core::ops::Add<Output = T>,
    {
        self.0.contains_bbox(&other.0)
        // self.contains(other.top_left())
        //     && self.contains(other.top_right())
        //     && self.contains(other.bottom_left())
        //     && self.contains(other.bottom_right())
    }

    pub fn containing(box1: Self, box2: Self) -> Self
    where
        T: traits::min::Min + traits::max::Max + Copy,
        T: core::ops::Sub<Output = T>,
        T: core::ops::Add<Output = T>,
    {
        Self(box1.0.merge(box2.0))
    }
}

impl<T: Num + core::ops::Sub<Output = T> + Copy> core::ops::Sub<T> for BBox<T> {
    type Output = BBox<T>;
    fn sub(mut self, rhs: T) -> Self::Output {
        self.0.point.iter_mut().for_each(|v| *v = *v - rhs);
        self.0.size.iter_mut().for_each(|v| *v = *v - rhs);
        self
    }
}

impl<T: Num + core::ops::Add<Output = T> + Copy> core::ops::Add<T> for BBox<T> {
    type Output = BBox<T>;
    fn add(mut self, rhs: T) -> Self::Output {
        self.0.point.iter_mut().for_each(|v| *v = *v + rhs);
        self.0.size.iter_mut().for_each(|v| *v = *v + rhs);
        self
    }
}
impl<T: Num + core::ops::Mul<Output = T> + Copy> core::ops::Mul<T> for BBox<T> {
    type Output = BBox<T>;
    fn mul(mut self, rhs: T) -> Self::Output {
        self.0.point.iter_mut().for_each(|v| *v = *v * rhs);
        self.0.size.iter_mut().for_each(|v| *v = *v * rhs);
        self
    }
}
impl<T: Num + core::ops::Div<Output = T> + Copy> core::ops::Div<T> for BBox<T> {
    type Output = BBox<T>;
    fn div(mut self, rhs: T) -> Self::Output {
        self.0.point.iter_mut().for_each(|v| *v = *v / rhs);
        self.0.size.iter_mut().for_each(|v| *v = *v / rhs);
        self
    }
}

impl<T> core::ops::Add<BBox<T>> for BBox<T>
where
    T: core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>
        + traits::min::Min
        + traits::max::Max
        + Copy
        + Num,
{
    type Output = BBox<T>;
    fn add(self, rhs: BBox<T>) -> Self::Output {
        let x1 = self.x1().min(rhs.x1());
        let y1 = self.y1().min(rhs.y1());
        let x2 = self.x2().max(rhs.x2());
        let y2 = self.y2().max(rhs.y2());
        BBox::new_xyxy(x1, y1, x2, y2)
    }
}

#[test]
fn test_bbox_add() {
    dbg!();
    let bbox1: BBox<usize> = BBox::new_xyxy(0, 0, 10, 10);
    dbg!();
    let bbox2: BBox<usize> = BBox::new_xyxy(5, 5, 15, 15);
    dbg!();
    let bbox3: BBox<usize> = bbox1 + bbox2;
    dbg!();
    assert_eq!(bbox3, BBox::new_xyxy(0, 0, 15, 15).cast());
    dbg!();
}

impl<I: Num + num::Zero> BBox<I>
where
    I: num::cast::AsPrimitive<usize>,
{
    pub fn zeros_ndarray_2d<T: num::Zero + Copy>(&self) -> ndarray::Array2<T> {
        ndarray::Array2::<T>::zeros((self.height().as_(), self.width().as_()))
    }
    pub fn zeros_ndarray_3d<T: num::Zero + Copy>(&self, channels: usize) -> ndarray::Array3<T> {
        ndarray::Array3::<T>::zeros((self.height().as_(), self.width().as_(), channels))
    }
    pub fn ones_ndarray_2d<T: num::One + Copy>(&self) -> ndarray::Array2<T> {
        ndarray::Array2::<T>::ones((self.height().as_(), self.width().as_()))
    }
}

impl<T: Num + num::Float> BBox<T> {
    pub fn round(&self) -> Self {
        Self(self.0.round())
    }
}

#[cfg(test)]
mod bbox_clamp_tests {
    use super::*;
    #[test]
    pub fn bbox_test_clamp_box() {
        let large_box = BBox::new(0, 0, 100, 100);
        let small_box = BBox::new(10, 10, 20, 20);
        let clamped = large_box.clamp_box(small_box);
        assert_eq!(clamped, small_box);
    }

    #[test]
    pub fn bbox_test_clamp_box_offset() {
        let box_a = BBox::new(0, 0, 100, 100);
        let box_b = BBox::new(-10, -10, 20, 20);
        let clamped = box_b.clamp_box(box_a);
        let expected = BBox::new(0, 0, 10, 10);
        assert_eq!(expected, clamped);
    }
}

#[cfg(test)]
mod bbox_padding_tests {
    use super::*;
    #[test]
    pub fn bbox_test_padding() {
        let bbox = BBox::new(0, 0, 10, 10);
        let padded = bbox.padding(2);
        assert_eq!(padded, BBox::new(-2, -2, 14, 14));
    }

    #[test]
    pub fn bbox_test_padding_height() {
        let bbox = BBox::new(0, 0, 10, 10);
        let padded = bbox.padding_height(2);
        assert_eq!(padded, BBox::new(0, -2, 10, 14));
    }

    #[test]
    pub fn bbox_test_padding_width() {
        let bbox = BBox::new(0, 0, 10, 10);
        let padded = bbox.padding_width(2);
        assert_eq!(padded, BBox::new(-2, 0, 14, 10));
    }

    #[test]
    pub fn bbox_test_clamped_padding() {
        let bbox = BBox::new(0, 0, 10, 10);
        let padded = bbox.padding(2);
        let clamp = BBox::new(0, 0, 12, 12);
        let clamped = padded.clamp_box(clamp);
        assert_eq!(clamped, clamp);
    }

    #[test]
    pub fn bbox_clamp_failure() {
        let og = BBox::new(475.0, 79.625, 37.0, 282.15);
        let padded = BBox::new_xywh(
            471.3,
            51.412499999999994,
            40.69999999999999,
            338.54999999999995,
        );
        let clamp = BBox::new(0.0, 0.0, 512.0, 512.0);
        let sus = padded.clamp_box(clamp);
        assert!(clamp.contains_bbox(sus));
    }
}

#[cfg(test)]
mod bbox_scale_tests {
    use super::*;
    #[test]
    pub fn bbox_test_scale_int() {
        let bbox = BBox::new(0, 0, 10, 10);
        let scaled = bbox.scale(2);
        assert_eq!(scaled, BBox::new(-5, -5, 20, 20));
    }

    #[test]
    pub fn bbox_test_scale_float() {
        let bbox = BBox::new(0, 0, 10, 10).cast();
        let scaled = bbox.scale(1.05); // 5% increase
        let l = 10.0 * 0.05;
        assert_eq!(scaled, BBox::new(-l / 2.0, -l / 2.0, 10.0 + l, 10.0 + l));
    }

    #[test]
    pub fn bbox_test_scale_float_negative() {
        let bbox = BBox::new(0, 0, 10, 10).cast();
        let scaled = bbox.scale(0.95); // 5% decrease
        let l = -10.0 * 0.05;
        assert_eq!(scaled, BBox::new(-l / 2.0, -l / 2.0, 10.0 + l, 10.0 + l));
    }

    #[test]
    pub fn bbox_scale_float() {
        let bbox = BBox::new_xywh(0, 0, 200, 200);
        let scaled = bbox.cast::<f64>().scale(1.1).cast::<i32>().clamp(0, 1000);
        let expected = BBox::new(0, 0, 220, 220);
        assert_eq!(scaled, expected);
    }
    #[test]
    pub fn add_padding_bbox_example() {
        // let result = add_padding_bbox(
        //     vec![Rect::new(100, 200, 300, 400)],
        //     (0.1, 0.1),
        //     (1000, 1000),
        // );
        //   assert_eq!(result[0], Rect::new(70, 160, 360, 480));
        let bbox = BBox::new(100, 200, 300, 400);
        let scaled = bbox.cast::<f64>().scale(1.2).cast::<i32>().clamp(0, 1000);
        assert_eq!(bbox, BBox::new(100, 200, 300, 400));
        assert_eq!(scaled, BBox::new(70, 160, 360, 480));
    }
    #[test]
    pub fn scale_bboxes() {
        // let result = scale_bboxes(Rect::new(100, 200, 300, 400), (1000, 1000), (500, 500));
        // assert_eq!(result[0], Rect::new(200, 400, 600, 800));
        let bbox = BBox::new(100, 200, 300, 400);
        let scaled = bbox.scale(2);
        assert_eq!(scaled, BBox::new(200, 400, 600, 800));
    }
}

pub mod traits {
    pub mod max {
        pub trait Max: Sized + Copy {
            fn max(self, other: Self) -> Self;
        }

        macro_rules! impl_max {
        ($($t:ty),*) => {
            $(
                impl Max for $t {
                    fn max(self, other: Self) -> Self {
                        Ord::max(self, other)
                    }
                }
            )*
        };
        (float $($t:ty),*) => {
            $(
                impl Max for $t {
                    fn max(self, other: Self) -> Self {
                        Self::max(self, other)
                    }
                }
            )*
        };
    }

        impl_max!(
            usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128
        );
        impl_max!(float f32, f64);
    }

    pub mod min {
        pub trait Min: Sized + Copy {
            fn min(self, other: Self) -> Self;
        }

        macro_rules! impl_min {
        ($($t:ty),*) => {
            $(
                impl Min for $t {
                    fn min(self, other: Self) -> Self {
                        Ord::min(self, other)
                    }
                }
            )*
        };
        (float $($t:ty),*) => {
            $(
                impl Min for $t {
                    fn min(self, other: Self) -> Self {
                        Self::min(self, other)
                    }
                }
            )*
        };
    }

        impl_min!(
            usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128
        );
        impl_min!(float f32, f64);
    }
}
