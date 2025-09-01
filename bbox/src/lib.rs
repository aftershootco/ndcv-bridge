pub mod traits;

/// A bounding box of co-ordinates whose origin is at the top-left corner.
#[derive(
    Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
pub struct BBox<T = f32> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> From<[T; 4]> for BBox<T> {
    fn from([x, y, width, height]: [T; 4]) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl<T: Copy> BBox<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Casts the internal values to another type using [as] keyword
    pub fn cast<T2>(self) -> BBox<T2>
    where
        T: num::cast::AsPrimitive<T2>,
        T2: Copy + 'static,
    {
        BBox {
            x: self.x.as_(),
            y: self.y.as_(),
            width: self.width.as_(),
            height: self.height.as_(),
        }
    }

    /// Clamps all the internal values to the given min and max.
    pub fn clamp(&self, min: T, max: T) -> Self
    where
        T: std::cmp::PartialOrd,
    {
        Self {
            x: num::clamp(self.x, min, max),
            y: num::clamp(self.y, min, max),
            width: num::clamp(self.width, min, max),
            height: num::clamp(self.height, min, max),
        }
    }

    pub fn clamp_box(&self, bbox: BBox<T>) -> Self
    where
        T: std::cmp::PartialOrd,
        T: num::Zero,
        T: core::ops::Add<Output = T>,
        T: core::ops::Sub<Output = T>,
    {
        let x1 = num::clamp(self.x1(), bbox.x1(), bbox.x2());
        let y1 = num::clamp(self.y1(), bbox.y1(), bbox.y2());
        let x2 = num::clamp(self.x2(), bbox.x1(), bbox.x2());
        let y2 = num::clamp(self.y2(), bbox.y1(), bbox.y2());
        Self::new_xyxy(x1, y1, x2, y2)
    }

    pub fn normalize(&self, width: T, height: T) -> Self
    where
        T: core::ops::Div<Output = T> + Copy,
    {
        Self {
            x: self.x / width,
            y: self.y / height,
            width: self.width / width,
            height: self.height / height,
        }
    }

    /// Normalize after casting to float
    pub fn normalize_f64(&self, width: T, height: T) -> BBox<f64>
    where
        T: core::ops::Div<Output = T> + Copy,
        T: num::cast::AsPrimitive<f64>,
    {
        BBox {
            x: self.x.as_() / width.as_(),
            y: self.y.as_() / height.as_(),
            width: self.width.as_() / width.as_(),
            height: self.height.as_() / height.as_(),
        }
    }

    pub fn denormalize(&self, width: T, height: T) -> Self
    where
        T: core::ops::Mul<Output = T> + Copy,
    {
        Self {
            x: self.x * width,
            y: self.y * height,
            width: self.width * width,
            height: self.height * height,
        }
    }

    pub fn height(&self) -> T {
        self.height
    }

    pub fn width(&self) -> T {
        self.width
    }

    pub fn padding(&self, padding: T) -> Self
    where
        T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + Copy,
    {
        Self {
            x: self.x - padding,
            y: self.y - padding,
            width: self.width + padding + padding,
            height: self.height + padding + padding,
        }
    }

    pub fn padding_height(&self, padding: T) -> Self
    where
        T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + Copy,
    {
        Self {
            x: self.x,
            y: self.y - padding,
            width: self.width,
            height: self.height + padding + padding,
        }
    }

    pub fn padding_width(&self, padding: T) -> Self
    where
        T: core::ops::Add<Output = T> + core::ops::Sub<Output = T> + Copy,
    {
        Self {
            x: self.x - padding,
            y: self.y,
            width: self.width + padding + padding,
            height: self.height,
        }
    }

    // Enlarge / shrink the bounding box by a factor while
    // keeping the center point and the aspect ratio fixed
    pub fn scale(&self, factor: T) -> Self
    where
        T: core::ops::Mul<Output = T>,
        T: core::ops::Sub<Output = T>,
        T: core::ops::Add<Output = T>,
        T: core::ops::Div<Output = T>,
        T: num::One + Copy,
    {
        let two = num::one::<T>() + num::one::<T>();
        let width = self.width * factor;
        let height = self.height * factor;
        let width_inc = width - self.width;
        let height_inc = height - self.height;
        Self {
            x: self.x - width_inc / two,
            y: self.y - height_inc / two,
            width,
            height,
        }
    }

    pub fn scale_x(&self, factor: T) -> Self
    where
        T: core::ops::Mul<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Add<Output = T>
            + core::ops::Div<Output = T>
            + num::One
            + Copy,
    {
        let two = num::one::<T>() + num::one::<T>();
        let width = self.width * factor;
        let width_inc = width - self.width;
        Self {
            x: self.x - width_inc / two,
            y: self.y,
            width,
            height: self.height,
        }
    }

    pub fn scale_y(&self, factor: T) -> Self
    where
        T: core::ops::Mul<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Add<Output = T>
            + core::ops::Div<Output = T>
            + num::One
            + Copy,
    {
        let two = num::one::<T>() + num::one::<T>();
        let height = self.height * factor;
        let height_inc = height - self.height;
        Self {
            x: self.x,
            y: self.y - height_inc / two,
            width: self.width,
            height,
        }
    }

    pub fn offset(&self, offset: Point<T>) -> Self
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        Self {
            x: self.x + offset.x,
            y: self.y + offset.y,
            width: self.width,
            height: self.height,
        }
    }

    /// Translate the bounding box by the given offset
    /// if they are in the same scale
    pub fn translate(&self, bbox: Self) -> Self
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        Self {
            x: self.x + bbox.x,
            y: self.y + bbox.y,
            width: self.width,
            height: self.height,
        }
    }

    pub fn with_top_left(&self, top_left: Point<T>) -> Self {
        Self {
            x: top_left.x,
            y: top_left.y,
            width: self.width,
            height: self.height,
        }
    }

    pub fn center(&self) -> Point<T>
    where
        T: core::ops::Add<Output = T> + core::ops::Div<Output = T> + Copy,
        T: num::One,
    {
        let two = T::one() + T::one();
        Point::new(self.x + self.width / two, self.y + self.height / two)
    }

    pub fn area(&self) -> T
    where
        T: core::ops::Mul<Output = T> + Copy,
    {
        self.width * self.height
    }

    // Corresponds to self.x1() and self.y1()
    pub fn top_left(&self) -> Point<T> {
        Point::new(self.x, self.y)
    }

    pub fn top_right(&self) -> Point<T>
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        Point::new(self.x + self.width, self.y)
    }

    pub fn bottom_left(&self) -> Point<T>
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        Point::new(self.x, self.y + self.height)
    }

    // Corresponds to self.x2() and self.y2()
    pub fn bottom_right(&self) -> Point<T>
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        Point::new(self.x + self.width, self.y + self.height)
    }

    pub const fn x1(&self) -> T {
        self.x
    }

    pub const fn y1(&self) -> T {
        self.y
    }

    pub fn x2(&self) -> T
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        self.x + self.width
    }

    pub fn y2(&self) -> T
    where
        T: core::ops::Add<Output = T> + Copy,
    {
        self.y + self.height
    }

    pub fn overlap(&self, other: &Self) -> T
    where
        T: std::cmp::PartialOrd
            + traits::min::Min
            + traits::max::Max
            + num::Zero
            + core::ops::Add<Output = T>
            + core::ops::Sub<Output = T>
            + core::ops::Mul<Output = T>
            + Copy,
    {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        let width = (x2 - x1).max(T::zero());
        let height = (y2 - y1).max(T::zero());
        width * height
    }

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
        let overlap = self.overlap(other);
        let union = self.area() + other.area() - overlap;
        overlap / union
    }

    pub fn contains(&self, point: Point<T>) -> bool
    where
        T: std::cmp::PartialOrd + core::ops::Add<Output = T> + Copy,
    {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    pub fn contains_bbox(&self, other: Self) -> bool
    where
        T: std::cmp::PartialOrd + Copy,
        T: core::ops::Add<Output = T>,
    {
        self.contains(other.top_left())
            && self.contains(other.top_right())
            && self.contains(other.bottom_left())
            && self.contains(other.bottom_right())
    }

    pub fn new_xywh(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
    pub fn new_xyxy(x1: T, y1: T, x2: T, y2: T) -> Self
    where
        T: core::ops::Sub<Output = T> + Copy,
    {
        Self {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
        }
    }

    pub fn containing(box1: Self, box2: Self) -> Self
    where
        T: traits::min::Min + traits::max::Max + Copy,
        T: core::ops::Sub<Output = T>,
        T: core::ops::Add<Output = T>,
    {
        let x1 = box1.x.min(box2.x);
        let y1 = box1.y.min(box2.y);
        let x2 = box1.x2().max(box2.x2());
        let y2 = box1.y2().max(box2.y2());
        Self::new_xyxy(x1, y1, x2, y2)
    }
}

impl<T: core::ops::Sub<Output = T> + Copy> core::ops::Sub<T> for BBox<T> {
    type Output = BBox<T>;
    fn sub(self, rhs: T) -> Self::Output {
        BBox {
            x: self.x - rhs,
            y: self.y - rhs,
            width: self.width - rhs,
            height: self.height - rhs,
        }
    }
}

impl<T: core::ops::Add<Output = T> + Copy> core::ops::Add<T> for BBox<T> {
    type Output = BBox<T>;
    fn add(self, rhs: T) -> Self::Output {
        BBox {
            x: self.x + rhs,
            y: self.y + rhs,
            width: self.width + rhs,
            height: self.height + rhs,
        }
    }
}
impl<T: core::ops::Mul<Output = T> + Copy> core::ops::Mul<T> for BBox<T> {
    type Output = BBox<T>;
    fn mul(self, rhs: T) -> Self::Output {
        BBox {
            x: self.x * rhs,
            y: self.y * rhs,
            width: self.width * rhs,
            height: self.height * rhs,
        }
    }
}
impl<T: core::ops::Div<Output = T> + Copy> core::ops::Div<T> for BBox<T> {
    type Output = BBox<T>;
    fn div(self, rhs: T) -> Self::Output {
        BBox {
            x: self.x / rhs,
            y: self.y / rhs,
            width: self.width / rhs,
            height: self.height / rhs,
        }
    }
}

impl<T> core::ops::Add<BBox<T>> for BBox<T>
where
    T: core::ops::Sub<Output = T>
        + core::ops::Add<Output = T>
        + traits::min::Min
        + traits::max::Max
        + Copy,
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
    let bbox1: BBox<usize> = BBox::new_xyxy(0, 0, 10, 10);
    let bbox2: BBox<usize> = BBox::new_xyxy(5, 5, 15, 15);
    let bbox3: BBox<usize> = bbox1 + bbox2;
    assert_eq!(bbox3, BBox::new_xyxy(0, 0, 15, 15).cast());
}

#[derive(
    Debug, Copy, Clone, serde::Serialize, serde::Deserialize, PartialEq, PartialOrd, Eq, Ord, Hash,
)]
pub struct Point<T = f32> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub const fn x(&self) -> T
    where
        T: Copy,
    {
        self.x
    }

    pub const fn y(&self) -> T
    where
        T: Copy,
    {
        self.y
    }

    pub fn cast<T2>(&self) -> Point<T2>
    where
        T: num::cast::AsPrimitive<T2>,
        T2: Copy + 'static,
    {
        Point {
            x: self.x.as_(),
            y: self.y.as_(),
        }
    }
}

impl<T: core::ops::Sub<T, Output = T> + Copy> core::ops::Sub<Point<T>> for Point<T> {
    type Output = Point<T>;
    fn sub(self, rhs: Point<T>) -> Self::Output {
        Point {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: core::ops::Add<T, Output = T> + Copy> core::ops::Add<Point<T>> for Point<T> {
    type Output = Point<T>;
    fn add(self, rhs: Point<T>) -> Self::Output {
        Point {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: core::ops::Sub<Output = T> + Copy> Point<T> {
    /// If both the boxes are in the same scale then make the translation of the origin to the
    /// other box
    pub fn with_origin(&self, origin: Self) -> Self {
        *self - origin
    }
}

impl<T: core::ops::Add<Output = T> + Copy> Point<T> {
    pub fn translate(&self, point: Point<T>) -> Self {
        *self + point
    }
}

impl<I: num::Zero> BBox<I>
where
    I: num::cast::AsPrimitive<usize>,
{
    pub fn zeros_ndarray_2d<T: num::Zero + Copy>(&self) -> ndarray::Array2<T> {
        ndarray::Array2::<T>::zeros((self.height.as_(), self.width.as_()))
    }
    pub fn zeros_ndarray_3d<T: num::Zero + Copy>(&self, channels: usize) -> ndarray::Array3<T> {
        ndarray::Array3::<T>::zeros((self.height.as_(), self.width.as_(), channels))
    }
    pub fn ones_ndarray_2d<T: num::One + Copy>(&self) -> ndarray::Array2<T> {
        ndarray::Array2::<T>::ones((self.height.as_(), self.width.as_()))
    }
}

impl<T: num::Float> BBox<T> {
    pub fn round(&self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            width: self.width.round(),
            height: self.height.round(),
        }
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
        let padded = BBox {
            x: 471.3,
            y: 51.412499999999994,
            width: 40.69999999999999,
            height: 338.54999999999995,
        };
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
    // #[test]
    // pub fn scale_bboxes() {
    //     // let result = scale_bboxes(Rect::new(100, 200, 300, 400), (1000, 1000), (500, 500));
    //     // assert_eq!(result[0], Rect::new(200, 400, 600, 800));
    //     let bbox = BBox::new(100, 200, 300, 400);
    //     let scaled = bbox.scale(2);
    //     assert_eq!(scaled, BBox::new(200, 400, 600, 800));
    // }
}
