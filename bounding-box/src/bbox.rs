// bbox compatibility
pub use crate::Aabb2;

pub trait BBoxCompat<T> {
    fn top_left(self) -> (T, T);
    fn with_top_left(self, x: T, y: T) -> Self;
    fn zeros_ndarray_2d<U: num::Zero + Copy>(&self) -> ndarray::Array2<U>;
    fn zeros_ndarray_3d<U: num::Zero + Copy>(&self, channels: usize) -> ndarray::Array3<U>;
}

impl<T: crate::Num> BBoxCompat<T> for Aabb2<T> {
    fn top_left(self) -> (T, T) {
        (self.x1(), self.y1())
    }

    fn with_top_left(self, x: T, y: T) -> Self {
        // self.translate((x - self.x1(), y - self.y1()))
        todo!()
    }

    fn zeros_ndarray_2d<U: num::Zero + Copy>(&self) -> ndarray::Array2<U> {
        // let width = self.x2() - self.x1();
        // let height = self.y2() - self.y1();
        // ndarray::Array2::<U>::zeros((height, width))
        todo!()
    }

    fn zeros_ndarray_3d<U: num::Zero + Copy>(&self, channels: usize) -> ndarray::Array3<U> {
        // let width = self.x2() - self.x1();
        // let height = self.y2() - self.y1();
        // ndarray::Array3::<U>::zeros((height, width, channels))
        todo!()
    }
}
