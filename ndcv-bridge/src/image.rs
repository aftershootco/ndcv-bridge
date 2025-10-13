/// Trait for converting images to ndarray
///
/// ```rust
/// use ndcv_bridge::NdImage;
/// use ndarray::Array3;
/// use image::{DynamicImage, GrayImage, RgbImage};
/// let rgb: RgbImage = RgbImage::new(10, 10);
/// let luma: GrayImage = DynamicImage::ImageRgb8(rgb).into_luma8();
/// let ndarray: Array3<u8> = rgb.to_ndarray();
/// let ndarray: Array2<u8> = luma.to_ndarray();
/// let ndarray: Array3<u8> = DynamicImage::ImageRgb8(rgb).to_ndarray();
/// ```
use ndarray::*;
pub trait NdImage {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn channels(&self) -> usize;
}

impl<T, S: RawData<Elem = T>> NdImage for ArrayBase<S, Ix3> {
    fn width(&self) -> usize {
        self.dim().1
    }
    fn height(&self) -> usize {
        self.dim().0
    }
    fn channels(&self) -> usize {
        self.dim().2
    }
}

impl<T, S: RawData<Elem = T>> NdImage for ArrayBase<S, Ix2> {
    fn width(&self) -> usize {
        self.dim().1
    }
    fn height(&self) -> usize {
        self.dim().0
    }
    fn channels(&self) -> usize {
        1
    }
}
