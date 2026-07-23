type Result<T, E = ndarray::ShapeError> = core::result::Result<T, E>;
fn shape_error() -> ndarray::ShapeError {
    ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape)
}

mod rgb8 {
    use super::Result;
    pub(super) fn image_as_ndarray(image: &image::RgbImage) -> Result<ndarray::ArrayView3<'_, u8>> {
        let (width, height) = image.dimensions();
        let data = image.as_raw();
        ndarray::ArrayView3::from_shape((height as usize, width as usize, 3), data)
    }
    pub(super) fn image_into_ndarray(image: image::RgbImage) -> Result<ndarray::Array3<u8>> {
        let (width, height) = image.dimensions();
        let data = image.into_raw();
        ndarray::Array3::from_shape_vec((height as usize, width as usize, 3), data)
    }
    pub(super) fn ndarray_to_image(array: &ndarray::ArrayView3<u8>) -> Result<image::RgbImage> {
        let (height, width, channels) = array.dim();
        let data = array.as_slice().ok_or_else(super::shape_error)?;
        if channels != 3 {
            return Err(super::shape_error());
        }
        image::RgbImage::from_raw(width as u32, height as u32, data.to_vec()).ok_or(
            ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape),
        )
    }
}

mod rgba8 {
    use super::Result;
    pub(super) fn image_as_ndarray(
        image: &image::RgbaImage,
    ) -> Result<ndarray::ArrayView3<'_, u8>> {
        let (width, height) = image.dimensions();
        let data = image.as_raw();
        ndarray::ArrayView3::from_shape((height as usize, width as usize, 4), data)
    }
    pub(super) fn image_into_ndarray(image: image::RgbaImage) -> Result<ndarray::Array3<u8>> {
        let (width, height) = image.dimensions();
        let data = image.into_raw();
        ndarray::Array3::from_shape_vec((height as usize, width as usize, 4), data)
    }
    pub(super) fn ndarray_to_image(array: &ndarray::ArrayView3<u8>) -> Result<image::RgbaImage> {
        let (height, width, channels) = array.dim();
        let data = array.as_slice().ok_or_else(super::shape_error)?;
        if channels != 4 {
            return Err(super::shape_error());
        }
        image::RgbaImage::from_raw(width as u32, height as u32, data.to_vec()).ok_or(
            ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape),
        )
    }
}

mod gray8 {
    use super::Result;
    pub(super) fn image_as_ndarray(
        image: &image::GrayImage,
    ) -> Result<ndarray::ArrayView2<'_, u8>> {
        let (width, height) = image.dimensions();
        let data = image.as_raw();
        ndarray::ArrayView2::from_shape((height as usize, width as usize), data)
    }
    pub(super) fn image_into_ndarray(image: image::GrayImage) -> Result<ndarray::Array2<u8>> {
        let (width, height) = image.dimensions();
        let data = image.into_raw();
        ndarray::Array2::from_shape_vec((height as usize, width as usize), data)
    }
    pub(super) fn ndarray_to_image(array: &ndarray::ArrayView2<u8>) -> Result<image::GrayImage> {
        let (height, width) = array.dim();
        let data = array.as_slice().ok_or_else(super::shape_error)?;
        image::GrayImage::from_raw(width as u32, height as u32, data.to_vec()).ok_or(
            ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape),
        )
    }
}

mod gray_alpha8 {
    use super::Result;
    pub(super) fn image_as_ndarray(
        image: &image::GrayAlphaImage,
    ) -> Result<ndarray::ArrayView3<'_, u8>> {
        let (width, height) = image.dimensions();
        let data = image.as_raw();
        ndarray::ArrayView3::from_shape((height as usize, width as usize, 2), data)
    }
    pub(super) fn image_into_ndarray(image: image::GrayAlphaImage) -> Result<ndarray::Array3<u8>> {
        let (width, height) = image.dimensions();
        let data = image.into_raw();
        ndarray::Array3::from_shape_vec((height as usize, width as usize, 2), data)
    }
    pub(super) fn ndarray_to_image(
        array: &ndarray::ArrayView3<u8>,
    ) -> Result<image::GrayAlphaImage> {
        let (height, width, channels) = array.dim();
        let data = array.as_slice().ok_or_else(super::shape_error)?;
        if channels != 2 {
            return Err(super::shape_error());
        }
        image::GrayAlphaImage::from_raw(width as u32, height as u32, data.to_vec()).ok_or(
            ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape),
        )
    }
}

mod dynamic_image {
    use super::*;
    pub fn image_as_ndarray(image: &image::DynamicImage) -> Result<ndarray::ArrayViewD<'_, u8>> {
        Ok(match image {
            image::DynamicImage::ImageRgb8(img) => rgb8::image_as_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageRgba8(img) => rgba8::image_as_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageLuma8(img) => gray8::image_as_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageLumaA8(img) => gray_alpha8::image_as_ndarray(img)?.into_dyn(),
            _ => {
                unimplemented!("Unsupported image format: {:?}", image);
            }
        })
    }
    pub fn image_into_ndarray(image: image::DynamicImage) -> Result<ndarray::ArrayD<u8>> {
        Ok(match image {
            image::DynamicImage::ImageRgb8(img) => rgb8::image_into_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageRgba8(img) => rgba8::image_into_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageLuma8(img) => gray8::image_into_ndarray(img)?.into_dyn(),
            image::DynamicImage::ImageLumaA8(img) => {
                gray_alpha8::image_into_ndarray(img)?.into_dyn()
            }
            _ => {
                unimplemented!("Unsupported image format: {:?}", image);
            }
        })
    }
}
pub trait ImageToNdarray {
    type OwnedOutput;
    type RefOutput<'a>
    where
        Self: 'a;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>>;
    fn to_ndarray(&self) -> Result<Self::OwnedOutput>;
    fn into_ndarray(self) -> Result<Self::OwnedOutput>;
}

pub trait NdarrayToImage<ImageOutput> {
    fn to_image(&self) -> Result<ImageOutput>;
}

impl NdarrayToImage<image::RgbImage> for ndarray::ArrayView3<'_, u8> {
    fn to_image(&self) -> Result<image::RgbImage> {
        rgb8::ndarray_to_image(self)
    }
}

impl NdarrayToImage<image::RgbaImage> for ndarray::ArrayView3<'_, u8> {
    fn to_image(&self) -> Result<image::RgbaImage> {
        rgba8::ndarray_to_image(self)
    }
}

impl NdarrayToImage<image::GrayImage> for ndarray::ArrayView2<'_, u8> {
    fn to_image(&self) -> Result<image::GrayImage> {
        gray8::ndarray_to_image(self)
    }
}

impl NdarrayToImage<image::GrayAlphaImage> for ndarray::ArrayView3<'_, u8> {
    fn to_image(&self) -> Result<image::GrayAlphaImage> {
        gray_alpha8::ndarray_to_image(self)
    }
}

impl ImageToNdarray for image::RgbImage {
    type OwnedOutput = ndarray::Array3<u8>;
    type RefOutput<'a> = ndarray::ArrayView3<'a, u8>;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>> {
        rgb8::image_as_ndarray(self)
    }

    fn to_ndarray(&self) -> Result<Self::OwnedOutput> {
        Ok(self.as_ndarray()?.to_owned())
    }

    fn into_ndarray(self) -> Result<Self::OwnedOutput> {
        rgb8::image_into_ndarray(self)
    }
}

impl ImageToNdarray for image::RgbaImage {
    type OwnedOutput = ndarray::Array3<u8>;
    type RefOutput<'a> = ndarray::ArrayView3<'a, u8>;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>> {
        rgba8::image_as_ndarray(self)
    }

    fn to_ndarray(&self) -> Result<Self::OwnedOutput> {
        Ok(self.as_ndarray()?.to_owned())
    }

    fn into_ndarray(self) -> Result<Self::OwnedOutput> {
        rgba8::image_into_ndarray(self)
    }
}

impl ImageToNdarray for image::GrayImage {
    type OwnedOutput = ndarray::Array2<u8>;
    type RefOutput<'a> = ndarray::ArrayView2<'a, u8>;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>> {
        gray8::image_as_ndarray(self)
    }

    fn to_ndarray(&self) -> Result<Self::OwnedOutput> {
        Ok(self.as_ndarray()?.to_owned())
    }

    fn into_ndarray(self) -> Result<Self::OwnedOutput> {
        gray8::image_into_ndarray(self)
    }
}

impl ImageToNdarray for image::GrayAlphaImage {
    type OwnedOutput = ndarray::Array3<u8>;
    type RefOutput<'a> = ndarray::ArrayView3<'a, u8>;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>> {
        gray_alpha8::image_as_ndarray(self)
    }

    fn to_ndarray(&self) -> Result<Self::OwnedOutput> {
        Ok(self.as_ndarray()?.to_owned())
    }

    fn into_ndarray(self) -> Result<Self::OwnedOutput> {
        gray_alpha8::image_into_ndarray(self)
    }
}

impl ImageToNdarray for image::DynamicImage {
    type OwnedOutput = ndarray::ArrayD<u8>;
    type RefOutput<'a> = ndarray::ArrayViewD<'a, u8>;

    fn as_ndarray<'a>(&'a self) -> Result<Self::RefOutput<'a>> {
        dynamic_image::image_as_ndarray(self)
    }

    fn to_ndarray(&self) -> Result<Self::OwnedOutput> {
        Ok(self.as_ndarray()?.to_owned())
    }

    fn into_ndarray(self) -> Result<Self::OwnedOutput> {
        dynamic_image::image_into_ndarray(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};
    use ndarray::{Array2, Array3};

    // Deterministic pixel data; every (y, x, channel) gets a distinct-ish value
    // so a round trip is only correct if the layout is preserved exactly.
    fn arr3(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| (y * 37 + x * 7 + ch * 2 + 1) as u8)
    }

    fn arr2(h: usize, w: usize) -> Array2<u8> {
        Array2::from_shape_fn((h, w), |(y, x)| (y * 37 + x * 7 + 1) as u8)
    }

    #[test]
    fn rgb8_round_trip_preserves_data_and_dims() {
        let arr = arr3(2, 3, 3);
        let img: RgbImage = arr.view().to_image().unwrap();
        // image reports (width, height); ndarray shape is (height, width, _).
        assert_eq!(img.dimensions(), (3, 2));
        let back: Array3<u8> = img.into_ndarray().unwrap();
        assert_eq!(back, arr);
    }

    #[test]
    fn rgba8_round_trip_preserves_data_and_dims() {
        let arr = arr3(2, 3, 4);
        let img: RgbaImage = arr.view().to_image().unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        // to_ndarray borrows and copies; into_ndarray consumes.
        assert_eq!(img.to_ndarray().unwrap(), arr);
        let back: Array3<u8> = img.into_ndarray().unwrap();
        assert_eq!(back, arr);
    }

    #[test]
    fn gray8_round_trip_preserves_data_and_dims() {
        let arr = arr2(2, 3);
        let img: GrayImage = arr.view().to_image().unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        assert_eq!(img.to_ndarray().unwrap(), arr);
        let back: Array2<u8> = img.into_ndarray().unwrap();
        assert_eq!(back, arr);
    }

    #[test]
    fn gray_alpha8_round_trip_preserves_data_and_dims() {
        let arr = arr3(2, 3, 2);
        let img: GrayAlphaImage = arr.view().to_image().unwrap();
        assert_eq!(img.dimensions(), (3, 2));
        assert_eq!(img.to_ndarray().unwrap(), arr);
        let back: Array3<u8> = img.into_ndarray().unwrap();
        assert_eq!(back, arr);
    }

    #[test]
    fn to_ndarray_borrows_without_consuming() {
        let arr = arr3(2, 3, 3);
        let img: RgbImage = arr.view().to_image().unwrap();
        let view = img.as_ndarray().unwrap();
        assert_eq!(view.dim(), (2, 3, 3));
        // to_ndarray copies; the source image is still usable afterwards.
        let owned = img.to_ndarray().unwrap();
        assert_eq!(owned, arr);
        assert_eq!(img.dimensions(), (3, 2));
    }

    #[test]
    fn ndarray_to_image_rejects_wrong_channel_count() {
        // RGB expects 3 channels, RGBA 4, gray-alpha 2 — a mismatch must error,
        // not silently produce an image.
        let three = arr3(2, 3, 3);
        let four = arr3(2, 3, 4);
        let rgb: Result<RgbImage> = four.view().to_image();
        assert!(rgb.is_err());
        let rgba: Result<RgbaImage> = three.view().to_image();
        assert!(rgba.is_err());
        let gray_alpha: Result<GrayAlphaImage> = three.view().to_image();
        assert!(gray_alpha.is_err());
    }

    #[test]
    fn dynamic_image_dispatches_every_variant() {
        let rgb = DynamicImage::ImageRgb8(arr3(2, 3, 3).view().to_image().unwrap());
        let rgba = DynamicImage::ImageRgba8(arr3(2, 3, 4).view().to_image().unwrap());
        let luma = DynamicImage::ImageLuma8(arr2(2, 3).view().to_image().unwrap());
        let luma_a = DynamicImage::ImageLumaA8(arr3(2, 3, 2).view().to_image().unwrap());

        // as_ndarray path (via to_ndarray)
        assert_eq!(rgb.to_ndarray().unwrap().shape(), &[2, 3, 3]);
        assert_eq!(rgba.to_ndarray().unwrap().shape(), &[2, 3, 4]);
        assert_eq!(luma.to_ndarray().unwrap().shape(), &[2, 3]);
        assert_eq!(luma_a.to_ndarray().unwrap().shape(), &[2, 3, 2]);

        // into_ndarray path (consuming) must also cover every arm
        assert_eq!(rgb.into_ndarray().unwrap().shape(), &[2, 3, 3]);
        assert_eq!(rgba.into_ndarray().unwrap().shape(), &[2, 3, 4]);
        assert_eq!(luma.into_ndarray().unwrap().shape(), &[2, 3]);
        assert_eq!(luma_a.into_ndarray().unwrap().shape(), &[2, 3, 2]);
    }

    #[test]
    fn dynamic_rgb8_round_trips_values() {
        let arr = arr3(2, 3, 3);
        let dynimg = DynamicImage::ImageRgb8(arr.view().to_image().unwrap());
        let nd = dynimg.into_ndarray().unwrap();
        assert_eq!(nd, arr.into_dyn());
    }
}
