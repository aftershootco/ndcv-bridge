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
