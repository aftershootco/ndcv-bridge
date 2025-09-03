use error_stack::*;
use fast_image_resize::*;
use images::{Image, ImageRef};
#[derive(Debug, Clone, thiserror::Error)]
#[error("NdFirError")]
pub struct NdFirError;
type Result<T, E = Report<NdFirError>> = std::result::Result<T, E>;

pub trait NdAsImage<T: seal::Sealed, D: ndarray::Dimension>: Sized {
    fn as_image_ref(&self) -> Result<ImageRef<'_>>;
}

pub trait NdAsImageMut<T: seal::Sealed, D: ndarray::Dimension>: Sized {
    fn as_image_ref_mut(&mut self) -> Result<Image<'_>>;
}

pub struct NdarrayImageContainer<'a, T: seal::Sealed, D: ndarray::Dimension> {
    #[allow(dead_code)]
    data: ndarray::ArrayView<'a, T, D>,
    pub _phantom: std::marker::PhantomData<(T, D)>,
}

impl<'a, T: seal::Sealed> NdarrayImageContainer<'a, T, ndarray::Ix3> {
    pub fn new<S: ndarray::Data<Elem = T>>(array: &'a ndarray::ArrayBase<S, ndarray::Ix3>) -> Self {
        Self {
            data: array.view(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T: seal::Sealed> NdarrayImageContainer<'a, T, ndarray::Ix2> {
    pub fn new<S: ndarray::Data<Elem = T>>(array: &'a ndarray::ArrayBase<S, ndarray::Ix2>) -> Self {
        Self {
            data: array.view(),
            _phantom: std::marker::PhantomData,
        }
    }
}
pub struct NdarrayImageContainerMut<'a, T: seal::Sealed, D: ndarray::Dimension> {
    #[allow(dead_code)]
    data: ndarray::ArrayViewMut<'a, T, D>,
}

impl<'a, T: seal::Sealed> NdarrayImageContainerMut<'a, T, ndarray::Ix3> {
    pub fn new<S: ndarray::DataMut<Elem = T>>(
        array: &'a mut ndarray::ArrayBase<S, ndarray::Ix3>,
    ) -> Self {
        Self {
            data: array.view_mut(),
        }
    }
}

impl<'a, T: seal::Sealed> NdarrayImageContainerMut<'a, T, ndarray::Ix2> {
    pub fn new<S: ndarray::DataMut<Elem = T>>(
        array: &'a mut ndarray::ArrayBase<S, ndarray::Ix2>,
    ) -> Self {
        Self {
            data: array.view_mut(),
        }
    }
}

pub struct NdarrayImageContainerTyped<'a, T: seal::Sealed, D: ndarray::Dimension, P: PixelTrait> {
    #[allow(dead_code)]
    data: ndarray::ArrayView<'a, T, D>,
    __marker: std::marker::PhantomData<P>,
}

// unsafe impl<'a, T: seal::Sealed + Sync + InnerPixel, P: PixelTrait> ImageView
//     for NdarrayImageContainerTyped<'a, T, ndarray::Ix3, P>
// where
//     T: bytemuck::Pod,
// {
//     type Pixel = P;
//     fn width(&self) -> u32 {
//         self.data.shape()[1] as u32
//     }
//     fn height(&self) -> u32 {
//         self.data.shape()[0] as u32
//     }
//     fn iter_rows(&self, start_row: u32) -> impl Iterator<Item = &[Self::Pixel]> {
//         self.data
//             .rows()
//             .into_iter()
//             .skip(start_row as usize)
//             .map(|row| {
//                 row.as_slice()
//                     .unwrap_or_default()
//                     .chunks_exact(P::CHANNELS as usize)
//             })
//     }
// }

// impl<'a, T: fast_image_resize::pixels::InnerPixel + seal::Sealed, D: ndarray::Dimension>
//     fast_image_resize::IntoImageView for NdarrayImageContainer<'a, T, D>
// {
//     fn pixel_type(&self) -> Option<PixelType> {
//         match D::NDIM {
//             Some(2) => Some(to_pixel_type::<T>(1).expect("Failed to convert to pixel type")),
//             Some(3) => Some(
//                 to_pixel_type::<T>(self.data.shape()[2]).expect("Failed to convert to pixel type"),
//             ),
//             _ => None,
//         }
//     }
//     fn width(&self) -> u32 {
//         self.data.shape()[1] as u32
//     }
//     fn height(&self) -> u32 {
//         self.data.shape()[0] as u32
//     }
//     fn image_view<P: PixelTrait>(&'a self) -> Option<NdarrayImageContainerTyped<'a, T, D, P>> {
//         Some(NdarrayImageContainerTyped {
//             data: self.data.view(),
//             __marker: std::marker::PhantomData,
//         })
//     }
// }

pub fn to_pixel_type<T: seal::Sealed>(u: usize) -> Result<PixelType> {
    match (core::any::type_name::<T>(), u) {
        ("u8", 1) => Ok(PixelType::U8),
        ("u8", 2) => Ok(PixelType::U8x2),
        ("u8", 3) => Ok(PixelType::U8x3),
        ("u8", 4) => Ok(PixelType::U8x4),
        ("u16", 1) => Ok(PixelType::U16),
        ("i32", 1) => Ok(PixelType::I32),
        ("f32", 1) => Ok(PixelType::F32),
        ("f32", 2) => Ok(PixelType::F32x2),
        ("f32", 3) => Ok(PixelType::F32x3),
        ("f32", 4) => Ok(PixelType::F32x4),
        _ => Err(Report::new(NdFirError).attach("Unsupported pixel type")),
    }
}

mod seal {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

impl<S: ndarray::Data<Elem = T>, T: seal::Sealed + bytemuck::Pod, D: ndarray::Dimension>
    NdAsImage<T, D> for ndarray::ArrayBase<S, D>
{
    /// Clones self and makes a new image
    fn as_image_ref(&self) -> Result<ImageRef<'_>> {
        let shape = self.shape();
        let rows = *shape
            .first()
            .ok_or_else(|| Report::new(NdFirError).attach("Failed to get rows"))?
            as u32;
        let cols = *shape.get(1).unwrap_or(&1) as u32;
        let channels = *shape.get(2).unwrap_or(&1);
        let data = self
            .as_slice()
            .ok_or(NdFirError)
            .attach("The ndarray is non continuous")?;
        let data_bytes: &[u8] = bytemuck::cast_slice(data);

        let pixel_type = to_pixel_type::<T>(channels)?;
        ImageRef::new(cols, rows, data_bytes, pixel_type)
            .change_context(NdFirError)
            .attach("Failed to create Image from ndarray")
    }
}

impl<S: ndarray::DataMut<Elem = T>, T: seal::Sealed + bytemuck::Pod, D: ndarray::Dimension>
    NdAsImageMut<T, D> for ndarray::ArrayBase<S, D>
{
    fn as_image_ref_mut(&mut self) -> Result<Image<'_>>
    where
        S: ndarray::DataMut<Elem = T>,
    {
        let shape = self.shape();
        let rows = *shape
            .first()
            .ok_or_else(|| Report::new(NdFirError).attach("Failed to get rows"))?
            as u32;
        let cols = *shape.get(1).unwrap_or(&1) as u32;
        let channels = *shape.get(2).unwrap_or(&1);
        let data = self
            .as_slice_mut()
            .ok_or(NdFirError)
            .attach("The ndarray is non continuous")?;
        let data_bytes: &mut [u8] = bytemuck::cast_slice_mut(data);

        let pixel_type = to_pixel_type::<T>(channels)?;
        Image::from_slice_u8(cols, rows, data_bytes, pixel_type)
            .change_context(NdFirError)
            .attach("Failed to create Image from ndarray")
    }
}

pub trait NdFir<T, D> {
    fn fast_resize<'o>(
        &self,
        height: usize,
        width: usize,
        options: impl Into<Option<&'o ResizeOptions>>,
    ) -> Result<ndarray::Array<T, D>>;
}

impl<T: seal::Sealed + bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdFir<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn fast_resize<'o>(
        &self,
        height: usize,
        width: usize,
        options: impl Into<Option<&'o ResizeOptions>>,
    ) -> Result<ndarray::Array3<T>> {
        let source = self.as_image_ref()?;
        let (_height, _width, channels) = self.dim();
        let mut dest = ndarray::Array3::<T>::zeros((height, width, channels));
        let mut dest_image = dest.as_image_ref_mut()?;
        let mut resizer = fast_image_resize::Resizer::default();
        resizer
            .resize(&source, &mut dest_image, options)
            .change_context(NdFirError)?;
        Ok(dest)
    }
}

impl<T: seal::Sealed + bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdFir<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn fast_resize<'o>(
        &self,
        height: usize,
        width: usize,
        options: impl Into<Option<&'o ResizeOptions>>,
    ) -> Result<ndarray::Array<T, ndarray::Ix2>> {
        let source = self.as_image_ref()?;
        let (_height, _width) = self.dim();
        let mut dest = ndarray::Array::<T, ndarray::Ix2>::zeros((height, width));
        let mut dest_image = dest.as_image_ref_mut()?;
        let mut resizer = fast_image_resize::Resizer::default();
        resizer
            .resize(&source, &mut dest_image, options)
            .change_context(NdFirError)?;
        Ok(dest)
    }
}

#[test]
pub fn test_ndarray_fast_image_resize_u8() {
    let source_fhd = ndarray::Array3::<u8>::ones((1920, 1080, 3));
    let mut resized_hd = ndarray::Array3::<u8>::zeros((1280, 720, 3));
    let mut resizer = fast_image_resize::Resizer::default();
    resizer
        .resize(
            &source_fhd.as_image_ref().unwrap(),
            &mut resized_hd.as_image_ref_mut().unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(resized_hd.shape(), [1280, 720, 3]);
}
