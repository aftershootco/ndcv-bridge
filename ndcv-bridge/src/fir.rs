use fast_image_resize::pixels::{InnerPixel, Pixel, PixelComponent};
pub use fast_image_resize::*;
use images::{Image, ImageRef};

use crate::image::NdImage;

#[derive(Debug, Clone, thiserror::Error)]
#[error("NdFirError")]
pub enum NdFirError {
    #[error("Invalid Pixel Type")]
    InvalidPixelType {
        type_name: &'static str,
        channels: usize,
    },
    #[error("Non continuous ndarray")]
    NonContinuousNdarray,
    #[error("Failed to access ndarray rows")]
    FailedToAccessNdarrayRows,
    #[error("Image Conversion Failed: {0}")]
    ImageConversionFailure(#[from] fast_image_resize::ImageBufferError),
    #[error("Image Resize Failed: {0}")]
    ImageResizeFailure(#[from] fast_image_resize::ResizeError),
}

impl NdFirError {
    pub fn into_error(self) -> impl std::error::Error + Send + Sync + 'static {
        self
    }
}

type Result<T, E = NdFirError> = std::result::Result<T, E>;

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
        _ => Err(NdFirError::InvalidPixelType {
            type_name: core::any::type_name::<T>(),
            channels: u,
        }),
    }
}

mod seal {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

pub struct NdarrayImageContainer<'a, T, D>
where
    T: seal::Sealed,
    D: ndarray::Dimension,
{
    data: ndarray::ArrayView<'a, T, D>,
    pub _phantom: std::marker::PhantomData<(T, D)>,
}

pub struct NdarrayImageContainerTyped<'a, T, P, D>
where
    D: ndarray::Dimension,
    T: seal::Sealed + PixelComponent,
    P: PixelTrait,
{
    data: ndarray::ArrayView<'a, T, D>,
    __marker: std::marker::PhantomData<(T, P)>,
}

unsafe impl<'a, T, P, const N: usize> fast_image_resize::ImageView
    for NdarrayImageContainerTyped<'a, T, P, ndarray::Ix3>
where
    T: seal::Sealed + Send + Sync + PixelComponent + bytemuck::Pod,
    P: PixelTrait<Component = T, CountOfComponents = pixels::Count<N>>
        + pixels::InnerPixel
        + bytemuck::Pod,
{
    type Pixel = P;

    fn width(&self) -> u32 {
        self.data.width() as u32
    }

    fn height(&self) -> u32 {
        self.data.height() as u32
    }

    fn iter_rows(&self, start_row: u32) -> impl Iterator<Item = &[Self::Pixel]> {
        let strides = self.data.strides();
        // if the last two strides are not contiguous, we cannot cast slice
        assert_eq!(
            strides[strides.len() - 1],
            1,
            "Ndarray is not contiguous in the channel axis"
        );
        assert_eq!(
            strides[strides.len() - 2],
            N as isize,
            "Ndarray is not contiguous in the width axis"
        );
        self.data
            .outer_iter()
            .into_iter()
            .skip(start_row as usize)
            .map(|row| {
                let row = row
                    .reborrow()
                    .to_slice()
                    .expect("Sliced into the last axis");
                bytemuck::cast_slice(row)
            })
    }
}

impl<'a> fast_image_resize::IntoImageView for NdarrayImageContainer<'a, u8, ndarray::Ix3>
// where
//     T: seal::Sealed + Sync + Default + bytemuck::Pod + PixelComponent,
{
    fn pixel_type(&self) -> Option<PixelType> {
        let channels = self.data.channels();
        to_pixel_type::<u8>(channels).ok()
    }

    fn width(&self) -> u32 {
        self.data.width() as u32
    }

    fn height(&self) -> u32 {
        self.data.height() as u32
    }

    fn image_view<P: PixelTrait>(
        &self,
    ) -> Option<NdarrayImageContainerTyped<'_, u8, P, ndarray::Ix3>> {
        to_pixel_type::<T>(self.data.channels())
            .ok()
            .and_then(|ptpt| {
                (ptpt == P::pixel_type()).then_some({
                    NdarrayImageContainerTyped {
                        data: self.data.view(),
                        __marker: std::marker::PhantomData,
                    }
                })
            })
    }
}

// pub trait NdFir<T, D> {
//     fn fast_resize<'o>(
//         &self,
//         height: usize,
//         width: usize,
//         options: impl Into<Option<&'o ResizeOptions>>,
//     ) -> Result<ndarray::Array<T, D>>;
// }
//
// impl<T: seal::Sealed + bytemuck::Pod + num::Zero + Send + Sync, S: ndarray::Data<Elem = T>>
//     NdFir<T, ndarray::Ix3> for ndarray::ArrayBase<S, ndarray::Ix3>
// {
//     fn fast_resize<'o>(
//         &self,
//         height: usize,
//         width: usize,
//         options: impl Into<Option<&'o ResizeOptions>>,
//     ) -> Result<ndarray::Array3<T>> {
//         let source = self.as_image_ref()?;
//         let (_height, _width, channels) = self.dim();
//         let mut dest = ndarray::Array3::<T>::zeros((height, width, channels));
//         let mut dest_image = dest.as_image_ref_mut()?;
//         let mut resizer = fast_image_resize::Resizer::default();
//         resizer.resize(&source, &mut dest_image, options)?;
//         Ok(dest)
//     }
// }
//
// impl<T: seal::Sealed + bytemuck::Pod + num::Zero + Send + Sync, S: ndarray::Data<Elem = T>>
//     NdFir<T, ndarray::Ix2> for ndarray::ArrayBase<S, ndarray::Ix2>
// {
//     fn fast_resize<'o>(
//         &self,
//         height: usize,
//         width: usize,
//         options: impl Into<Option<&'o ResizeOptions>>,
//     ) -> Result<ndarray::Array<T, ndarray::Ix2>> {
//         let source = self.as_image_ref()?;
//         let (_height, _width) = self.dim();
//         let mut dest = ndarray::Array::<T, ndarray::Ix2>::zeros((height, width));
//         let mut dest_image = dest.as_image_ref_mut()?;
//         let mut resizer = fast_image_resize::Resizer::default();
//         resizer.resize(&source, &mut dest_image, options)?;
//         Ok(dest)
//     }
// }
//
// #[test]
// pub fn test_ndarray_fast_image_resize_u8() {
//     let source_fhd = ndarray::Array3::<u8>::ones((1920, 1080, 3));
//     let mut resized_hd = ndarray::Array3::<u8>::zeros((1280, 720, 3));
//     let mut resizer = fast_image_resize::Resizer::default();
//     resizer
//         .resize(
//             &source_fhd.as_image_ref().unwrap(),
//             &mut resized_hd.as_image_ref_mut().unwrap(),
//             None,
//         )
//         .unwrap();
//     assert_eq!(resized_hd.shape(), [1280, 720, 3]);
// }

#[test]
pub fn impl_test_fir() {
    use bounding_box::roi::Roi as _;
    use ndarray_image::{ImageToNdarray as _, NdarrayToImage as _};
    let image = image::open("/Users/fs0c131y/Downloads/monke.jpeg")
        .unwrap()
        .into_rgb8();
    let array: ndarray::ArrayView3<u8> = image.as_ndarray().unwrap();
    let half_sized_bbox = bounding_box::Aabb2::from_xywh(
        image.width() / 4,
        image.height() / 4,
        image.width() / 2,
        image.height() / 2,
    );
    let roi = array.roi(half_sized_bbox.cast()).unwrap();
    let roi = roi.fast_resize(256, 256, None).unwrap();
    let image = roi.as_standard_layout().to_owned().to_image().unwrap();
    image.save("../resized.jpg").unwrap();
}
