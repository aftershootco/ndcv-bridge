use crate::type_depth;
use crate::types::CvType;
use crate::{MatAsNd, NdAsImage, image::NdImage};

#[derive(Debug, Clone, derive_builder::Builder)]
#[builder(setter(into), pattern = "owned")]
pub struct SobelArgs {
    dxy: glam::IVec2,
    #[builder(default = "Ksize::K3")]
    ksize: Ksize,
    #[builder(default = "1.0")]
    scale: f64,
    #[builder(default = "0.0")]
    delta: f64,
    #[builder(default = "opencv::core::BorderTypes::BORDER_REFLECT_101")]
    border_type: opencv::core::BorderTypes,
}

impl SobelArgs {
    pub fn builder(dxy: impl Into<glam::IVec2>) -> SobelArgsBuilder {
        SobelArgsBuilder::default().dxy(dxy)
    }
    pub fn dxy(dxy: impl Into<glam::IVec2>) -> SobelArgsBuilder {
        Self::builder(dxy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Ksize {
    K1 = 1,
    K3 = 3,
    K5 = 5,
    K7 = 7,
}

#[derive(Debug, thiserror::Error)]
pub enum NdCvSobelError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvSobel<T: CvType, D: ndarray::Dimension>: crate::image::NdImage {
    fn sobel<U: CvType>(&self, args: SobelArgs) -> Result<ndarray::Array<U, D>, NdCvSobelError>;
}

impl<T, D, S> NdCvSobel<T, D> for ndarray::ArrayBase<S, D>
where
    T: CvType,
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = T> + ndarray::RawDataMut<Elem = T>,
    ndarray::ArrayBase<S, D>: NdAsImage<T, D>,
    ndarray::ArrayBase<S, D>: NdImage,
{
    fn sobel<U: CvType>(&self, args: SobelArgs) -> Result<ndarray::Array<U, D>, NdCvSobelError> {
        let img = self.as_image_mat()?;
        let mut dst = opencv::core::Mat::default();
        let ddepth = type_depth::<U>();
        opencv::imgproc::sobel(
            &*img,
            &mut dst,
            ddepth,
            args.dxy.x,
            args.dxy.y,
            args.ksize as i32,
            args.scale,
            args.delta,
            args.border_type as i32,
        )?;
        Ok(dst.as_ndarray()?.into_owned())
    }
}

#[test]
fn sobel_api_test_default() {
    let img = ndarray::Array::<u8, _>::from_shape_fn((10, 10), |(i, j)| (i * j) as u8);
    let result = img.sobel::<i16>(SobelArgs::dxy((1, 1)).build().unwrap());
    dbg!(&result);
    assert!(result.is_ok());
}
