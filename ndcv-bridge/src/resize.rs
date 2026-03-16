use crate::{NdAsImage, NdAsImageMut, types::CvType};

/// Resize ndarray using OpenCV resize functions
pub trait NdCvResize<T: CvType, D: ndarray::Dimension>: NdAsImage<T, D> {
    /// The input array must be a continuous 2D or 3D ndarray
    fn resize(
        &self,
        height: u16,
        width: u16,
        interpolation: Interpolation,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>, ResizeError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ResizeError {
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
}

#[repr(i32)]
#[derive(Debug, Copy, Clone)]
pub enum Interpolation {
    Linear = opencv::imgproc::INTER_LINEAR,
    LinearExact = opencv::imgproc::INTER_LINEAR_EXACT,
    Max = opencv::imgproc::INTER_MAX,
    Area = opencv::imgproc::INTER_AREA,
    Cubic = opencv::imgproc::INTER_CUBIC,
    Nearest = opencv::imgproc::INTER_NEAREST,
    NearestExact = opencv::imgproc::INTER_NEAREST_EXACT,
    Lanczos4 = opencv::imgproc::INTER_LANCZOS4,
}

impl<T, S, D> NdCvResize<T, D> for ndarray::ArrayBase<S, D>
where
    T: CvType + num::Zero,
    T: core::fmt::Debug,
    S: ndarray::Data<Elem = T>,
    ndarray::ArrayBase<S, D>: NdAsImage<T, D>,
    ndarray::Array<T, D>: NdAsImageMut<T, D>,
    D: ndarray::Dimension,
{
    fn resize(
        &self,
        height: u16,
        width: u16,
        interpolation: Interpolation,
    ) -> Result<ndarray::Array<T, D>, ResizeError> {
        let mat = self.as_image_mat()?;
        let mut size = ndarray::Dim(self.dim());
        let mut size_mut = size.as_array_view_mut();
        size_mut[0] = height as usize;
        size_mut[1] = width as usize;
        // size.0 = height as usize;
        // size.1 = width as usize;
        let mut dest: ndarray::Array<T, D> = ndarray::Array::zeros(size);
        let mut dest_mat = dest.as_image_mat_mut()?;
        // let mut output = opencv::core::Mat::default();
        opencv::imgproc::resize(
            mat.as_ref(),
            &mut dest_mat,
            opencv::core::Size {
                height: height.into(),
                width: width.into(),
            },
            0.,
            0.,
            interpolation as i32,
        )?;

        // let dest = output.as_ndarray()?.to_owned();
        Ok(dest)
    }
}

#[test]
fn test_resize_simple() {
    let foo = ndarray::Array2::<u8>::ones((10, 10));
    let foo_resized = foo.resize(15, 20, Interpolation::Linear).unwrap();
    assert_eq!(foo_resized, ndarray::Array2::<u8>::ones((15, 20)));
}

#[test]
fn test_resize_3d() {
    let foo = ndarray::Array3::<u8>::ones((10, 10, 3));
    let foo_resized = foo.resize(15, 20, Interpolation::Linear).unwrap();
    assert_eq!(foo_resized, ndarray::Array3::<u8>::ones((15, 20, 3)));
}
