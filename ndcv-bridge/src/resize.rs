use crate::{NdAsImage, NdAsImageMut, prelude_::*};

/// Resize ndarray using OpenCV resize functions
pub trait NdCvResize<T, D>: seal::SealedInternal {
    /// The input array must be a continuous 2D or 3D ndarray
    fn resize(
        &self,
        height: u16,
        width: u16,
        interpolation: Interpolation,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>, NdCvError>;
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

mod seal {
    pub trait SealedInternal {}
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> SealedInternal
        for ndarray::ArrayBase<S, ndarray::Ix3>
    {
    }
    impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> SealedInternal
        for ndarray::ArrayBase<S, ndarray::Ix2>
    {
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvResize<T, ndarray::Ix2>
    for ndarray::ArrayBase<S, ndarray::Ix2>
{
    fn resize(
        &self,
        height: u16,
        width: u16,
        interpolation: Interpolation,
    ) -> Result<ndarray::Array2<T>, NdCvError> {
        let mat = self.as_image_mat()?;
        let mut dest = ndarray::Array2::zeros((height.into(), width.into()));
        let mut dest_mat = dest.as_image_mat_mut()?;
        opencv::imgproc::resize(
            mat.as_ref(),
            dest_mat.as_mut(),
            opencv::core::Size {
                height: height.into(),
                width: width.into(),
            },
            0.,
            0.,
            interpolation as i32,
        )
        .change_context(NdCvError)?;
        Ok(dest)
    }
}

impl<T: bytemuck::Pod + num::Zero, S: ndarray::Data<Elem = T>> NdCvResize<T, ndarray::Ix3>
    for ndarray::ArrayBase<S, ndarray::Ix3>
{
    fn resize(
        &self,
        height: u16,
        width: u16,
        interpolation: Interpolation,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, ndarray::Ix3>, NdCvError> {
        let mat = self.as_image_mat()?;
        let mut dest =
            ndarray::Array3::zeros((height.into(), width.into(), self.len_of(ndarray::Axis(2))));
        let mut dest_mat = dest.as_image_mat_mut()?;
        opencv::imgproc::resize(
            mat.as_ref(),
            dest_mat.as_mut(),
            opencv::core::Size {
                height: height.into(),
                width: width.into(),
            },
            0.,
            0.,
            interpolation as i32,
        )
        .change_context(NdCvError)?;
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
