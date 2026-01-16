//! Colorspace conversion functions
//! ## Example
//! ```rust
//! let arr = Array3::<u8>::ones((100, 100, 3));
//! let out: Array3<u8> = arr.cvt::<Rgba<u8>, Rgb<u8>>()
//!
//! let arr = Array3::<u8>::ones((100, 100, 3));
//! let out: Array3<i8> = arr.cvt::<Rgb<u8>, Lab<i8>>()
//! ```
use crate::{NdAsImage, NdAsImageMut, prelude_::*};
use ndarray::*;

pub trait ColorSpace<Elem> {
    type Dim: ndarray::Dimension;
    const CHANNELS: usize;
}

mod seal {
    pub trait Sealed: bytemuck::Pod {}
    impl Sealed for u8 {} // 0 to 255
    impl Sealed for i8 {} // -128 to 128
    impl Sealed for u16 {} // 0 to 65535
    impl Sealed for f32 {} // 0 to 1
}

macro_rules! define_color_space {
    ($name:ident, $channels:expr, $depth:ty) => {
        pub struct $name<T: seal::Sealed> {
            __phantom: core::marker::PhantomData<T>,
        }
        impl<T: seal::Sealed> ColorSpace<T> for $name<T> {
            type Dim = $depth;
            const CHANNELS: usize = $channels;
        }
    };
}

define_color_space!(Rgb, 3, Ix3);
define_color_space!(Bgr, 3, Ix3);
define_color_space!(Rgba, 4, Ix3);
define_color_space!(Lab, 3, Ix3);

pub trait ToColorSpace<SrcT, DstT, Dst>: ColorSpace<SrcT>
where
    SrcT: seal::Sealed,
    Dst: ColorSpace<DstT>,
{
    fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes;
}

macro_rules! impl_color_converter {
    ($src:tt, $dst:tt, $code:expr) => {
        impl<T: seal::Sealed> ToColorSpace<T, T, $dst<T>> for $src<T> {
            fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
                $code
            }
        }
    };
}

impl_color_converter!(
    Rgb,
    Bgr,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2RGB
);
impl_color_converter!(
    Bgr,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2RGB
);
impl_color_converter!(
    Rgba,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_BGRA2BGR
);
impl_color_converter!(
    Rgb,
    Rgba,
    opencv::imgproc::ColorConversionCodes::COLOR_BGR2BGRA
);

impl ToColorSpace<u8, i8, Lab<i8>> for Rgb<u8> {
    fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
        opencv::imgproc::ColorConversionCodes::COLOR_RGB2Lab
    }
}
impl ToColorSpace<i8, u8, Rgb<u8>> for Lab<i8> {
    fn cv_colorspace_code() -> opencv::imgproc::ColorConversionCodes {
        opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB
    }
}

impl_color_converter!(
    Lab,
    Rgb,
    opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB
);

pub trait ConvertColor<T, T2, S>
where
    T: seal::Sealed,
    T2: seal::Sealed,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(&self) -> Result<ArrayBase<CowRepr<'_, T2>, Dst::Dim>, NdCvError>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>;
    fn cvt<Src, Dst>(&self) -> ArrayBase<CowRepr<'_, T2>, Dst::Dim>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>,
    {
        self.try_cvt::<Src, Dst>().expect("Color conversion failed")
    }
}

impl<T, S, T2> ConvertColor<T, T2, S> for ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>
where
    T: seal::Sealed + num::Zero,
    T2: seal::Sealed + num::Zero,
    S: ndarray::Data<Elem = T>,
{
    fn try_cvt<Src, Dst>(&self) -> Result<ArrayBase<CowRepr<'_, T2>, Dst::Dim>, NdCvError>
    where
        Src: ToColorSpace<T, T2, Dst>,
        Dst: ColorSpace<T2>,
        ArrayBase<OwnedRepr<T2>, <Dst as ColorSpace<T2>>::Dim>:
            NdAsImageMut<T2, <Dst as ColorSpace<T2>>::Dim>,
        ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>: NdAsImage<T, <Rgb<T> as ColorSpace<T>>::Dim>,
    {
        let size = self.shape();
        let mut new_size = Dst::Dim::zeros(Dst::Dim::NDIM.unwrap_or(self.ndim()));
        size.iter()
            .cloned()
            .take(size.len() - 1)
            .chain(std::iter::once(Dst::CHANNELS))
            .enumerate()
            .for_each(|(idx, val)| {
                new_size[idx] = val;
            });
        let mut dst_ndarray = ArrayBase::<ndarray::OwnedRepr<T2>, Dst::Dim>::zeros(new_size);
        let mat = self.as_image_mat()?;
        let mut dst_mat = dst_ndarray.as_image_mat_mut()?;
        opencv::imgproc::cvt_color(
            &*mat,
            &mut *dst_mat,
            <Src as ToColorSpace<T, T2, Dst>>::cv_colorspace_code().into(),
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .change_context(NdCvError)?;
        Ok(dst_ndarray.into())
    }
}

#[test]
fn test_usage() {
    use ndarray::Array3;
    let arr = Array3::<u8>::ones((100, 100, 3));
    let out: CowArray<i8, Ix3> = arr.cvt::<Rgb<u8>, Lab<i8>>();
    let expected = Array3::<i8>::zeros((100, 100, 3));
    assert_eq!(out, expected);
}
