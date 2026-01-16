//! Colorspace conversion functions
//! ## Example
//! ```rust
//! let arr = Array3::<u8>::ones((100, 100, 3));
//! let out: Array3<u8> = arr.cvt::<Rgba<u8>, Rgb<u8>>()
//! ```
use crate::{NdAsImage, NdAsImageMut, prelude_::*};
use ndarray::*;

pub trait ColorSpace<Elem> {
    type Dim: ndarray::Dimension;
    const CHANNELS: usize;
}

mod seal {
    pub trait Sealed: bytemuck::Pod {}
    // impl<T> Sealed for T {}
    impl Sealed for u8 {} // 0 to 255
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

pub trait ColorConverter<T, Dst>
where
    T: seal::Sealed,
    Dst: ColorSpace<T>,
{
    fn cv_color_code() -> opencv::imgproc::ColorConversionCodes;
}

macro_rules! impl_color_converter {
    ($src:ty, $dst:ty, $code:expr) => {
        impl<T: seal::Sealed> ColorConverter<T, $dst<T>> for $src<T> {
            fn cv_color_code() -> opencv::imgproc::ColorConversionCodes {
                $code
            }
        }
    };
}

impl<T: seal::Sealed> ColorConverter<T, Lab<T>> for Rgb<T> {
    fn cv_color_code() -> opencv::imgproc::ColorConversionCodes {
        (opencv::imgproc::ColorConversionCodes::COLOR_RGB2Lab)
    }
}
impl<T: seal::Sealed> ColorConverter<T, Rgb<T>> for Lab<T> {
    fn cv_color_code() -> opencv::imgproc::ColorConversionCodes {
        (opencv::imgproc::ColorConversionCodes::COLOR_Lab2RGB)
    }
}
// impl_color_converter!(
//     Rgb,
//     Bgr,
//     opencv::imgproc::ColorConversionCodes::COLOR_RGB2BGR
// );
// impl_color_converter!(
//     Bgr,
//     Rgb,
//     opencv::imgproc::ColorConversionCodes::COLOR_BGR2RGB
// );
// impl_color_converter!(
//     Rgba,
//     Rgb,
//     opencv::imgproc::ColorConversionCodes::COLOR_RGBA2RGB
// );
// impl_color_converter!(
//     Rgb,
//     Rgba,
//     opencv::imgproc::ColorConversionCodes::COLOR_RGB2RGBA
// );

pub trait ConvertColor<T, S, Src>
where
    T: seal::Sealed,
    ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
    Src: ColorSpace<T>,
    S: ndarray::Data<Elem = T>,
{
    fn cvt<Dst: ColorSpace<T>>(&self) -> ArrayBase<CowRepr<'_, T>, Dst::Dim>;
}

impl<T, S, Src> ConvertColor<T, S, Src> for ArrayBase<S, <Src as ColorSpace<T>>::Dim>
where
    T: seal::Sealed,
    T: num::Zero,
    S: ndarray::Data<Elem = T>,
    Self: NdAsImage<T, Src::Dim>,
    ArrayBase<S, <Src as ColorSpace<T>>::Dim>: NdAsImage<T, <Src as ColorSpace<T>>::Dim>,
    Src: ColorSpace<T>,
{
    fn cvt<Dst>(&self) -> ArrayBase<CowRepr<'_, T>, <Dst as ColorSpace<T>>::Dim>
    where
        Dst: ColorSpace<T>,
        // ArrayBase<ndarray::OwnedRepr<T>, Dst::Dim>: NdAsImageMut<T, Dst::Dim>,
    {
        let size = self.shape();
        let new_size: Vec<usize> = size
            .iter()
            .cloned()
            .take(size.len() - 1)
            .chain(std::iter::once(Dst::CHANNELS))
            .collect();
        let mut dst_ndarray = ArrayBase::<ndarray::OwnedRepr<T>, Dst::Dim>::zeros(new_size);
        // let mat = self.as_image_mat().expect("Failed to convert to Mat");
        // let mut dst_mat = dst_ndarray
        //     .as_image_mat_mut()
        //     .expect("Failed to convert to Mat");
        todo!()
        // opencv::imgproc::cvt_color(
        //     &*mat,
        //     &mut *dst_mat,
        //     <() as ColorConverter<T, Src, Dst>>::cv_color_code(),
        //     0,
        //     opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        // )
        // .expect("Failed to convert color");
        // dst_ndarray.into()
    }
}

// impl<T: seal::Sealed, S: ndarray::Data<Elem = T>> ConvertColor<T, Rgb<T>, Bgr<T>>
//     for ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>
// {
//     fn cvt(&self) -> ArrayBase<CowRepr<'_, T>, <Bgr<T> as ColorSpace<T>>::Dim> {
//         self.view().permuted_axes([2, 1, 0]).into()
//     }
// }
//
// impl<T: seal::Sealed, S: ndarray::Data<Elem = T>> ConvertColor<T, Bgr<T>, Rgb<T>>
//     for ArrayBase<S, <Bgr<T> as ColorSpace<T>>::Dim>
// {
//     fn cvt(&self) -> ArrayBase<CowRepr<'_, T>, <Rgb<T> as ColorSpace<T>>::Dim> {
//         self.view().permuted_axes([2, 1, 0]).into()
//     }
// }
//
// impl<T: seal::Sealed + num::Zero, S: ndarray::Data<Elem = T>> ConvertColor<T, Rgb<T>, Lab<T>>
//     for ArrayBase<S, <Rgb<T> as ColorSpace<T>>::Dim>
// {
//     fn cvt(&self) -> ArrayBase<CowRepr<'_, T>, <Lab<T> as ColorSpace<T>>::Dim> {
//         let mat = self.as_image_mat().expect("Failed to convert to Mat");
//         let mut lab_ndarray = Array3::<T>::zeros((
//             self.dim().0,
//             self.dim().1,
//             <Lab<T> as ColorSpace<T>>::CHANNELS,
//         ));
//         let mut lab_mat = lab_ndarray
//             .as_image_mat_mut()
//             .expect("Failed to convert to Mat");
//         opencv::imgproc::cvt_color(
//             &*mat,
//             &mut *lab_mat,
//             opencv::imgproc::COLOR_RGB2Lab,
//             0,
//             opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
//         )
//         .expect("Failed to convert color");
//         lab_ndarray.into()
//     }
// }
//
//
// // impl<T: seal::Sealed + num::Zero, S: ndarray::Data<Elem = T>> ConvertColor<T, Lab<T>, Rgb<T>>
// //     for ArrayBase<S, <Lab<T> as ColorSpace<T>>::Dim>
// // {
// //     fn cvt(&self) -> ArrayBase<CowRepr<'_, T>, <Rgb<T> as ColorSpace<T>>::Dim> {
// //         let mat = self.as_image_mat().expect("Failed to convert to Mat");
// //         let mut rgb_ndarray = Array3::<T>::zeros((
// //             self.dim().0,
// //             self.dim().1,
// //             <Rgb<T> as ColorSpace<T>>::CHANNELS,
// //         ));
// //         let mut rgb_mat = rgb_ndarray
// //             .as_image_mat_mut()
// //             .expect("Failed to convert to Mat");
// //         opencv::imgproc::cvt_color(
// //             &*mat,
// //             &mut *rgb_mat,
// //             opencv::imgproc::COLOR_Lab2RGB,
// //             0,
// //             opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
// //         )
// //         .expect("Failed to convert color");
// //         rgb_ndarray.into()
// //     }
// // }
