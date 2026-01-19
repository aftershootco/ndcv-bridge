//! Colorspace conversion functions
//! ## Example
//! ```rust
//! let arr = Array3::<u8>::ones((100, 100, 3));
//! let out: Array3<u8> = arr.cvt::<Rgba<u8>, Rgb<u8>>()
//! ```
use crate::prelude_::*;
use ndarray::*;

pub trait ColorSpace {
    type Elem: seal::Sealed;
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
        pub struct $name<T> {
            __phantom: core::marker::PhantomData<T>,
        }
        impl<T: seal::Sealed> ColorSpace for $name<T> {
            type Elem = T;
            type Dim = $depth;
            const CHANNELS: usize = $channels;
        }
    };
}

define_color_space!(Rgb, 3, Ix3);
define_color_space!(Bgr, 3, Ix3);
define_color_space!(Rgba, 4, Ix3);

pub trait NdArray<T, D: ndarray::Dimension> {}
impl<T, D: ndarray::Dimension, S: ndarray::Data<Elem = T>> NdArray<S, D> for ArrayBase<S, D> {}

pub trait ConvertColor<T, U>
where
    T: ColorSpace,
    U: ColorSpace,
    Self: NdArray<T::Elem, T::Dim>,
{
    type Output: NdArray<U::Elem, U::Dim>;
    fn cvt(&self) -> Self::Output;
}

// impl<T: seal::Sealed, S: ndarray::Data<Elem = T>> ConvertColor<Rgb<T>, Bgr<T>> for ArrayBase<S, Ix3>
// where
//     Self: NdArray<T, Ix3>,
// {
//     type Output = ArrayView3<'a, T>;
//     fn cvt(&self) -> CowArray<T, Ix3> {
//         self.view().permuted_axes([2, 1, 0]).into()
//     }
// }
//
// impl<T: seal::Sealed, S: ndarray::Data<Elem = T>> ConvertColor<Bgr<T>, Rgb<T>> for ArrayBase<S, Ix3>
// where
//     Self: NdArray<T, Ix3>,
// {
//     type Output = ArrayView3<'a, T>;
//     fn cvt(&self) -> CowArray<T, Ix3> {
//         self.view().permuted_axes([2, 1, 0]).into()
//     }
// }

// impl<T: seal::Sealed + num::One + num::Zero, S: ndarray::Data<Elem = T>>
//     ConvertColor<Rgb<T>, Rgba<T>> for ArrayBase<S, Ix3>
// {
//     fn cvt(&self) -> CowArray<T, Ix3> {
//         let mut out = Array3::<T>::zeros((self.height(), self.width(), 4));
//         // Zip::from(&mut out).and(self).for_each(|out, &in_| {
//         //     out[0] = in_[0];
//         //     out[1] = in_[1];
//         //     out[2] = in_[2];
//         //     out[3] = T::one();
//         // });
//         out.into()
//     }
// }
