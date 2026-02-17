use bounding_box::{Aabb2, nalgebra::Point2};
use error_stack::ResultExt;
use ndarray::{Array3, ArrayView2, ArrayViewMut3, Axis, s};
use tap::{Pipe, TapOptional};

// TODO: move ts somewhere else
#[derive(Debug, thiserror::Error)]
#[error("Paste Error")]
pub struct PasteError;

pub trait Paste<T> {
    type Out;

    fn paste(self, other: T) -> error_stack::Result<Self::Out, PasteError>;
}

// pub trait Paste: Sized {
//     type Out;
//
//     fn paste<T>(self, other: T) -> error_stack::Result<Self::Out, PasteError>
//     where
//         T: PastableOver<Self>,
//         <T as PastableOver<Self>>::Out: Into<Self::Out>,
//     {
//         other.paste_over(self).map(Into::into)
//     }
// }

// pub trait PastableOver<T: Paste> {
//     type Out: Into<T::Out>;
//
//     fn paste_over(self, src: T) -> error_stack::Result<Self::Out, PasteError>;
// }

pub trait PasteConfig<'a> {
    type Out;

    fn with_opts(self) -> Self::Out;
}
