//! Methods and type conversions for ndarray to opencv and vice versa
mod blend;
// mod dilate;
pub mod fir;
mod image;
mod inplace;
pub mod percentile;
mod roi;

#[cfg(feature = "opencv")]
pub mod bounding_rect;
// #[cfg(feature = "opencv")]
// pub mod color_space;
#[cfg(feature = "opencv")]
pub mod connected_components;
#[cfg(feature = "opencv")]
pub mod contours;
#[cfg(feature = "opencv")]
pub mod conversions;
#[cfg(feature = "opencv")]
pub mod gaussian;
#[cfg(feature = "opencv")]
pub mod resize;

// pub mod codec;
pub mod orient;
pub use blend::NdBlend;
pub use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
pub use fir::NdFir;
pub use gaussian::{BorderType, NdCvGaussianBlur, NdCvGaussianBlurInPlace};
pub use roi::{NdRoiZeroPadded, Roi as NdRoi, RoiMut as NdRoiMut};

#[cfg(feature = "opencv")]
pub use contours::{
    ContourApproximationMethod, ContourHierarchy, ContourResult, ContourRetrievalMode,
    NdCvContourArea, NdCvFindContours,
};

#[cfg(feature = "opencv")]
pub use bounding_rect::BoundingRect;
#[cfg(feature = "opencv")]
pub use connected_components::{Connectivity, NdCvConnectedComponents};
#[cfg(feature = "opencv")]
pub use conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat, NdAsMatMut, NdCvConversion};
#[cfg(feature = "opencv")]
pub use resize::{Interpolation, NdCvResize};

pub(crate) mod prelude_ {
    pub use crate::NdCvError;
    pub use error_stack::*;
    pub type Result<T, C> = core::result::Result<T, Report<C>>;
}

#[derive(Debug, thiserror::Error)]
#[error("NdCvError")]
pub struct NdCvError;

#[cfg(feature = "opencv")]
pub fn type_depth<T>() -> i32 {
    match std::any::type_name::<T>() {
        "u8" => opencv::core::CV_8U,
        "i8" => opencv::core::CV_8S,
        "u16" => opencv::core::CV_16U,
        "i16" => opencv::core::CV_16S,
        "i32" => opencv::core::CV_32S,
        "f32" => opencv::core::CV_32F,
        "f64" => opencv::core::CV_64F,
        _ => panic!("Unsupported type"),
    }
}

#[cfg(feature = "opencv")]
pub const fn depth_type(depth: i32) -> &'static str {
    match depth {
        opencv::core::CV_8U => "u8",
        opencv::core::CV_8S => "i8",
        opencv::core::CV_16U => "u16",
        opencv::core::CV_16S => "i16",
        opencv::core::CV_32S => "i32",
        opencv::core::CV_32F => "f32",
        opencv::core::CV_64F => "f64",
        _ => panic!("Unsupported depth"),
    }
}
