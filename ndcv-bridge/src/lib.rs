//! Methods and type conversions for ndarray to opencv and vice versa

#[cfg(all(feature = "ndarray_0_16", not(feature = "ndarray_0_17")))]
extern crate ndarray_0_16 as ndarray;
#[cfg(feature = "ndarray_0_17")]
extern crate ndarray_0_17 as ndarray;

#[cfg(all(not(feature = "ndarray_0_17"), not(feature = "ndarray_0_16")))]
compile_error!(
    "Either 'ndarray_0_16' or 'ndarray_0_17' feature is required for this crate to work. Please enable either"
);

mod blend;
mod errors;
pub mod fir;
mod image;
pub mod percentile;
pub use errors::NdCvError;

#[cfg(feature = "opencv")]
pub mod blur;
#[cfg(feature = "opencv")]
pub mod bounding_rect;
#[cfg(feature = "opencv")]
pub mod color_space;
#[cfg(feature = "opencv")]
pub mod connected_components;
#[cfg(feature = "opencv")]
pub mod contours;
#[cfg(feature = "opencv")]
pub mod conversions;
#[cfg(feature = "opencv")]
pub mod gaussian;
#[cfg(feature = "opencv")]
mod inplace;
#[cfg(feature = "opencv")]
pub mod orient;
#[cfg(feature = "opencv")]
pub mod resize;

#[cfg(feature = "roi")]
mod roi;
#[cfg(feature = "roi")]
pub use roi::{NdRoiZeroPadded, Roi as NdRoi, RoiMut as NdRoiMut};

pub use blend::NdBlend;
pub use blur::NdCvBlur;
pub use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
pub use fir::NdFir;

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
pub use conversions::NdCvConversion;
#[cfg(feature = "opencv")]
pub use conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat, NdAsMatMut};
#[cfg(feature = "opencv")]
pub use gaussian::{BorderType, NdCvGaussianBlur, NdCvGaussianBlurInPlace};
#[cfg(feature = "opencv")]
pub use resize::{Interpolation, NdCvResize};

pub(crate) mod prelude_ {
    pub use crate::errors::NdCvError;
    pub use error_stack::*;
    pub type Result<T, C> = core::result::Result<T, Report<C>>;
}

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
