//! Methods and type conversions for ndarray to opencv and vice versa
mod blend;
mod errors;
// mod dilate;
pub mod fir;
mod image;
mod inplace;
pub mod percentile;
mod roi;
pub use errors::NdCvError;

#[cfg(feature = "opencv")]
pub mod absdiff;
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
pub mod conversions_v2;
#[cfg(feature = "opencv")]
pub mod gaussian;
#[cfg(feature = "opencv")]
pub mod resize;
#[cfg(feature = "opencv")]
pub mod sobel;
#[cfg(feature = "opencv")]
pub mod types;
#[cfg(feature = "opencv")]
pub mod xdog;

// pub mod codec;
pub mod orient;
pub use blend::NdBlend;
pub use blur::NdCvBlur;
pub use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
pub use fir::NdFir;
pub use gaussian::{BorderType, NdCvGaussianBlur, NdCvGaussianBlurInPlace};
pub use roi::{NdRoiZeroPadded, Roi as NdRoi, RoiMut as NdRoiMut};
pub use sobel::{Ksize, NdCvSobel, NdCvSobelError, SobelArgs};
pub use xdog::{NdCvXDoG, XDoGArgs, XDoGError};

#[cfg(feature = "opencv")]
pub use contours::{
    ContourApproximationMethod, ContourHierarchy, ContourResult, ContourRetrievalMode,
    NdCvContourArea, NdCvFindContours,
};

#[allow(deprecated)]
pub use conversions::NdCvConversion;

#[cfg(feature = "opencv")]
pub use absdiff::{NdCvAbsDiff, NdCvAbsDiffInPlace};
#[cfg(feature = "opencv")]
pub use bounding_rect::BoundingRect;
#[cfg(feature = "opencv")]
pub use connected_components::{Connectivity, NdCvConnectedComponents};
#[cfg(feature = "opencv")]
pub use conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat, NdAsMatMut};
#[cfg(feature = "opencv")]
pub use resize::{Interpolation, NdCvResize};

pub(crate) mod prelude_ {
    pub use crate::errors::NdCvError;
    pub use error_stack::*;
    pub type Result<T, C> = core::result::Result<T, Report<C>>;
}

#[cfg(feature = "opencv")]
pub fn type_depth<T: types::CvType>() -> i32 {
    use types::CvType;
    <T as CvType>::cv_depth()
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
