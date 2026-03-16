//! Methods and type conversions for ndarray to opencv and vice versa
pub mod dilate;
pub mod fir;
pub mod image;
mod inplace;
mod roi;

pub(crate) mod sealer {
    #[doc(hidden)]
    pub struct __Sealed__;
}

pub mod absdiff;
pub mod blur;
pub mod bounding_rect;
pub mod color_space;
pub mod connected_components;
pub mod contours;
pub mod conversions;
pub mod conversions_v2;
pub mod gaussian;
pub mod resize;
pub mod sobel;
pub mod types;

// pub mod codec;
pub use blur::NdCvBlur;
pub use dilate::{DilateError, NdCvDilate, NdCvDilateInPlace};
pub use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
pub use fir::NdFir;
pub use gaussian::{BorderType, NdCvGaussianBlur, NdCvGaussianBlurInPlace};
pub use roi::{NdRoiZeroPadded, Roi as NdRoi, RoiMut as NdRoiMut};
pub use sobel::{Ksize, NdCvSobel, NdCvSobelError, SobelArgs};

pub use contours::{
    ContourApproximationMethod, ContourHierarchy, ContourResult, ContourRetrievalMode,
    NdCvContourArea, NdCvFindContours,
};

#[allow(deprecated)]
pub use conversions::NdCvConversion;

pub use absdiff::{NdCvAbsDiff, NdCvAbsDiffInPlace};
pub use bounding_rect::BoundingRect;
pub use connected_components::{Connectivity, NdCvConnectedComponents};
pub use conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat, NdAsMatMut};
pub use resize::{Interpolation, NdCvResize};

pub fn type_depth<T: types::CvType>() -> i32 {
    use types::CvType;
    <T as CvType>::cv_depth()
}

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
