//! Methods and type conversions for ndarray to opencv and vice versa
//!
//! # Public-API surface (PR #13, issue 4)
//!
//! Every operation trait should be reachable from the crate root, the way `NdCvResize`,
//! `NdCvNormalize`, `NdCvWarpAffine` and `NdCvDilate` are. The `blob` and `morphology`
//! modules, and part of `affine`, are declared `pub mod` but never re-exported, so the
//! following does NOT compile today. The `compile_fail` marker pins that gap; once the
//! re-exports are added this doctest starts failing and should be removed.
//!
//! ```compile_fail
//! use ndcv_bridge::{
//!     NdCvMorphologyEx, NdCvBlobFromImage, NdCvInvertWarpAffine, NdCvEstimateAffinePartial2D,
//!     MorphType, EstimateAffineMethod,
//! };
//! ```
//!
//! For reference, these ones already resolve (they are re-exported):
//!
//! ```
//! use ndcv_bridge::{NdCvNormalize, NdCvWarpAffine, NormType};
//! ```
mod blend;
#[cfg(feature = "opencv")]
pub mod dilate;
mod errors;
pub mod fir;
mod image;
mod inplace;
pub mod percentile;
mod roi;
pub use errors::NdCvError;

#[cfg(feature = "opencv")]
pub mod affine;
#[cfg(feature = "opencv")]
pub mod blob;
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
pub mod morphology;
#[cfg(feature = "opencv")]
pub mod normalize;
#[cfg(feature = "opencv")]
pub mod resize;
#[cfg(feature = "opencv")]
pub mod types;

// pub mod codec;
pub mod orient;
pub use blend::NdBlend;
pub use blur::NdCvBlur;
pub use dilate::{DilateError, NdCvDilate, NdCvDilateInPlace};
pub use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
pub use fir::NdFir;
pub use gaussian::{BorderType, NdCvGaussianBlur, NdCvGaussianBlurInPlace};
pub use image::NdImage;
pub use roi::{NdRoiZeroPadded, Roi as NdRoi, RoiMut as NdRoiMut};

#[cfg(feature = "opencv")]
pub use contours::{
    ContourApproximationMethod, ContourHierarchy, ContourResult, ContourRetrievalMode,
    NdCvContourArea, NdCvFindContours,
};

#[allow(deprecated)]
pub use conversions::NdCvConversion;

#[cfg(feature = "opencv")]
pub use affine::{NdCvEstimateAffinePartial2D, NdCvInvertWarpAffine, NdCvWarpAffine};
#[cfg(feature = "opencv")]
pub use bounding_rect::BoundingRect;
#[cfg(feature = "opencv")]
pub use connected_components::{Connectivity, NdCvConnectedComponents};
#[cfg(feature = "opencv")]
pub use conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat, NdAsMatMut};
#[cfg(feature = "opencv")]
pub use morphology::{MorphType, NdCvMorphologyEx};
#[cfg(feature = "opencv")]
pub use normalize::{NdCvNormalize, NormType};
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

#[cfg(all(test, feature = "opencv"))]
mod depth_type_tests {
    use super::depth_type;

    #[test]
    fn each_depth_maps_to_its_rust_type_name() {
        assert_eq!(depth_type(opencv::core::CV_8U), "u8");
        assert_eq!(depth_type(opencv::core::CV_8S), "i8");
        assert_eq!(depth_type(opencv::core::CV_16U), "u16");
        assert_eq!(depth_type(opencv::core::CV_16S), "i16");
        assert_eq!(depth_type(opencv::core::CV_32S), "i32");
        assert_eq!(depth_type(opencv::core::CV_32F), "f32");
        assert_eq!(depth_type(opencv::core::CV_64F), "f64");
    }

    #[test]
    #[should_panic(expected = "Unsupported depth")]
    fn unsupported_depth_panics() {
        depth_type(999);
    }
}
