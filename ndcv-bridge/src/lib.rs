//! Methods and type conversions for ndarray to opencv and vice versa
pub mod dilate;
pub mod image;
mod inplace;

pub(crate) mod sealer {
    #[doc(hidden)]
    pub struct __Sealed__;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! seal {
        () => {
            #[doc(hidden)]
            fn __sealed() -> $crate::sealer::__Sealed__;
        };
        (impl) => {
            #[doc(hidden)]
            fn __sealed() -> $crate::sealer::__Sealed__ {
                $crate::sealer::__Sealed__
            }
        };
        (impl, $name: ident, $($typ: ty),*) => {
            $(
                impl $name for $typ {
                    $crate::seal!(impl);
                }
             )*
        };
        ($name: ident, $($typ: ty),*) => {
            pub trait $name {
                $crate::seal!();
            }
            $crate::seal!(impl, $name, $($typ),*);

        };
    }
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
pub use gaussian::{NdCvGaussianBlur, NdCvGaussianBlurInPlace};
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

use opencv::core::AlgorithmHint as OpencvAlgorithmHint;
#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub enum BorderType {
    #[default]
    BorderConstant = 0,
    BorderReplicate = 1,
    BorderReflect = 2,
    BorderWrap = 3,
    BorderReflect101 = 4,
    BorderTransparent = 5,
    BorderIsolated = 16,
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub enum AlgorithmHint {
    #[default]
    AlgoHintDefault = OpencvAlgorithmHint::ALGO_HINT_DEFAULT as isize,
    AlgoHintAccurate = OpencvAlgorithmHint::ALGO_HINT_ACCURATE as isize,
    AlgoHintApprox = OpencvAlgorithmHint::ALGO_HINT_APPROX as isize,
}

impl AlgorithmHint {
    pub fn to_opencv(self) -> OpencvAlgorithmHint {
        match self {
            AlgorithmHint::AlgoHintDefault => OpencvAlgorithmHint::ALGO_HINT_DEFAULT,
            AlgorithmHint::AlgoHintAccurate => OpencvAlgorithmHint::ALGO_HINT_ACCURATE,
            AlgorithmHint::AlgoHintApprox => OpencvAlgorithmHint::ALGO_HINT_APPROX,
        }
    }
}
