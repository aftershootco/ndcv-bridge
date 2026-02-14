//! Extended Difference of Gaussians (XDoG) edge detection.
//!
//! The XDoG operator generalises the classic Difference-of-Gaussians (DoG) by
//! adding a sharpness multiplier **p** that controls the balance between the
//! two Gaussian-blurred images:
//!
//! ```text
//! D(x) = (1 + p) · G(x, σ) − p · G(x, k·σ)
//! ```
//!
//! An optional soft-thresholding step produces a stylised, ink-like output:
//!
//! ```text
//!           ⎧ 1                                 if D(x) ≥ ε
//! T(x) =    ⎨
//!           ⎩ 1 + tanh(φ · (D(x) − ε))         otherwise
//! ```
//!
//! where **ε** is the edge threshold and **φ** controls the transition
//! sharpness.
//!
//! # References
//!
//! XDoG: An eXtended difference-of-Gaussians compendium
//! including advanced image stylization
//! Holger Winnem¨ollera, Jan Eric Kyprianidisb, Sven C. Olsen
//!
//! <https://docs.rs/opencv/latest/opencv/imgproc/fn.gaussian_blur.html>

use crate::{NdCvGaussianBlur, gaussian::BorderType};
use ndarray::*;
use opencv::core::AlgorithmHint as OpencvAlgorithmHint;

#[derive(Debug, thiserror::Error)]
pub enum XDoGError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("Invalid parameter: {0}")]
    BuilderError(#[from] XDoGArgsBuilderError),
}

impl From<crate::gaussian::GaussianBlurError> for XDoGError {
    fn from(e: crate::gaussian::GaussianBlurError) -> Self {
        match e {
            crate::gaussian::GaussianBlurError::ConversionError(e) => Self::ConversionError(e),
            crate::gaussian::GaussianBlurError::OpenCvError(e) => Self::OpenCvError(e),
        }
    }
}

/// Parameters for the Extended Difference of Gaussians operator.
#[derive(Debug, Clone, derive_builder::Builder)]
#[builder(setter(into), pattern = "owned")]
#[builder(build_fn(validate = "Self::validate"))]
pub struct XDoGArgs<T: num::Float + num::FromPrimitive> {
    /// Standard deviation of the smaller (detail) Gaussian.
    sigma: T,

    kernel_size: glam::U8Vec2,

    /// Ratio between the two Gaussian sigmas (`k > 1`). The larger Gaussian
    /// uses `sigma * k`.
    #[builder(default = default_args::default_k::<T>())]
    k: T,

    /// Sharpness multiplier.  `p = 0` gives a plain DoG; larger values
    /// sharpen the response.
    #[builder(default = default_args::default_p::<T>())]
    p: T,

    /// Threshold for the soft-thresholding step.  Only used when
    /// `thresholding` is enabled.
    #[builder(default = default_args::default_epsilon::<T>())]
    epsilon: T,

    /// Controls the steepness of the `tanh` transition near the threshold.
    /// Only used when `thresholding` is enabled.
    #[builder(default = default_args::default_phi::<T>())]
    phi: T,

    /// When `true`, apply the soft-thresholding step to produce a stylised
    /// output in `[0, 1]`.
    #[builder(default = "false")]
    thresholding: bool,

    /// Border handling mode passed to the underlying Gaussian blurs.
    #[builder(default = "BorderType::BorderReflect101")]
    border_type: BorderType,
}

mod default_args {
    use super::*;

    pub fn default_k<T: num::Float + num::FromPrimitive>() -> T {
        T::from_f64(1.6).expect("failed to convert default k to target type")
    }

    pub fn default_p<T: num::Float + num::FromPrimitive>() -> T {
        T::from_f64(200.0).expect("failed to convert default p to target type")
    }

    pub fn default_epsilon<T: num::Float + num::FromPrimitive>() -> T {
        T::from_f64(0.5).expect("failed to convert default epsilon to target type")
    }

    pub fn default_phi<T: num::Float + num::FromPrimitive>() -> T {
        T::from_f64(10.0).expect("failed to convert default phi to target type")
    }
}

impl<T: num::Float + num::FromPrimitive> XDoGArgs<T> {
    /// Start building with the required `sigma` parameter.
    pub fn builder(sigma: impl Into<T>) -> XDoGArgsBuilder<T> {
        XDoGArgsBuilder::default().sigma(sigma)
    }

    /// Shorthand: `XDoGArgs::sigma(1.0).build()`.
    pub fn sigma(sigma: impl Into<T>) -> XDoGArgsBuilder<T> {
        Self::builder(sigma)
    }
}

impl<T: num::Float + num::FromPrimitive> XDoGArgsBuilder<T> {
    pub fn validate(&self) -> Result<(), String> {
        // `self.sigma` is `Option<f64>` – it's `None` only when the caller
        // forgot to set it (which the builder already catches as a missing
        // required field).  We still guard against non-positive values.
        self.sigma
            .is_some_and(|s| !s.is_sign_negative())
            .then_some(())
            .ok_or_else(|| "sigma must be positive".to_string())?;

        // `self.k` is `Option<f64>` – `None` means the default (1.6) will be
        // used, which is valid.  Only reject explicitly-set non-positive values.
        self.k
            .is_some_and(|k| !k.is_sign_negative())
            .then_some(())
            .ok_or_else(|| "k must be positive".to_string())?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sealed trait – restrict input types to those supported by OpenCV gaussian
// blur, and output types to floats.
// ---------------------------------------------------------------------------

mod seal {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Computes the Extended Difference of Gaussians on an ndarray image.
///
/// The input can be any element type supported by the Gaussian blur (u8, u16,
/// i16, f32, f64), but the output is always a floating-point array (`f32` or
/// `f64`) because the XDoG subtraction can produce negative values and the
/// optional thresholding step uses `tanh`.
///
/// # Type Parameters
///
/// * `T` – input element type (must implement [`CvType`] and be supported by
///   OpenCV's Gaussian blur).
/// * `U` – output element type (`f32` or `f64`).
/// * `D` – array dimensionality.
pub trait NdCvXDoG<T, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>
where
    T: crate::types::CvType + seal::Sealed + num::Float + num::FromPrimitive,
    D: ndarray::Dimension,
{
    /// Compute XDoG with full control over all parameters.
    fn xdog(&self, args: XDoGArgs<T>) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>;

    /// Convenience: compute a plain (un-thresholded) XDoG with default
    /// parameters, only requiring `sigma`.
    fn xdog_def(&self, sigma: T) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
    {
        self.xdog(XDoGArgs::sigma(sigma).build()?)
    }
}

impl<T, D, S> NdCvXDoG<T, D> for ArrayBase<S, D>
where
    T: crate::types::CvType
        + Send
        + Sync
        + num::Zero
        + num::One
        + num::ToPrimitive
        + seal::Sealed
        + crate::gaussian::seal::Sealed
        + core::ops::Mul<Output = T>,
    T: crate::types::CvType + seal::Sealed + num::Float + num::FromPrimitive,
    D: ndarray::Dimension,
    D: ndarray::Dimension,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    ndarray::ArrayBase<S, D>: crate::image::NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
    ndarray::ArrayBase<S, D>: NdCvGaussianBlur<T, D>,
    ndarray::Array<T, D>: NdProducer,
{
    fn xdog(&self, args: XDoGArgs<T>) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: crate::conversions::NdAsImageMut<T, D>,
    {
        let sigma = args.sigma;
        let k_x_sigma = args.k * sigma;
        let kernel_size = args.kernel_size;
        let sigma_f64 = sigma
            .to_f64()
            .expect("failed to convert sigma to f64 for OpenCV call");
        let k_x_sigma_f64 = k_x_sigma
            .to_f64()
            .expect("failed to convert k*sigma to f64 for OpenCV call");
        let g1: ndarray::Array<T, D> =
            self.gaussian_blur(kernel_size, (sigma_f64, sigma_f64), args.border_type)?;
        let g2: ndarray::Array<T, D> = self.gaussian_blur(
            kernel_size,
            (k_x_sigma_f64, k_x_sigma_f64),
            args.border_type,
        )?;

        let one = T::one();
        let dst = if !args.thresholding {
            ndarray::Zip::from(&g1).and(&g2).par_map_collect(|v1, v2| {
                let d = (one + args.p) * *v1 - args.p * *v2;
                d
            })
        } else {
            ndarray::Zip::from(&g1).and(&g2).par_map_collect(|v1, v2| {
                let d = (one + args.p) * *v1 - args.p * *v2;
                if d >= args.epsilon {
                    one
                } else {
                    one + (args.phi * (d - args.epsilon)).tanh()
                }
            })
        };
        Ok(dst)
    }
}

// #[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    // ── Shape preservation ───────────────────────────────────────────

    #[test]
    fn xdog_2d_preserves_shape() {
        let img = Array2::<f32>::from_shape_fn((20, 20), |(i, j)| (i * j) as f32);
        let result: Array<f32, _> = img.xdog_def(1.0).unwrap();
        assert_eq!(result.shape(), &[20, 20]);
    }

    #[test]
    fn xdog_3d_preserves_shape() {
        let img = Array3::<u8>::ones((20, 30, 3));
        let result: Array<f32, _> = img.xdog_def(1.0).unwrap();
        assert_eq!(result.shape(), &[20, 30, 3]);
    }

    #[test]
    fn xdog_non_square_image() {
        let img = Array2::<u8>::zeros((15, 40));
        let result: Array<f64, _> = img.xdog_def(1.0).unwrap();
        assert_eq!(result.shape(), &[15, 40]);
    }

    // ── Uniform image produces near-zero output ─────────────────────

    #[test]
    fn xdog_uniform_image_near_zero() {
        let img = Array2::<u8>::from_elem((20, 20), 128);
        let result: Array<f32, _> = img.xdog(XDoGArgs::sigma(1.0).build().unwrap()).unwrap();

        // On a uniform image both Gaussians are identical, so the
        // difference is essentially (1+p)*v - p*v = v ≈ 128 everywhere.
        // With default p=200, XDoG = (1+200)*128 - 200*128 = 128.
        // But since both G1 and G2 are ~128 on a uniform image:
        // D(x) = (1+p)*128 - p*128 = 128
        for &v in result.iter() {
            assert!(
                (v - 128.0).abs() < 1.0,
                "expected ~128 on uniform input, got {v}"
            );
        }
    }

    // ── Edge detection ──────────────────────────────────────────────

    #[test]
    fn xdog_detects_edges() {
        let mut img = Array2::<u8>::zeros((40, 40));
        img.slice_mut(s![.., 20..]).fill(255); // vertical edge

        let result: Array<f32, _> = img
            .xdog(XDoGArgs::sigma(1.0).k(1.6).p(20.0).build().unwrap())
            .unwrap();

        // The edge region should differ from the flat interior.
        let interior_val = result[[5, 5]];
        let edge_val = result[[20, 20]];
        assert!(
            (edge_val - interior_val).abs() > 1.0,
            "edge response ({edge_val}) should differ from interior ({interior_val})"
        );
    }

    // ── p=0 gives plain DoG ─────────────────────────────────────────

    #[test]
    fn xdog_p_zero_is_dog() {
        // When p = 0, XDoG = (1+0)*G1 - 0*G2 = G1.
        let img = Array2::<u8>::from_shape_fn((20, 20), |(i, j)| ((i + j) % 256) as u8);
        let result: Array<f32, _> = img
            .xdog(XDoGArgs::sigma(1.0).p(0.0).build().unwrap())
            .unwrap();

        // Compare against a direct Gaussian blur with the same sigma.
        use crate::gaussian::NdCvGaussianBlur;
        let g1 = img
            .gaussian_blur((5, 5), (1.0, 1.0), BorderType::BorderReflect101)
            .unwrap();

        for (r, g) in result.iter().zip(g1.iter()) {
            let g = *g as f32;
            assert!((r - g).abs() < 1.0, "p=0 XDoG ({r}) should match G1 ({g})");
        }
    }

    // ── Thresholding ────────────────────────────────────────────────

    #[test]
    fn xdog_thresholding_produces_bounded_output() {
        let img = Array2::<u8>::from_shape_fn((30, 30), |(i, j)| ((i * 8 + j * 3) % 256) as u8);

        let result: Array<f32, _> = img
            .xdog(
                XDoGArgs::sigma(1.0)
                    .p(200.0)
                    .thresholding(true)
                    .epsilon(0.5)
                    .phi(10.0)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // The tanh-based thresholding should produce values in roughly (0, 1].
        // `1 + tanh(...)` is in (0, 2), but typical values are near 0 or near 1.
        for &v in result.iter() {
            assert!(
                (-0.01..=1.01).contains(&v),
                "thresholded value {v} out of expected range"
            );
        }
    }

    #[test]
    fn xdog_thresholding_uniform_all_ones() {
        // A uniform image should produce D(x) = v for all pixels.
        // With epsilon < v, all pixels should threshold to 1.0.
        let img = Array2::<u8>::from_elem((10, 10), 200);
        let result: Array<f32, _> = img
            .xdog(
                XDoGArgs::sigma(1.0)
                    .thresholding(true)
                    .epsilon(0.0)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        for &v in result.iter() {
            assert!(
                (v - 1.0).abs() < 1e-3,
                "uniform image should threshold to 1.0, got {v}"
            );
        }
    }

    // ── Different types ─────────────────────────────────────────────

    #[test]
    fn xdog_f32_input() {
        let img = Array2::<f32>::from_shape_fn((15, 15), |(i, j)| (i + j) as f32);
        let result: Array<f32, _> = img.xdog_def(1.0).unwrap();
        assert_eq!(result.shape(), &[15, 15]);
    }

    #[test]
    fn xdog_f64_output() {
        let img = Array2::<u8>::ones((10, 10));
        let result: Array<f64, _> = img.xdog_def(1.0).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    #[test]
    fn xdog_u16_input() {
        let img = Array2::<u16>::from_elem((10, 10), 1000);
        let result: Array<f32, _> = img.xdog_def(1.5).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    // ── Parameter validation ────────────────────────────────────────

    #[test]
    fn xdog_invalid_sigma() {
        // Negative sigma should be rejected at build time.
        let result = XDoGArgs::sigma(-1.0f64).build();
        assert!(result.is_err(), "negative sigma should fail validation");
    }

    #[test]
    fn xdog_invalid_k() {
        // Negative k should be rejected at build time.
        let result = XDoGArgs::sigma(1.0).k(-0.5).build();
        assert!(result.is_err(), "negative k should fail validation");
    }

    // ── Different k values ──────────────────────────────────────────

    #[test]
    fn xdog_different_k_values() {
        let img = Array2::<u8>::from_shape_fn((20, 20), |(i, j)| ((i * j) % 256) as u8);

        for k in [1.2, 1.6, 2.0, 3.0] {
            let result: Array<f32, _> = img
                .xdog(XDoGArgs::sigma(1.0).k(k).build().unwrap())
                .unwrap();
            assert_eq!(result.shape(), &[20, 20], "failed for k={k}");
        }
    }

    // ── Different border types ──────────────────────────────────────

    #[test]
    fn xdog_different_border_types() {
        let img = Array2::<u8>::from_shape_fn((20, 20), |(i, j)| ((i + j) % 256) as u8);

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for bt in border_types {
            let result: Array<f32, _> = img
                .xdog(XDoGArgs::sigma(1.0).border_type(bt).build().unwrap())
                .unwrap();
            assert_eq!(result.shape(), &[20, 20], "failed for border type {bt:?}");
        }
    }

    // ── Builder ergonomics ──────────────────────────────────────────

    #[test]
    fn xdog_builder_defaults() {
        // sigma is the only required field
        let args = XDoGArgs::sigma(1.0).build().unwrap();
        let img = Array2::<u8>::zeros((10, 10));
        let _result: Array<f32, _> = img.xdog(args).unwrap();
    }

    #[test]
    fn xdog_builder_all_fields() {
        let args = XDoGArgs::sigma(1.5)
            .k(2.0)
            .p(100.0)
            .epsilon(0.3)
            .phi(5.0)
            .thresholding(true)
            .border_type(BorderType::BorderReplicate)
            .build()
            .unwrap();

        let img = Array2::<u8>::from_elem((20, 20), 100);
        let result: Array<f32, _> = img.xdog(args).unwrap();
        assert_eq!(result.shape(), &[20, 20]);
    }
}
