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

use ndarray::*;
use ndcv_bridge::{NdCvGaussianBlur, gaussian::BorderType};

#[derive(Debug, thiserror::Error)]
pub enum XDoGError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ndcv_bridge::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("Invalid parameter: {0}")]
    BuilderError(#[from] XDoGArgsBuilderError),
}

impl From<ndcv_bridge::gaussian::GaussianBlurError> for XDoGError {
    fn from(e: ndcv_bridge::gaussian::GaussianBlurError) -> Self {
        match e {
            ndcv_bridge::gaussian::GaussianBlurError::ConversionError(e) => {
                Self::ConversionError(e)
            }
            ndcv_bridge::gaussian::GaussianBlurError::OpenCvError(e) => Self::OpenCvError(e),
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
    #[builder(default = default_t::<T>(1.6))]
    k: T,

    /// Sharpness multiplier.  `p = 0` gives a plain DoG; larger values
    /// sharpen the response.
    #[builder(default = default_t::<T>(0.0))]
    p: T,

    /// Threshold for the soft-thresholding step.  Only used when
    /// `thresholding` is enabled.
    #[builder(default = default_t::<T>(0.0))]
    epsilon: T,

    /// Controls the steepness of the `tanh` transition near the threshold.
    /// Only used when `thresholding` is enabled.
    #[builder(default = default_t::<T>(0.0))]
    phi: T,

    /// When `true`, apply the soft-thresholding step to produce a stylised
    /// output in `[0, 1]`.
    #[builder(default = "false")]
    thresholding: bool,

    /// Border handling mode passed to the underlying Gaussian blurs.
    #[builder(default = "BorderType::BorderReflect101")]
    border_type: BorderType,
}

fn default_t<T: num::Float + num::FromPrimitive>(v: f64) -> T {
    T::from_f64(v).expect("failed to convert default t to target type")
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
        // `self.sigma` is `Option<T>` – it's `None` only when the caller
        // forgot to set it (which the builder already catches as a missing
        // required field).  We still guard against non-positive values.
        if let Some(s) = self.sigma
            && s.is_sign_negative()
        {
            return Err("sigma must be positive".to_string());
        }

        // `self.k` is `Option<T>` – `None` means the default (1.6) will be
        // used, which is valid.  Only reject explicitly-set non-positive values.
        if let Some(k) = self.k
            && k.is_sign_negative()
        {
            return Err("k must be positive".to_string());
        }

        if let Some(kernel_size) = self.kernel_size {
            if kernel_size.x % 2 == 0 || kernel_size.y % 2 == 0 {
                return Err("kernel size must be odd".to_string());
            }
        }

        Ok(())
    }
}

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
/// * `D` – array dimensionality.
pub trait NdCvXDoG<T, D>:
    ndcv_bridge::image::NdImage + ndcv_bridge::conversions::NdAsImage<T, D>
where
    T: ndcv_bridge::types::CvType + seal::Sealed + num::Float + num::FromPrimitive,
    D: ndarray::Dimension,
{
    /// Compute XDoG with full control over all parameters.
    fn xdog(&self, args: XDoGArgs<T>) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: ndcv_bridge::conversions::NdAsImageMut<T, D>;

    /// Convenience: compute a plain (un-thresholded) XDoG with default
    /// parameters, only requiring `sigma`.
    fn xdog_def(&self, sigma: T) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: ndcv_bridge::conversions::NdAsImageMut<T, D>,
    {
        self.xdog(XDoGArgs::sigma(sigma).build()?)
    }
}

impl<T, D, S> NdCvXDoG<T, D> for ArrayBase<S, D>
where
    T: ndcv_bridge::types::CvType
        + Send
        + Sync
        + num::Zero
        + num::One
        + num::ToPrimitive
        + seal::Sealed
        + ndcv_bridge::gaussian::seal::Sealed
        + core::ops::Mul<Output = T>,
    T: ndcv_bridge::types::CvType + seal::Sealed + num::Float + num::FromPrimitive,
    D: ndarray::Dimension,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    ndarray::ArrayBase<S, D>:
        ndcv_bridge::image::NdImage + ndcv_bridge::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: ndcv_bridge::conversions::NdAsImageMut<T, D>,
    ndarray::ArrayBase<S, D>: NdCvGaussianBlur<T, D>,
{
    fn xdog(&self, args: XDoGArgs<T>) -> Result<ndarray::Array<T, D>, XDoGError>
    where
        ndarray::Array<T, D>: ndcv_bridge::conversions::NdAsImageMut<T, D>,
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
        let dst = if args.thresholding {
            ndarray::Zip::from(&g1).and(&g2).par_map_collect(|v1, v2| {
                let d = (one + args.p) * *v1 - args.p * *v2;
                if d >= args.epsilon {
                    one
                } else {
                    one + (args.phi * (d - args.epsilon)).tanh()
                }
            })
        } else {
            ndarray::Zip::from(&g1)
                .and(&g2)
                .par_map_collect(|v1, v2| (one + args.p) * *v1 - args.p * *v2)
        };
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::U8Vec2;
    use ndarray::Array3;

    /// Default kernel size used across tests.
    const KSIZE: U8Vec2 = U8Vec2::new(5, 5);

    // ── Shape preservation ───────────────────────────────────────────

    #[test]
    fn xdog_3d_preserves_shape() {
        let img = Array3::<f32>::ones((20, 30, 3));
        let result = img
            .xdog(XDoGArgs::sigma(1.0f32).kernel_size(KSIZE).build().unwrap())
            .unwrap();
        assert_eq!(result.shape(), &[20, 30, 3]);
    }

    #[test]
    fn xdog_non_square_3d() {
        let img = Array3::<f32>::zeros((15, 40, 3));
        let result = img
            .xdog(XDoGArgs::sigma(1.0f32).kernel_size(KSIZE).build().unwrap())
            .unwrap();
        assert_eq!(result.shape(), &[15, 40, 3]);
    }

    #[test]
    fn xdog_f64_preserves_shape() {
        let img = Array3::<f64>::ones((10, 10, 3));
        let result = img
            .xdog(XDoGArgs::sigma(1.0f64).kernel_size(KSIZE).build().unwrap())
            .unwrap();
        assert_eq!(result.shape(), &[10, 10, 3]);
    }

    // ── Uniform image produces near-identity output ─────────────────

    #[test]
    fn xdog_uniform_image_near_identity() {
        let img = Array3::<f32>::from_elem((20, 20, 3), 128.0);
        let result = img
            .xdog(XDoGArgs::sigma(1.0f32).kernel_size(KSIZE).build().unwrap())
            .unwrap();

        // On a uniform image both Gaussians are identical, so
        // D(x) = (1+p)*v - p*v = v ≈ 128 everywhere.
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
        let mut img = Array3::<f32>::zeros((40, 40, 1));
        img.slice_mut(s![.., 20.., ..]).fill(255.0); // vertical edge

        let result = img
            .xdog(
                XDoGArgs::sigma(1.0f32)
                    .kernel_size(KSIZE)
                    .k(1.6f32)
                    .p(20.0f32)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // The edge region should differ from the flat interior.
        let interior_val = result[[5, 5, 0]];
        let edge_val = result[[20, 20, 0]];
        assert!(
            (edge_val - interior_val).abs() > 1.0,
            "edge response ({edge_val}) should differ from interior ({interior_val})"
        );
    }

    // ── p=0 gives plain DoG ─────────────────────────────────────────

    #[test]
    fn xdog_p_zero_is_gaussian() {
        // When p = 0, XDoG = (1+0)*G1 - 0*G2 = G1.
        let img = Array3::<f32>::from_shape_fn((20, 20, 3), |(i, j, _c)| ((i + j) % 256) as f32);
        let result = img
            .xdog(
                XDoGArgs::sigma(1.0f32)
                    .kernel_size(KSIZE)
                    .p(0.0f32)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // Compare against a direct Gaussian blur with the same sigma.
        use ndcv_bridge::gaussian::NdCvGaussianBlur;
        let g1: Array3<f32> = img
            .gaussian_blur((5, 5), (1.0, 1.0), BorderType::BorderReflect101)
            .unwrap();

        for (r, g) in result.iter().zip(g1.iter()) {
            assert!((r - g).abs() < 1e-3, "p=0 XDoG ({r}) should match G1 ({g})");
        }
    }

    // ── Thresholding ────────────────────────────────────────────────

    #[test]
    fn xdog_thresholding_produces_bounded_output() {
        let img =
            Array3::<f32>::from_shape_fn((30, 30, 3), |(i, j, _c)| ((i * 8 + j * 3) % 256) as f32);

        let result = img
            .xdog(
                XDoGArgs::sigma(1.0f32)
                    .kernel_size(KSIZE)
                    .p(200.0f32)
                    .thresholding(true)
                    .epsilon(0.5f32)
                    .phi(10.0f32)
                    .build()
                    .unwrap(),
            )
            .unwrap();

        // The tanh-based thresholding should produce values in roughly (0, 1].
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
        let img = Array3::<f32>::from_elem((10, 10, 3), 200.0);
        let result = img
            .xdog(
                XDoGArgs::sigma(1.0f32)
                    .kernel_size(KSIZE)
                    .thresholding(true)
                    .epsilon(0.0f32)
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
    fn xdog_f32_input_output() {
        let img = Array3::<f32>::from_shape_fn((15, 15, 3), |(i, j, _)| (i + j) as f32);
        let result = img
            .xdog(XDoGArgs::sigma(1.0f32).kernel_size(KSIZE).build().unwrap())
            .unwrap();
        assert_eq!(result.shape(), &[15, 15, 3]);
    }

    #[test]
    fn xdog_f64_input_output() {
        let img = Array3::<f64>::from_shape_fn((15, 15, 3), |(i, j, _)| (i + j) as f64);
        let result = img
            .xdog(XDoGArgs::sigma(1.0f64).kernel_size(KSIZE).build().unwrap())
            .unwrap();
        assert_eq!(result.shape(), &[15, 15, 3]);
    }

    // ── Parameter validation ────────────────────────────────────────

    #[test]
    fn xdog_invalid_sigma() {
        // Negative sigma should be rejected at build time.
        let result = XDoGArgs::<f64>::sigma(-1.0f64).kernel_size(KSIZE).build();
        assert!(result.is_err(), "negative sigma should fail validation");
    }

    #[test]
    fn xdog_invalid_k() {
        // Negative k should be rejected at build time.
        let result = XDoGArgs::<f64>::sigma(1.0f64)
            .kernel_size(KSIZE)
            .k(-0.5f64)
            .build();
        assert!(result.is_err(), "negative k should fail validation");
    }

    // ── Different k values ──────────────────────────────────────────

    #[test]
    fn xdog_different_k_values() {
        let img = Array3::<f32>::from_shape_fn((20, 20, 3), |(i, j, _)| ((i * j) % 256) as f32);

        for k in [1.2f32, 1.6, 2.0, 3.0] {
            let result = img
                .xdog(
                    XDoGArgs::sigma(1.0f32)
                        .kernel_size(KSIZE)
                        .k(k)
                        .build()
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(result.shape(), &[20, 20, 3], "failed for k={k}");
        }
    }

    // ── Different border types ──────────────────────────────────────

    #[test]
    fn xdog_different_border_types() {
        let img = Array3::<f32>::from_shape_fn((20, 20, 3), |(i, j, _)| ((i + j) % 256) as f32);

        let border_types = [
            BorderType::BorderConstant,
            BorderType::BorderReplicate,
            BorderType::BorderReflect,
            BorderType::BorderReflect101,
        ];

        for bt in border_types {
            let result = img
                .xdog(
                    XDoGArgs::sigma(1.0f32)
                        .kernel_size(KSIZE)
                        .border_type(bt)
                        .build()
                        .unwrap(),
                )
                .unwrap();
            assert_eq!(
                result.shape(),
                &[20, 20, 3],
                "failed for border type {bt:?}"
            );
        }
    }

    // ── Builder ergonomics ──────────────────────────────────────────

    #[test]
    fn xdog_builder_defaults() {
        // sigma and kernel_size are the required fields
        let args = XDoGArgs::sigma(1.0f32).kernel_size(KSIZE).build().unwrap();
        let img = Array3::<f32>::zeros((10, 10, 3));
        let _result = img.xdog(args).unwrap();
    }

    #[test]
    fn xdog_builder_all_fields() {
        let args = XDoGArgs::sigma(1.5f32)
            .kernel_size(U8Vec2::new(7, 7))
            .k(2.0f32)
            .p(100.0f32)
            .epsilon(0.3f32)
            .phi(5.0f32)
            .thresholding(true)
            .border_type(BorderType::BorderReplicate)
            .build()
            .unwrap();

        let img = Array3::<f32>::from_elem((20, 20, 3), 100.0);
        let result = img.xdog(args).unwrap();
        assert_eq!(result.shape(), &[20, 20, 3]);
    }

    // ── Different kernel sizes ──────────────────────────────────────

    #[test]
    fn xdog_different_kernel_sizes() {
        let img = Array3::<f32>::from_elem((20, 20, 3), 50.0);

        for ksize in [U8Vec2::new(3, 3), U8Vec2::new(5, 5), U8Vec2::new(7, 7)] {
            let result = img
                .xdog(XDoGArgs::sigma(1.0f32).kernel_size(ksize).build().unwrap())
                .unwrap();
            assert_eq!(result.shape(), &[20, 20, 3], "failed for ksize={ksize:?}");
        }
    }
}
