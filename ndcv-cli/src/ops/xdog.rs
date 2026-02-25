use anyhow::{Context, Result, anyhow};
use clap::Args;
use glam::U8Vec2;
use ndarray::Array3;

use crate::io::NdImage;
use crate::ops::blur::BorderTypeArg;
use ndarray_image::NdCvXDoG;
use ndarray_image::xdog::XDoGArgs as LibXDoGArgs;

use ndcv_bridge::BorderType;

#[derive(Debug, Clone, Args)]
pub struct XDoGArgs {
    /// Standard deviation of the smaller (detail) Gaussian
    #[arg(short = 'x', long)]
    pub sigma: f32,

    /// Gaussian kernel size (must be odd and positive)
    #[arg(long, default_value_t = 5)]
    pub kernel: u8,

    /// Ratio between the two Gaussian sigmas (k > 1)
    #[arg(short, long, default_value_t = 1.6)]
    pub k: f32,

    /// Sharpness multiplier (p = 0 gives plain DoG)
    #[arg(short, long, default_value_t = 200.0)]
    pub p: f32,

    /// Edge threshold for soft-thresholding step
    #[arg(short, long, default_value_t = 0.5)]
    pub epsilon: f32,

    /// Steepness of the tanh transition near the threshold
    #[arg(long, default_value_t = 10.0)]
    pub phi: f32,

    /// Enable soft-thresholding for stylised (ink-like) output
    #[arg(short, long, default_value_t = false)]
    pub threshold: bool,

    /// Border extrapolation method
    #[arg(short, long, value_enum, default_value_t = BorderTypeArg::Reflect101)]
    pub border: BorderTypeArg,
}

pub fn run(image: &NdImage, args: &XDoGArgs) -> Result<NdImage> {
    let border = BorderType::from(args.border);
    let kernel_size = U8Vec2::new(args.kernel, args.kernel);

    let lib_args = LibXDoGArgs::sigma(args.sigma)
        .kernel_size(kernel_size)
        .k(args.k)
        .p(args.p)
        .epsilon(args.epsilon)
        .phi(args.phi)
        .thresholding(args.threshold)
        .border_type(border)
        .build()
        .map_err(|e| anyhow!("invalid XDoG args: {e}"))
        .context("failed to build XDoG args")?;

    match image {
        NdImage::Color(arr) => {
            let arr_f32 = arr.mapv(|v| v as f32 / u8::MAX as f32);
            let result: Array3<f32> = arr_f32.xdog(lib_args).context("XDoG operation failed")?;
            let result_u8 = to_u8_3d(&result, args.threshold);
            Ok(NdImage::Color(result_u8))
        }
        NdImage::Gray(_) => {
            let color = image
                .ensure_color()
                .context("failed to convert to color for XDoG")?;
            let color_f32 = color.mapv(|v| v as f32);
            let result: Array3<f32> = color_f32.xdog(lib_args).context("XDoG operation failed")?;
            let result_u8 = to_u8_3d(&result, args.threshold);
            let gray_img = NdImage::Color(result_u8);
            let gray = gray_img.ensure_gray()?;
            Ok(NdImage::Gray(gray))
        }
    }
}

/// Convert an f32 array to u8 for saving.
///
/// When thresholding is enabled, the output is in `[0, 1]` and is scaled to
/// `[0, 255]`. Otherwise, values are clamped to `[0, 255]` directly.
fn to_u8_3d(arr: &Array3<f32>, thresholded: bool) -> Array3<u8> {
    if thresholded {
        arr.mapv(|v| (v * 255.0).clamp(0.0, 255.0) as u8)
    } else {
        arr.mapv(|v| v.clamp(0.0, 255.0) as u8)
    }
}
