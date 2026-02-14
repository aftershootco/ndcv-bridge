use anyhow::{Context, Result, bail};
use clap::Args;

use crate::io::NdImage;
use ndcv_bridge::{BorderType, NdCvGaussianBlur};

/// Gaussian blur argument values for --border
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BorderTypeArg {
    Constant,
    Replicate,
    Reflect,
    Wrap,
    Reflect101,
    Transparent,
    Isolated,
}

impl From<BorderTypeArg> for BorderType {
    fn from(b: BorderTypeArg) -> Self {
        match b {
            BorderTypeArg::Constant => BorderType::BorderConstant,
            BorderTypeArg::Replicate => BorderType::BorderReplicate,
            BorderTypeArg::Reflect => BorderType::BorderReflect,
            BorderTypeArg::Wrap => BorderType::BorderWrap,
            BorderTypeArg::Reflect101 => BorderType::BorderReflect101,
            BorderTypeArg::Transparent => BorderType::BorderTransparent,
            BorderTypeArg::Isolated => BorderType::BorderIsolated,
        }
    }
}

#[derive(Debug, Clone, Args)]
pub struct BlurArgs {
    /// Gaussian kernel size (must be odd and positive)
    #[arg(short, long, default_value_t = 3)]
    pub kernel: u16,

    /// Sigma value for X direction
    #[arg(short = 'x', long, default_value_t = 1.0)]
    pub sigma: f64,

    /// Sigma value for Y direction (defaults to same as sigma-x)
    #[arg(short = 'y', long)]
    pub sigma_y: Option<f64>,

    /// Border extrapolation method
    #[arg(short, long, value_enum, default_value_t = BorderTypeArg::Reflect101)]
    pub border: BorderTypeArg,
}

pub fn run(image: &NdImage, args: &BlurArgs) -> Result<NdImage> {
    if args.kernel.is_multiple_of(2) {
        bail!("kernel size must be odd, got {}", args.kernel);
    }

    let sigma_y = args.sigma_y.unwrap_or(args.sigma);
    let kernel = (args.kernel as i32, args.kernel as i32);
    let border = BorderType::from(args.border);

    match image {
        NdImage::Color(arr) => {
            let result = arr
                .gaussian_blur(kernel, (args.sigma, sigma_y), border)
                .context("gaussian blur failed")?;
            Ok(NdImage::Color(result))
        }
        NdImage::Gray(_) => {
            let color = image
                .ensure_color()
                .context("failed to convert to color for blur")?;
            let result = color
                .gaussian_blur(kernel, (args.sigma, sigma_y), border)
                .context("gaussian blur failed")?;
            // Convert back to gray
            let gray_img = NdImage::Color(result);
            let gray = gray_img.ensure_gray()?;
            Ok(NdImage::Gray(gray))
        }
    }
}
