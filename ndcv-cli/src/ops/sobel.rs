use anyhow::{Context, Result, bail};
use clap::Args;

use crate::io::NdImage;
use ndcv_bridge::{Ksize, NdCvSobel, SobelArgs as LibSobelArgs};

/// Kernel size for Sobel operator
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum KsizeArg {
    Scharr = -1,
    #[value(name = "1")]
    K1,
    #[value(name = "3")]
    K3,
    #[value(name = "5")]
    K5,
    #[value(name = "7")]
    K7,
}

impl From<KsizeArg> for Ksize {
    fn from(k: KsizeArg) -> Self {
        match k {
            KsizeArg::Scharr => Ksize::Scharr,
            KsizeArg::K1 => Ksize::K1,
            KsizeArg::K3 => Ksize::K3,
            KsizeArg::K5 => Ksize::K5,
            KsizeArg::K7 => Ksize::K7,
        }
    }
}

#[derive(Debug, Clone, Args)]
pub struct SobelArgsCli {
    /// Derivative order in X direction
    #[arg(long, default_value_t = 1)]
    pub dx: i32,

    /// Derivative order in Y direction
    #[arg(long, default_value_t = 0)]
    pub dy: i32,

    /// Aperture kernel size
    #[arg(short, long, value_enum, default_value_t = KsizeArg::K3)]
    pub ksize: KsizeArg,

    /// Scale factor for computed derivative values
    #[arg(short, long, default_value_t = 1.0)]
    pub scale: f64,

    /// Delta value added to the results
    #[arg(short, long, default_value_t = 0.0)]
    pub delta: f64,
}

pub fn run(image: &NdImage, args: &SobelArgsCli) -> Result<NdImage> {
    if args.dx == 0 && args.dy == 0 {
        bail!("at least one of --dx or --dy must be non-zero");
    }

    let sobel_args = LibSobelArgs::dxy([args.dx, args.dy])
        .ksize(Ksize::from(args.ksize))
        .scale(args.scale)
        .delta(args.delta)
        .build()
        .context("failed to build Sobel args")?;

    let color = image
        .ensure_color()
        .context("failed to convert to color for Sobel")?;

    // Sobel outputs i16 for u8 input; we normalize to u8 for saving.
    let result: ndarray::Array3<i16> = color.sobel(sobel_args).context("Sobel operation failed")?;

    // Normalize i16 to u8: take absolute value, clamp to [0, 255]
    let result_u8 = result.mapv(|v| v.unsigned_abs().min(255) as u8);

    Ok(NdImage::Color(result_u8))
}
