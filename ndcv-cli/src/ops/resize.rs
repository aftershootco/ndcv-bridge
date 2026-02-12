use anyhow::{bail, Result};
use clap::Args;

use crate::io::NdImage;
use ndcv_bridge::fir::NdFir;
use ndcv_bridge::{Interpolation, NdCvResize};

/// Resize interpolation method
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum InterpolationArg {
    Linear,
    LinearExact,
    Area,
    Cubic,
    Nearest,
    NearestExact,
    Lanczos4,
}

impl From<InterpolationArg> for Interpolation {
    fn from(i: InterpolationArg) -> Self {
        match i {
            InterpolationArg::Linear => Interpolation::Linear,
            InterpolationArg::LinearExact => Interpolation::LinearExact,
            InterpolationArg::Area => Interpolation::Area,
            InterpolationArg::Cubic => Interpolation::Cubic,
            InterpolationArg::Nearest => Interpolation::Nearest,
            InterpolationArg::NearestExact => Interpolation::NearestExact,
            InterpolationArg::Lanczos4 => Interpolation::Lanczos4,
        }
    }
}

/// Resize backend
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum ResizeBackend {
    /// Use OpenCV for resizing
    #[default]
    Opencv,
    /// Use fast_image_resize (no OpenCV dependency)
    Fir,
}

#[derive(Debug, Clone, Args)]
pub struct ResizeArgs {
    /// Target width in pixels (if omitted, computed from height to preserve aspect ratio)
    #[arg(short = 'W', long)]
    pub width: Option<u32>,

    /// Target height in pixels (if omitted, computed from width to preserve aspect ratio)
    #[arg(short = 'H', long)]
    pub height: Option<u32>,

    /// Interpolation method (only used with opencv backend)
    #[arg(short, long, value_enum, default_value_t = InterpolationArg::Linear)]
    pub interpolation: InterpolationArg,

    /// Resize backend to use
    #[arg(short, long, value_enum, default_value_t = ResizeBackend::Opencv)]
    pub backend: ResizeBackend,
}

pub fn run(image: &NdImage, args: &ResizeArgs) -> Result<NdImage> {
    if args.width.is_none() && args.height.is_none() {
        bail!("at least one of --width or --height must be specified");
    }

    let src_w = image.width() as u32;
    let src_h = image.height() as u32;

    // Compute target dimensions, preserving aspect ratio if one is omitted
    let (target_w, target_h) = match (args.width, args.height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => {
            let h = (src_h as f64 * w as f64 / src_w as f64).round() as u32;
            (w, h)
        }
        (None, Some(h)) => {
            let w = (src_w as f64 * h as f64 / src_h as f64).round() as u32;
            (w, h)
        }
        (None, None) => unreachable!(),
    };

    if target_w == 0 || target_h == 0 {
        bail!(
            "target dimensions must be non-zero (computed {}x{})",
            target_w,
            target_h
        );
    }

    match args.backend {
        ResizeBackend::Opencv => resize_opencv(image, target_w, target_h, args.interpolation),
        ResizeBackend::Fir => resize_fir(image, target_w, target_h),
    }
}

fn resize_opencv(image: &NdImage, w: u32, h: u32, interp: InterpolationArg) -> Result<NdImage> {
    let interpolation = Interpolation::from(interp);
    match image {
        NdImage::Color(arr) => {
            let result = arr
                .resize(h as u16, w as u16, interpolation)
                .map_err(|e| anyhow::anyhow!("OpenCV resize failed: {e}"))?;
            Ok(NdImage::Color(result))
        }
        NdImage::Gray(arr) => {
            let result = arr
                .resize(h as u16, w as u16, interpolation)
                .map_err(|e| anyhow::anyhow!("OpenCV resize failed: {e}"))?;
            Ok(NdImage::Gray(result))
        }
    }
}

fn resize_fir(image: &NdImage, w: u32, h: u32) -> Result<NdImage> {
    match image {
        NdImage::Color(arr) => {
            let result = arr
                .fast_resize(h as usize, w as usize, None)
                .map_err(|e| anyhow::anyhow!("FIR resize failed: {e}"))?;
            Ok(NdImage::Color(result))
        }
        NdImage::Gray(arr) => {
            let result = arr
                .fast_resize(h as usize, w as usize, None)
                .map_err(|e| anyhow::anyhow!("FIR resize failed: {e}"))?;
            Ok(NdImage::Gray(result))
        }
    }
}
