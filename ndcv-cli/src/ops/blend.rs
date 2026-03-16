use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

use crate::io::NdImage;
use ndarray_image::NdBlend;

#[derive(Debug, Clone, Args)]
pub struct BlendArgs {
    /// Path to the overlay image (blended on top of the input)
    #[arg(long)]
    pub overlay: PathBuf,

    /// Path to the mask image (grayscale; white = overlay, black = base)
    #[arg(long)]
    pub mask: PathBuf,

    /// Alpha blending factor (0.0 - 1.0)
    #[arg(short, long, default_value_t = 1.0)]
    pub alpha: f32,
}

pub fn run(image: &NdImage, args: &BlendArgs) -> Result<NdImage> {
    // Load overlay and mask
    let overlay_img = NdImage::load(&args.overlay).context("failed to load overlay image")?;
    let mask_img = NdImage::load(&args.mask).context("failed to load mask image")?;

    // Get base as color Array3<u8>
    let base_u8 = image
        .ensure_color()
        .context("failed to convert base to color")?;
    let overlay_u8 = overlay_img
        .ensure_color()
        .context("failed to convert overlay to color")?;
    let mask_u8 = mask_img
        .ensure_gray()
        .context("failed to convert mask to grayscale")?;

    // Convert to f32 (normalized 0.0-1.0) as required by NdBlend
    let base_f32 = base_u8.mapv(|v| v as f32 / 255.0);
    let overlay_f32 = overlay_u8.mapv(|v| v as f32 / 255.0);
    let mask_f32 = mask_u8.mapv(|v| v as f32 / 255.0);

    // Perform blend
    let result_f32 = base_f32
        .blend(mask_f32.view(), overlay_f32.view(), args.alpha)
        .context("blend operation failed")?;

    // Convert back to u8
    let result_u8 = result_f32.mapv(|v| (v.clamp(0.0, 1.0) * 255.0) as u8);

    Ok(NdImage::Color(result_u8))
}
