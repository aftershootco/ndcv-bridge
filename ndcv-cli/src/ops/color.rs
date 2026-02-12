use anyhow::{Context, Result, bail};
use clap::Args;

use crate::io::NdImage;
use ndcv_bridge::color_space::{Bgr, ConvertColor, Gray, Lab, Rgb, Rgba};

/// Target color space
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum ColorSpaceArg {
    Rgb,
    Bgr,
    Rgba,
    Gray,
    Lab,
}

/// Source color space (for explicit specification)
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum SrcColorSpaceArg {
    Rgb,
    Bgr,
    Rgba,
    Gray,
    Lab,
}

#[derive(Debug, Clone, Args)]
pub struct ColorArgs {
    /// Target color space to convert to
    #[arg(short, long, value_enum)]
    pub to: ColorSpaceArg,

    /// Source color space (default: auto-detect based on channel count; rgb for 3-channel, gray for 1-channel)
    #[arg(short, long, value_enum)]
    pub from: Option<SrcColorSpaceArg>,
}

pub fn run(image: &NdImage, args: &ColorArgs) -> Result<NdImage> {
    // Determine source color space
    let src = args.from.unwrap_or_else(|| match image {
        NdImage::Gray(_) => SrcColorSpaceArg::Gray,
        NdImage::Color(arr) => match arr.shape()[2] {
            4 => SrcColorSpaceArg::Rgba,
            _ => SrcColorSpaceArg::Rgb,
        },
    });

    // Perform conversion based on src -> dst combination
    match (src, args.to) {
        // Identity conversions
        (SrcColorSpaceArg::Rgb, ColorSpaceArg::Rgb)
        | (SrcColorSpaceArg::Bgr, ColorSpaceArg::Bgr)
        | (SrcColorSpaceArg::Rgba, ColorSpaceArg::Rgba)
        | (SrcColorSpaceArg::Gray, ColorSpaceArg::Gray) => Ok(image.clone()),

        // RGB -> X
        (SrcColorSpaceArg::Rgb, ColorSpaceArg::Bgr) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Rgb<u8>, Bgr<u8>>()
                .context("RGB -> BGR failed")?;
            Ok(NdImage::Color(out.into_owned()))
        }
        (SrcColorSpaceArg::Rgb, ColorSpaceArg::Rgba) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Rgb<u8>, Rgba<u8>>()
                .context("RGB -> RGBA failed")?;
            Ok(NdImage::Color(out.into_owned()))
        }
        (SrcColorSpaceArg::Rgb, ColorSpaceArg::Gray) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Rgb<u8>, Gray<u8>>()
                .context("RGB -> Gray failed")?;
            Ok(NdImage::Gray(out.into_owned()))
        }
        (SrcColorSpaceArg::Rgb, ColorSpaceArg::Lab) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Rgb<u8>, Lab<i8>>()
                .context("RGB -> Lab failed")?;
            // Lab with i8 values; shift to u8 range for saving (L: 0-100 -> 0-255, a,b: -128..127 -> 0-255)
            let out_u8 = out.mapv(|v| (v as i16 + 128) as u8);
            Ok(NdImage::Color(out_u8))
        }

        // BGR -> X
        (SrcColorSpaceArg::Bgr, ColorSpaceArg::Rgb) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Bgr<u8>, Rgb<u8>>()
                .context("BGR -> RGB failed")?;
            Ok(NdImage::Color(out.into_owned()))
        }
        (SrcColorSpaceArg::Bgr, ColorSpaceArg::Rgba) => {
            // BGR -> RGB -> RGBA
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Bgr<u8>, Rgb<u8>>()
                .context("BGR -> RGB failed")?;
            let rgba = rgb
                .try_cvt::<Rgb<u8>, Rgba<u8>>()
                .context("RGB -> RGBA failed")?;
            Ok(NdImage::Color(rgba.into_owned()))
        }
        (SrcColorSpaceArg::Bgr, ColorSpaceArg::Gray) => {
            // BGR -> RGB -> Gray
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Bgr<u8>, Rgb<u8>>()
                .context("BGR -> RGB failed")?;
            let gray = rgb
                .try_cvt::<Rgb<u8>, Gray<u8>>()
                .context("RGB -> Gray failed")?;
            Ok(NdImage::Gray(gray.into_owned()))
        }
        (SrcColorSpaceArg::Bgr, ColorSpaceArg::Lab) => {
            // BGR -> RGB -> Lab
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Bgr<u8>, Rgb<u8>>()
                .context("BGR -> RGB failed")?;
            let lab = rgb
                .try_cvt::<Rgb<u8>, Lab<i8>>()
                .context("RGB -> Lab failed")?;
            let out_u8 = lab.mapv(|v| (v as i16 + 128) as u8);
            Ok(NdImage::Color(out_u8))
        }

        // RGBA -> X
        (SrcColorSpaceArg::Rgba, ColorSpaceArg::Rgb) => {
            let arr = image.ensure_color()?;
            let out = arr
                .try_cvt::<Rgba<u8>, Rgb<u8>>()
                .context("RGBA -> RGB failed")?;
            Ok(NdImage::Color(out.into_owned()))
        }
        (SrcColorSpaceArg::Rgba, ColorSpaceArg::Bgr) => {
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Rgba<u8>, Rgb<u8>>()
                .context("RGBA -> RGB failed")?;
            let bgr = rgb
                .try_cvt::<Rgb<u8>, Bgr<u8>>()
                .context("RGB -> BGR failed")?;
            Ok(NdImage::Color(bgr.into_owned()))
        }
        (SrcColorSpaceArg::Rgba, ColorSpaceArg::Gray) => {
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Rgba<u8>, Rgb<u8>>()
                .context("RGBA -> RGB failed")?;
            let gray = rgb
                .try_cvt::<Rgb<u8>, Gray<u8>>()
                .context("RGB -> Gray failed")?;
            Ok(NdImage::Gray(gray.into_owned()))
        }
        (SrcColorSpaceArg::Rgba, ColorSpaceArg::Lab) => {
            let arr = image.ensure_color()?;
            let rgb = arr
                .try_cvt::<Rgba<u8>, Rgb<u8>>()
                .context("RGBA -> RGB failed")?;
            let lab = rgb
                .try_cvt::<Rgb<u8>, Lab<i8>>()
                .context("RGB -> Lab failed")?;
            let out_u8 = lab.mapv(|v| (v as i16 + 128) as u8);
            Ok(NdImage::Color(out_u8))
        }

        // Gray -> X
        (SrcColorSpaceArg::Gray, ColorSpaceArg::Rgb) => {
            let arr = image.ensure_gray()?;
            let out = arr
                .try_cvt::<Gray<u8>, Rgb<u8>>()
                .context("Gray -> RGB failed")?;
            Ok(NdImage::Color(out.into_owned()))
        }
        (SrcColorSpaceArg::Gray, ColorSpaceArg::Bgr) => {
            let arr = image.ensure_gray()?;
            let rgb = arr
                .try_cvt::<Gray<u8>, Rgb<u8>>()
                .context("Gray -> RGB failed")?;
            let bgr = rgb
                .try_cvt::<Rgb<u8>, Bgr<u8>>()
                .context("RGB -> BGR failed")?;
            Ok(NdImage::Color(bgr.into_owned()))
        }
        (SrcColorSpaceArg::Gray, ColorSpaceArg::Rgba) => {
            let arr = image.ensure_gray()?;
            let rgb = arr
                .try_cvt::<Gray<u8>, Rgb<u8>>()
                .context("Gray -> RGB failed")?;
            let rgba = rgb
                .try_cvt::<Rgb<u8>, Rgba<u8>>()
                .context("RGB -> RGBA failed")?;
            Ok(NdImage::Color(rgba.into_owned()))
        }
        (SrcColorSpaceArg::Gray, ColorSpaceArg::Lab) => {
            let arr = image.ensure_gray()?;
            let rgb = arr
                .try_cvt::<Gray<u8>, Rgb<u8>>()
                .context("Gray -> RGB failed")?;
            let lab = rgb
                .try_cvt::<Rgb<u8>, Lab<i8>>()
                .context("RGB -> Lab failed")?;
            let out_u8 = lab.mapv(|v| (v as i16 + 128) as u8);
            Ok(NdImage::Color(out_u8))
        }

        // Lab -> X  (Lab is stored as shifted u8, so we can't easily decode it back)
        (SrcColorSpaceArg::Lab, _) => {
            bail!(
                "converting FROM Lab is not supported in the CLI (Lab data would need special encoding)"
            )
        }
    }
}
