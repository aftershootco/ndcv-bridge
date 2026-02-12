use anyhow::{Result, bail};
use clap::Args;

use crate::io::NdImage;
use ndcv_bridge::orient::{FlipFlag, Orient, Orientation, RotationFlag};

/// Rotation angle
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum RotationArg {
    /// 90 degrees clockwise
    #[value(name = "90")]
    Cw90,
    /// 180 degrees
    #[value(name = "180")]
    Cw180,
    /// 270 degrees clockwise (= 90 degrees counter-clockwise)
    #[value(name = "270")]
    Cw270,
}

impl From<RotationArg> for RotationFlag {
    fn from(r: RotationArg) -> Self {
        match r {
            RotationArg::Cw90 => RotationFlag::Clock90,
            RotationArg::Cw180 => RotationFlag::Clock180,
            RotationArg::Cw270 => RotationFlag::Clock270,
        }
    }
}

/// Flip direction
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum FlipArg {
    /// Mirror horizontally (left-right)
    Mirror,
    /// Flip vertically (top-bottom)
    Water,
    /// Both mirror and flip
    Both,
}

impl From<FlipArg> for FlipFlag {
    fn from(f: FlipArg) -> Self {
        match f {
            FlipArg::Mirror => FlipFlag::Mirror,
            FlipArg::Water => FlipFlag::Water,
            FlipArg::Both => FlipFlag::Both,
        }
    }
}

#[derive(Debug, Clone, Args)]
#[group(required = true, multiple = false)]
pub struct OrientArgs {
    /// Rotate the image (90, 180, or 270 degrees clockwise)
    #[arg(short, long, value_enum)]
    pub rotate: Option<RotationArg>,

    /// Flip the image (mirror = horizontal, water = vertical, both)
    #[arg(short, long, value_enum)]
    pub flip: Option<FlipArg>,

    /// Apply EXIF orientation (1-8)
    #[arg(short, long)]
    pub orientation: Option<u8>,
}

pub fn run(image: &NdImage, args: &OrientArgs) -> Result<NdImage> {
    if let Some(rot) = args.rotate {
        let flag = RotationFlag::from(rot);
        return match image {
            NdImage::Color(arr) => Ok(NdImage::Color(arr.rotate(flag))),
            NdImage::Gray(arr) => Ok(NdImage::Gray(arr.rotate(flag))),
        };
    }

    if let Some(flip) = args.flip {
        let flag = FlipFlag::from(flip);
        return match image {
            NdImage::Color(arr) => Ok(NdImage::Color(arr.flip(flag))),
            NdImage::Gray(arr) => Ok(NdImage::Gray(arr.flip(flag))),
        };
    }

    if let Some(exif) = args.orientation {
        if exif == 0 || exif > 8 {
            bail!("EXIF orientation must be between 1 and 8, got {}", exif);
        }
        let orientation = Orientation::from_raw(exif);
        return match image {
            NdImage::Color(arr) => Ok(NdImage::Color(arr.orient(orientation))),
            NdImage::Gray(arr) => Ok(NdImage::Gray(arr.orient(orientation))),
        };
    }

    bail!("one of --rotate, --flip, or --orientation must be specified")
}
