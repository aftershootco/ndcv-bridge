use anyhow::{bail, Context, Result};
use ndarray::{Array2, Array3};
use ndarray_image::{ImageToNdarray, NdarrayToImage};
use std::path::Path;

/// Represents a loaded image as an ndarray, either grayscale (2D) or color (3D).
#[derive(Debug, Clone)]
pub enum NdImage {
    Gray(Array2<u8>),
    Color(Array3<u8>),
}

impl NdImage {
    /// Load an image from a file path. Grayscale images become `Gray(Array2)`,
    /// everything else becomes `Color(Array3)`.
    pub fn load(path: &Path) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("failed to open image: {}", path.display()))?;

        match img {
            image::DynamicImage::ImageLuma8(gray) => {
                let arr = gray
                    .into_ndarray()
                    .context("failed to convert grayscale image to ndarray")?;
                Ok(NdImage::Gray(arr))
            }
            other => {
                // Convert to RGB8 for uniform handling
                let rgb = other.into_rgb8();
                let arr = rgb
                    .into_ndarray()
                    .context("failed to convert RGB image to ndarray")?;
                Ok(NdImage::Color(arr))
            }
        }
    }

    /// Save the image to a file. Format is inferred from the file extension.
    pub fn save(&self, path: &Path) -> Result<()> {
        match self {
            NdImage::Gray(arr) => {
                let view = arr.view();
                let img: image::GrayImage = NdarrayToImage::to_image(&view)
                    .context("failed to convert grayscale ndarray to image")?;
                img.save(path)
                    .with_context(|| format!("failed to save image to {}", path.display()))?;
            }
            NdImage::Color(arr) => {
                let view = arr.view();
                let channels = arr.shape()[2];
                match channels {
                    3 => {
                        let img: image::RgbImage = NdarrayToImage::to_image(&view)
                            .context("failed to convert RGB ndarray to image")?;
                        img.save(path).with_context(|| {
                            format!("failed to save image to {}", path.display())
                        })?;
                    }
                    4 => {
                        let img: image::RgbaImage = NdarrayToImage::to_image(&view)
                            .context("failed to convert RGBA ndarray to image")?;
                        img.save(path).with_context(|| {
                            format!("failed to save image to {}", path.display())
                        })?;
                    }
                    _ => bail!(
                        "unsupported number of channels for saving: {} (expected 3 or 4)",
                        channels
                    ),
                }
            }
        }
        Ok(())
    }

    /// Get the width of the image.
    pub fn width(&self) -> usize {
        match self {
            NdImage::Gray(arr) => arr.shape()[1],
            NdImage::Color(arr) => arr.shape()[1],
        }
    }

    /// Get the height of the image.
    pub fn height(&self) -> usize {
        match self {
            NdImage::Gray(arr) => arr.shape()[0],
            NdImage::Color(arr) => arr.shape()[0],
        }
    }

    /// Get the number of channels (1 for gray, 3 or 4 for color).
    pub fn channels(&self) -> usize {
        match self {
            NdImage::Gray(_) => 1,
            NdImage::Color(arr) => arr.shape()[2],
        }
    }

    /// Ensure the image is Color (3-channel). If grayscale, convert to RGB.
    pub fn ensure_color(&self) -> Result<Array3<u8>> {
        match self {
            NdImage::Color(arr) => Ok(arr.clone()),
            NdImage::Gray(arr) => {
                use ndcv_bridge::color_space::{ConvertColor, Gray, Rgb};
                let rgb = arr
                    .try_cvt::<Gray<u8>, Rgb<u8>>()
                    .context("failed to convert grayscale to RGB")?;
                Ok(rgb.into_owned())
            }
        }
    }

    /// Ensure the image is Gray (1-channel). If color, convert to grayscale.
    pub fn ensure_gray(&self) -> Result<Array2<u8>> {
        match self {
            NdImage::Gray(arr) => Ok(arr.clone()),
            NdImage::Color(arr) => {
                use ndcv_bridge::color_space::{ConvertColor, Gray, Rgb};
                let gray = arr
                    .try_cvt::<Rgb<u8>, Gray<u8>>()
                    .context("failed to convert RGB to grayscale")?;
                Ok(gray.into_owned())
            }
        }
    }
}
