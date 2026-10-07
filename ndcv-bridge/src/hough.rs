use crate::{NdAsImage, conversions::ConversionError};
use glam::DVec2;
use ndarray::{Array2, ArrayBase, Data, Ix2};

#[derive(Debug, thiserror::Error)]
pub enum HoughError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvHoughLinesP: crate::image::NdImage + NdAsImage<u8, Ix2> {
    fn hough_lines_p(
        &self,
        resolution: impl Into<DVec2>,
        threshold: i32,
        min_line_length: f64,
        max_line_gap: f64,
    ) -> Result<Array2<i32>, HoughError>;
}

impl<S: Data<Elem = u8>> NdCvHoughLinesP for ArrayBase<S, Ix2> {
    fn hough_lines_p(
        &self,
        resolution: impl Into<DVec2>,
        threshold: i32,
        min_line_length: f64,
        max_line_gap: f64,
    ) -> Result<Array2<i32>, HoughError> {
        let resolution = resolution.into();
        let src = self.as_image_mat()?;
        let mut lines = opencv::core::Vector::<opencv::core::Vec4i>::new();
        opencv::imgproc::hough_lines_p(
            &*src,
            &mut lines,
            resolution.x,
            resolution.y,
            threshold,
            min_line_length,
            max_line_gap,
        )?;

        let mut dst = Array2::zeros((lines.len(), 4));
        for (mut row, line) in dst.outer_iter_mut().zip(lines.iter()) {
            for (value, coordinate) in row.iter_mut().zip(line.0) {
                *value = coordinate;
            }
        }
        Ok(dst)
    }
}
