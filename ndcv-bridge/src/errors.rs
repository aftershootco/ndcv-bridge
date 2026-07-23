#[derive(Debug, thiserror::Error)]
pub enum NdCvError {
    #[error("empty input")]
    EmptyInput,
    #[error("qth percentile must be between 0 and 1, got {0}")]
    InvalidQuantile(f64),
    #[cfg(feature = "opencv")]
    #[error(transparent)]
    Conversion(#[from] crate::conversions::ConversionError),
    #[cfg(feature = "opencv")]
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
}
