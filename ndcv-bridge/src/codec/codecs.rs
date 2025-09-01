use super::decode::Decoder;
use super::encode::Encoder;
use crate::NdCvError;
use crate::conversions::matref::MatRef;
use error_stack::*;
use img_parts::{
    Bytes,
    jpeg::{Jpeg, markers},
};
use opencv::{
    core::{Mat, Vector, VectorToVec},
    imgcodecs::{ImreadModes, ImwriteFlags, imdecode, imencode},
};

#[derive(Debug)]
pub enum CvEncoder {
    Jpeg(CvJpegEncFlags),
    Tiff(CvTiffEncFlags),
}

pub enum EncKind {
    Jpeg,
    Tiff,
}

impl CvEncoder {
    fn kind(&self) -> EncKind {
        match self {
            Self::Jpeg(_) => EncKind::Jpeg,
            Self::Tiff(_) => EncKind::Tiff,
        }
    }

    fn extension(&self) -> &'static str {
        match self {
            Self::Jpeg(_) => ".jpg",
            Self::Tiff(_) => ".tiff",
        }
    }

    fn to_cv_param_list(&self) -> Vector<i32> {
        match self {
            Self::Jpeg(flags) => flags.to_cv_param_list(),
            Self::Tiff(flags) => flags.to_cv_param_list(),
        }
    }
}

#[derive(Default, Debug)]
pub struct CvJpegEncFlags {
    quality: Option<usize>,
    progressive: Option<bool>,
    optimize: Option<bool>,
    remove_app0: Option<bool>,
}

#[derive(Default, Debug)]
pub struct CvTiffEncFlags {
    compression: Option<i32>,
}

impl CvTiffEncFlags {
    pub fn new() -> Self {
        Self::default().with_compression(1)
    }

    pub fn with_compression(mut self, compression: i32) -> Self {
        self.compression = Some(compression);
        self
    }

    fn to_cv_param_list(&self) -> Vector<i32> {
        let iter = [(
            ImwriteFlags::IMWRITE_TIFF_COMPRESSION as i32,
            self.compression.map(|i| i),
        )]
        .into_iter()
        .filter_map(|(flag, opt)| opt.map(|o| [flag, o]))
        .flatten();

        Vector::from_iter(iter)
    }
}

impl CvJpegEncFlags {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_quality(mut self, quality: usize) -> Self {
        self.quality = Some(quality);
        self
    }

    pub fn remove_app0_marker(mut self, val: bool) -> Self {
        self.remove_app0 = Some(val);
        self
    }

    fn to_cv_param_list(&self) -> Vector<i32> {
        let iter = [
            (
                ImwriteFlags::IMWRITE_JPEG_QUALITY as i32,
                self.quality.map(|i| i as i32),
            ),
            (
                ImwriteFlags::IMWRITE_JPEG_PROGRESSIVE as i32,
                self.progressive.map(|i| i as i32),
            ),
            (
                ImwriteFlags::IMWRITE_JPEG_OPTIMIZE as i32,
                self.optimize.map(|i| i as i32),
            ),
        ]
        .into_iter()
        .filter_map(|(flag, opt)| opt.map(|o| [flag, o]))
        .flatten();

        Vector::from_iter(iter)
    }
}

impl Encoder for CvEncoder {
    type Input<'a>
        = MatRef<'a>
    where
        Self: 'a;

    fn encode(&self, input: Self::Input<'_>) -> Result<Vec<u8>, NdCvError> {
        let mut buf = Vector::default();

        let params = self.to_cv_param_list();

        imencode(self.extension(), &input.as_ref(), &mut buf, &params).change_context(NdCvError)?;

        match self.kind() {
            EncKind::Jpeg => {
                let bytes = Bytes::from(buf.to_vec());
                let mut jpg = Jpeg::from_bytes(bytes).change_context(NdCvError)?;
                jpg.remove_segments_by_marker(markers::APP0);
                let bytes = jpg.encoder().bytes();
                Ok(bytes.to_vec())
            }
            EncKind::Tiff => Ok(buf.to_vec()),
        }
    }
}

pub enum CvDecoder {
    Jpeg(CvJpegDecFlags),
}

impl CvDecoder {
    fn to_cv_decode_flag(&self) -> i32 {
        match self {
            Self::Jpeg(flags) => flags.to_cv_decode_flag(),
        }
    }
}

#[derive(Default)]
pub enum ColorMode {
    #[default]
    Color,
    GrayScale,
}

impl ColorMode {
    fn to_cv_decode_flag(&self) -> i32 {
        match self {
            Self::Color => ImreadModes::IMREAD_ANYCOLOR as i32,
            Self::GrayScale => ImreadModes::IMREAD_GRAYSCALE as i32,
        }
    }
}

#[derive(Default)]
pub struct CvJpegDecFlags {
    color_mode: ColorMode,
    ignore_orientation: bool,
}

impl CvJpegDecFlags {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_color_mode(mut self, color_mode: ColorMode) -> Self {
        self.color_mode = color_mode;
        self
    }

    pub fn with_ignore_orientation(mut self, ignore_orientation: bool) -> Self {
        self.ignore_orientation = ignore_orientation;
        self
    }

    fn to_cv_decode_flag(&self) -> i32 {
        let flag = self.color_mode.to_cv_decode_flag();

        if self.ignore_orientation {
            flag | ImreadModes::IMREAD_IGNORE_ORIENTATION as i32
        } else {
            flag
        }
    }
}

impl Decoder for CvDecoder {
    type Output = Mat;

    fn decode(&self, input: impl AsRef<[u8]>) -> Result<Self::Output, NdCvError> {
        let flag = self.to_cv_decode_flag();
        let out = imdecode(&Vector::from_slice(input.as_ref()), flag).change_context(NdCvError)?;

        Ok(out)
    }
}
