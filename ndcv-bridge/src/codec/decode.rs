#![deny(warnings)]

use super::codecs::CvDecoder;
use super::error::ErrorReason;
use crate::NdCvError;
use crate::{NdAsImage, conversions::NdCvConversion};
use error_stack::*;
use ndarray::Array;
use std::path::Path;

pub trait Decodable<D: Decoder>: Sized {
    fn decode(buf: impl AsRef<[u8]>, decoder: &D) -> Result<Self, NdCvError> {
        let output = decoder.decode(buf)?;
        Self::transform(output)
    }

    fn read(&self, path: impl AsRef<Path>, decoder: &D) -> Result<Self, NdCvError> {
        let buf = std::fs::read(path)
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    Report::new(e).attach_printable(ErrorReason::ImageWriteFileNotFound)
                }
                std::io::ErrorKind::PermissionDenied => {
                    Report::new(e).attach_printable(ErrorReason::ImageWritePermissionDenied)
                }
                std::io::ErrorKind::OutOfMemory => {
                    Report::new(e).attach_printable(ErrorReason::OutOfMemory)
                }
                std::io::ErrorKind::StorageFull => {
                    Report::new(e).attach_printable(ErrorReason::OutOfStorage)
                }
                _ => Report::new(e).attach_printable(ErrorReason::ImageWriteOtherError),
            })
            .change_context(NdCvError)?;
        Self::decode(buf, decoder)
    }

    fn transform(input: D::Output) -> Result<Self, NdCvError>;
}

pub trait Decoder {
    type Output: Sized;
    fn decode(&self, buf: impl AsRef<[u8]>) -> Result<Self::Output, NdCvError>;
}

impl<T: bytemuck::Pod + Copy, D: ndarray::Dimension> Decodable<CvDecoder> for Array<T, D>
where
    Self: NdAsImage<T, D>,
{
    fn transform(input: <CvDecoder as Decoder>::Output) -> Result<Self, NdCvError> {
        Self::from_mat(input)
    }
}

#[test]
#[ignore = "Files are not included in the repo"]
fn decode_image() {
    use crate::codec::codecs::*;
    let img = std::fs::read("assets/test_image.jpg").unwrap();
    let decoder = CvDecoder::Jpeg(CvJpegDecFlags::new().with_ignore_orientation(true));
    let _out = ndarray::Array3::<u8>::decode(img, &decoder).unwrap();
}
