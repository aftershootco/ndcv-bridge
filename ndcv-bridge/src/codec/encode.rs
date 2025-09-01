use super::codecs::CvEncoder;
use super::error::ErrorReason;
use crate::NdCvError;
use crate::conversions::NdAsImage;
use error_stack::*;
use ndarray::ArrayBase;
use std::path::Path;

pub trait Encodable<E: Encoder> {
    fn encode(&self, encoder: &E) -> Result<Vec<u8>, NdCvError> {
        let input = self.transform()?;
        encoder.encode(input)
    }

    fn write(&self, path: impl AsRef<Path>, encoder: &E) -> Result<(), NdCvError> {
        let buf = self.encode(encoder)?;

        std::fs::write(path, buf)
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
            .change_context(NdCvError)
    }

    fn transform(&self) -> Result<<E as Encoder>::Input<'_>, NdCvError>;
}

pub trait Encoder {
    type Input<'a>
    where
        Self: 'a;

    fn encode(&self, input: Self::Input<'_>) -> Result<Vec<u8>, NdCvError>;
}

impl<T: bytemuck::Pod + Copy, S: ndarray::Data<Elem = T>, D: ndarray::Dimension>
    Encodable<CvEncoder> for ArrayBase<S, D>
where
    Self: NdAsImage<T, D>,
{
    fn transform(&self) -> Result<<CvEncoder as Encoder>::Input<'_>, NdCvError> {
        self.as_image_mat()
    }
}
