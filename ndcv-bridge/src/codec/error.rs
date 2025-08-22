#[derive(Debug)]
pub enum ErrorReason {
    ImageReadFileNotFound,
    ImageReadPermissionDenied,
    ImageReadOtherError,

    ImageWriteFileNotFound,
    ImageWritePermissionDenied,
    ImageWriteOtherError,

    OutOfMemory,
    OutOfStorage,
}

impl std::fmt::Display for ErrorReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
