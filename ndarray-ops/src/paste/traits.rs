use super::PasteError;

pub trait Paste<T> {
    type Out;

    fn paste(self, other: T) -> error_stack::Result<Self::Out, PasteError>;
}

pub trait PasteConfig<'a> {
    type Out;

    fn with_opts(self) -> Self::Out;
}
