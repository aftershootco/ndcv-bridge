use super::PasteError;
use error_stack::Report;

pub trait Paste<T> {
    type Out;

    fn paste(self, other: T) -> Result<Self::Out, Report<PasteError>>;
}

pub trait PasteConfig<I> {
    type Out;

    fn with_opts(self, opts: I) -> Self::Out;
}
