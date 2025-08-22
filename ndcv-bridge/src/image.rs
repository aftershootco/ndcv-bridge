use ndarray::*;
pub trait NdImage {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn channels(&self) -> usize;
}

impl<T, S: RawData<Elem = T>> NdImage for ArrayBase<S, Ix3> {
    fn width(&self) -> usize {
        self.dim().1
    }
    fn height(&self) -> usize {
        self.dim().0
    }
    fn channels(&self) -> usize {
        self.dim().2
    }
}

impl<T, S: RawData<Elem = T>> NdImage for ArrayBase<S, Ix2> {
    fn width(&self) -> usize {
        self.dim().1
    }
    fn height(&self) -> usize {
        self.dim().0
    }
    fn channels(&self) -> usize {
        1
    }
}
