//! <https://docs.rs/opencv/latest/opencv/core/fn.absdiff.html>
use crate::{
    conversions::{NdAsImageMut, *},
    image::NdImage,
    types::CvType,
};
use ndarray::*;

#[derive(Debug, thiserror::Error)]
pub enum AbsDiffError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

/// Calculates the per-element absolute difference between two arrays.
///
/// Both arrays must have the same shape and element type. The result is
/// `dst(I) = saturate(|self(I) - other(I)|)` for every element index `I`.
///
/// Note: Saturation is not applied when the depth is `i32` (`CV_32S`).
pub trait NdCvAbsDiff<T, D>: NdImage + crate::conversions::NdAsImage<T, D>
where
    T: CvType,
    D: ndarray::Dimension,
{
    fn absdiff<S2>(&self, other: &ArrayBase<S2, D>) -> Result<ndarray::Array<T, D>, AbsDiffError>
    where
        ArrayBase<S2, D>: NdImage + crate::conversions::NdAsImage<T, D>,
        S2: ndarray::RawData + ndarray::Data<Elem = T>;
}

impl<T, S, D> NdCvAbsDiff<T, D> for ArrayBase<S, D>
where
    ndarray::ArrayBase<S, D>: NdImage + crate::conversions::NdAsImage<T, D>,
    ndarray::Array<T, D>: NdAsImageMut<T, D>,
    T: CvType,
    S: ndarray::RawData + ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    fn absdiff<S2>(&self, other: &ArrayBase<S2, D>) -> Result<ndarray::Array<T, D>, AbsDiffError>
    where
        ArrayBase<S2, D>: NdImage + crate::conversions::NdAsImage<T, D>,
        S2: ndarray::RawData + ndarray::Data<Elem = T>,
    {
        let mut dst = ndarray::Array::default(self.dim());
        let cv_self = self.as_image_mat()?;
        let cv_other = other.as_image_mat()?;
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::core::absdiff(&*cv_self, &*cv_other, &mut *cv_dst)?;
        Ok(dst)
    }
}

/// In-place variant: computes `self(I) = saturate(|self(I) - other(I)|)`.
pub trait NdCvAbsDiffInPlace<T: CvType, D: ndarray::Dimension>:
    NdImage + NdAsImageMut<T, D>
{
    fn absdiff_inplace<S2>(&mut self, other: &ArrayBase<S2, D>) -> Result<&mut Self, AbsDiffError>
    where
        ArrayBase<S2, D>: NdImage + crate::conversions::NdAsImage<T, D>,
        S2: ndarray::RawData + ndarray::Data<Elem = T>;
}

impl<T, S, D> NdCvAbsDiffInPlace<T, D> for ArrayBase<S, D>
where
    Self: NdImage + NdAsImageMut<T, D>,
    T: CvType,
    S: ndarray::RawData + ndarray::DataMut<Elem = T>,
    D: ndarray::Dimension,
{
    fn absdiff_inplace<S2>(&mut self, other: &ArrayBase<S2, D>) -> Result<&mut Self, AbsDiffError>
    where
        ArrayBase<S2, D>: NdImage + crate::conversions::NdAsImage<T, D>,
        S2: ndarray::RawData + ndarray::Data<Elem = T>,
    {
        let cv_other = other.as_image_mat()?;
        let mut cv_self = self.as_image_mat_mut()?;

        unsafe {
            crate::inplace::op_inplace(&mut cv_self, |this, out| {
                opencv::core::absdiff(this, &*cv_other, out)
            })
        }?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn test_absdiff_basic_3d() {
        let a = Array3::<u8>::from_elem((10, 10, 3), 200);
        let b = Array3::<u8>::from_elem((10, 10, 3), 50);
        let res = a.absdiff(&b).unwrap();
        assert_eq!(res.shape(), &[10, 10, 3]);
        assert!(res.iter().all(|&v| v == 150));
    }

    #[test]
    fn test_absdiff_basic_2d() {
        let a = Array2::<u8>::from_elem((10, 10), 100);
        let b = Array2::<u8>::from_elem((10, 10), 40);
        let res = a.absdiff(&b).unwrap();
        assert_eq!(res.shape(), &[10, 10]);
        assert!(res.iter().all(|&v| v == 60));
    }

    #[test]
    fn test_absdiff_commutative() {
        let a = Array3::<u8>::from_elem((8, 8, 3), 200);
        let b = Array3::<u8>::from_elem((8, 8, 3), 50);
        let ab = a.absdiff(&b).unwrap();
        let ba = b.absdiff(&a).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_absdiff_identical_arrays() {
        let a = Array3::<u8>::from_elem((5, 5, 3), 123);
        let res = a.absdiff(&a).unwrap();
        assert!(res.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_absdiff_zeros() {
        let a = Array3::<u8>::zeros((5, 5, 1));
        let b = Array3::<u8>::zeros((5, 5, 1));
        let res = a.absdiff(&b).unwrap();
        assert!(res.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_absdiff_u8_saturation() {
        // |0 - 255| should be 255, not wrap around
        let a = Array3::<u8>::zeros((4, 4, 1));
        let b = Array3::<u8>::from_elem((4, 4, 1), 255);
        let res = a.absdiff(&b).unwrap();
        assert!(res.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_absdiff_f32() {
        let a = Array3::<f32>::from_elem((6, 6, 3), 1.5);
        let b = Array3::<f32>::from_elem((6, 6, 3), 0.25);
        let res = a.absdiff(&b).unwrap();
        assert_eq!(res.shape(), &[6, 6, 3]);
        assert!(res.iter().all(|&v| (v - 1.25).abs() < 1e-6));
    }

    #[test]
    fn test_absdiff_f64() {
        let a = Array2::<f64>::from_elem((4, 4), 3.0);
        let b = Array2::<f64>::from_elem((4, 4), 5.0);
        let res = a.absdiff(&b).unwrap();
        assert!(res.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn test_absdiff_i16() {
        let a = Array2::<i16>::from_elem((4, 4), -100);
        let b = Array2::<i16>::from_elem((4, 4), 100);
        let res = a.absdiff(&b).unwrap();
        assert!(res.iter().all(|&v| v == 200));
    }

    #[test]
    fn test_absdiff_preserves_shape() {
        let shapes: Vec<(usize, usize, usize)> =
            vec![(1, 1, 1), (1, 10, 3), (10, 1, 3), (100, 50, 4)];
        for shape in shapes {
            let a = Array3::<u8>::zeros(shape);
            let b = Array3::<u8>::zeros(shape);
            let res = a.absdiff(&b).unwrap();
            assert_eq!(res.shape(), &[shape.0, shape.1, shape.2]);
        }
    }

    #[test]
    fn test_absdiff_inplace_basic() {
        let mut a = Array3::<u8>::from_elem((10, 10, 3), 200);
        let b = Array3::<u8>::from_elem((10, 10, 3), 50);
        a.absdiff_inplace(&b).unwrap();
        assert!(a.iter().all(|&v| v == 150));
    }

    #[test]
    fn test_absdiff_inplace_chaining() {
        let mut a = Array3::<u8>::from_elem((4, 4, 1), 100);
        let b = Array3::<u8>::from_elem((4, 4, 1), 30);
        let c = Array3::<u8>::from_elem((4, 4, 1), 20);
        // |100 - 30| = 70, then |70 - 20| = 50
        a.absdiff_inplace(&b).unwrap().absdiff_inplace(&c).unwrap();
        assert!(a.iter().all(|&v| v == 50));
    }
}
