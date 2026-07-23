#[derive(Debug, Clone)]
pub struct MatRef<'a> {
    pub(crate) mat: opencv::core::Mat,
    pub(crate) _marker: core::marker::PhantomData<&'a opencv::core::Mat>,
}

impl MatRef<'_> {
    pub fn clone_pointee(&self) -> opencv::core::Mat {
        self.mat.clone()
    }
}

impl<'a> opencv::core::ToOutputArray for MatRefMut<'a> {
    fn output_array(
        &mut self,
    ) -> std::result::Result<
        opencv::boxed_ref::BoxedRefMut<'a, opencv::core::_OutputArray>,
        opencv::Error,
    > {
        unsafe { core::mem::transmute(self.mat.output_array()) }
    }
}

impl<'a> opencv::core::ToInputArray for MatRef<'a> {
    fn input_array(
        &self,
    ) -> std::result::Result<
        opencv::boxed_ref::BoxedRef<'a, opencv::core::_InputArray>,
        opencv::Error,
    > {
        unsafe { core::mem::transmute(self.mat.input_array()) }
    }
}

impl MatRef<'_> {
    pub fn new<'a>(mat: opencv::core::Mat) -> MatRef<'a> {
        MatRef {
            mat,
            _marker: core::marker::PhantomData,
        }
    }
}

impl AsRef<opencv::core::Mat> for MatRef<'_> {
    fn as_ref(&self) -> &opencv::core::Mat {
        &self.mat
    }
}

impl AsRef<opencv::core::Mat> for MatRefMut<'_> {
    fn as_ref(&self) -> &opencv::core::Mat {
        &self.mat
    }
}

impl AsMut<opencv::core::Mat> for MatRefMut<'_> {
    fn as_mut(&mut self) -> &mut opencv::core::Mat {
        &mut self.mat
    }
}

#[derive(Debug, Clone)]
pub struct MatRefMut<'a> {
    pub(crate) mat: opencv::core::Mat,
    pub(crate) _marker: core::marker::PhantomData<&'a mut opencv::core::Mat>,
}

impl MatRefMut<'_> {
    pub fn new<'a>(mat: opencv::core::Mat) -> MatRefMut<'a> {
        MatRefMut {
            mat,
            _marker: core::marker::PhantomData,
        }
    }
}

impl core::ops::Deref for MatRef<'_> {
    type Target = opencv::core::Mat;
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl core::ops::Deref for MatRefMut<'_> {
    type Target = opencv::core::Mat;
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl core::ops::DerefMut for MatRefMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mat
    }
}

#[cfg(test)]
mod tests {
    use crate::conversions::{NdAsMat, NdAsMatMut};
    use ndarray::Array2;
    use opencv::core::MatTraitConst;

    #[test]
    fn test_clone_pointee_copies_matrix() {
        // A mutated `Default::default()` would hand back an empty 0x0 Mat.
        let arr = Array2::<u8>::from_elem((3, 5), 7u8);
        let mref = arr.as_single_channel_mat().unwrap();
        let cloned = mref.clone_pointee();
        assert_eq!(cloned.rows(), 3);
        assert_eq!(cloned.cols(), 5);
        assert_eq!(*cloned.at_2d::<u8>(1, 1).unwrap(), 7);
    }

    #[test]
    fn test_mat_ref_mut_as_ref_and_deref_expose_real_mat() {
        // Both accessors must borrow the wrapped Mat, not a leaked empty one.
        let mut arr = Array2::<u8>::from_elem((3, 5), 7u8);
        let mref = arr.as_single_channel_mat_mut().unwrap();
        let via_as_ref: &opencv::core::Mat = mref.as_ref();
        assert_eq!(via_as_ref.rows(), 3);
        assert_eq!(via_as_ref.cols(), 5);
        // Deref path.
        assert_eq!(mref.rows(), 3);
        assert_eq!(mref.cols(), 5);
        assert_eq!(*mref.at_2d::<u8>(1, 1).unwrap(), 7);
    }
}
