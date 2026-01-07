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
