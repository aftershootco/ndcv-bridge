//! Mat <--> ndarray conversion traits
//!
//! Conversion Table
//!
//! | ndarray           | Mat     |
//! |---------          |-----    |
//! | Array<T, Ix1>     | Mat(ndims = 1, channels = 1)   |
//! | Array<T, Ix2>     | Mat(ndims = 2, channels = 1)   |
//! | Array<T, Ix2>     | Mat(ndims = 1, channels = X)   |
//! | Array<T, Ix3>     | Mat(ndims = 3, channels = 1)   |
//! | Array<T, Ix3>     | Mat(ndims = 2, channels = X)   |
//! | Array<T, Ix4>     | Mat(ndims = 4, channels = 1)   |
//! | Array<T, Ix4>     | Mat(ndims = 3, channels = X)   |
//! | Array<T, Ix5>     | Mat(ndims = 5, channels = 1)   |
//! | Array<T, Ix5>     | Mat(ndims = 4, channels = X)   |
//! | Array<T, Ix6>     | Mat(ndims = 6, channels = 1)   |
//! | Array<T, Ix6>     | Mat(ndims = 5, channels = X)   |
//!
//! // X is the last dimension
use crate::type_depth;
use ndarray::{Ix2, Ix3};
pub mod impls;
pub(crate) mod matref;
use matref::{MatRef, MatRefMut};

pub(crate) mod seal {
    pub trait SealedInternal {}
    impl<T, S: ndarray::Data<Elem = T>, D> SealedInternal for ndarray::ArrayBase<S, D> {}
}

#[derive(Debug, thiserror::Error)]
pub enum ConversionErrorKind {
    #[error("Opencv Error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("Ndarray Shape Error: {0}")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
    #[error("Unsupported ndarray shape for conversion")]
    UnsupportedNdarrayShape,
    #[error("Invalid number of channels for conversion max: {max}, found: {found}")]
    InvalidNumberOfChannels { max: usize, found: usize },
    #[error("Data is not contiguous")]
    NonContiguousData,
    #[error("Data type {0} is not supported for conversion")]
    UnsupportedDataType(&'static str),
    #[error(
        "Incompatible dimensions: Mat with dims {mat_dims} (rows {rows}, cols {cols}, channels {channels}) cannot be converted to ndarray with dims {ndarray_dims}"
    )]
    IncompatibleDimensions {
        mat_dims: usize,
        rows: usize,
        cols: usize,
        channels: usize,
        ndarray_dims: usize,
    },
    #[error("Expected Mat<{expected}>, but got Mat<{got}>")]
    TypeMismatch {
        expected: &'static str,
        got: &'static str,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("Conversion error at {location}: {kind}")]
pub struct ConversionError {
    pub kind: ConversionErrorKind,
    pub location: &'static core::panic::Location<'static>,
}

impl ConversionError {
    pub fn into_error(self) -> impl std::error::Error + Send + Sync + 'static {
        self
    }
}

impl From<ConversionErrorKind> for ConversionError {
    fn from(kind: ConversionErrorKind) -> Self {
        ConversionError {
            kind,
            location: core::panic::Location::caller(),
        }
    }
}

impl From<opencv::Error> for ConversionError {
    #[track_caller]
    fn from(err: opencv::Error) -> Self {
        ConversionError {
            kind: ConversionErrorKind::OpenCvError(err),
            location: core::panic::Location::caller(),
        }
    }
}

impl From<ndarray::ShapeError> for ConversionError {
    #[track_caller]
    fn from(err: ndarray::ShapeError) -> Self {
        ConversionError {
            kind: ConversionErrorKind::NdarrayShapeError(err),
            location: core::panic::Location::caller(),
        }
    }
}

#[deprecated = "Use NdAsMat and NdAsImage traits instead"]
pub trait NdCvConversion<T: bytemuck::Pod + Copy, D: ndarray::Dimension>:
    seal::SealedInternal + Sized
{
    #[deprecated = "Use NdAsMat and NdAsImage traits instead"]
    fn to_mat(&self) -> Result<opencv::core::Mat, ConversionError>;
    #[deprecated = "Use NdAsMat and NdAsImage traits instead"]
    fn from_mat(
        mat: opencv::core::Mat,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>, ConversionError>;
}

#[allow(deprecated)]
impl<T: bytemuck::Pod + Copy, S: ndarray::Data<Elem = T>, D: ndarray::Dimension>
    NdCvConversion<T, D> for ndarray::ArrayBase<S, D>
where
    Self: NdAsImage<T, D>,
{
    fn to_mat(&self) -> Result<opencv::core::Mat, ConversionError> {
        Ok(self.as_image_mat()?.mat.clone())
    }

    fn from_mat(
        mat: opencv::core::Mat,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>, ConversionError> {
        let ndarray = unsafe { impls::mat_to_ndarray::<T, D>(&mat) }?;
        Ok(ndarray.to_owned())
    }
}

pub trait MatAsNd {
    fn as_ndarray<T: bytemuck::Pod, D: ndarray::Dimension>(
        &self,
    ) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError>;
}

impl MatAsNd for opencv::core::Mat {
    fn as_ndarray<T: bytemuck::Pod, D: ndarray::Dimension>(
        &self,
    ) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError> {
        unsafe { impls::mat_to_ndarray::<T, D>(self) }
    }
}

pub trait NdAsMat<T: bytemuck::Pod + Copy, D: ndarray::Dimension> {
    fn as_single_channel_mat(&self) -> Result<MatRef<'_>, ConversionError>;
    fn as_multi_channel_mat(&self) -> Result<MatRef<'_>, ConversionError>;
}

pub trait NdAsMatMut<T: bytemuck::Pod + Copy, D: ndarray::Dimension>: NdAsMat<T, D> {
    fn as_single_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
    fn as_multi_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
}

impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>, D: ndarray::Dimension> NdAsMat<T, D>
    for ndarray::ArrayBase<S, D>
{
    fn as_single_channel_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        let mat = unsafe { impls::ndarray_to_mat_regular(self) }?;
        Ok(MatRef::new(mat))
    }
    fn as_multi_channel_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        let mat = unsafe { impls::ndarray_to_mat_consolidated(self) }?;
        Ok(MatRef::new(mat))
    }
}

impl<T: bytemuck::Pod, S: ndarray::DataMut<Elem = T>, D: ndarray::Dimension> NdAsMatMut<T, D>
    for ndarray::ArrayBase<S, D>
{
    fn as_single_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        let mat = unsafe { impls::ndarray_to_mat_regular(self) }?;
        Ok(MatRefMut::new(mat))
    }

    fn as_multi_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        let mat = unsafe { impls::ndarray_to_mat_consolidated(self) }?;
        Ok(MatRefMut::new(mat))
    }
}

pub trait NdAsImage<T: bytemuck::Pod, D: ndarray::Dimension> {
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError>;
}

pub trait NdAsImageMut<T: bytemuck::Pod, D: ndarray::Dimension> {
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
}

impl<T, S> NdAsImage<T, Ix2> for ndarray::ArrayBase<S, Ix2>
where
    T: bytemuck::Pod + Copy,
    S: ndarray::Data<Elem = T>,
{
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        self.as_single_channel_mat()
    }
}

impl<T, S> NdAsImageMut<T, Ix2> for ndarray::ArrayBase<S, Ix2>
where
    T: bytemuck::Pod + Copy,
    S: ndarray::DataMut<Elem = T>,
{
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        self.as_single_channel_mat_mut()
    }
}

impl<T, S> NdAsImage<T, Ix3> for ndarray::ArrayBase<S, Ix3>
where
    T: bytemuck::Pod + Copy,
    S: ndarray::Data<Elem = T>,
{
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        self.as_multi_channel_mat()
    }
}

impl<T, S> NdAsImageMut<T, Ix3> for ndarray::ArrayBase<S, Ix3>
where
    T: bytemuck::Pod + Copy,
    S: ndarray::DataMut<Elem = T>,
{
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        self.as_multi_channel_mat_mut()
    }
}

#[test]
fn test_1d_mat_to_ndarray() {
    let mat = opencv::core::Mat::new_nd_with_default(
        &[10],
        opencv::core::CV_MAKE_TYPE(opencv::core::CV_8U, 1),
        200.into(),
    )
    .expect("failed");
    let array: ndarray::ArrayView1<u8> = mat.as_ndarray().expect("failed");
    array.into_iter().for_each(|&x| assert_eq!(x, 200));
}

#[test]
fn test_2d_mat_to_ndarray() {
    let mat = opencv::core::Mat::new_nd_with_default(
        &[10],
        opencv::core::CV_16SC3,
        (200, 200, 200).into(),
    )
    .expect("failed");
    let array2: ndarray::ArrayView2<i16> = mat.as_ndarray().expect("failed");
    assert_eq!(array2.shape(), [10, 3]);
    array2.into_iter().for_each(|&x| {
        assert_eq!(x, 200);
    });
    let array2: ndarray::ArrayView3<i16> = mat.as_ndarray().expect("failed");
    assert_eq!(array2.shape(), [10, 1, 3]);
    array2.into_iter().for_each(|&x| {
        assert_eq!(x, 200);
    });
}

#[test]
fn test_3d_mat_to_ndarray() {
    let mat = opencv::core::Mat::new_nd_with_default(
        &[20, 30],
        opencv::core::CV_32FC3,
        (200, 200, 200).into(),
    )
    .expect("failed");
    let array2: ndarray::ArrayView3<f32> = mat.as_ndarray().expect("failed");
    array2.into_iter().for_each(|&x| {
        assert_eq!(x, 200f32);
    });
}

#[test]
fn test_mat_to_dyn_ndarray() {
    let mat = opencv::core::Mat::new_nd_with_default(&[10], opencv::core::CV_8UC1, 200.into())
        .expect("failed");
    let array2: ndarray::ArrayViewD<u8> = mat.as_ndarray().expect("failed");
    array2.into_iter().for_each(|&x| assert_eq!(x, 200));
}

#[test]
fn test_3d_mat_to_ndarray_4k() {
    let mat = opencv::core::Mat::new_nd_with_default(
        &[4096, 4096],
        opencv::core::CV_8UC3,
        (255, 0, 255).into(),
    )
    .expect("failed");
    let array2: ndarray::ArrayView3<u8> = (mat).as_ndarray().expect("failed");
    array2.exact_chunks((1, 1, 3)).into_iter().for_each(|x| {
        assert_eq!(x[(0, 0, 0)], 255);
        assert_eq!(x[(0, 0, 1)], 0);
        assert_eq!(x[(0, 0, 2)], 255);
    });
}

// #[test]
// fn test_3d_mat_to_ndarray_8k() {
//     let mat = opencv::core::Mat::new_nd_with_default(
//         &[8192, 8192],
//         opencv::core::CV_8UC3,
//         (255, 0, 255).into(),
//     )
//     .expect("failed");
//     let array2 = ndarray::Array3::<u8>::from_mat(mat).expect("failed");
//     array2.exact_chunks((1, 1, 3)).into_iter().for_each(|x| {
//         assert_eq!(x[(0, 0, 0)], 255);
//         assert_eq!(x[(0, 0, 1)], 0);
//         assert_eq!(x[(0, 0, 2)], 255);
//     });
// }

#[test]
pub fn test_mat_to_nd_default_strides() {
    let mat = opencv::core::Mat::new_rows_cols_with_default(
        10,
        10,
        opencv::core::CV_8UC3,
        opencv::core::VecN([10f64, 0.0, 0.0, 0.0]),
    )
    .expect("failed");
    let array = unsafe { impls::mat_to_ndarray::<u8, Ix3>(&mat) }.expect("failed");
    assert_eq!(array.shape(), [10, 10, 3]);
    assert_eq!(array.strides(), [30, 3, 1]);
    assert_eq!(array[(0, 0, 0)], 10);
}

// #[test]
// pub fn test_mat_to_nd_custom_strides() {
//     let mat = opencv::core::Mat::new_rows_cols_with_default(
//         10,
//         10,
//         opencv::core::CV_8UC3,
//         opencv::core::VecN([10f64, 0.0, 0.0, 0.0]),
//     )
//     .unwrap();
//     let mat_roi = opencv::core::Mat::roi(&mat, opencv::core::Rect::new(3, 2, 3, 5))
//         .expect("failed to get roi");
//     let array = unsafe { impls::mat_to_ndarray::<u8, Ix3>(&mat_roi) }.expect("failed");
//     assert_eq!(array.shape(), [5, 3, 3]);
//     assert_eq!(array.strides(), [30, 3, 1]);
//     assert_eq!(array[(0, 0, 0)], 10);
// }

#[test]
pub fn test_non_continuous_3d() {
    let array = ndarray::Array3::<f32>::from_shape_fn((10, 10, 4), |(i, j, k)| {
        ((i + 1) * (j + 1) * (k + 1)) as f32
    });
    let slice = array.slice(ndarray::s![3..7, 3..7, 0..4]);
    let mat = unsafe { impls::ndarray_to_mat_consolidated(&slice) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, Ix3>(&mat).unwrap() };
    assert!(slice == arr);
}

#[test]
pub fn test_5d_array() {
    let array = ndarray::Array5::<f32>::ones((1, 2, 3, 4, 5));
    let mat = unsafe { impls::ndarray_to_mat_consolidated(&array) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix5>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
pub fn test_3d_array() {
    let array = ndarray::Array3::<f32>::ones((23, 31, 33));
    let mat = unsafe { impls::ndarray_to_mat_consolidated(&array) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix3>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
pub fn test_2d_array_consolidated() {
    let array = ndarray::Array2::<f32>::ones((23, 31));
    let mat = unsafe { impls::ndarray_to_mat_consolidated(&array) }.unwrap();
    dbg!(&mat);
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix2>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
#[should_panic]
pub fn test_1d_array_consolidated() {
    let array = ndarray::Array1::<f32>::ones(23);
    let mat = unsafe { impls::ndarray_to_mat_consolidated(&array) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix1>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
pub fn test_1d_array_regular() {
    let array = ndarray::Array1::<f32>::ones(23);
    let mat = unsafe { impls::ndarray_to_mat_regular(&array) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix1>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
pub fn test_2d_array_regular() {
    let array = ndarray::Array2::<f32>::ones((23, 31));
    let mat = unsafe { impls::ndarray_to_mat_regular(&array) }.unwrap();
    let arr = unsafe { impls::mat_to_ndarray::<f32, ndarray::Ix2>(&mat).unwrap() };
    assert_eq!(array, arr);
}

#[test]
#[allow(deprecated)]
pub fn test_ndcv_1024_1024_to_mat() {
    let array = ndarray::Array2::<f32>::ones((1024, 1024));
    let _mat = array.to_mat().unwrap();
}

#[test]
fn test_3d_mat_to_ndarray_with_broadcasted_1d() {
    let mat1 =
        opencv::core::Mat::new_nd_with_default(&[10, 1], opencv::core::CV_8UC1, (200).into())
            .expect("failed");
    let mat2 =
        opencv::core::Mat::new_nd_with_default(&[10, 1, 1], opencv::core::CV_8UC1, (200).into())
            .expect("failed");

    let array1: ndarray::ArrayView2<u8> = mat1.as_ndarray().expect("failed");
    let array2: ndarray::ArrayView3<u8> = mat2.as_ndarray().expect("failed");
    dbg!(array1.shape());
    dbg!(array2.shape());
    array2.into_iter().for_each(|&x| {
        assert_eq!(x, 200u8);
    });
}

#[test]
fn test_3d_2x1_with_3_channels() {
    // let mat1 =
    //     opencv::core::Mat::new_nd_with_default(&[10, 1], opencv::core::CV_8UC1, (200).into())
    //         .expect("failed");
    let mat2 = opencv::core::Mat::new_nd_with_default(
        &[2, 1],
        opencv::core::CV_8UC3,
        (129, 74, 50).into(),
    )
    .expect("failed");

    dbg!(&mat2);

    // let array1: ndarray::ArrayView2<u8> = mat1.as_ndarray().expect("failed");
    let array2: ndarray::ArrayView3<u8> = mat2.as_ndarray().expect("failed");
    dbg!(array2.shape());
    // array2.into_iter().for_each(|&x| {
    //     assert_eq!(x, 200u8);
    // });
}
