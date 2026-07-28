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
use ndarray::{Ix2, Ix3};
pub mod impls;
pub(crate) mod matref;
use matref::{MatRef, MatRefMut};

pub(crate) mod seal {
    use crate::types::CvType;
    pub trait SealedInternal {}
    impl<T: CvType, S: ndarray::Data<Elem = T>, D> SealedInternal for ndarray::ArrayBase<S, D> {}
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
    #[error(
        "Mat data pointer is not aligned for the target element type ({align}-byte alignment required)"
    )]
    MisalignedData { align: usize },
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
    #[track_caller]
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

// #[deprecated = "Use NdAsMat and NdAsImage traits instead"]
pub trait NdCvConversion<T: crate::types::CvType, D: ndarray::Dimension>:
    seal::SealedInternal + Sized
{
    // #[deprecated = "Use NdAsMat and NdAsImage traits instead"]
    fn to_mat(&self) -> Result<opencv::core::Mat, ConversionError>;
    // #[deprecated = "Use NdAsMat and NdAsImage traits instead"]
    fn from_mat(
        mat: opencv::core::Mat,
    ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>, ConversionError>;
}

#[allow(deprecated)]
impl<T: crate::types::CvType, S: ndarray::Data<Elem = T>, D: ndarray::Dimension>
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
    fn as_ndarray<T: crate::types::CvType, D: ndarray::Dimension>(
        &self,
    ) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError>;
}

impl MatAsNd for opencv::core::Mat {
    fn as_ndarray<T: crate::types::CvType, D: ndarray::Dimension>(
        &self,
    ) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError> {
        unsafe { impls::mat_to_ndarray::<T, D>(self) }
    }
}

pub trait NdAsMat<T: crate::types::CvType, D: ndarray::Dimension> {
    fn as_single_channel_mat(&self) -> Result<MatRef<'_>, ConversionError>;
    fn as_multi_channel_mat(&self) -> Result<MatRef<'_>, ConversionError>;
}

pub trait NdAsMatMut<T: crate::types::CvType, D: ndarray::Dimension>: NdAsMat<T, D> {
    fn as_single_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
    fn as_multi_channel_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
}

impl<T: crate::types::CvType, S: ndarray::Data<Elem = T>, D: ndarray::Dimension> NdAsMat<T, D>
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

impl<T: crate::types::CvType, S: ndarray::DataMut<Elem = T>, D: ndarray::Dimension> NdAsMatMut<T, D>
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

pub trait NdAsImage<T: crate::types::CvType, D: ndarray::Dimension> {
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError>;
}

pub trait NdAsImageMut<T: crate::types::CvType, D: ndarray::Dimension> {
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError>;
}

impl<T, S> NdAsImage<T, Ix2> for ndarray::ArrayBase<S, Ix2>
where
    T: crate::types::CvType,
    S: ndarray::Data<Elem = T>,
{
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        self.as_single_channel_mat()
    }
}

impl<T, S> NdAsImageMut<T, Ix2> for ndarray::ArrayBase<S, Ix2>
where
    T: crate::types::CvType,
    S: ndarray::DataMut<Elem = T>,
{
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        self.as_single_channel_mat_mut()
    }
}

impl<T, S> NdAsImage<T, Ix3> for ndarray::ArrayBase<S, Ix3>
where
    T: crate::types::CvType,
    S: ndarray::Data<Elem = T>,
{
    fn as_image_mat(&self) -> Result<MatRef<'_>, ConversionError> {
        self.as_multi_channel_mat()
    }
}

impl<T, S> NdAsImageMut<T, Ix3> for ndarray::ArrayBase<S, Ix3>
where
    T: crate::types::CvType,
    S: ndarray::DataMut<Elem = T>,
{
    fn as_image_mat_mut(&mut self) -> Result<MatRefMut<'_>, ConversionError> {
        self.as_multi_channel_mat_mut()
    }
}

#[test]
#[allow(deprecated)]
fn test_nd_cv_conversion_to_mat_from_mat_roundtrip() {
    use opencv::core::MatTraitConst;
    let arr = ndarray::Array2::<u8>::from_shape_fn((3, 5), |(r, c)| (r * 5 + c) as u8);
    // to_mat must clone the real backing Mat, not an empty default.
    let mat = arr.to_mat().unwrap();
    assert_eq!(mat.rows(), 3);
    assert_eq!(mat.cols(), 5);
    // from_mat must reconstruct the array, not an empty default.
    let back = <ndarray::Array2<u8> as NdCvConversion<u8, ndarray::Ix2>>::from_mat(mat).unwrap();
    assert_eq!(back, arr);
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
pub fn test_1d_array_consolidated() {
    // The consolidated path folds the last axis into channels, leaving a 1d
    // array with no spatial axes; it is rejected rather than misread.
    let array = ndarray::Array1::<f32>::ones(23);
    let err = unsafe { impls::ndarray_to_mat_consolidated(&array) }
        .expect_err("1d arrays should be rejected by the consolidated path");
    assert!(
        matches!(err.kind, ConversionErrorKind::UnsupportedNdarrayShape),
        "unexpected error kind: {:?}",
        err.kind
    );
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
pub fn test_2d_array_of_rgb_pixels_as_image_mat_preserves_channels() {
    use opencv::prelude::MatTraitConst;

    let array = ndarray::Array2::from_elem((2, 3), [10_u8, 20, 30]);
    let mat = array.as_image_mat().unwrap();

    assert_eq!(mat.channels(), 3);
    assert_eq!(mat.typ(), opencv::core::CV_8UC3);
    assert_eq!(mat.rows(), 2);
    assert_eq!(mat.cols(), 3);
}

#[test]
pub fn test_2d_array_of_rgb_pixels_roundtrips_through_image_mat() {
    let array = ndarray::arr2(&[
        [[10_u8, 20, 30], [40, 50, 60], [70, 80, 90]],
        [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
    ]);
    let mat = array.as_image_mat().unwrap();
    let roundtrip: ndarray::ArrayView2<[u8; 3]> = mat.as_ndarray().unwrap();

    assert_eq!(roundtrip, array.view());
}

#[test]
pub fn test_single_row_of_rgb_pixels_roundtrips_through_image_mat() {
    // A 1xN image of pixel-typed elements should round-trip preserving values
    // and column order (exercises the multi-channel stride path with mat.rows() == 1).
    let array = ndarray::arr2(&[[[10_u8, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]]]);
    let mat = array.as_image_mat().unwrap();
    let roundtrip: ndarray::ArrayView2<[u8; 3]> = mat.as_ndarray().unwrap();

    assert_eq!(roundtrip.shape(), &[1, 4]);
    assert_eq!(roundtrip, array.view());
}

#[test]
pub fn test_glam_vec3_pixels_roundtrip_through_image_mat() {
    use opencv::prelude::MatTraitConst;

    let array = ndarray::Array2::from_shape_fn((2, 3), |(r, c)| {
        glam::Vec3::new(r as f32, c as f32, (r + c) as f32)
    });
    let mat = array.as_image_mat().unwrap();

    assert_eq!(mat.typ(), opencv::core::CV_32FC3);
    assert_eq!(mat.channels(), 3);

    let roundtrip: ndarray::ArrayView2<glam::Vec3> = mat.as_ndarray().unwrap();
    assert_eq!(roundtrip, array.view());
}

#[test]
pub fn test_mat_to_ndarray_errors_when_pixel_channels_mismatch() {
    // CV_8UC3 Mat read as a 4-channel pixel type must error, not read out of bounds.
    let array = ndarray::Array2::from_elem((2, 3), [10_u8, 20, 30]);
    let mat = array.as_image_mat().unwrap();

    let err = mat
        .as_ndarray::<[u8; 4], ndarray::Ix2>()
        .expect_err("4-channel read of a 3-channel Mat should fail");
    assert!(
        matches!(err.kind, ConversionErrorKind::IncompatibleDimensions { .. }),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_mat_to_ndarray_errors_reading_single_channel_as_pixel() {
    // CV_8UC1 Mat (matching depth) read as a multi-channel pixel type must error.
    let array = ndarray::Array2::<u8>::ones((4, 5));
    let mat = array.as_image_mat().unwrap();

    let err = mat
        .as_ndarray::<[u8; 3], ndarray::Ix2>()
        .expect_err("3-channel read of a single-channel Mat should fail");
    assert!(
        matches!(err.kind, ConversionErrorKind::IncompatibleDimensions { .. }),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_3d_array_of_pixel_elements_errors_as_image_mat() {
    // The consolidated path folds the last axis into channels, which is only
    // meaningful for scalar elements. An Array3 of pixel-typed elements must
    // error instead of silently building a Mat with a mismatched element size.
    let array = ndarray::Array3::from_elem((2, 3, 4), glam::Vec3::ONE);

    let err = array
        .as_image_mat()
        .expect_err("pixel-typed elements must be rejected by the multi-channel path");
    assert!(
        matches!(err.kind, ConversionErrorKind::UnsupportedDataType(_)),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_mat_step_not_multiple_of_pixel_errors() {
    // A Mat whose row step (10 bytes) is not a whole number of 3-byte pixels
    // cannot be expressed as a stride over [u8; 3]; flooring would produce a
    // garbage view.
    let data = [0_u8; 32];
    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            &[2, 3],
            opencv::core::CV_8UC3,
            data.as_ptr() as *mut core::ffi::c_void,
            Some(&[10]),
        )
    }
    .unwrap();

    let err = mat
        .as_ndarray::<[u8; 3], ndarray::Ix2>()
        .expect_err("step not divisible by pixel size should fail");
    assert!(
        matches!(err.kind, ConversionErrorKind::IncompatibleDimensions { .. }),
        "unexpected error kind: {:?}",
        err.kind
    );

    // The scalar view of the same Mat is still representable.
    let scalar: ndarray::ArrayView3<u8> = mat.as_ndarray().unwrap();
    assert_eq!(scalar.shape(), &[2, 3, 3]);
}

#[test]
pub fn test_mat_with_misaligned_data_errors_for_simd_pixel() {
    // glam::Vec4 is 16-byte aligned on SIMD targets; a Mat over a buffer that
    // breaks that alignment must error instead of constructing a UB view.
    let align = core::mem::align_of::<glam::Vec4>();
    if align <= core::mem::align_of::<f32>() {
        // Scalar-math build of glam; nothing to misalign.
        return;
    }

    let backing = [0_f32; 24];
    let mut offset = 0;
    while (backing.as_ptr() as usize + offset * 4).is_multiple_of(align) {
        offset += 1;
    }
    let ptr = unsafe { backing.as_ptr().add(offset) };

    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            &[1, 2],
            opencv::core::CV_32FC4,
            ptr as *mut core::ffi::c_void,
            None,
        )
    }
    .unwrap();

    let err = mat
        .as_ndarray::<glam::Vec4, ndarray::Ix2>()
        .expect_err("misaligned Mat data should fail for a 16-byte-aligned pixel type");
    assert!(
        matches!(err.kind, ConversionErrorKind::MisalignedData { .. }),
        "unexpected error kind: {:?}",
        err.kind
    );
}

// --- Issue #16: non-contiguous / negative-stride views ---------------------
//
// `ndarray_to_mat_regular` drops the innermost stride (assuming it is 1) and
// maps the remaining strides through `as usize`, which wraps negative strides
// into astronomically large Mat steps. Views produced by `.slice(...)` reach
// this path unchecked. Each test below asserts the only two acceptable
// outcomes: a `NonContiguousData` error, or a Mat that reads the same elements
// the view does.

#[test]
pub fn test_column_strided_view_regular_is_not_silently_wrong() {
    // Failure mode 1: the skipped inner stride (2) makes OpenCV read
    // consecutive elements, so row 0 reads [0, 1, 2, 3, 4] instead of the
    // view's [0, 2, 4, 6, 8]. No error is raised; the pixels are just wrong.
    let base = ndarray::Array2::<f32>::from_shape_fn((10, 10), |(r, c)| (r * 10 + c) as f32);
    let view = base.slice(ndarray::s![.., ..;2]);
    assert_eq!(view.shape(), &[10, 5]);

    match unsafe { impls::ndarray_to_mat_regular(&view) } {
        Err(err) => assert!(
            matches!(err.kind, ConversionErrorKind::NonContiguousData),
            "unexpected error kind: {:?}",
            err.kind
        ),
        Ok(mat) => {
            // Reading is in-bounds here (the view spans the whole allocation),
            // so the corruption can be observed directly.
            let roundtrip = unsafe { impls::mat_to_ndarray::<f32, Ix2>(&mat) }.unwrap();
            assert_eq!(
                roundtrip, view,
                "column-strided view built a Mat over the wrong elements"
            );
        }
    }
}

#[test]
pub fn test_inner_reversed_view_regular_is_rejected() {
    // Failure mode 2: strides (10, -1). The dropped -1 makes the Mat read rows
    // forward from the view's base pointer, which sits at the *end* of each
    // logical row, so the last row runs past the allocation. The resulting Mat
    // is deliberately never read here.
    let base = ndarray::Array2::<f32>::from_shape_fn((10, 10), |(r, c)| (r * 10 + c) as f32);
    let view = base.slice(ndarray::s![.., ..;-1]);
    assert_eq!(view.strides(), &[10, -1]);

    let err = unsafe { impls::ndarray_to_mat_regular(&view) }
        .map(|_| ())
        .expect_err("inner-reversed view must not build a Mat that reads out of bounds");
    assert!(
        matches!(err.kind, ConversionErrorKind::NonContiguousData),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_outer_reversed_view_regular_is_rejected() {
    // Failure mode 3: stride -10 becomes `(-10i64 as usize) * 4` (~2^64) as a
    // Mat step, handed straight to OpenCV pointer arithmetic. The resulting Mat
    // is deliberately never read here.
    let base = ndarray::Array2::<f32>::from_shape_fn((10, 10), |(r, c)| (r * 10 + c) as f32);
    let view = base.slice(ndarray::s![..;-1, ..]);
    assert_eq!(view.strides(), &[-10, 1]);

    let err = unsafe { impls::ndarray_to_mat_regular(&view) }
        .map(|_| ())
        .expect_err("outer-reversed view must not build a Mat with a wrapped step");
    assert!(
        matches!(err.kind, ConversionErrorKind::NonContiguousData),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_outer_reversed_view_consolidated_is_rejected() {
    // The consolidated path's middle-stride check does not catch a negative
    // outer stride: shape.last() == 4 still matches strides[len-2] == 4.
    let base = ndarray::Array3::<f32>::from_shape_fn((10, 10, 4), |(i, j, k)| {
        ((i * 40) + (j * 4) + k) as f32
    });
    let view = base.slice(ndarray::s![..;-1, .., ..]);
    assert_eq!(view.strides(), &[-40, 4, 1]);

    let err = unsafe { impls::ndarray_to_mat_consolidated(&view) }
        .map(|_| ())
        .expect_err("outer-reversed view must not build a Mat with a wrapped step");
    assert!(
        matches!(err.kind, ConversionErrorKind::NonContiguousData),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_broadcast_view_is_rejected() {
    // A broadcast axis has stride 0: every step along it aliases the same row.
    // OpenCV's step invariant forbids that, and the `_mut` paths would turn it
    // into aliased writes.
    let base = ndarray::Array2::<f32>::from_shape_fn((10, 10), |(r, c)| (r * 10 + c) as f32);
    let view = base.broadcast((3, 10, 10)).unwrap();
    assert_eq!(view.strides(), &[0, 10, 1]);

    let err = unsafe { impls::ndarray_to_mat_regular(&view) }
        .map(|_| ())
        .expect_err("broadcast view must not build a Mat with a zero step");
    assert!(
        matches!(err.kind, ConversionErrorKind::NonContiguousData),
        "unexpected error kind: {:?}",
        err.kind
    );
}

#[test]
pub fn test_degenerate_innermost_axis_with_odd_stride_is_accepted() {
    // The innermost stride is only dropped, never used, when its axis has
    // length 1 -- the Mat never advances along it, and ndarray normalizes the
    // stride to 0. Such a view is exact, so it must be accepted rather than
    // swept up by the contiguity guard.
    let base = ndarray::Array2::<f32>::from_shape_fn((10, 10), |(r, c)| (r * 10 + c) as f32);
    let view = base.slice(ndarray::s![.., ..1;2]);
    assert_eq!(view.shape(), &[10, 1]);
    assert_eq!(view.strides(), &[10, 0]);

    let mat = unsafe { impls::ndarray_to_mat_regular(&view) }
        .expect("a length-1 innermost axis is exact regardless of its stride");
    let roundtrip = unsafe { impls::mat_to_ndarray::<f32, Ix2>(&mat) }.unwrap();
    assert_eq!(roundtrip, view);
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
