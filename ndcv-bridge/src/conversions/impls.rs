use super::ConversionError;
use super::ConversionErrorKind;
use super::type_depth;
use core::ffi::*;
use opencv::core::prelude::*;
pub(crate) unsafe fn ndarray_to_mat_regular<
    T,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    input: &ndarray::ArrayBase<S, D>,
) -> Result<opencv::core::Mat, ConversionError> {
    let shape = input.shape();
    let strides = input.strides();

    // let channels = shape.last().copied().unwrap_or(1);
    // if channels > opencv::core::CV_CN_MAX as usize {
    //     Err(Report::new(ConversionError).attach(format!(
    //             "Number of channels({channels}) exceeds CV_CN_MAX({}) use the regular version of the function", opencv::core::CV_CN_MAX
    //         )))?;
    // }

    // let size_len = shape.len();
    let size = shape.iter().copied().map(|f| f as i32).collect::<Vec<_>>();
    // Step len for ndarray is always 1 less than ndims
    let step_len = strides.len() - 1;
    let step = strides
        .iter()
        .take(step_len)
        .copied()
        .map(|f| f as usize * core::mem::size_of::<T>())
        .collect::<Vec<_>>();

    let data_ptr = input.as_ptr() as *const c_void;

    let typ = opencv::core::CV_MAKETYPE(type_depth::<T>(), 1);
    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            size.as_slice(),
            typ,
            data_ptr.cast_mut(),
            Some(step.as_slice()),
        )?
    };

    Ok(mat)
}

pub(crate) unsafe fn ndarray_to_mat_consolidated<
    T,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    input: &ndarray::ArrayBase<S, D>,
) -> Result<opencv::core::Mat, ConversionError> {
    let shape = input.shape();
    let strides = input.strides();

    let channels = shape.last().copied().unwrap_or(1);
    if channels > opencv::core::CV_CN_MAX as usize {
        Err(ConversionErrorKind::InvalidNumberOfChannels {
            max: opencv::core::CV_CN_MAX as usize,
            found: channels,
        })?;
    }

    if shape.len() > 2 {
        // Basically the second last stride is used to jump from one column to next
        // But opencv only keeps ndims - 1 strides so we can't have the column stride as that
        // will be lost
        if shape.last() != strides.get(strides.len() - 2).map(|x| *x as usize).as_ref() {
            Err(ConversionErrorKind::NonContiguousData)?;
        }
    } else if shape.len() == 1 {
        return Err(ConversionErrorKind::UnsupportedNdarrayShape)?;
    }

    // Since this is the consolidated version we should always only have ndims - 1 sizes and
    // ndims - 2 strides

    let size_len = shape.len() - 1; // Since we move last axis into the channel
    let size = shape
        .iter()
        .take(size_len)
        .map(|f| *f as i32)
        .collect::<Vec<_>>();

    let step_len = strides.len() - 1;
    let step = strides
        .iter()
        .take(step_len)
        .map(|f| *f as usize * core::mem::size_of::<T>())
        .collect::<Vec<_>>();

    let data_ptr = input.as_ptr() as *const c_void;

    let typ = opencv::core::CV_MAKETYPE(type_depth::<T>(), channels as i32);

    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            size.as_slice(),
            typ,
            data_ptr.cast_mut(),
            Some(step.as_slice()),
        )
    }?;

    Ok(mat)
}

pub(crate) unsafe fn mat_to_ndarray<T: bytemuck::Pod, D: ndarray::Dimension>(
    mat: &opencv::core::Mat,
) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError> {
    let depth = mat.depth();
    if type_depth::<T>() != depth {
        Err(ConversionErrorKind::TypeMismatch {
            expected: std::any::type_name::<T>()
                .rsplit_once("::")
                .expect("Impossible")
                .1,
            got: crate::depth_type(depth),
        })?;
        // return Err(Report::new(ConversionError).attach(format!(
        //     "Expected type Mat<{}> ({}), got Mat<{}> ({})",
        //     std::any::type_name::<T>(),
        //     type_depth::<T>(),
        //     crate::depth_type(depth),
        //     depth,
        // )));
    }

    let channels = mat.channels();
    let multi_channel = channels > 1;

    let mat_dims = mat.dims(); // dims is always >= 2
    let maybe_1d = mat_dims == 2
        && ((mat.rows() == mat.total() as i32 && mat.cols() == 1)
            || (mat.cols() == mat.total() as i32 && mat.rows() == 1));

    let (are_dims_compatible, dim) = match (D::NDIM, maybe_1d, multi_channel) {
        (Some(ndim), false, false) => {
            // for example a 3d mat with shape (2,3,4) and data type CV_8UC1 maps to a ndarray with
            // the same shape (2,3,4)
            (ndim == mat_dims as usize, ndim)
        }
        (Some(ndim), false, true) => {
            // for example a 3d mat with shape (2,3,4) and data type CV_8UC3 maps to a ndarray with
            // shape (2,3,4,3)
            (ndim == (mat_dims as usize + 1), ndim)
        }
        (Some(ndim), true, false) => {
            // for example a 2d mat with shape (1,12) and data type CV_8UC1 can map to ndarray with
            // shapes (12) or (1,12) or (12,1)
            // So either a 1d or 2d ndarray is compatible
            (ndim == 1 || ndim == 2, ndim)
        }
        (Some(ndim), true, true) => {
            // for example a 2d mat with shape (1,12) and data type CV_8UC3 can map to ndarray with
            // shapes (12,3) or (1,12,3) or (12,1,3)
            // So either a 2d or 3d ndarray is compatible
            (ndim == 2 || ndim == 3, ndim)
        }
        (None, false, false) => {
            // Dynamic dimension ndarray is always compatible but we need to determine the final dims
            (true, mat_dims as usize)
        }
        (None, false, true) => {
            // if multi channel we need to add an extra dim
            (true, mat_dims as usize + 1)
        }
        (None, true, false) => {
            // It's probably better to return 1d and let the user upcast to 2d if they want to
            (true, 1)
        }
        (None, true, true) => {
            // if multi channel we need to add an extra dim It's probably better to return 2d and let the user upcast to 3d if they want to
            (true, 2)
        }
    };

    let multi_channel_1d = maybe_1d && multi_channel && D::NDIM.is_some_and(|d| d == 2);

    if !are_dims_compatible {
        return Err(ConversionErrorKind::IncompatibleDimensions {
            mat_dims: mat_dims as _,
            rows: mat.rows() as _,
            cols: mat.cols() as _,
            channels: channels as _,
            ndarray_dims: D::NDIM.unwrap_or(0),
        })?;
    }

    let mat_size = mat.mat_size();

    use ndarray::ShapeBuilder;
    let sizes = (0..(mat.dims() - multi_channel_1d as i32))
        .map(|i| mat_size.get(i).map_err(ConversionError::from))
        .chain([Ok(channels)])
        .map(|x| x.map(|x| x as usize))
        .take(dim)
        .collect::<Result<Vec<_>, ConversionError>>()?;
    let strides = (0..(mat.dims() - 1 - multi_channel_1d as i32))
        .map(|i| mat.step1(i).map_err(ConversionError::from))
        .chain([Ok(channels as usize), Ok(1)])
        .take(dim)
        .collect::<Result<Vec<_>, ConversionError>>()?;
    let shape = sizes.strides(strides);

    let raw_array = unsafe {
        ndarray::RawArrayView::from_shape_ptr(shape, mat.data() as *const T)
            .into_dimensionality()?
    };
    Ok(unsafe { raw_array.deref_into_view() })
}

#[test]
fn mat_test_all_types() {
    fn print_all_mat<T: opencv::core::MatTraitConst>(mat: &T, name: impl AsRef<str>) {
        println!(
            "Mat({:^10}): dims {} rows {:>3}, cols {:>3}, total {:>3}, channels {}, depth {}",
            name.as_ref(),
            mat.dims(),
            mat.rows(),
            mat.cols(),
            mat.total(),
            mat.channels(),
            mat.depth()
        );
    }
    let mat1 = opencv::core::Mat::from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let mat1a =
        opencv::core::Mat::new_nd_with_default(&[8], opencv::core::CV_8UC1, (10).into()).unwrap();

    let mat2 = opencv::core::Mat::from_slice_2d(&[[1u8, 2, 3, 4], [5, 6, 7, 8]]).unwrap();
    let mat2a = opencv::core::Mat::new_nd_with_default(&[8, 1], opencv::core::CV_8UC1, (10).into())
        .unwrap();
    let mat2b = opencv::core::Mat::new_nd_with_default(&[1, 8], opencv::core::CV_8UC1, (10).into())
        .unwrap();
    let mat2c = opencv::core::Mat::new_nd_with_default(&[2, 4], opencv::core::CV_8UC1, (10).into())
        .unwrap();
    let mat3 =
        opencv::core::Mat::new_nd_with_default(&[2, 2, 2], opencv::core::CV_8UC1, (10).into())
            .unwrap();
    let mat3a =
        opencv::core::Mat::new_nd_with_default(&[2, 2], opencv::core::CV_8UC3, (10, 10, 10).into())
            .unwrap();
    let mat3b = opencv::core::Mat::new_nd_with_default(
        &[3, 1, 1],
        opencv::core::CV_8UC1,
        (10, 10, 10).into(),
    )
    .unwrap();
    let mat4 =
        opencv::core::Mat::new_nd_with_default(&[2, 2, 2, 2], opencv::core::CV_8UC1, (10).into())
            .unwrap();
    let mat4a = opencv::core::Mat::new_nd_with_default(
        &[2, 2, 2],
        opencv::core::CV_8UC3,
        (10, 10, 10).into(),
    )
    .unwrap();
    print_all_mat(&mat1, "mat1");
    print_all_mat(&mat1a, "mat1a");
    print_all_mat(&mat2, "mat2");
    print_all_mat(&mat2a, "mat2a");
    print_all_mat(&mat2b, "mat2b");
    print_all_mat(&mat2c, "mat2c");
    print_all_mat(&mat3, "mat3");
    print_all_mat(&mat3a, "mat3a");
    print_all_mat(&mat3b, "mat3b");
    print_all_mat(&mat4, "mat4");
    print_all_mat(&mat4a, "mat4a");
}
