use super::type_depth;
use crate::prelude_::*;
use core::ffi::*;
use opencv::core::prelude::*;
pub(crate) unsafe fn ndarray_to_mat_regular<
    T,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    input: &ndarray::ArrayBase<S, D>,
) -> Result<opencv::core::Mat, NdCvError> {
    let shape = input.shape();
    let strides = input.strides();

    // let channels = shape.last().copied().unwrap_or(1);
    // if channels > opencv::core::CV_CN_MAX as usize {
    //     Err(Report::new(NdCvError).attach(format!(
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
        )
        .change_context(NdCvError)?
    };

    Ok(mat)
}

pub(crate) unsafe fn ndarray_to_mat_consolidated<
    T,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    input: &ndarray::ArrayBase<S, D>,
) -> Result<opencv::core::Mat, NdCvError> {
    let shape = input.shape();
    let strides = input.strides();

    let channels = shape.last().copied().unwrap_or(1);
    if channels > opencv::core::CV_CN_MAX as usize {
        Err(Report::new(NdCvError).attach(format!(
                "Number of channels({channels}) exceeds CV_CN_MAX({}) use the regular version of the function", opencv::core::CV_CN_MAX
            )))?;
    }

    if shape.len() > 2 {
        // Basically the second last stride is used to jump from one column to next
        // But opencv only keeps ndims - 1 strides so we can't have the column stride as that
        // will be lost
        if shape.last() != strides.get(strides.len() - 2).map(|x| *x as usize).as_ref() {
            Err(Report::new(NdCvError)
                .attach("You cannot slice into the last axis in ndarray when converting to mat"))?;
        }
    } else if shape.len() == 1 {
        return Err(Report::new(NdCvError).attach(
            "You cannot convert a 1D array to a Mat while using the consolidated version",
        ));
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
        .change_context(NdCvError)?
    };

    Ok(mat)
}

pub(crate) unsafe fn mat_to_ndarray<T: bytemuck::Pod, D: ndarray::Dimension>(
    mat: &opencv::core::Mat,
) -> Result<ndarray::ArrayView<'_, T, D>, NdCvError> {
    let depth = mat.depth();
    if type_depth::<T>() != depth {
        return Err(Report::new(NdCvError).attach(format!(
            "Expected type Mat<{}> ({}), got Mat<{}> ({})",
            std::any::type_name::<T>(),
            type_depth::<T>(),
            crate::depth_type(depth),
            depth,
        )));
    }

    let channels = mat.channels();
    let multi_channel = channels > 1;

    let mat_dims = mat.dims(); // dims is always >= 2
    let maybe_1d = mat_dims == 2
        && ((mat.rows() == mat.total() as i32 && mat.cols() == 1)
            || (mat.cols() == mat.total() as i32 && mat.rows() == 1));

    // if we don't have any expected dims we'll just convert the
    let expected_mat_dims = D::NDIM.map(|d| d - multi_channel.then_some(1).unwrap_or(0));
    let are_dims_compatible = expected_mat_dims.map_or(true, |expected| {});

    let mat_actual_dims = if maybe_1d { 1 } else { mat_dims };

    // If the mat is multi channel we add an extra dimension at the end for channels in ndarray
    let final_dims = if multi_channel {
        mat_actual_dims + 1
    } else {
        mat_actual_dims
    } as usize;

    use ndarray::ShapeBuilder;
    let mat_size = mat.mat_size();
    let shape = if multi_channel {
        let sizes = (0..mat_actual_dims)
            .map(|i| mat_size.get(i as i32).change_context(NdCvError))
            .chain([Ok(channels)])
            .map(|x| x.map(|x| x as usize))
            .take(final_dims)
            .collect::<Result<Vec<_>, NdCvError>>()
            .change_context(NdCvError)?;
        let strides = (0..(mat_actual_dims - 1))
            .map(|i| mat.step1(i as i32).change_context(NdCvError))
            .chain([Ok(channels as usize), Ok(if channels == 1 { 0 } else { 1 })])
            .take(final_dims)
            .collect::<Result<Vec<_>, NdCvError>>()
            .change_context(NdCvError)?;
        sizes.strides(strides)
    } else {
        let sizes: Vec<_> = [mat_size.get(0).change_context(NdCvError)? as usize]
            .into_iter()
            .chain(core::iter::repeat(1))
            .take(final_dims)
            .collect();
        let strides: Vec<_> = [mat.step1(0).change_context(NdCvError)?]
            .into_iter()
            .chain(core::iter::repeat(0))
            .take(final_dims)
            .collect();
        sizes.strides(strides)
    };

    use ::tap::*;
    let raw_array = unsafe {
        ndarray::RawArrayView::from_shape_ptr(shape, mat.data() as *const T)
            .tap(|d| {
                dbg!(d.shape());
            })
            .into_dimensionality()
            .change_context(NdCvError)?
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
    // print_all_mat(&mat2, "mat2");
    print_all_mat(&mat2a, "mat2a");
    // print_all_mat(&mat2b, "mat2b");
    // print_all_mat(&mat2c, "mat2c");
    // print_all_mat(&mat3, "mat3");
    // print_all_mat(&mat3a, "mat3a");
    // print_all_mat(&mat3b, "mat3b");
    // print_all_mat(&mat4, "mat4");
    // print_all_mat(&mat4a, "mat4a");
    dbg!(&mat1a);
    dbg!(mat2a);
}
