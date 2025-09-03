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

    // Since a dims always returns >= 2 we can't use this to check if it's a 1D array
    // So we compare the first axis to the total to see if its a 1D array
    let channels = mat.channels();
    let is_1d = mat.total() as i32 == mat.rows() && channels == 1;
    let dims = is_1d.then_some(1).unwrap_or(mat.dims());

    let ndarray_size = (channels != 1).then_some(dims + 1).unwrap_or(dims) as usize;
    if let Some(ndim) = D::NDIM {
        // When channels is not 1,
        // the last dimension is the channels
        // Array1 -> Mat(ndims = 1, channels = 1)
        // Array2 -> Mat(ndims = 1, channels = X)
        // Array2 -> Mat(ndims = 2, channels = 1)
        // Array3 -> Mat(ndims = 2, channels = X)
        // Array3 -> Mat(ndims = 3, channels = 1)
        // ...
        // 1D arrays can alwyas be upcasted to and N-Dimentional matrix with 1 in the other channels
        if (ndim != dims as usize && channels == 1) && !is_1d {
            return Err(Report::new(NdCvError)
                .attach(format!("Expected {}D array, got {}D", ndim, ndarray_size)));
        }
    }

    let mat_size = mat.mat_size();
    let sizes = (0..dims)
        .map(|i| mat_size.get(i).change_context(NdCvError))
        .chain([Ok(channels)])
        .map(|x| x.map(|x| x as usize))
        .take(ndarray_size)
        .collect::<Result<Vec<_>, NdCvError>>()
        .change_context(NdCvError)?;
    let strides = (0..(dims - 1))
        .map(|i| mat.step1(i).change_context(NdCvError))
        .chain([Ok(channels as usize), Ok(if channels == 1 { 0 } else { 1 })])
        .take(ndarray_size)
        .collect::<Result<Vec<_>, NdCvError>>()
        .change_context(NdCvError)?;
    use ndarray::ShapeBuilder;
    let shape = sizes.strides(strides);
    let raw_array: ndarray::RawArrayView<T, D> = unsafe {
        if is_1d
            && channels == 1
            && let Some(ndims) = D::NDIM
            && ndims > 1
        {
            // if we compute the size and it turns out to be 1D but the target dimension is more than 1D
            // we need to insert extra axis at the front
            // let arr = ndarray::RawArrayView::from_shape_ptr(shape, mat.data() as *const T)
            //     .into_dimensionality::<ndarray::Ix1>()
            //     .change_context(NdCvError)?;

            // for _ in 1..ndims {
            //     let arr = arr.insert_axis(ndarray::Axis(0));
            // }
            // arr.insert_axis(ndarray::Axis(0));
            let sizes: Vec<_> = [mat_size.get(0).change_context(NdCvError)? as usize]
                .into_iter()
                .chain(core::iter::repeat(1))
                .take(ndims)
                .collect();
            let strides: Vec<_> = [mat.step1(0).change_context(NdCvError)?]
                .into_iter()
                .chain(core::iter::repeat(0))
                .take(ndims)
                .collect();
            use ndarray::ShapeBuilder;
            let shape = sizes.strides(strides);
            ndarray::RawArrayView::from_shape_ptr(shape, mat.data() as *const T)
                .into_dimensionality()
                .change_context(NdCvError)?
        } else {
            ndarray::RawArrayView::from_shape_ptr(shape, mat.data() as *const T)
                .into_dimensionality()
                .change_context(NdCvError)?
        }
    };
    Ok(unsafe { raw_array.deref_into_view() })
}
