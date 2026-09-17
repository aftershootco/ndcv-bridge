use crate::types::CvType;

use super::ConversionError;
use super::ConversionErrorKind;
use core::ffi::*;
use opencv::core::prelude::*;

/// Translates ndarray strides into OpenCV `Mat` steps.
///
/// ndarray reports a stride of `0` for every axis of length 1 (and for every axis of an
/// empty array) because such a stride is never used to address an element. OpenCV is
/// stricter: `Mat::locateROI` asserts `step[0] > 0`, and every filter that does not pass
/// `BORDER_ISOLATED` calls it. For those axes the contiguous step is substituted; it
/// addresses the same single element and keeps OpenCV happy.
///
/// * `mat_sizes` - extents of the `Mat` axes, outermost first. For the consolidated layout
///   this excludes the trailing ndarray axis that becomes the channel count.
/// * `strides` - the ndarray strides, counted in elements of `T`.
/// * `elem_size` - size of one `Mat` element in bytes (`size_of::<T>()`, times the channel
///   count for the consolidated layout).
/// * `t_size` - `size_of::<T>()`, the unit the ndarray strides are expressed in.
///
/// Returns `mat_sizes.len() - 1` steps, which is what `Mat::new_nd_with_data_unsafe`
/// expects; the innermost step is always the element size and OpenCV fills it in itself.
fn mat_steps(
    mat_sizes: &[usize],
    strides: &[isize],
    elem_size: usize,
    t_size: usize,
) -> Result<Vec<usize>, ConversionError> {
    let mut steps = vec![0usize; mat_sizes.len().saturating_sub(1)];
    let mut contiguous = elem_size;
    for i in (0..steps.len()).rev() {
        contiguous *= mat_sizes[i + 1];
        steps[i] = if mat_sizes[i] > 1 {
            if strides[i] == 0 {
                // A zero stride on an axis longer than one element is a broadcast view.
                // OpenCV cannot express it and would read neighbouring memory instead of
                // repeating the axis, so reject it rather than hand back a wrong Mat.
                Err(ConversionErrorKind::NonContiguousData)?;
            }
            strides[i] as usize * t_size
        } else {
            contiguous
        };
    }
    Ok(steps)
}

pub(crate) unsafe fn ndarray_to_mat_regular<
    T: CvType,
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
    let esz = core::mem::size_of::<T>();
    let step = mat_steps(shape, strides, esz, esz)?;

    let data_ptr = input.as_ptr() as *const c_void;

    let typ = <T as CvType>::cv_type();
    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            size.as_slice(),
            typ,
            data_ptr.cast_mut(),
            (!step.is_empty()).then_some(step.as_slice()),
        )?
    };

    Ok(mat)
}

pub(crate) unsafe fn ndarray_to_mat_consolidated<
    T: CvType,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
>(
    input: &ndarray::ArrayBase<S, D>,
) -> Result<opencv::core::Mat, ConversionError> {
    // The consolidated layout folds the last ndarray axis into Mat channels,
    // which only makes sense for scalar elements: a pixel-typed T would need
    // CV_MAKETYPE(depth, last_dim * T::cv_channels()) and nothing downstream
    // expects such Mats. Use the regular path (Ix2 of pixel T) instead.
    if <T as CvType>::cv_channels() > 1 {
        Err(ConversionErrorKind::UnsupportedDataType(
            std::any::type_name::<T>(),
        ))?;
    }

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
        //
        // A column axis of length 1 reports a stride of 0 in ndarray: there is no second
        // column to jump to, so the layout is trivially compatible and the check does not
        // apply.
        let column_axis = shape.len() - 2;
        if shape[column_axis] > 1
            && shape.last() != strides.get(column_axis).map(|x| *x as usize).as_ref()
        {
            Err(ConversionErrorKind::NonContiguousData)?;
        }
    } else if shape.len() == 1 {
        Err(ConversionErrorKind::UnsupportedNdarrayShape)?;
    }

    // Since this is the consolidated version we should always only have ndims - 1 sizes and
    // ndims - 2 strides

    let size_len = shape.len() - 1; // Since we move last axis into the channel
    let mat_sizes = &shape[..size_len];
    let size = mat_sizes.iter().map(|f| *f as i32).collect::<Vec<_>>();

    // One Mat element is a whole pixel here, since the last ndarray axis became the channels.
    let esz = core::mem::size_of::<T>();
    let step = mat_steps(mat_sizes, strides, esz * channels, esz)?;

    let data_ptr = input.as_ptr() as *const c_void;

    let typ = opencv::core::CV_MAKETYPE(<T as CvType>::cv_depth(), channels as i32);

    let mat = unsafe {
        opencv::core::Mat::new_nd_with_data_unsafe(
            size.as_slice(),
            typ,
            data_ptr.cast_mut(),
            (!step.is_empty()).then_some(step.as_slice()),
        )
    }?;

    Ok(mat)
}

pub(crate) unsafe fn mat_to_ndarray<T: CvType, D: ndarray::Dimension>(
    mat: &opencv::core::Mat,
) -> Result<ndarray::ArrayView<'_, T, D>, ConversionError> {
    let depth = mat.depth();
    if crate::type_depth::<T>() != depth {
        Err(ConversionErrorKind::TypeMismatch {
            expected: std::any::type_name::<T>()
                .rsplit_once("::")
                .map_or_else(|| std::any::type_name::<T>(), |(_, name)| name),
            got: crate::depth_type(depth),
        })?;
    }

    let channels = mat.channels();
    let type_channels = <T as CvType>::cv_channels();
    let multi_channel = channels > 1 && type_channels == 1;

    if type_channels > 1 && channels != type_channels {
        Err(ConversionErrorKind::IncompatibleDimensions {
            mat_dims: mat.dims() as _,
            rows: mat.rows() as _,
            cols: mat.cols() as _,
            channels: channels as _,
            ndarray_dims: D::NDIM.unwrap_or(0),
        })?;
    }

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
        Err(ConversionErrorKind::IncompatibleDimensions {
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
    let strides = if type_channels > 1 {
        (0..(mat.dims() - 1))
            .map(|i| {
                let step = mat.step1(i)?;
                // A step that is not a whole number of pixels cannot be
                // expressed as a stride over T; flooring it would produce a
                // garbage view.
                if !step.is_multiple_of(type_channels as usize) {
                    return Err(ConversionErrorKind::IncompatibleDimensions {
                        mat_dims: mat.dims() as _,
                        rows: mat.rows() as _,
                        cols: mat.cols() as _,
                        channels: channels as _,
                        ndarray_dims: D::NDIM.unwrap_or(0),
                    }
                    .into());
                }
                Ok(step / type_channels as usize)
            })
            .chain([Ok(1)])
            .take(dim)
            .collect::<Result<Vec<_>, ConversionError>>()?
    } else {
        (0..(mat.dims() - 1 - multi_channel_1d as i32))
            .map(|i| mat.step1(i).map_err(ConversionError::from))
            .chain([Ok(channels as usize), Ok(1)])
            .take(dim)
            .collect::<Result<Vec<_>, ConversionError>>()?
    };
    let shape = sizes.strides(strides);

    // RawArrayView::from_shape_ptr requires the pointer to be aligned for T.
    // SIMD-backed pixel types (e.g. glam::Vec4, 16-byte aligned) can exceed
    // the alignment of a Mat built over a foreign buffer. All strides are in
    // whole units of T, so checking the base pointer is sufficient.
    if !(mat.data() as usize).is_multiple_of(core::mem::align_of::<T>()) {
        Err(ConversionErrorKind::MisalignedData {
            align: core::mem::align_of::<T>(),
        })?;
    }

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

#[cfg(test)]
mod unit_axis_stride_tests {
    //! ndarray reports a stride of 0 for an axis of length 1, which used to be copied
    //! verbatim into the OpenCV `Mat` step. Any single-row crop then tripped
    //! `(-215) dims <= 2 && step[0] > 0` inside `Mat::locateROI`.
    use crate::conversions::{ConversionErrorKind, NdAsImage};
    use crate::erode::NdCvErode;
    use crate::{NdCvDilate, NdCvGaussianBlur, NdRoi};
    use bounding_box::Aabb2;
    use ndarray::{Array2, Array3};
    use opencv::core::MatTraitConst;

    fn kernel() -> Array2<u8> {
        Array2::<u8>::ones((3, 3))
    }

    #[test]
    fn single_row_roi_has_positive_step() {
        let parent = Array2::<u8>::zeros((40, 40));
        let roi = parent
            .roi(Aabb2::from_xywh(5usize, 5, 7, 1))
            .unwrap()
            .to_owned();
        // Guards the premise: if ndarray ever stops zeroing unit-axis strides this test
        // stops covering anything.
        assert_eq!(roi.strides(), [0, 1]);

        let mat = roi.as_image_mat().unwrap();
        assert_eq!(mat.dims(), 2);
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 7);
        assert_eq!(mat.step1(0).unwrap(), 7);
    }

    #[test]
    fn single_row_roi_survives_morphology() {
        let parent = Array2::<u8>::zeros((40, 40));
        let roi = parent
            .roi(Aabb2::from_xywh(5usize, 5, 7, 1))
            .unwrap()
            .to_owned();
        let k = kernel();

        assert!(roi.dilate_def(k.view(), 2).is_ok());
        assert!(roi.erode_def(k.view(), 4).is_ok());
        assert!(roi.gaussian_blur_def((31, 31), 0.0).is_ok());
    }

    #[test]
    fn one_by_one_roi_survives_morphology() {
        let parent = Array2::<u8>::zeros((40, 40));
        let roi = parent
            .roi(Aabb2::from_xywh(5usize, 5, 1, 1))
            .unwrap()
            .to_owned();
        assert_eq!(roi.strides(), [0, 0]);

        let k = kernel();
        assert!(roi.dilate_def(k.view(), 2).is_ok());
        assert!(roi.erode_def(k.view(), 4).is_ok());
        assert!(roi.gaussian_blur_def((31, 31), 0.0).is_ok());
    }

    #[test]
    fn single_row_roi_reads_the_right_bytes() {
        // A substituted step must still address the crop itself, not neighbouring memory.
        let mut parent = Array2::<u8>::zeros((40, 40));
        parent
            .iter_mut()
            .enumerate()
            .for_each(|(i, p)| *p = (i % 251) as u8);
        let roi = parent
            .roi(Aabb2::from_xywh(5usize, 5, 7, 1))
            .unwrap()
            .to_owned();

        let mat = roi.as_image_mat().unwrap();
        let round_tripped = crate::MatAsNd::as_ndarray::<u8, ndarray::Ix2>(&*mat).unwrap();
        assert_eq!(round_tripped, roi);
    }

    #[test]
    fn single_row_rgb_roi_converts() {
        let parent = Array3::<u8>::zeros((40, 40, 3));
        let roi = parent
            .roi(Aabb2::from_xywh(5usize, 5, 7, 1))
            .unwrap()
            .to_owned();
        assert_eq!(roi.strides(), [0, 3, 1]);

        let mat = roi.as_image_mat().unwrap();
        assert_eq!(mat.dims(), 2);
        assert_eq!(mat.channels(), 3);
        assert_eq!(mat.step1(0).unwrap(), 21);
        assert!(roi.gaussian_blur_def((31, 31), 0.0).is_ok());
    }

    #[test]
    fn single_column_rgb_view_converts() {
        // The width-1 column axis reports stride 0, which used to be rejected outright by
        // the consolidated layout's column-stride check.
        let parent = Array3::<u8>::zeros((40, 40, 3));
        let view = parent.roi(Aabb2::from_xywh(5usize, 5, 1, 7)).unwrap();
        assert_eq!(view.strides(), [120, 0, 1]);

        let mat = view.as_image_mat().unwrap();
        assert_eq!(mat.rows(), 7);
        assert_eq!(mat.cols(), 1);
        assert_eq!(mat.channels(), 3);
        assert_eq!(mat.step1(0).unwrap(), 120);
    }

    #[test]
    fn broadcast_view_is_rejected() {
        // shape > 1 with stride 0 is a real broadcast; substituting a contiguous step
        // would silently read the wrong memory, so it must be an error instead.
        let row = Array2::<u8>::zeros((1, 8));
        let broadcast = row.broadcast((5, 8)).unwrap();
        assert_eq!(broadcast.strides(), [0, 1]);

        let err = broadcast.as_image_mat().unwrap_err();
        assert!(matches!(err.kind, ConversionErrorKind::NonContiguousData));
    }
}
