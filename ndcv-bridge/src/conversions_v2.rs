// use crate::types::{CvDepth, CvType};
//
// /// Get a view into the mat using ndarray
// pub trait MatAsNdarray<T: CvType, CD: CvDepth, D: ndarray::Dimension> {
//     fn as_ndarray<'a>(&'a self) -> ndarray::ArrayView<'a, T, D>;
// }
//
// pub trait NdarrayAsMat<T: CvType, CD: CvDepth, D: ndarray::Dimension> {
//     fn as_mat(&self) -> opencv::core::Mat;
// }
//
// mod impls {
//     use crate::{conversions::ConversionError, types::CvType};
//
//     pub(crate) unsafe fn ndarray_to_mat_regular<T, S, D>(
//         input: &ndarray::ArrayBase<S, D>,
//     ) -> Result<opencv::core::Mat, ConversionError>
//     where
//         T: CvType,
//         <T as CvType>::Depth: crate::types::CvDepth,
//         S: ndarray::Data<Elem = T>,
//         D: ndarray::Dimension,
//     {
//         let shape = input.shape();
//         let strides = input.strides();
//
//         // let channels = shape.last().copied().unwrap_or(1);
//         // if channels > opencv::core::CV_CN_MAX as usize {
//         //     Err(Report::new(ConversionError).attach(format!(
//         //             "Number of channels({channels}) exceeds CV_CN_MAX({}) use the regular version of the function", opencv::core::CV_CN_MAX
//         //         )))?;
//         // }
//
//         // let size_len = shape.len();
//         let size = shape.iter().copied().map(|f| f as i32).collect::<Vec<_>>();
//         // Step len for ndarray is always 1 less than ndims
//         let step_len = strides.len() - 1;
//         let step = strides
//             .iter()
//             .take(step_len)
//             .copied()
//             .map(|f| f as usize * core::mem::size_of::<T>())
//             .collect::<Vec<_>>();
//
//         let data_ptr = input.as_ptr() as *const c_void;
//
//         let typ = opencv::core::CV_MAKETYPE(type_depth::<T>(), 1);
//         let mat = unsafe {
//             opencv::core::Mat::new_nd_with_data_unsafe(
//                 size.as_slice(),
//                 typ,
//                 data_ptr.cast_mut(),
//                 Some(step.as_slice()),
//             )?
//         };
//
//         Ok(mat)
//     }
// }
