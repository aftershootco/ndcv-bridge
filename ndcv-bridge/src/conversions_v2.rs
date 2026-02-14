// //! opencv::Mat to ndarray::ArrayBase conversions and vice versa, for OpenCV 4.7+ (ndarray 0.17+).
// use crate::conversions::{ConversionError, ConversionErrorKind};
// use crate::types::{CvDepth, CvType};
// mod impls {
//     use super::*;
//     pub(crate) unsafe fn ndarray_to_mat<
//         T: crate::types::CvType,
//         S: ndarray::Data<Elem = T>,
//         D: ndarray::Dimension,
//     >(
//         input: &ndarray::ArrayBase<S, D>,
//     ) -> Result<opencv::core::Mat, ConversionError> {
//         let shape = input.shape();
//         let strides = input.strides();
//
//         let channels = T::channels();
//         if channels > opencv::core::CV_CN_MAX {
//             Err(ConversionErrorKind::InvalidNumberOfChannels {
//                 max: opencv::core::CV_CN_MAX as usize,
//                 found: channels as usize,
//             })?;
//         }
//
//         if shape.len() > 2 {
//             // Basically the second last stride is used to jump from one column to next
//             // But opencv only keeps ndims - 1 strides so we can't have the column stride as that
//             // will be lost
//             if shape.last() != strides.get(strides.len() - 2).map(|x| *x as usize).as_ref() {
//                 Err(ConversionErrorKind::NonContiguousData)?;
//             }
//         } else if shape.len() == 1 {
//             Err(ConversionErrorKind::UnsupportedNdarrayShape)?;
//         }
//
//         // Since this is the consolidated version we should always only have ndims - 1 sizes and
//         // ndims - 2 strides
//
//         let size_len = shape.len() - 1; // Since we move last axis into the channel
//         let size = shape
//             .iter()
//             .take(size_len)
//             .map(|f| *f as i32)
//             .collect::<Vec<_>>();
//
//         let step_len = strides.len() - 1;
//         let step = strides
//             .iter()
//             .take(step_len)
//             .map(|f| *f as usize * core::mem::size_of::<T>())
//             .collect::<Vec<_>>();
//
//         let data_ptr = input.as_ptr() as *const c_void;
//
//         let typ = opencv::core::CV_MAKETYPE(type_depth::<T>(), channels as i32);
//
//         let mat = unsafe {
//             opencv::core::Mat::new_nd_with_data_unsafe(
//                 size.as_slice(),
//                 typ,
//                 data_ptr.cast_mut(),
//                 Some(step.as_slice()),
//             )
//         }?;
//
//         Ok(mat)
//     }
// }
