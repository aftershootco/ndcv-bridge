#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array1, Array2, Array3};
use ndcv_bridge::conversions::{MatAsNd, NdAsMat};

#[derive(Arbitrary, Debug)]
struct ConversionInput {
    height: u8,
    width: u8,
    channels: u8,
    elem_type: ElemType,
    conversion_mode: ConversionMode,
    data: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
enum ElemType {
    U8,
    I16,
    F32,
    F64,
}

#[derive(Arbitrary, Debug)]
enum ConversionMode {
    /// 1D array -> single channel Mat -> back
    Array1d,
    /// 2D array -> single channel Mat -> back
    Array2dSingleChannel,
    /// 2D array -> consolidated (last dim = channels) Mat -> back
    Array2dConsolidated,
    /// 3D array -> consolidated Mat -> back
    Array3dConsolidated,
}

macro_rules! fuzz_roundtrip_1d {
    ($t:ty, $data:expr, $size:expr) => {{
        let byte_size = $size * std::mem::size_of::<$t>();
        if $data.len() < byte_size {
            return;
        }
        let vals: Vec<$t> = bytemuck::cast_slice(&$data[..byte_size]).to_vec();
        let arr = Array1::from_vec(vals);
        if let Ok(mat_ref) = arr.as_single_channel_mat() {
            if let Ok(view) = mat_ref.as_ndarray::<$t, ndarray::Ix1>() {
                assert_eq!(arr, view);
            }
        }
    }};
}

macro_rules! fuzz_roundtrip_2d_single {
    ($t:ty, $data:expr, $h:expr, $w:expr) => {{
        let total = $h * $w;
        let byte_size = total * std::mem::size_of::<$t>();
        if $data.len() < byte_size {
            return;
        }
        let vals: Vec<$t> = bytemuck::cast_slice(&$data[..byte_size]).to_vec();
        let arr = Array2::from_shape_vec(($h, $w), vals).unwrap();
        if let Ok(mat_ref) = arr.as_single_channel_mat() {
            if let Ok(view) = mat_ref.as_ndarray::<$t, ndarray::Ix2>() {
                assert_eq!(arr, view);
            }
        }
    }};
}

macro_rules! fuzz_roundtrip_3d_consolidated {
    ($t:ty, $data:expr, $h:expr, $w:expr, $c:expr) => {{
        let total = $h * $w * $c;
        let byte_size = total * std::mem::size_of::<$t>();
        if $data.len() < byte_size {
            return;
        }
        let vals: Vec<$t> = bytemuck::cast_slice(&$data[..byte_size]).to_vec();
        let arr = Array3::from_shape_vec(($h, $w, $c), vals).unwrap();
        if let Ok(mat_ref) = arr.as_multi_channel_mat() {
            if let Ok(view) = mat_ref.as_ndarray::<$t, ndarray::Ix3>() {
                assert_eq!(arr, view);
            }
        }
    }};
}

fuzz_target!(|input: ConversionInput| {
    let height = (input.height as usize).clamp(1, 64);
    let width = (input.width as usize).clamp(1, 64);
    let channels = (input.channels as usize).clamp(1, 4);

    match (&input.conversion_mode, &input.elem_type) {
        (ConversionMode::Array1d, ElemType::U8) => {
            fuzz_roundtrip_1d!(u8, input.data, width);
        }
        (ConversionMode::Array1d, ElemType::I16) => {
            fuzz_roundtrip_1d!(i16, input.data, width);
        }
        (ConversionMode::Array1d, ElemType::F32) => {
            fuzz_roundtrip_1d!(f32, input.data, width);
        }
        (ConversionMode::Array1d, ElemType::F64) => {
            fuzz_roundtrip_1d!(f64, input.data, width);
        }
        (ConversionMode::Array2dSingleChannel, ElemType::U8) => {
            fuzz_roundtrip_2d_single!(u8, input.data, height, width);
        }
        (ConversionMode::Array2dSingleChannel, ElemType::I16) => {
            fuzz_roundtrip_2d_single!(i16, input.data, height, width);
        }
        (ConversionMode::Array2dSingleChannel, ElemType::F32) => {
            fuzz_roundtrip_2d_single!(f32, input.data, height, width);
        }
        (ConversionMode::Array2dSingleChannel, ElemType::F64) => {
            fuzz_roundtrip_2d_single!(f64, input.data, height, width);
        }
        (ConversionMode::Array2dConsolidated, ElemType::U8) => {
            fuzz_roundtrip_2d_single!(u8, input.data, height, width);
        }
        (ConversionMode::Array2dConsolidated, ElemType::I16) => {
            fuzz_roundtrip_2d_single!(i16, input.data, height, width);
        }
        (ConversionMode::Array2dConsolidated, ElemType::F32) => {
            fuzz_roundtrip_2d_single!(f32, input.data, height, width);
        }
        (ConversionMode::Array2dConsolidated, ElemType::F64) => {
            fuzz_roundtrip_2d_single!(f64, input.data, height, width);
        }
        (ConversionMode::Array3dConsolidated, ElemType::U8) => {
            fuzz_roundtrip_3d_consolidated!(u8, input.data, height, width, channels);
        }
        (ConversionMode::Array3dConsolidated, ElemType::I16) => {
            fuzz_roundtrip_3d_consolidated!(i16, input.data, height, width, channels);
        }
        (ConversionMode::Array3dConsolidated, ElemType::F32) => {
            fuzz_roundtrip_3d_consolidated!(f32, input.data, height, width, channels);
        }
        (ConversionMode::Array3dConsolidated, ElemType::F64) => {
            fuzz_roundtrip_3d_consolidated!(f64, input.data, height, width, channels);
        }
    }
});
