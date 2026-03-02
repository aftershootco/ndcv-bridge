#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array2, Array3};
use ndcv_bridge::fir::{NdAsImage, NdFir};

#[derive(Arbitrary, Debug)]
struct FirInput {
    height: u8,
    width: u8,
    channels: u8,
    target_height: u8,
    target_width: u8,
    use_3d: bool,
    elem_type: ElemType,
    data: Vec<u8>,
}

#[derive(Arbitrary, Debug)]
enum ElemType {
    U8,
    U16,
    F32,
}

fuzz_target!(|input: FirInput| {
    let height = (input.height as usize).clamp(1, 64);
    let width = (input.width as usize).clamp(1, 64);
    let target_height = (input.target_height as usize).clamp(1, 64);
    let target_width = (input.target_width as usize).clamp(1, 64);

    if input.use_3d {
        let channels = (input.channels as usize).clamp(1, 4);
        let total = height * width * channels;

        match input.elem_type {
            ElemType::U8 => {
                if input.data.len() < total {
                    return;
                }
                let arr =
                    Array3::from_shape_vec((height, width, channels), input.data[..total].to_vec())
                        .unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
            ElemType::U16 => {
                let byte_total = total * 2;
                if input.data.len() < byte_total {
                    return;
                }
                let vals: Vec<u16> = input.data[..byte_total]
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                let arr = Array3::from_shape_vec((height, width, channels), vals).unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
            ElemType::F32 => {
                let byte_total = total * 4;
                if input.data.len() < byte_total {
                    return;
                }
                let vals: Vec<f32> = input.data[..byte_total]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let arr = Array3::from_shape_vec((height, width, channels), vals).unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
        }
    } else {
        let total = height * width;
        match input.elem_type {
            ElemType::U8 => {
                if input.data.len() < total {
                    return;
                }
                let arr =
                    Array2::from_shape_vec((height, width), input.data[..total].to_vec()).unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
            ElemType::U16 => {
                let byte_total = total * 2;
                if input.data.len() < byte_total {
                    return;
                }
                let vals: Vec<u16> = input.data[..byte_total]
                    .chunks_exact(2)
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect();
                let arr = Array2::from_shape_vec((height, width), vals).unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
            ElemType::F32 => {
                let byte_total = total * 4;
                if input.data.len() < byte_total {
                    return;
                }
                let vals: Vec<f32> = input.data[..byte_total]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let arr = Array2::from_shape_vec((height, width), vals).unwrap();
                let _ = arr.as_image_ref();
                let _ = arr.fast_resize(target_height, target_width, None);
            }
        }
    }
});
