#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array2, Array3};
use ndcv_bridge::NdBlend;

#[derive(Arbitrary, Debug)]
struct BlendInput {
    height: u8,
    width: u8,
    channels: u8,
    alpha_bits: u32,
    src_data: Vec<u8>,
    other_data: Vec<u8>,
    mask_data: Vec<u8>,
}

fuzz_target!(|input: BlendInput| {
    let height = (input.height as usize).clamp(1, 32);
    let width = (input.width as usize).clamp(1, 32);
    // Channels 1-4 are interesting; blend uses f32x4 SIMD which reads 4 elements
    let channels = (input.channels as usize).clamp(1, 4);

    let img_total = height * width * channels;
    let mask_total = height * width;

    if input.src_data.len() < img_total * 4
        || input.other_data.len() < img_total * 4
        || input.mask_data.len() < mask_total * 4
    {
        return;
    }

    // Reinterpret bytes as f32
    let src_f32: Vec<f32> = input.src_data[..img_total * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let other_f32: Vec<f32> = input.other_data[..img_total * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let mask_f32: Vec<f32> = input.mask_data[..mask_total * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let src = Array3::from_shape_vec((height, width, channels), src_f32).unwrap();
    let other = Array3::from_shape_vec((height, width, channels), other_f32).unwrap();
    let mask = Array2::from_shape_vec((height, width), mask_f32).unwrap();

    let alpha = f32::from_bits(input.alpha_bits);

    // Test blend (allocating)
    let _ = src.blend(mask.view(), other.view(), alpha);

    // Test blend_inplace
    let mut src_mut = src.clone();
    let _ = src_mut.blend_inplace(mask.view(), other.view(), alpha);

    // Test with mismatched shapes -- should return errors, not panic
    if height > 1 && width > 1 {
        let small_other = Array3::from_shape_vec(
            (height - 1, width, channels),
            vec![0.0f32; (height - 1) * width * channels],
        )
        .unwrap();
        let result = src.blend(mask.view(), small_other.view(), 1.0);
        assert!(result.is_err());
    }
});
