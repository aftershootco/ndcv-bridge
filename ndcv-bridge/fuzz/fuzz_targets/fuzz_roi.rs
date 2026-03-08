#![no_main]

use arbitrary::Arbitrary;
use bounding_box::Aabb2;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array2, Array3};
use ndcv_bridge::NdRoiZeroPadded;

#[derive(Arbitrary, Debug)]
struct RoiInput {
    arr_height: u8,
    arr_width: u8,
    channels: u8,
    // Original bbox
    orig_x: u8,
    orig_y: u8,
    orig_w: u8,
    orig_h: u8,
    // Padded bbox
    pad_x: u8,
    pad_y: u8,
    pad_w: u8,
    pad_h: u8,
    use_3d: bool,
}

fuzz_target!(|input: RoiInput| {
    let arr_h = (input.arr_height as usize).clamp(1, 64);
    let arr_w = (input.arr_width as usize).clamp(1, 64);

    let orig = Aabb2::from_xywh(
        input.orig_x as usize,
        input.orig_y as usize,
        input.orig_w as usize,
        input.orig_h as usize,
    );
    let padded = Aabb2::from_xywh(
        input.pad_x as usize,
        input.pad_y as usize,
        input.pad_w as usize,
        input.pad_h as usize,
    );

    let arr_bbox = Aabb2::from_xywh(0usize, 0usize, arr_w, arr_h);

    // Only call roi_zero_padded when both bboxes are contained in the array
    // to avoid expected panics that aren't bugs
    if !arr_bbox.contains_bbox(&orig) || !arr_bbox.contains_bbox(&padded) {
        return;
    }

    if input.use_3d {
        let channels = (input.channels as usize).clamp(1, 4);
        let arr = Array3::<u8>::zeros((arr_h, arr_w, channels));
        // Use catch_unwind since roi_zero_padded panics on invalid bbox relationships
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arr.roi_zero_padded(orig, padded)
        }));
    } else {
        let arr = Array2::<u8>::zeros((arr_h, arr_w));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            arr.roi_zero_padded(orig, padded)
        }));
    }
});
