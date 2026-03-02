#![no_main]

use arbitrary::Arbitrary;
use bounding_box::roi::{MultiRoi, Roi, RoiError, RoiMut};
use bounding_box::Aabb2;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array2, Array3};

#[derive(Debug, Arbitrary)]
struct RoiInput {
    // Image dimensions (clamped to reasonable sizes)
    img_h: u8,
    img_w: u8,
    channels: u8,
    // ROI coordinates
    x1: u16,
    y1: u16,
    x2: u16,
    y2: u16,
    // Additional ROIs for multi_roi
    extra_rois: Vec<(u16, u16, u16, u16)>,
}

fuzz_target!(|input: RoiInput| {
    // Clamp dimensions to avoid huge allocations
    let h = (input.img_h as usize).max(1).min(64);
    let w = (input.img_w as usize).max(1).min(64);
    let c = (input.channels as usize).max(1).min(4);

    let x1 = input.x1 as usize;
    let y1 = input.y1 as usize;
    let x2 = input.x2 as usize;
    let y2 = input.y2 as usize;

    // Test ROI on Array3
    fuzz_roi_3d(h, w, c, x1, y1, x2, y2);

    // Test ROI on Array2
    fuzz_roi_2d(h, w, x1, y1, x2, y2);

    // Test RoiMut on Array3
    fuzz_roi_mut_3d(h, w, c, x1, y1, x2, y2);

    // Test RoiMut on Array2
    fuzz_roi_mut_2d(h, w, x1, y1, x2, y2);

    // Test MultiRoi on Array3<u8>
    fuzz_multi_roi_3d(h, w, c, &input.extra_rois);
});

fn fuzz_roi_3d(h: usize, w: usize, c: usize, x1: usize, y1: usize, x2: usize, y2: usize) {
    let arr = Array3::<u8>::zeros((h, w, c));

    // from_x1y1x2y2 panics if x2 < x1 or y2 < y1, so use from_xywh
    // But we want to test both valid and invalid ROIs via from_xywh
    let (bx, by, bw, bh) = if x2 >= x1 && y2 >= y1 {
        (x1, y1, x2 - x1, y2 - y1)
    } else {
        // Use the raw values as xywh
        (x1, y1, x2, y2)
    };
    let aabb = Aabb2::from_xywh(bx, by, bw, bh);

    match arr.roi(aabb) {
        Ok(view) => {
            // Valid ROI: verify the shape matches
            assert_eq!(view.shape()[0], bh, "ROI height mismatch");
            assert_eq!(view.shape()[1], bw, "ROI width mismatch");
            assert_eq!(view.shape()[2], c, "ROI channels mismatch");
        }
        Err(RoiError::RoiOutOfBounds { max, got }) => {
            // Confirm it actually is out of bounds
            assert!(
                !max.contains_bbox(&got),
                "RoiOutOfBounds error but bbox is contained"
            );
        }
        Err(RoiError::InvalidRoi { .. }) => {
            // Must have had a negative coordinate (impossible with usize, but
            // the error path exists for the trait)
        }
    }
}

fn fuzz_roi_2d(h: usize, w: usize, x1: usize, y1: usize, x2: usize, y2: usize) {
    let arr = Array2::<u8>::zeros((h, w));

    let (bx, by, bw, bh) = if x2 >= x1 && y2 >= y1 {
        (x1, y1, x2 - x1, y2 - y1)
    } else {
        (x1, y1, x2, y2)
    };
    let aabb = Aabb2::from_xywh(bx, by, bw, bh);

    match arr.roi(aabb) {
        Ok(view) => {
            assert_eq!(view.shape()[0], bh);
            assert_eq!(view.shape()[1], bw);
        }
        Err(_) => {}
    }
}

fn fuzz_roi_mut_3d(h: usize, w: usize, c: usize, x1: usize, y1: usize, x2: usize, y2: usize) {
    let mut arr = Array3::<u8>::zeros((h, w, c));

    let (bx, by, bw, bh) = if x2 >= x1 && y2 >= y1 {
        (x1, y1, x2 - x1, y2 - y1)
    } else {
        (x1, y1, x2, y2)
    };
    let aabb = Aabb2::from_xywh(bx, by, bw, bh);

    match arr.roi_mut(aabb) {
        Ok(mut view) => {
            assert_eq!(view.shape()[0], bh);
            assert_eq!(view.shape()[1], bw);
            assert_eq!(view.shape()[2], c);
            // Write into the mutable view
            view.fill(42);
        }
        Err(_) => {}
    }
}

fn fuzz_roi_mut_2d(h: usize, w: usize, x1: usize, y1: usize, x2: usize, y2: usize) {
    let mut arr = Array2::<u8>::zeros((h, w));

    let (bx, by, bw, bh) = if x2 >= x1 && y2 >= y1 {
        (x1, y1, x2 - x1, y2 - y1)
    } else {
        (x1, y1, x2, y2)
    };
    let aabb = Aabb2::from_xywh(bx, by, bw, bh);

    match arr.roi_mut(aabb) {
        Ok(mut view) => {
            assert_eq!(view.shape()[0], bh);
            assert_eq!(view.shape()[1], bw);
            view.fill(42);
        }
        Err(_) => {}
    }
}

fn fuzz_multi_roi_3d(h: usize, w: usize, c: usize, extra_rois: &[(u16, u16, u16, u16)]) {
    let arr = Array3::<u8>::zeros((h, w, c));

    // Build a list of ROIs, capping at 16 to avoid blowup
    let rois: Vec<Aabb2<usize>> = extra_rois
        .iter()
        .take(16)
        .filter_map(|&(rx1, ry1, rx2, ry2)| {
            let (rx1, ry1, rx2, ry2) = (rx1 as usize, ry1 as usize, rx2 as usize, ry2 as usize);
            if rx2 >= rx1 && ry2 >= ry1 {
                Some(Aabb2::from_xywh(rx1, ry1, rx2 - rx1, ry2 - ry1))
            } else {
                None
            }
        })
        .collect();

    match arr.multi_roi(&rois) {
        Ok(views) => {
            assert_eq!(views.len(), rois.len());
        }
        Err(_) => {}
    }
}
