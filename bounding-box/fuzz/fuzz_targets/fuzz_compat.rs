#![no_main]

use arbitrary::Arbitrary;
use bounding_box::compat;
use libfuzzer_sys::fuzz_target;
use nalgebra::{Point2, Vector2};

/// Arbitrary input for compat::BBox operations.
#[derive(Debug, Arbitrary)]
struct CompatInput {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    x2: f32,
    y2: f32,
    w2: f32,
    h2: f32,
    scalar: f32,
    padding: f32,
    scale_factor: f32,
    offset_x: f32,
    offset_y: f32,
    norm_w: f32,
    norm_h: f32,
    clamp_min: f32,
    clamp_max: f32,
}

#[derive(Debug, Arbitrary)]
struct CompatIntInput {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    x2: i32,
    y2: i32,
    w2: i32,
    h2: i32,
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    float_input: CompatInput,
    int_input: CompatIntInput,
}

fuzz_target!(|input: FuzzInput| {
    fuzz_compat_f32(&input.float_input);
    fuzz_compat_i32(&input.int_input);
});

fn fuzz_compat_f32(input: &CompatInput) {
    if ![input.x, input.y, input.w, input.h]
        .iter()
        .all(|v| v.is_finite())
    {
        return;
    }

    let bbox = compat::BBox::new_xywh(input.x, input.y, input.w, input.h);

    // Accessors via Deref to Aabb2
    let _ = bbox.x1();
    let _ = bbox.y1();
    let _ = bbox.x2();
    let _ = bbox.y2();
    let _ = bbox.width();
    let _ = bbox.height();
    let _ = bbox.area();
    let _ = bbox.center();
    let _ = bbox.top_left();
    let _ = bbox.into_inner();

    // Serde round-trip (JSON)
    let json = serde_json::to_string(&bbox);
    if let Ok(json_str) = json {
        let deserialized: Result<compat::BBox<f32>, _> = serde_json::from_str(&json_str);
        if let Ok(recovered) = deserialized {
            assert_eq!(bbox.x1(), recovered.x1(), "Serde round-trip x1 mismatch");
            assert_eq!(bbox.y1(), recovered.y1(), "Serde round-trip y1 mismatch");
            assert_eq!(
                bbox.width(),
                recovered.width(),
                "Serde round-trip width mismatch"
            );
            assert_eq!(
                bbox.height(),
                recovered.height(),
                "Serde round-trip height mismatch"
            );
        }
    }

    // compat::Point serde round-trip
    let pt = compat::Point::new(input.x, input.y);
    let pt_json = serde_json::to_string(&pt);
    if let Ok(json_str) = pt_json {
        let deserialized: Result<compat::Point<f32>, _> = serde_json::from_str(&json_str);
        if let Ok(recovered) = deserialized {
            assert_eq!(pt.x(), recovered.x());
            assert_eq!(pt.y(), recovered.y());
        }
    }

    // Arithmetic operators (catch div-by-zero)
    if input.scalar.is_finite() {
        let _ = bbox + input.scalar;
        let _ = bbox - input.scalar;
        let _ = bbox * input.scalar;
        if input.scalar != 0.0 {
            let _ = bbox / input.scalar;
        }
    }

    // BBox + BBox (merge via compat Add impl)
    if [input.x2, input.y2, input.w2, input.h2]
        .iter()
        .all(|v| v.is_finite())
    {
        let bbox2 = compat::BBox::new_xywh(input.x2, input.y2, input.w2.abs(), input.h2.abs());
        if input.w.abs() > 0.0
            && input.h.abs() > 0.0
            && input.w2.abs() > 0.0
            && input.h2.abs() > 0.0
        {
            // new_xyxy used internally in Add panics if result has x2 < x1
            // The min/max logic should guarantee x1 <= x2 for non-negative-size inputs
            let bbox_valid = compat::BBox::new_xywh(input.x, input.y, input.w.abs(), input.h.abs());
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bbox_valid + bbox2));
        }

        // overlap
        let _ = bbox.overlap(&bbox2);

        // contains_bbox
        let _ = bbox.contains_bbox(bbox2);

        // clamp_box
        let _ = bbox.clamp_box(bbox2);
    }

    // contains point
    if input.offset_x.is_finite() && input.offset_y.is_finite() {
        let _ = bbox.contains(Point2::new(input.offset_x, input.offset_y));
    }

    // padding operations
    if input.padding.is_finite() {
        let _ = bbox.padding(input.padding);
        let _ = bbox.padding_height(input.padding);
        let _ = bbox.padding_width(input.padding);
    }

    // scale
    if input.scale_factor.is_finite() {
        let _ = bbox.scale(input.scale_factor);
        let _ = bbox.scale_x(input.scale_factor);
        let _ = bbox.scale_y(input.scale_factor);
    }

    // offset / translate
    if input.offset_x.is_finite() && input.offset_y.is_finite() {
        let _ = bbox.offset(Vector2::new(input.offset_x, input.offset_y));
    }

    // normalize / denormalize
    if input.norm_w.is_finite()
        && input.norm_h.is_finite()
        && input.norm_w != 0.0
        && input.norm_h != 0.0
    {
        let normed = bbox.normalize(input.norm_w, input.norm_h);
        let denormed = normed.denormalize(input.norm_w, input.norm_h);
        // Check approximate round-trip
        if input.norm_w.abs() > 1e-5 && input.norm_h.abs() > 1e-5 {
            let eps =
                (bbox.x1().abs() + bbox.y1().abs() + bbox.width().abs() + bbox.height().abs())
                    * 1e-4
                    + 1e-4;
            assert!(
                (denormed.x1() - bbox.x1()).abs() < eps,
                "normalize/denormalize x1 round-trip: {} vs {}",
                denormed.x1(),
                bbox.x1()
            );
        }
    }

    // component clamp
    if input.clamp_min.is_finite() && input.clamp_max.is_finite() {
        let _ = bbox.clamp(input.clamp_min, input.clamp_max);
    }

    // cast
    let _ = bbox.cast::<f64>();
    let _ = bbox.as_::<f64>();

    // round
    let _ = bbox.round();

    // with_top_left
    if input.offset_x.is_finite() && input.offset_y.is_finite() {
        let _ = bbox.with_top_left(Point2::new(input.offset_x, input.offset_y));
    }

    // From<[T; 4]>
    let _ = compat::BBox::from([input.x, input.y, input.w, input.h]);
}

fn fuzz_compat_i32(input: &CompatIntInput) {
    let bbox = compat::BBox::new_xywh(input.x, input.y, input.w, input.h);

    // Accessors (may overflow on max_vertex computation)
    let _ = bbox.x1();
    let _ = bbox.y1();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bbox.x2()));
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bbox.y2()));
    let _ = bbox.width();
    let _ = bbox.height();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bbox.area()));

    // Serde round-trip
    let json = serde_json::to_string(&bbox);
    if let Ok(json_str) = json {
        let deserialized: Result<compat::BBox<i32>, _> = serde_json::from_str(&json_str);
        if let Ok(recovered) = deserialized {
            assert_eq!(bbox.x1(), recovered.x1());
            assert_eq!(bbox.y1(), recovered.y1());
            assert_eq!(bbox.width(), recovered.width());
            assert_eq!(bbox.height(), recovered.height());
        }
    }

    // BBox + BBox merge (integer, may overflow or panic from new_xyxy)
    if let (Some(_x2), Some(_y2), Some(w2), Some(h2)) = (
        input.x2.checked_abs(),
        input.y2.checked_abs(),
        input.w2.checked_abs(),
        input.h2.checked_abs(),
    ) {
        let bbox2 = compat::BBox::new_xywh(input.x2, input.y2, w2, h2);
        let _ =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bbox.contains_bbox(bbox2)));
    }

    // cast to f64 (always safe from i32)
    let _ = bbox.cast::<f64>();

    // zeros_ndarray_2d / 3d: only for non-negative sizes that fit in usize
    if input.w >= 0 && input.h >= 0 && input.w <= 64 && input.h <= 64 {
        let pos_bbox = compat::BBox::new_xywh(0i32, 0i32, input.w, input.h);
        let arr2: ndarray::Array2<u8> = pos_bbox.zeros_ndarray_2d();
        assert_eq!(arr2.shape(), &[input.h as usize, input.w as usize]);
        let arr3: ndarray::Array3<u8> = pos_bbox.zeros_ndarray_3d(3);
        assert_eq!(arr3.shape(), &[input.h as usize, input.w as usize, 3]);
        let arr_ones: ndarray::Array2<u8> = pos_bbox.ones_ndarray_2d();
        assert_eq!(arr_ones.shape(), &[input.h as usize, input.w as usize]);
    }
}
