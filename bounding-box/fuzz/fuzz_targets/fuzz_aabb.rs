#![no_main]

use arbitrary::Arbitrary;
use bounding_box::{Aabb2, Aabb3};
use libfuzzer_sys::fuzz_target;
use nalgebra::{Point2, Point3, SVector, Vector2, Vector3};

/// Arbitrary input for 2D bounding box operations.
/// Uses `from_xywh` to avoid panics from invalid min/max ordering,
/// then exercises the full API surface.
#[derive(Debug, Arbitrary)]
struct Aabb2Input {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    // Second bbox for binary operations
    x2: f64,
    y2: f64,
    w2: f64,
    h2: f64,
    // Point for containment test
    px: f64,
    py: f64,
    // Scalars
    padding: f64,
    scale_x: f64,
    scale_y: f64,
    tx: f64,
    ty: f64,
    clamp_min: f64,
    clamp_max: f64,
    norm_x: f64,
    norm_y: f64,
}

/// Arbitrary input for 3D bounding box operations.
#[derive(Debug, Arbitrary)]
struct Aabb3Input {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
    h: f64,
    d: f64,
    padding: f64,
}

/// Arbitrary input for integer 2D bounding box operations.
#[derive(Debug, Arbitrary)]
struct Aabb2IntInput {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    x2: i32,
    y2: i32,
    w2: i32,
    h2: i32,
    px: i32,
    py: i32,
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    f64_input: Aabb2Input,
    i32_input: Aabb2IntInput,
    aabb3_input: Aabb3Input,
}

fuzz_target!(|input: FuzzInput| {
    fuzz_aabb2_f64(&input.f64_input);
    fuzz_aabb2_i32(&input.i32_input);
    fuzz_aabb3_f64(&input.aabb3_input);
});

fn fuzz_aabb2_f64(input: &Aabb2Input) {
    // Filter out NaN/Inf to keep operations meaningful
    if ![
        input.x, input.y, input.w, input.h, input.x2, input.y2, input.w2, input.h2,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return;
    }

    // Ensure non-negative sizes — negative sizes create semantically invalid bboxes
    // where min_vertex > max_vertex, causing panics in merge/new.
    let w = input.w.abs();
    let h = input.h.abs();
    let w2 = input.w2.abs();
    let h2 = input.h2.abs();

    let a = Aabb2::from_xywh(input.x, input.y, w, h);
    let b = Aabb2::from_xywh(input.x2, input.y2, w2, h2);

    // Basic accessors
    let _ = a.min_vertex();
    let _ = a.max_vertex();
    let _ = a.size();
    let _ = a.center();
    let _ = a.width();
    let _ = a.height();
    let _ = a.area();
    let _ = a.measure();
    let _ = a.is_positive();
    let _ = a.x1();
    let _ = a.y1();
    let _ = a.x2();
    let _ = a.y2();
    let _ = a.corners();
    let _ = a.x1y1();
    let _ = a.x2y2();
    let _ = a.x1y2();
    let _ = a.x2y1();

    // try_new: should succeed since w, h >= 0 (we use abs)
    {
        let min = Point2::new(input.x, input.y);
        let max = Point2::new(input.x + w, input.y + h);
        // max >= min is guaranteed when w, h >= 0 and addition doesn't overflow to Inf
        if (input.x + w).is_finite() && (input.y + h).is_finite() {
            let via_try = Aabb2::try_new(min, max);
            assert!(
                via_try.is_some(),
                "try_new should succeed for non-negative size: min={min:?}, max={max:?}"
            );
        }
    }

    // Exercise try_new with intentionally inverted coordinates (should return None or Some
    // depending on floating-point precision, so just call without asserting)
    {
        let _ = Aabb2::<f64>::try_new(
            Point2::new(input.x + w, input.y + h),
            Point2::new(input.x, input.y),
        );
    }

    // Binary operations (should not panic)
    let _ = a.intersection(b);
    let _ = a.overlap(b);
    let _ = a.contains_bbox(&b);
    let _ = a.iou(b);
    let _ = a.union(b);
    let _ = a.merge(b);
    let _ = a.clamp(b);

    // contains_point
    if input.px.is_finite() && input.py.is_finite() {
        let pt = Point2::new(input.px, input.py);
        let _ = a.contains_point(pt);
    }

    // Unary transformations
    if input.padding.is_finite() {
        let _ = a.padding_uniform(input.padding);
    }
    if input.scale_x.is_finite() && input.scale_y.is_finite() {
        let _ = a.scale(Vector2::new(input.scale_x, input.scale_y));
        let _ = a.scale_uniform(input.scale_x);
    }
    if input.tx.is_finite() && input.ty.is_finite() {
        let _ = a.translate(Vector2::new(input.tx, input.ty));
        let _ = a.move_to(Point2::new(input.tx, input.ty));
        let _ = a.move_origin(Point2::new(input.tx, input.ty));
    }

    // component_clamp
    if input.clamp_min.is_finite() && input.clamp_max.is_finite() {
        let _ = a.component_clamp(input.clamp_min, input.clamp_max);
    }

    // normalize / denormalize
    if input.norm_x.is_finite()
        && input.norm_y.is_finite()
        && input.norm_x != 0.0
        && input.norm_y != 0.0
    {
        let factor = Vector2::new(input.norm_x, input.norm_y);
        let normed = a.normalize(factor);
        let denormed = normed.denormalize(factor);
        // Check approximate round-trip only for well-behaved values where
        // the normalized result is still finite (avoids overflow to inf)
        let normed_finite = normed.x1().is_finite()
            && normed.y1().is_finite()
            && normed.width().is_finite()
            && normed.height().is_finite();
        if input.norm_x.abs() > 1e-10 && input.norm_y.abs() > 1e-10 && normed_finite {
            let diff_x = (denormed.x1() - a.x1()).abs();
            let diff_y = (denormed.y1() - a.y1()).abs();
            let diff_w = (denormed.width() - a.width()).abs();
            let diff_h = (denormed.height() - a.height()).abs();
            let eps =
                (a.x1().abs() + a.y1().abs() + a.width().abs() + a.height().abs()) * 1e-10 + 1e-10;
            assert!(
                diff_x < eps && diff_y < eps && diff_w < eps && diff_h < eps,
                "normalize/denormalize round-trip failed: original={a:?}, recovered={denormed:?}"
            );
        }
    }

    // cast / try_cast
    let as_f32: Aabb2<f32> = a.as_();
    let _ = as_f32.as_::<f64>();
    let _ = a.try_cast::<f32>();

    // round (Float-specific)
    let _ = a.round();

    // from_vertices
    let _ = Aabb2::from_vertices([a.x1y1(), a.x2y1(), a.x2y2(), a.x1y2()]);

    // Display
    let _ = format!("{a}");

    // zero
    let z = Aabb2::<f64>::zero();
    assert_eq!(z.measure(), 0.0);
}

fn fuzz_aabb2_i32(input: &Aabb2IntInput) {
    // Clamp inputs to a safe range to avoid i32 overflow in arithmetic operations.
    // The library uses unchecked arithmetic internally, so extreme values would cause
    // overflow panics that aren't meaningful bugs.
    const LIMIT: i32 = 10_000;
    let clamp = |v: i32| v.clamp(-LIMIT, LIMIT);

    let x = clamp(input.x);
    let y = clamp(input.y);
    // Ensure non-negative sizes — negative sizes create semantically invalid bboxes
    // that cause panics in merge/new but aren't real bugs.
    let w = clamp(input.w).unsigned_abs() as i32;
    let h = clamp(input.h).unsigned_abs() as i32;
    let x2 = clamp(input.x2);
    let y2 = clamp(input.y2);
    let w2 = clamp(input.w2).unsigned_abs() as i32;
    let h2 = clamp(input.h2).unsigned_abs() as i32;
    let px = clamp(input.px);
    let py = clamp(input.py);

    let a = Aabb2::from_xywh(x, y, w, h);
    let b = Aabb2::from_xywh(x2, y2, w2, h2);

    // Basic accessors — all safe with clamped inputs
    let _ = a.min_vertex();
    let _ = a.max_vertex();
    let _ = a.size();
    let _ = a.center();
    let _ = a.width();
    let _ = a.height();
    let _ = a.measure();
    let _ = a.is_positive();

    // Binary operations
    let _ = a.intersection(b);
    let _ = a.contains_bbox(&b);
    let _ = a.iou(b);
    let _ = a.merge(b);
    let _ = a.union(b);
    let _ = a.overlap(b);
    let _ = a.clamp(b);

    // Point containment
    let pt = Point2::new(px, py);
    let _ = a.contains_point(pt);

    // Scale, translate — use small scale values to avoid overflow in component_mul
    let sx = clamp(input.x2).clamp(-100, 100);
    let sy = clamp(input.y2).clamp(-100, 100);
    let _ = a.scale(Vector2::new(sx, sy));
    let _ = a.translate(Vector2::new(px, py));

    // Casts
    let _ = a.try_cast::<usize>();
    let _ = a.as_::<f64>();

    // Display
    let _ = format!("{a}");
}

fn fuzz_aabb3_f64(input: &Aabb3Input) {
    if ![input.x, input.y, input.z, input.w, input.h, input.d]
        .iter()
        .all(|v| v.is_finite())
    {
        return;
    }

    let a = Aabb3::new_point_size(
        Point3::new(input.x, input.y, input.z),
        Vector3::new(input.w, input.h, input.d),
    );

    let _ = a.min_vertex();
    let _ = a.max_vertex();
    let _ = a.size();
    let _ = a.center();
    let _ = a.volume();
    let _ = a.is_positive();

    if input.padding.is_finite() {
        let _ = a.padding_uniform(input.padding);
        let _ = a.padding(SVector::from_element(input.padding));
    }

    let _ = a.round();
    let _ = format!("{a}");
}
