#![no_main]

use arbitrary::Arbitrary;
use bounding_box::nms::{nms, NmsError};
use bounding_box::Aabb2;
use libfuzzer_sys::fuzz_target;

/// A single bounding box with a score, using f32 for NMS.
#[derive(Debug, Arbitrary)]
struct BoxWithScore {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
}

#[derive(Debug, Arbitrary)]
struct NmsInput {
    boxes: Vec<BoxWithScore>,
    score_threshold: f32,
    nms_threshold: f32,
    /// Extra scores to test length-mismatch error path
    extra_scores: Vec<f32>,
    /// If true, test the mismatch path
    test_mismatch: bool,
}

fuzz_target!(|input: NmsInput| {
    // Cap the number of boxes to avoid quadratic blowup in NMS
    let max_boxes = 128;
    let boxes_slice = if input.boxes.len() > max_boxes {
        &input.boxes[..max_boxes]
    } else {
        &input.boxes[..]
    };

    // Build valid boxes and scores
    let boxes: Vec<Aabb2<f32>> = boxes_slice
        .iter()
        .filter(|b| b.x.is_finite() && b.y.is_finite() && b.w.is_finite() && b.h.is_finite())
        .map(|b| Aabb2::from_xywh(b.x, b.y, b.w.abs(), b.h.abs()))
        .collect();

    let scores: Vec<f32> = boxes_slice
        .iter()
        .filter(|b| b.x.is_finite() && b.y.is_finite() && b.w.is_finite() && b.h.is_finite())
        .map(|b| if b.score.is_finite() { b.score } else { 0.0 })
        .collect();

    let score_threshold = if input.score_threshold.is_finite() {
        input.score_threshold
    } else {
        0.0
    };
    let nms_threshold = if input.nms_threshold.is_finite() {
        input.nms_threshold
    } else {
        0.5
    };

    assert_eq!(boxes.len(), scores.len());

    // Normal NMS path: should never panic
    let result = nms(&boxes, &scores, score_threshold, nms_threshold);
    match result {
        Ok(kept) => {
            // All returned indices should be valid
            for &idx in &kept {
                assert!(idx < boxes.len(), "NMS returned out-of-bounds index {idx}");
            }
            // All kept scores should exceed the threshold
            for &idx in &kept {
                assert!(
                    scores[idx] > score_threshold,
                    "Kept box {idx} has score {} <= threshold {}",
                    scores[idx],
                    score_threshold
                );
            }
            // Verify suppression: no two kept boxes should have IoU > nms_threshold
            let kept_vec: Vec<usize> = kept.iter().copied().collect();
            for i in 0..kept_vec.len() {
                for j in (i + 1)..kept_vec.len() {
                    let iou = boxes[kept_vec[i]].iou(boxes[kept_vec[j]]);
                    // IoU check: the lower-scored one should have been suppressed
                    // NMS suppresses later (lower-scored) boxes, so both being kept
                    // means their IoU should not exceed the threshold
                    if iou.is_finite() {
                        assert!(
                            iou <= nms_threshold,
                            "Kept boxes {} and {} have IoU {iou} > threshold {nms_threshold}",
                            kept_vec[i],
                            kept_vec[j]
                        );
                    }
                }
            }
        }
        Err(NmsError::BoxesAndScoresLengthMismatch { .. }) => {
            // Should not happen here since we ensured equal lengths
            panic!("Unexpected length mismatch with equal-length inputs");
        }
    }

    // Test the mismatch error path
    if input.test_mismatch && !input.extra_scores.is_empty() {
        let bad_scores: Vec<f32> = input
            .extra_scores
            .iter()
            .take(64)
            .map(|s| if s.is_finite() { *s } else { 0.0 })
            .collect();
        if bad_scores.len() != boxes.len() {
            let result = nms(&boxes, &bad_scores, score_threshold, nms_threshold);
            assert!(
                matches!(result, Err(NmsError::BoxesAndScoresLengthMismatch { .. })),
                "Expected length mismatch error, got {result:?}"
            );
        }
    }

    // Edge case: empty input
    let result = nms::<f32>(&[], &[], score_threshold, nms_threshold);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
});
