use std::collections::{HashSet, VecDeque};

use itertools::Itertools;
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NmsError {
    #[error("Boxes and scores length mismatch (boxes: {boxes}, scores: {scores})")]
    BoxesAndScoresLengthMismatch { boxes: usize, scores: usize },
}

use crate::*;
/// Apply Non-Maximum Suppression to a set of bounding boxes.
///
/// # Arguments
///
/// * `boxes` - A slice of bounding boxes to apply NMS on.
/// * `scores` - A slice of confidence scores corresponding to the bounding boxes.
/// * `score_threshold` - The minimum score threshold for consideration.
/// * `nms_threshold` - The IoU threshold for suppression.
///
/// # Returns
///
/// A vector of indices of the bounding boxes that are kept after applying NMS.
pub fn nms<T>(
    boxes: &[Aabb2<T>],
    scores: &[T],
    score_threshold: T,
    nms_threshold: T,
) -> Result<HashSet<usize>, NmsError>
where
    T: Num
        + ordered_float::FloatCore
        + core::ops::Neg<Output = T>
        + core::iter::Product<T>
        + core::ops::AddAssign
        + core::ops::SubAssign
        + core::ops::MulAssign
        + nalgebra::SimdValue
        + nalgebra::SimdPartialOrd,
{
    if boxes.len() != scores.len() {
        return Err(NmsError::BoxesAndScoresLengthMismatch {
            boxes: boxes.len(),
            scores: scores.len(),
        });
    }
    let mut combined: VecDeque<(usize, Aabb2<T>, T, bool)> = boxes
        .iter()
        .enumerate()
        .zip(scores)
        .filter_map(|((idx, bbox), score)| {
            (*score > score_threshold).then_some((idx, *bbox, *score, true))
        })
        .sorted_by_cached_key(|(_, _, score, _)| -ordered_float::OrderedFloat(*score))
        .collect();

    for i in 0..combined.len() {
        let first = combined[i];
        if !first.3 {
            continue;
        }
        let bbox = first.1;
        for item in combined.iter_mut().skip(i + 1) {
            if bbox.iou(item.1) > nms_threshold {
                item.3 = false
            }
        }
    }

    Ok(combined
        .into_iter()
        .filter_map(|(idx, _, _, keep)| keep.then_some(idx))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nms_keeps_best_of_overlaps_and_drops_low_scores() {
        let boxes = vec![
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),   // 0: best of the overlap group
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),   // 1: identical, IoU 1.0 -> suppressed
            Aabb2::from_x1y1x2y2(10.0, 10.0, 12.0, 12.0), // 2: disjoint -> kept
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),   // 3: below score threshold -> filtered
        ];
        let scores = vec![0.9_f64, 0.8, 0.7, 0.1];
        let keep = nms(&boxes, &scores, 0.5, 0.5).unwrap();
        // Box 1 is suppressed by the higher-scoring identical box 0; box 3 never
        // enters because its score is below the threshold.
        assert_eq!(keep, HashSet::from([0, 2]));
    }

    #[test]
    fn nms_errors_on_length_mismatch() {
        let boxes = vec![Aabb2::from_x1y1x2y2(0.0, 0.0, 1.0, 1.0)];
        let scores = vec![0.9_f64, 0.8];
        let err = nms(&boxes, &scores, 0.5, 0.5).unwrap_err();
        assert_eq!(
            err,
            NmsError::BoxesAndScoresLengthMismatch { boxes: 1, scores: 2 }
        );
    }

    #[test]
    fn nms_thresholds_are_strict() {
        // Two identical boxes: IoU is exactly 1.0. With nms_threshold = 1.0 the
        // strict `>` keeps both; a `>=` would suppress one.
        let boxes = vec![
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),
        ];
        let scores = vec![0.9_f64, 0.8];
        let keep = nms(&boxes, &scores, 0.5, 1.0).unwrap();
        assert_eq!(keep, HashSet::from([0, 1]));

        // A score exactly equal to score_threshold is excluded by the strict `>`.
        let boxes = vec![Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0)];
        let scores = vec![0.5_f64];
        let keep = nms(&boxes, &scores, 0.5, 0.5).unwrap();
        assert!(keep.is_empty());
    }

    #[test]
    fn nms_below_threshold_iou_keeps_both() {
        // Two boxes with IoU = 1/7 < 0.5 must both survive.
        let boxes = vec![
            Aabb2::from_x1y1x2y2(0.0, 0.0, 2.0, 2.0),
            Aabb2::from_x1y1x2y2(1.0, 1.0, 3.0, 3.0),
        ];
        let scores = vec![0.9_f64, 0.8];
        let keep = nms(&boxes, &scores, 0.5, 0.5).unwrap();
        assert_eq!(keep, HashSet::from([0, 1]));
    }
}
