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
            if bbox.iou(&item.1) > nms_threshold {
                item.3 = false
            }
        }
    }

    Ok(combined
        .into_iter()
        .filter_map(|(idx, _, _, keep)| keep.then_some(idx))
        .collect())
}
