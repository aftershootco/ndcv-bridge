use std::ops::{Add, Mul};

use super::PasteInput;

/// Blends based on subject mask (foreground = 1.)
pub fn blend<T>(mut input: PasteInput<T>) -> T
where
    T: Copy,
    T: Add<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Mul<f32, Output = T>,
{
    input.mask = 1. - input.mask;
    blend_inverted_mask(input)
}

/// Blends based on background mask (background = 1.)
pub fn blend_inverted_mask<T>(input: PasteInput<T>) -> T
where
    T: Copy,
    T: Add<T, Output = T>,
    T: Mul<T, Output = T>,
    T: Mul<f32, Output = T>,
{
    let PasteInput {
        this,
        other,
        mask,
        alpha,
    } = input;
    let mask = mask * alpha;

    this * (1.0 - mask) + other * mask
}
