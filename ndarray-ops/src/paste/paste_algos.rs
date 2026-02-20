use std::ops::{Add, Mul};

use super::PasteInput;

pub fn blend<T>(input: PasteInput<T>) -> T
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

pub fn overwrite<T>(input: PasteInput<T>) -> T
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
        alpha: _,
    } = input;

    this * (1.0 - mask) + other * mask
}
