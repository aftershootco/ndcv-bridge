mod paste;

#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct Rgb<T>(pub [T; 3]);

impl<T> Rgb<T> {
    pub const CHANNELS: usize = 3;

    pub const fn channels(&self) -> usize {
        Self::CHANNELS
    }

    pub fn new(r: T, g: T, b: T) -> Self {
        Self([r, g, b])
    }
}

impl<T> From<[T; 3]> for Rgb<T> {
    fn from(value: [T; 3]) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod test_utils {
    use std::{
        f64,
        ops::{Add, Div, Mul},
    };

    use ndarray::Array2;
    use ndarray_image::NdarrayToImage;

    pub fn circular_wave_mask(
        h: usize,
        w: usize,
        vertical_f: f64,
        horizontal_f: f64,
    ) -> Array2<u8> {
        let mask = Array2::from_shape_fn((h, w), |(h, w)| {
            use f64::consts::PI;

            ((h as f64).div(vertical_f).mul(PI).cos() + (w as f64).div(horizontal_f).mul(PI).sin())
                .add(1.)
                .div(2.)
                .mul(255.)
                .clamp(0., 255.)
                .floor() as u8
        });

        mask
    }

    pub fn save_rgb(img: impl NdarrayToImage<image::RgbImage>, name: &str) {
        if option_env!("TEST_SAVE_IMG").is_some_and(|x| x.eq("1")) {
            let img = img.to_image().unwrap();
            img.save(name).unwrap();
        }
    }

    pub fn save_rgba(img: impl NdarrayToImage<image::RgbaImage>, name: &str) {
        if option_env!("TEST_SAVE_IMG").is_some_and(|x| x.eq("1")) {
            let img = img.to_image().unwrap();
            img.save(name).unwrap();
        }
    }
}
