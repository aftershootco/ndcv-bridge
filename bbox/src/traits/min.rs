pub trait Min: Sized + Copy {
    fn min(self, other: Self) -> Self;
}

macro_rules! impl_min {
    ($($t:ty),*) => {
        $(
            impl Min for $t {
                fn min(self, other: Self) -> Self {
                    Ord::min(self, other)
                }
            }
        )*
    };
    (float $($t:ty),*) => {
        $(
            impl Min for $t {
                fn min(self, other: Self) -> Self {
                    Self::min(self, other)
                }
            }
        )*
    };
}

impl_min!(
    usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128
);
impl_min!(float f32, f64);
