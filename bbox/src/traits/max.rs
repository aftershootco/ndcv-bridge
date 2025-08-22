pub trait Max: Sized + Copy {
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_max {
    ($($t:ty),*) => {
        $(
            impl Max for $t {
                fn max(self, other: Self) -> Self {
                    Ord::max(self, other)
                }
            }
        )*
    };
    (float $($t:ty),*) => {
        $(
            impl Max for $t {
                fn max(self, other: Self) -> Self {
                    Self::max(self, other)
                }
            }
        )*
    };
}

impl_max!(usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128);
impl_max!(float f32, f64);
