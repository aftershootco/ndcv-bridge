pub(crate) mod seal {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
pub trait CvType: seal::Sealed + bytemuck::Pod {
    fn cv_depth() -> i32;
}

macro_rules! impl_cv_depth {
    ($($t:ty => $cv_const:ident),*) => {
        $(
            impl CvDepth for $t {
                fn cv_depth() -> i32 {
                    opencv::core::$cv_const
                }
            }
        )*
    };
}

impl_cv_depth!(
    u8 => CV_8U,
    i8 => CV_8S,
    u16 => CV_16U,
    i16 => CV_16S,
    i32 => CV_32S,
    f32 => CV_32F,
    f64 => CV_64F
);
