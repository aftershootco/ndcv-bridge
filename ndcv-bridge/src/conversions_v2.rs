pub fn to_cv_type<T: seal::Sealed>() -> i32 {
    <T as seal::Sealed>::cv_type()
}

mod seal {
    use glam::*;
    pub trait Sealed {
        fn cv_type() -> i32;
        // fn channels() -> i32;
    }
    macro_rules! seal {
        ($type:ty, $cv_type: expr) => {
            impl Sealed for $type {
                fn cv_type() -> i32 {
                    $cv_type
                }
            }
        };
    }
    seal!(f32, opencv::core::CV_32FC1);
    seal!(Vec2, opencv::core::CV_32FC2);
    seal!(Vec3, opencv::core::CV_32FC3);
    seal!(Vec4, opencv::core::CV_32FC4);
    seal!(f64, opencv::core::CV_64FC1);
    seal!(DVec2, opencv::core::CV_64FC2);
    seal!(DVec3, opencv::core::CV_64FC3);
    seal!(DVec4, opencv::core::CV_64FC4);
    seal!(u8, opencv::core::CV_8UC1);
    seal!(U8Vec2, opencv::core::CV_8UC2);
    seal!(U8Vec3, opencv::core::CV_8UC3);
    seal!(U8Vec4, opencv::core::CV_8UC4);
    seal!(i8, opencv::core::CV_8SC1);
    seal!(I8Vec2, opencv::core::CV_8SC2);
    seal!(I8Vec3, opencv::core::CV_8SC3);
    seal!(I8Vec4, opencv::core::CV_8SC4);
    seal!(u16, opencv::core::CV_16UC1);
    seal!(U16Vec2, opencv::core::CV_16UC2);
    seal!(U16Vec3, opencv::core::CV_16UC3);
    seal!(U16Vec4, opencv::core::CV_16UC4);
    seal!(i16, opencv::core::CV_16SC1);
    seal!(I16Vec2, opencv::core::CV_16SC2);
    seal!(I16Vec3, opencv::core::CV_16SC3);
    seal!(I16Vec4, opencv::core::CV_16SC4);
    seal!(i32, opencv::core::CV_32SC1);
    seal!(IVec2, opencv::core::CV_32SC2);
    seal!(IVec3, opencv::core::CV_32SC3);
    seal!(IVec4, opencv::core::CV_32SC4);
    // seal!(u64, opencv::core::CV_64FC1);
    // seal!(U64Vec2, opencv::core::CV_64FC2);
    // seal!(U64Vec3, opencv::core::CV_64FC3);
    // seal!(U64Vec4, opencv::core::CV_64FC4);
    // seal!(i64, opencv::core::CV_64SC1);
    // seal!(I64Vec2, opencv::core::CV_64SC2);
    // seal!(I64Vec3, opencv::core::CV_64SC3);
    // seal!(I64Vec4, opencv::core::CV_64SC4);
}
