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
    fn channels() -> i32 {
        1
    }
    fn cv_type() -> i32 {
        opencv::core::CV_MAKETYPE(Self::cv_depth(), Self::channels())
    }
}

macro_rules! impl_cv_depth {
    ($($t:ty => $cv_const:ident $(=> $channels: expr)?),*) => {
        $(
            impl CvType for $t {
                fn cv_depth() -> i32 {
                    opencv::core::$cv_const
                }
                $(fn channels() -> i32 {
                    $channels
                })?
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
    f64 => CV_64F,
    glam::Vec2 => CV_32FC2 => 2,
    glam::Vec3 => CV_32FC3 => 3,
    glam::Vec4 => CV_32FC4 => 4,
    glam::DVec2 => CV_64FC2 => 2,
    glam::DVec3 => CV_64FC3 => 3,
    glam::DVec4 => CV_64FC4 => 4,
    glam::U8Vec2 => CV_8UC2 => 2,
    glam::U8Vec3 => CV_8UC3 => 3,
    glam::U8Vec4 => CV_8UC4 => 4,
    glam::I8Vec2 => CV_8SC2 => 2,
    glam::I8Vec3 => CV_8SC3 => 3,
    glam::I8Vec4 => CV_8SC4 => 4,
    glam::U16Vec2 => CV_16UC2 => 2,
    glam::U16Vec3 => CV_16UC3 => 3,
    glam::U16Vec4 => CV_16UC4 => 4,
    glam::I16Vec2 => CV_16SC2 => 2,
    glam::I16Vec3 => CV_16SC3 => 3,
    glam::I16Vec4 => CV_16SC4 => 4,
    glam::IVec2 => CV_32SC2 => 2,
    glam::IVec3 => CV_32SC3 => 3,
    glam::IVec4 => CV_32SC4 => 4
);
