pub(crate) mod seal {
    pub trait Sealed {}
    macro_rules! seal {
        ($($t:ty),*) => {
            $(
                impl Sealed for $t {}
            )*
        };
    }
    seal!(
        u8,
        i8,
        u16,
        i16,
        i32,
        f32,
        f64,
        glam::Vec2,
        glam::Vec3,
        glam::Vec4,
        glam::DVec2,
        glam::DVec3,
        glam::DVec4,
        glam::U8Vec2,
        glam::U8Vec3,
        glam::U8Vec4,
        glam::I8Vec2,
        glam::I8Vec3,
        glam::I8Vec4,
        glam::U16Vec2,
        glam::U16Vec3,
        glam::U16Vec4,
        glam::I16Vec2,
        glam::I16Vec3,
        glam::I16Vec4,
        glam::IVec2,
        glam::IVec3,
        glam::IVec4
    );
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
    glam::Vec2 => CV_32F => 2,
    glam::Vec3 => CV_32F => 3,
    glam::Vec4 => CV_32F => 4,
    glam::DVec2 => CV_64F => 2,
    glam::DVec3 => CV_64F => 3,
    glam::DVec4 => CV_64F => 4,
    glam::U8Vec2 => CV_8U => 2,
    glam::U8Vec3 => CV_8U => 3,
    glam::U8Vec4 => CV_8U => 4,
    glam::I8Vec2 => CV_8S => 2,
    glam::I8Vec3 => CV_8S => 3,
    glam::I8Vec4 => CV_8S => 4,
    glam::U16Vec2 => CV_16U => 2,
    glam::U16Vec3 => CV_16U => 3,
    glam::U16Vec4 => CV_16U => 4,
    glam::I16Vec2 => CV_16S => 2,
    glam::I16Vec3 => CV_16S => 3,
    glam::I16Vec4 => CV_16S => 4,
    glam::IVec2 => CV_32S => 2,
    glam::IVec3 => CV_32S => 3,
    glam::IVec4 => CV_32S => 4
);

#[test]
fn test_cv_type() {
    assert_eq!(<u8 as CvType>::cv_type(), opencv::core::CV_8UC1);
    assert_eq!(<glam::Vec3 as CvType>::cv_type(), opencv::core::CV_32FC3);
}
