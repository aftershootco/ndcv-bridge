/// Primitive types that can be used as Type depth for OpenCV Mat.
pub trait CvDepth: seal::Sealed + bytemuck::Pod + Default {
    fn cv_depth() -> i32;
}

/// Types that can be used as OpenCV Mat types (combination of depth and channels).
pub trait CvType: seal::Sealed + bytemuck::Pod + Default {
    type Depth: CvDepth;
    fn cv_depth() -> i32 {
        Self::Depth::cv_depth()
    }
    fn channels() -> i32 {
        1
    }
    fn cv_type() -> i32 {
        opencv::core::CV_MAKETYPE(Self::cv_depth(), Self::channels())
    }
}

pub(crate) mod seal {
    pub trait Sealed {}
    macro_rules! seal {
        ($($t:ty),*) => {
            $(
                impl Sealed for $t {}
            )*
        };
    }
    macro_rules! seal_array {
        ($t:ty => $($n:expr),*) => {
            $(
                impl Sealed for [$t; $n] {}
            )*
        };
        ($($t:ty),*) => {
            $(seal_array!($t => 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25);)*
        }
    }

    seal!(u8, i8, u16, i16, i32, f32, f64);
    seal_array!(u8, i8, u16, i16, i32, f32, f64);

    #[cfg(feature = "glam")]
    seal!(
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

    #[cfg(feature = "nalgebra")]
    const _: () = {
        impl<T: Sealed, const N: usize> Sealed for nalgebra::SVector<T, N> {}
    };
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

macro_rules! impl_cv_type {
    ($($t:ty),*) => {
        $(
            impl CvType for $t {
                type Depth = $t;
            }
        )*
    };
    (glam: $($t:ty => $depth:ty => $channels:expr),*) => {
        $(
            impl CvType for $t {
                type Depth = $depth;
                fn channels() -> i32 {
                    $channels
                }
            }
        )*
    };
}

impl<T, const N: usize> CvType for [T; N]
where
    T: CvDepth,
    [T; N]: seal::Sealed + Default + bytemuck::Pod,
{
    type Depth = T;
    fn channels() -> i32 {
        N as i32
    }
}

impl_cv_type!(u8, i8, u16, i16, i32, f32, f64);

#[cfg(feature = "glam")]
impl_cv_type!(glam:
    glam::Vec2 => f32 => 2,
    glam::Vec3 => f32 => 3,
    glam::Vec4 => f32 => 4,
    glam::DVec2 => f64 => 2,
    glam::DVec3 => f64 => 3,
    glam::DVec4 => f64 => 4,
    glam::U8Vec2 => u8 => 2,
    glam::U8Vec3 => u8 => 3,
    glam::U8Vec4 => u8 => 4,
    glam::I8Vec2 => i8 => 2,
    glam::I8Vec3 => i8 => 3,
    glam::I8Vec4 => i8 => 4,
    glam::U16Vec2 => u16 => 2,
    glam::U16Vec3 => u16 => 3,
    glam::U16Vec4 => u16 => 4,
    glam::I16Vec2 => i16 => 2,
    glam::I16Vec3 => i16 => 3,
    glam::I16Vec4 => i16 => 4,
    glam::IVec2 => i32 => 2,
    glam::IVec3 => i32 => 3,
    glam::IVec4 => i32 => 4
);

#[cfg(feature = "nalgebra")]
const _: () = {
    use nalgebra::SVector;

    impl<T, const N: usize> CvType for SVector<T, N>
    where
        T: CvDepth,
        SVector<T, N>: seal::Sealed + Default + bytemuck::Pod,
    {
        type Depth = T;
        fn channels() -> i32 {
            if N > opencv::core::CV_CN_MAX as usize {
                panic!("Number of channels exceeds OpenCV's maximum");
            }
            N as i32
        }
    }
};

#[test]
fn test_cv_type() {
    assert_eq!(<u8 as CvType>::cv_type(), opencv::core::CV_8UC1);
    assert_eq!(<glam::Vec3 as CvType>::cv_type(), opencv::core::CV_32FC3);
}
