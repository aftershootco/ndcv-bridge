use crate::prelude_::*;
use ndarray::*;

type Result<T, E = Report<NdCvError>> = std::result::Result<T, E>;

mod seal {
    pub trait Sealed {}
    impl<T: ndarray::Data<Elem = f32>> Sealed for ndarray::ArrayBase<T, ndarray::Ix3> {}
}
pub trait NdBlend<T, D: ndarray::Dimension>: seal::Sealed {
    fn blend(
        &self,
        mask: ndarray::ArrayView<T, D::Smaller>,
        other: ndarray::ArrayView<T, D>,
        alpha: T,
    ) -> Result<ndarray::Array<T, D>>;
    fn blend_inplace(
        &mut self,
        mask: ndarray::ArrayView<T, D::Smaller>,
        other: ndarray::ArrayView<T, D>,
        alpha: T,
    ) -> Result<()>;
}

impl<S> NdBlend<f32, Ix3> for ndarray::ArrayBase<S, Ix3>
where
    S: ndarray::DataMut<Elem = f32>,
{
    fn blend(
        &self,
        mask: ndarray::ArrayView<f32, Ix2>,
        other: ndarray::ArrayView<f32, Ix3>,
        alpha: f32,
    ) -> Result<ndarray::Array<f32, Ix3>> {
        if self.shape() != other.shape() {
            return Err(NdCvError).attach("Shapes of image and other image do not match");
        }
        if self.shape()[0] != mask.shape()[0] || self.shape()[1] != mask.shape()[1] {
            return Err(NdCvError).attach("Shapes of image and mask do not match");
        }

        let mut output = ndarray::Array3::zeros(self.dim());
        let (_height, _width, channels) = self.dim();

        Zip::from(output.lanes_mut(Axis(2)))
            .and(self.lanes(Axis(2)))
            .and(other.lanes(Axis(2)))
            .and(mask)
            .par_for_each(|mut out, this, other, mask| {
                let this = wide::f32x4::from(this.as_slice().expect("Invalid self array"));
                let other = wide::f32x4::from(other.as_slice().expect("Invalid other array"));
                let mask = wide::f32x4::splat(mask * alpha);
                let o = this * (1.0 - mask) + other * mask;
                out.as_slice_mut()
                    .expect("Failed to get mutable slice")
                    .copy_from_slice(&o.as_array()[..channels]);
            });

        Ok(output)
    }

    fn blend_inplace(
        &mut self,
        mask: ndarray::ArrayView<f32, <Ix3 as Dimension>::Smaller>,
        other: ndarray::ArrayView<f32, Ix3>,
        alpha: f32,
    ) -> Result<()> {
        if self.shape() != other.shape() {
            return Err(NdCvError).attach("Shapes of image and other imagge do not match");
        }
        if self.shape()[0] != mask.shape()[0] || self.shape()[1] != mask.shape()[1] {
            return Err(NdCvError).attach("Shapes of image and mask do not match");
        }

        let (_height, _width, channels) = self.dim();

        // Zip::from(self.lanes_mut(Axis(2)))
        //     .and(other.lanes(Axis(2)))
        //     .and(mask)
        //     .par_for_each(|mut this, other, mask| {
        //         let this_wide = wide::f32x4::from(this.as_slice().expect("Invalid self array"));
        //         let other = wide::f32x4::from(other.as_slice().expect("Invalid other array"));
        //         let mask = wide::f32x4::splat(mask * alpha);
        //         let o = this_wide * (1.0 - mask) + other * mask;
        //         this.as_slice_mut()
        //             .expect("Failed to get mutable slice")
        //             .copy_from_slice(&o.as_array()[..channels]);
        //     });
        let this = self
            .as_slice_mut()
            .ok_or(NdCvError)
            .attach("Failed to get source image as a continuous slice")?;
        let other = other
            .as_slice()
            .ok_or(NdCvError)
            .attach("Failed to get other image as a continuous slice")?;
        let mask = mask
            .as_slice()
            .ok_or(NdCvError)
            .attach("Failed to get mask as a continuous slice")?;

        use rayon::prelude::*;
        this.par_chunks_exact_mut(channels)
            .zip(other.par_chunks_exact(channels))
            .zip(mask)
            .for_each(|((this, other), mask)| {
                let this_wide = wide::f32x4::from(&*this);
                let other = wide::f32x4::from(other);
                let mask = wide::f32x4::splat(mask * alpha);
                this.copy_from_slice(
                    &(this_wide * (1.0 - mask) + other * mask).as_array()[..channels],
                );
            });

        // for h in 0.._height {
        //     for w in 0.._width {
        //         let mask_index = h * _width + w;
        //         let mask = mask[mask_index];
        //         let mask = wide::f32x4::splat(mask * alpha);
        //         let this = &mut this[mask_index * channels..(mask_index + 1) * channels];
        //         let other = &other[mask_index * channels..(mask_index + 1) * channels];
        //         let this_wide = wide::f32x4::from(&*this);
        //         let other = wide::f32x4::from(other);
        //         let o = this_wide * (1.0 - mask) + other * mask;
        //         this.copy_from_slice(&o.as_array()[..channels]);
        //     }
        // }
        Ok(())
    }
}

#[test]
pub fn test_blend() {
    let img = Array3::<f32>::from_shape_fn((10, 10, 3), |(i, j, k)| match (i, j, k) {
        (0..=3, _, 0) => 1f32, // red
        (4..=6, _, 1) => 1f32, // green
        (7..=9, _, 2) => 1f32, // blue
        _ => 0f32,
    });
    let other = img.clone().permuted_axes([1, 0, 2]).to_owned();
    let mask = Array2::<f32>::from_shape_fn((10, 10), |(_, j)| if j > 5 { 1f32 } else { 0f32 });
    // let other = Array3::<f32>::zeros((10, 10, 3));
    let out = img.blend(mask.view(), other.view(), 1f32).unwrap();
    let out_u8 = out.mapv(|v| (v * 255f32) as u8);
    let expected = Array3::<u8>::from_shape_fn((10, 10, 3), |(i, j, k)| {
        match (i, j, k) {
            (0..=3, 0..=5, 0) => u8::MAX,                  // red
            (4..=6, 0..=5, 1) | (_, 6, 1) => u8::MAX,      // green
            (7..=9, 0..=5, 2) | (_, 7..=10, 2) => u8::MAX, // blue
            _ => u8::MIN,
        }
    });
    assert_eq!(out_u8, expected);
}

// #[test]
// pub fn test_blend_inplace() {
//     let mut img = Array3::<f32>::from_shape_fn((10, 10, 3), |(i, j, k)| match (i, j, k) {
//         (0..=3, _, 0) => 1f32, // red
//         (4..=6, _, 1) => 1f32, // green
//         (7..=9, _, 2) => 1f32, // blue
//         _ => 0f32,
//     });
//     let other = img.clone().permuted_axes([1, 0, 2]);
//     let mask = Array2::<f32>::from_shape_fn((10, 10), |(_, j)| if j > 5 { 1f32 } else { 0f32 });
//     // let other = Array3::<f32>::zeros((10, 10, 3));
//     img.blend_inplace(mask.view(), other.view(), 1f32).unwrap();
//     let out_u8 = img.mapv(|v| (v * 255f32) as u8);
//     let expected = Array3::<u8>::from_shape_fn((10, 10, 3), |(i, j, k)| {
//         match (i, j, k) {
//             (0..=3, 0..=5, 0) => u8::MAX,                  // red
//             (4..=6, 0..=5, 1) | (_, 6, 1) => u8::MAX,      // green
//             (7..=9, 0..=5, 2) | (_, 7..=10, 2) => u8::MAX, // blue
//             _ => u8::MIN,
//         }
//     });
//     assert_eq!(out_u8, expected);
// }
