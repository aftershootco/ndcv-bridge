use crate::types::CvType;
use crate::{NdAsImage, image::NdImage};
use crate::{NdAsImageMut, type_depth};

#[derive(Debug, Clone, derive_builder::Builder)]
#[builder(setter(into), pattern = "owned")]
pub struct SobelArgs {
    dxy: glam::IVec2,
    #[builder(default = "Ksize::K3")]
    ksize: Ksize,
    #[builder(default = "1.0")]
    scale: f64,
    #[builder(default = "0.0")]
    delta: f64,
    #[builder(default = "opencv::core::BorderTypes::BORDER_REFLECT_101")]
    border_type: opencv::core::BorderTypes,
}

impl SobelArgs {
    pub fn builder(dxy: impl Into<glam::IVec2>) -> SobelArgsBuilder {
        SobelArgsBuilder::default().dxy(dxy)
    }
    pub fn dxy(dxy: impl Into<glam::IVec2>) -> SobelArgsBuilder {
        Self::builder(dxy)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Ksize {
    K1 = 1,
    K3 = 3,
    K5 = 5,
    K7 = 7,
}

#[derive(Debug, thiserror::Error)]
pub enum NdCvSobelError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
}

pub trait NdCvSobel<T: CvType, D: ndarray::Dimension>: crate::image::NdImage {
    fn sobel<U: CvType + Default>(
        &self,
        args: SobelArgs,
    ) -> Result<ndarray::Array<U, D>, NdCvSobelError>
    where
        ndarray::Array<U, D>: NdAsImageMut<U, D>;
}

impl<T, D, S> NdCvSobel<T, D> for ndarray::ArrayBase<S, D>
where
    T: CvType,
    D: ndarray::Dimension,
    S: ndarray::RawData<Elem = T> + ndarray::RawDataMut<Elem = T>,
    ndarray::ArrayBase<S, D>: NdAsImage<T, D>,
    ndarray::ArrayBase<S, D>: NdImage,
{
    fn sobel<U: CvType + Default>(
        &self,
        args: SobelArgs,
    ) -> Result<ndarray::Array<U, D>, NdCvSobelError>
    where
        ndarray::Array<U, D>: NdAsImageMut<U, D>,
    {
        let img = self.as_image_mat()?;
        let ddepth = type_depth::<U>();
        let mut dst = ndarray::Array::<U, D>::default(self.raw_dim());
        let mut dst_mat = dst.as_image_mat_mut()?;
        opencv::imgproc::sobel(
            &img,
            &mut dst_mat,
            ddepth,
            args.dxy.x,
            args.dxy.y,
            args.ksize as i32,
            args.scale,
            args.delta,
            args.border_type as i32,
        )?;
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3, s};

    /// Helper: create a 2D grayscale image with a vertical edge in the middle.
    /// Left half = 0, right half = 255.
    fn vertical_edge_image(rows: usize, cols: usize) -> Array2<u8> {
        Array2::from_shape_fn(
            (rows, cols),
            |(_i, j)| {
                if j < cols / 2 { 0u8 } else { 255u8 }
            },
        )
    }

    /// Helper: create a 2D grayscale image with a horizontal edge in the middle.
    /// Top half = 0, bottom half = 255.
    fn horizontal_edge_image(rows: usize, cols: usize) -> Array2<u8> {
        Array2::from_shape_fn(
            (rows, cols),
            |(i, _j)| {
                if i < rows / 2 { 0u8 } else { 255u8 }
            },
        )
    }

    // ── Basic API / shape preservation ──────────────────────────────

    #[test]
    fn sobel_2d_preserves_shape() {
        let img = Array2::<u8>::from_shape_fn((10, 10), |(i, j)| (i * j) as u8);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    #[test]
    fn sobel_3d_preserves_shape() {
        let img = Array3::<u8>::ones((20, 30, 3));
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[20, 30, 3]);
    }

    #[test]
    fn sobel_non_square_image() {
        let img = Array2::<u8>::zeros((15, 40));
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((0, 1)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[15, 40]);
    }

    // ── Edge detection correctness ──────────────────────────────────

    #[test]
    fn sobel_detects_vertical_edge_with_dx() {
        let img = vertical_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();

        // The Sobel dx derivative should have high magnitude near the vertical edge (col ~10)
        let edge_col = result.column(10);
        let interior_col = result.column(0);

        let edge_energy: i64 = edge_col.iter().map(|&v| (v as i64).abs()).sum();
        let interior_energy: i64 = interior_col.iter().map(|&v| (v as i64).abs()).sum();
        assert!(
            edge_energy > interior_energy,
            "edge energy ({edge_energy}) should exceed interior energy ({interior_energy})"
        );
    }

    #[test]
    fn sobel_detects_horizontal_edge_with_dy() {
        let img = horizontal_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((0, 1)).build().unwrap()).unwrap();

        // The Sobel dy derivative should have high magnitude near the horizontal edge (row ~10)
        let edge_row = result.row(10);
        let interior_row = result.row(0);

        let edge_energy: i64 = edge_row.iter().map(|&v| (v as i64).abs()).sum();
        let interior_energy: i64 = interior_row.iter().map(|&v| (v as i64).abs()).sum();
        assert!(
            edge_energy > interior_energy,
            "edge energy ({edge_energy}) should exceed interior energy ({interior_energy})"
        );
    }

    #[test]
    fn sobel_uniform_image_produces_near_zero() {
        let img = Array2::<u8>::from_elem((20, 20), 128);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();

        // A completely uniform image should produce all-zero derivatives
        assert!(result.iter().all(|&v| v == 0));
    }

    // ── Kernel sizes ────────────────────────────────────────────────

    #[test]
    fn sobel_different_kernel_sizes() {
        let img = vertical_edge_image(30, 30);

        for ksize in [Ksize::K1, Ksize::K3, Ksize::K5, Ksize::K7] {
            let args = SobelArgs::dxy((1, 0)).ksize(ksize).build().unwrap();
            let result: ndarray::Array<i16, _> = img.sobel(args).unwrap();
            assert_eq!(result.shape(), &[30, 30], "failed for ksize {ksize:?}");
        }
    }

    // ── Scale and delta ─────────────────────────────────────────────

    #[test]
    fn sobel_scale_amplifies_output() {
        let img = vertical_edge_image(20, 20);

        let result_1x: ndarray::Array<f32, _> = img
            .sobel(SobelArgs::dxy((1, 0)).scale(1.0).build().unwrap())
            .unwrap();
        let result_2x: ndarray::Array<f32, _> = img
            .sobel(SobelArgs::dxy((1, 0)).scale(2.0).build().unwrap())
            .unwrap();

        let energy_1x: f64 = result_1x.iter().map(|&v| (v as f64).abs()).sum();
        let energy_2x: f64 = result_2x.iter().map(|&v| (v as f64).abs()).sum();

        let ratio = energy_2x / energy_1x;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "expected 2x scaling, got ratio {ratio}"
        );
    }

    #[test]
    fn sobel_delta_shifts_output() {
        let img = Array2::<u8>::from_elem((20, 20), 128);

        let result_no_delta: ndarray::Array<f32, _> = img
            .sobel(SobelArgs::dxy((1, 0)).delta(0.0).build().unwrap())
            .unwrap();
        let result_with_delta: ndarray::Array<f32, _> = img
            .sobel(SobelArgs::dxy((1, 0)).delta(42.0).build().unwrap())
            .unwrap();

        // On a uniform image the derivative is 0, so with delta=42 every pixel should be ~42
        assert!(result_no_delta.iter().all(|&v| v.abs() < 1e-6));
        assert!(result_with_delta.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    // ── Border types ────────────────────────────────────────────────

    #[test]
    fn sobel_different_border_types() {
        let img = vertical_edge_image(20, 20);

        let border_types = [
            opencv::core::BorderTypes::BORDER_CONSTANT,
            opencv::core::BorderTypes::BORDER_REPLICATE,
            opencv::core::BorderTypes::BORDER_REFLECT,
            opencv::core::BorderTypes::BORDER_REFLECT_101,
        ];

        for bt in border_types {
            let args = SobelArgs::dxy((1, 0)).border_type(bt).build().unwrap();
            let result: ndarray::Array<i16, _> = img.sobel(args).unwrap();
            assert_eq!(result.shape(), &[20, 20], "failed for border type {bt:?}");
        }
    }

    // ── Type combinations ───────────────────────────────────────────

    #[test]
    fn sobel_u8_to_i16() {
        let img = Array2::<u8>::ones((10, 10));
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    #[test]
    fn sobel_u8_to_f32() {
        let img = Array2::<u8>::ones((10, 10));
        let result: ndarray::Array<f32, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    #[test]
    fn sobel_u8_to_f64() {
        let img = Array2::<u8>::ones((10, 10));
        let result: ndarray::Array<f64, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    #[test]
    fn sobel_f32_input_to_f32_output() {
        let img = Array2::<f32>::from_shape_fn((10, 10), |(i, j)| (i * j) as f32);
        let result: ndarray::Array<f32, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        assert_eq!(result.shape(), &[10, 10]);
    }

    // ── Builder ergonomics ──────────────────────────────────────────

    #[test]
    fn sobel_builder_defaults() {
        // dxy is the only required field; everything else should use defaults
        let args = SobelArgs::dxy((1, 0)).build().unwrap();
        let img = Array2::<u8>::zeros((10, 10));
        let _result: ndarray::Array<i16, _> = img.sobel(args).unwrap();
    }

    #[test]
    fn sobel_builder_all_fields() {
        let args = SobelArgs::dxy((1, 0))
            .ksize(Ksize::K5)
            .scale(2.0)
            .delta(10.0)
            .border_type(opencv::core::BorderTypes::BORDER_REPLICATE)
            .build()
            .unwrap();

        let img = Array2::<u8>::zeros((20, 20));
        let result: ndarray::Array<f32, _> = img.sobel(args).unwrap();
        assert_eq!(result.shape(), &[20, 20]);
    }

    // ── Mixed derivative orders ─────────────────────────────────────

    #[test]
    fn sobel_dx_only() {
        let img = vertical_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();
        // dx on a vertical edge should produce non-zero output
        assert!(result.iter().any(|&v| v != 0));
    }

    #[test]
    fn sobel_dy_only() {
        let img = horizontal_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((0, 1)).build().unwrap()).unwrap();
        // dy on a horizontal edge should produce non-zero output
        assert!(result.iter().any(|&v| v != 0));
    }

    #[test]
    fn sobel_dx_on_horizontal_edge_is_weak() {
        let img = horizontal_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((1, 0)).build().unwrap()).unwrap();

        // dx derivative of a purely horizontal edge should be zero in the interior
        let interior = result.slice(s![5..15, 2..18]);
        assert!(
            interior.iter().all(|&v| v == 0),
            "dx derivative should be zero in the interior of a horizontal edge image"
        );
    }

    #[test]
    fn sobel_dy_on_vertical_edge_is_weak() {
        let img = vertical_edge_image(20, 20);
        let result: ndarray::Array<i16, _> =
            img.sobel(SobelArgs::dxy((0, 1)).build().unwrap()).unwrap();

        // dy derivative of a purely vertical edge should be zero in the interior
        let interior = result.slice(s![2..18, 5..15]);
        assert!(
            interior.iter().all(|&v| v == 0),
            "dy derivative should be zero in the interior of a vertical edge image"
        );
    }
}
