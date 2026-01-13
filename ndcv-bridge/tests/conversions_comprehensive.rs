//! Comprehensive unit tests for ndcv-bridge conversion functionality
//!
//! This test suite covers:
//! - All supported data types (u8, i8, u16, i16, i32, f32, f64)
//! - Various dimensionalities (1D through 3D)
//! - Channel configurations
//! - Error conditions and edge cases
//! - Memory safety and data consistency

use ndarray::{Array1, Array2, Array3, ArrayView, Ix1, Ix2, Ix3};
use ndcv_bridge::conversions::{MatAsNd, NdAsImage, NdAsImageMut, NdAsMat};
use opencv::core::{CV_8S, CV_8U, CV_16S, CV_16U, CV_32F, CV_32S, CV_64F, Mat};
use opencv::prelude::*;

#[test]
fn test_all_numeric_types_1d_conversion() {
    // Test 1D array conversion for all supported numeric types
    let test_cases = vec![
        (CV_8U, "u8"),
        (CV_8S, "i8"),
        (CV_16U, "u16"),
        (CV_16S, "i16"),
        (CV_32S, "i32"),
        (CV_32F, "f32"),
        (CV_64F, "f64"),
    ];

    for (cv_type, type_name) in test_cases {
        println!("Testing 1D conversion for type: {}", type_name);

        let mat = Mat::new_nd_with_default(&[20], cv_type, 100.into()).unwrap();

        match cv_type {
            CV_8U => {
                let array: ArrayView<u8, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100));

                // Test round-trip conversion
                let new_array = Array1::<u8>::from_elem(20, 200);
                let new_mat = new_array.as_single_channel_mat().unwrap();
                let back_array: ArrayView<u8, Ix1> = new_mat.as_ndarray().unwrap();
                assert!(back_array.iter().all(|&x| x == 200));
            }
            CV_8S => {
                let array: ArrayView<i8, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100));
            }
            CV_16U => {
                let array: ArrayView<u16, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100));
            }
            CV_16S => {
                let array: ArrayView<i16, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100));
            }
            CV_32S => {
                let array: ArrayView<i32, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100));
            }
            CV_32F => {
                let array: ArrayView<f32, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100.0));
            }
            CV_64F => {
                let array: ArrayView<f64, Ix1> = mat.as_ndarray().unwrap();
                assert_eq!(array.len(), 20);
                assert!(array.iter().all(|&x| x == 100.0));
            }
            _ => panic!("Unsupported type"),
        }
    }
}

#[test]
fn test_2d_arrays_all_types() {
    // Test 2D arrays for all supported types
    let shapes = vec![(5, 4), (10, 8), (15, 12)];

    for (height, width) in shapes {
        for (cv_type, type_name) in [(CV_32F, "f32"), (CV_8U, "u8"), (CV_16S, "i16")] {
            println!(
                "Testing 2D ({}, {}) conversion for type: {}",
                height, width, type_name
            );

            let mat =
                Mat::new_rows_cols_with_default(height as i32, width as i32, cv_type, 42.into())
                    .unwrap();

            match cv_type {
                CV_32F => {
                    let array: ArrayView<f32, Ix2> = mat.as_ndarray().unwrap();
                    assert_eq!(array.shape(), [height, width]);
                    assert!(array.iter().all(|&x| x == 42.0));
                }
                CV_8U => {
                    let array: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();
                    assert_eq!(array.shape(), [height, width]);
                    assert!(array.iter().all(|&x| x == 42));
                }
                CV_16S => {
                    let array: ArrayView<i16, Ix2> = mat.as_ndarray().unwrap();
                    assert_eq!(array.shape(), [height, width]);
                    assert!(array.iter().all(|&x| x == 42));
                }
                _ => panic!("Unsupported type in test"),
            }
        }
    }
}

#[test]
fn test_3d_arrays_multichannel() {
    // Test 3D arrays representing multi-channel images
    let configurations = vec![(64, 64, 3, "rgb"), (32, 48, 4, "rgba")];

    for (height, width, channels, desc) in configurations {
        println!("Testing 3D {} ({}x{}x{})", desc, height, width, channels);

        let mat = Mat::new_nd_with_default(
            &[height, width],
            opencv::core::CV_MAKETYPE(CV_8U, channels),
            opencv::core::VecN([128f64; 4]),
        )
        .unwrap();

        let array: ArrayView<u8, Ix3> = mat.as_ndarray().unwrap();
        assert_eq!(
            array.shape(),
            [height as usize, width as usize, channels as usize]
        );
        assert!(array.iter().all(|&x| x == 128));
    }

    // Test single channel case (should be 2D)
    let mat_single =
        Mat::new_nd_with_default(&[16, 16], opencv::core::CV_MAKETYPE(CV_8U, 1), 128.into())
            .unwrap();

    let array_single: ArrayView<u8, Ix2> = mat_single.as_ndarray().unwrap();
    assert_eq!(array_single.shape(), [16, 16]);
    assert!(array_single.iter().all(|&x| x == 128));
}

#[test]
fn test_channel_configurations() {
    // Test various channel configurations
    let channel_configs = vec![
        (1, "single channel"),
        (2, "dual channel"),
        (3, "RGB"),
        (4, "RGBA"),
    ];

    for (channels, desc) in channel_configs {
        println!("Testing {}", desc);

        let mat = Mat::new_rows_cols_with_default(
            10,
            10,
            opencv::core::CV_MAKETYPE(CV_8U, channels),
            opencv::core::VecN([128f64; 4]),
        )
        .unwrap();

        match channels {
            1 => {
                let array: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();
                assert_eq!(array.shape(), [10, 10]);
                assert!(array.iter().all(|&x| x == 128));
            }
            2 | 3 | 4 => {
                let array: ArrayView<u8, Ix3> = mat.as_ndarray().unwrap();
                assert_eq!(array.shape(), [10, 10, channels as usize]);
                assert!(array.iter().all(|&x| x == 128));
            }
            _ => panic!("Unsupported channel count"),
        }
    }
}

#[test]
fn test_non_contiguous_arrays() {
    // Test conversion with non-contiguous arrays (sliced)
    let original =
        Array3::<f32>::from_shape_fn((20, 30, 4), |(i, j, k)| (i * 100 + j * 10 + k) as f32);

    // Test middle slice
    let middle_slice = original.slice(ndarray::s![5..15, 10..20, ..]);

    // Convert to mat and back
    let mat_ref = middle_slice.as_multi_channel_mat().unwrap();
    let reconstructed: ArrayView<f32, Ix3> = mat_ref.as_ndarray().unwrap();

    // Verify shapes match
    assert_eq!(middle_slice.shape(), reconstructed.shape());

    // Verify all values match
    for (original_val, reconstructed_val) in middle_slice.iter().zip(reconstructed.iter()) {
        assert_eq!(original_val, reconstructed_val);
    }
}

#[test]
fn test_large_arrays() {
    // Test performance and correctness with large arrays
    let sizes = vec![(512, 512), (1024, 768)];

    for (height, width) in sizes {
        println!("Testing large array: {}x{}", height, width);

        // Create large array with pattern
        let array =
            Array2::<u16>::from_shape_fn((height, width), |(i, j)| ((i + j) % 65536) as u16);

        // Convert to mat
        let mat = array.as_single_channel_mat().unwrap();
        assert_eq!(mat.rows(), height as i32);
        assert_eq!(mat.cols(), width as i32);

        // Convert back and verify
        let reconstructed: ArrayView<u16, Ix2> = mat.as_ndarray().unwrap();
        assert_eq!(array.shape(), reconstructed.shape());

        // Verify data integrity
        for (orig, recon) in array.iter().zip(reconstructed.iter()) {
            assert_eq!(orig, recon);
        }
    }
}

#[test]
fn test_edge_cases() {
    // Test edge cases and boundary conditions

    // Single element arrays
    let single_1d = Array1::<f32>::from_elem(1, 42.0);
    let mat = single_1d.as_single_channel_mat().unwrap();
    let back: ArrayView<f32, Ix1> = mat.as_ndarray().unwrap();
    assert_eq!(single_1d.shape(), back.shape());

    // Very small arrays
    let tiny_2d = Array2::<i32>::from_elem((1, 1), -123);
    let mat = tiny_2d.as_single_channel_mat().unwrap();
    let back: ArrayView<i32, Ix2> = mat.as_ndarray().unwrap();
    assert_eq!(tiny_2d.shape(), back.shape());

    // Arrays with one dimension being 1
    let thin_array = Array3::<u8>::from_elem((100, 1, 3), 200);
    let mat = thin_array.as_multi_channel_mat().unwrap();
    let back: ArrayView<u8, Ix3> = mat.as_ndarray().unwrap();
    assert_eq!(thin_array.shape(), back.shape());
}

#[test]
fn test_image_specific_conversions() {
    // Test image-specific conversion traits

    // 2D grayscale image
    let gray_img = Array2::<u8>::from_shape_fn((100, 150), |(i, j)| ((i + j) % 256) as u8);

    let mat_ref = gray_img.as_image_mat().unwrap();
    assert_eq!(mat_ref.rows(), 100);
    assert_eq!(mat_ref.cols(), 150);
    assert_eq!(mat_ref.channels(), 1);

    // 3D color image
    let color_img =
        Array3::<u8>::from_shape_fn((100, 150, 3), |(i, j, k)| ((i + j + k) % 256) as u8);

    let mat_ref = color_img.as_image_mat().unwrap();
    assert_eq!(mat_ref.rows(), 100);
    assert_eq!(mat_ref.cols(), 150);
    assert_eq!(mat_ref.channels(), 3);

    // Test mutable versions
    let mut gray_img_mut = Array2::<u8>::zeros((50, 60));
    {
        let _mat_ref_mut = gray_img_mut.as_image_mat_mut().unwrap();
        // Could modify through mat reference here
    }

    let mut color_img_mut = Array3::<u8>::zeros((50, 60, 3));
    {
        let _mat_ref_mut = color_img_mut.as_image_mat_mut().unwrap();
        // Could modify through mat reference here
    }
}

#[test]
fn test_regular_vs_consolidated_conversion() {
    // Test the difference between regular and consolidated conversion methods

    let array_3d = Array3::<f32>::from_shape_fn((5, 6, 4), |(i, j, k)| (i * 24 + j * 4 + k) as f32);

    // Test regular conversion (treats last dimension as separate)
    let mat_regular = array_3d.as_single_channel_mat().unwrap();
    println!(
        "Regular conversion - dims: {}, channels: {}",
        mat_regular.dims(),
        mat_regular.channels()
    );

    // Test consolidated conversion (treats last dimension as channels)
    let mat_consolidated = array_3d.as_multi_channel_mat().unwrap();
    println!(
        "Consolidated conversion - dims: {}, channels: {}",
        mat_consolidated.dims(),
        mat_consolidated.channels()
    );

    // Verify they represent the same data differently
    assert_eq!(
        mat_regular.total() * mat_regular.channels() as usize,
        mat_consolidated.total() * mat_consolidated.channels() as usize
    );

    // Convert back and verify data integrity
    let back_regular: ArrayView<f32, Ix3> = mat_regular.as_ndarray().unwrap();
    let back_consolidated: ArrayView<f32, Ix3> = mat_consolidated.as_ndarray().unwrap();

    assert_eq!(array_3d.shape(), back_regular.shape());
    assert_eq!(array_3d.shape(), back_consolidated.shape());
}

#[test]
fn test_round_trip_data_integrity() {
    // Comprehensive round-trip testing with various data patterns

    // Test zeros pattern
    let zeros = Array2::<u8>::zeros((50, 75));
    for round in 0..3 {
        let mat = zeros.as_single_channel_mat().unwrap();
        let reconstructed: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();

        assert_eq!(
            zeros.shape(),
            reconstructed.shape(),
            "Round-trip {} failed for zeros pattern - shape mismatch",
            round + 1
        );

        for (orig, recon) in zeros.iter().zip(reconstructed.iter()) {
            assert_eq!(
                orig,
                recon,
                "Round-trip {} failed for zeros pattern - value mismatch",
                round + 1
            );
        }
    }

    // Test ones pattern
    let ones = Array2::<u8>::from_elem((50, 75), 255);
    let mat = ones.as_single_channel_mat().unwrap();
    let reconstructed: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();

    assert_eq!(ones.shape(), reconstructed.shape());
    for (orig, recon) in ones.iter().zip(reconstructed.iter()) {
        assert_eq!(orig, recon);
    }

    // Test gradient pattern
    let gradient = Array2::<u8>::from_shape_fn((50, 75), |(i, j)| ((i + j) % 256) as u8);
    let mat = gradient.as_single_channel_mat().unwrap();
    let reconstructed: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();

    assert_eq!(gradient.shape(), reconstructed.shape());
    for (orig, recon) in gradient.iter().zip(reconstructed.iter()) {
        assert_eq!(orig, recon);
    }
}

#[test]
fn test_basic_performance() {
    // Basic performance test to ensure no major regressions
    use std::time::Instant;

    let array = Array2::<u8>::from_shape_fn((256, 256), |(i, j)| ((i + j) % 256) as u8);

    // Time conversion to Mat
    let start = Instant::now();
    let mat = array.as_single_channel_mat().unwrap();
    let to_mat_time = start.elapsed();

    // Time conversion back to Array
    let start = Instant::now();
    let _back: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();
    let from_mat_time = start.elapsed();

    println!(
        "Performance: to_mat: {:?}, from_mat: {:?}",
        to_mat_time, from_mat_time
    );

    // Basic sanity check that times aren't unreasonably long (1 second is very generous)
    assert!(
        to_mat_time.as_millis() < 1000,
        "Conversion to Mat took too long: {:?}",
        to_mat_time
    );
    assert!(
        from_mat_time.as_millis() < 1000,
        "Conversion from Mat took too long: {:?}",
        from_mat_time
    );
}

#[test]
fn test_memory_consistency() {
    // Test that data remains consistent through conversions
    let mut original = Array2::<f32>::zeros((10, 10));
    original[[5, 5]] = 100.0;

    // Convert to Mat and back
    let mat = original.as_single_channel_mat().unwrap();
    let reconstructed: ArrayView<f32, Ix2> = mat.as_ndarray().unwrap();

    // Verify the data matches
    assert_eq!(original.shape(), reconstructed.shape());
    for ((i, j), &orig_val) in original.indexed_iter() {
        let recon_val = reconstructed[[i, j]];
        assert_eq!(orig_val, recon_val, "Mismatch at position ({}, {})", i, j);
    }
}
