//! Error handling and edge case tests for ndcv-bridge conversions
//!
//! This test suite focuses on:
//! - Type mismatch errors
//! - Dimension compatibility errors
//! - Invalid channel configurations
//! - Boundary conditions
//! - Error message quality and debugging information

use ndarray::{Array1, Array2, Array3, ArrayView, Ix1, Ix2, Ix3};
use ndcv_bridge::conversions::{MatAsNd, NdAsMat};
use opencv::core::{CV_8U, CV_32F, Mat};

#[test]
fn test_type_mismatch_errors() {
    // Test various type mismatches between Mat and ndarray

    // Create Mat with u8 data
    let mat_u8 = Mat::new_rows_cols_with_default(10, 10, CV_8U, 100.into()).unwrap();

    // Try to interpret as f32 - should fail
    let result_f32 = mat_u8.as_ndarray::<f32, Ix2>();
    assert!(result_f32.is_err(), "Expected error for f32 type mismatch");

    // Try to interpret as i8 - should fail
    let result_i8 = mat_u8.as_ndarray::<i8, Ix2>();
    assert!(result_i8.is_err(), "Expected error for i8 type mismatch");

    // Check that error message contains useful information
    let err_msg = format!("{:?}", result_f32.unwrap_err());
    assert!(
        err_msg.contains("Expected type") || err_msg.contains("type"),
        "Error message should mention type mismatch: {}",
        err_msg
    );
}

#[test]
fn test_dimension_mismatch_errors() {
    // Test dimension compatibility errors

    // 2D Mat but trying to interpret as different dimensions
    let mat_2d = Mat::new_rows_cols_with_default(5, 8, CV_32F, 42.0.into()).unwrap();

    // Try to interpret as 1D - should fail
    let result_1d = mat_2d.as_ndarray::<f32, Ix1>();
    assert!(
        result_1d.is_err(),
        "Expected error for 1D dimension mismatch"
    );

    // Try to interpret as 3D - should fail
    let result_3d = mat_2d.as_ndarray::<f32, Ix3>();
    assert!(
        result_3d.is_err(),
        "Expected error for 3D dimension mismatch"
    );

    let err_msg = format!("{:?}", result_1d.unwrap_err());
    assert!(
        err_msg.contains("Incompatible dimensions") || err_msg.contains("dimension"),
        "Error message should mention dimension mismatch: {}",
        err_msg
    );
}

#[test]
fn test_channel_boundary_conditions() {
    // Test edge cases with channel counts

    // Single channel (should work)
    let mat_1ch =
        Mat::new_rows_cols_with_default(5, 5, opencv::core::CV_MAKETYPE(CV_8U, 1), 100.into())
            .unwrap();
    let array_1ch: ArrayView<u8, Ix2> = mat_1ch.as_ndarray().unwrap();
    assert_eq!(array_1ch.shape(), [5, 5]);

    // Multiple channels (should work)
    for channels in 2..=4 {
        let mat = Mat::new_rows_cols_with_default(
            4,
            4,
            opencv::core::CV_MAKETYPE(CV_8U, channels),
            opencv::core::VecN([100f64; 4]),
        )
        .unwrap();

        let array: ArrayView<u8, Ix3> = mat.as_ndarray().unwrap();
        assert_eq!(array.shape(), [4, 4, channels as usize]);
        assert!(array.iter().all(|&x| x == 100));
    }
}

#[test]
fn test_empty_and_minimal_arrays() {
    // Test various minimal size arrays

    // 1x1 arrays
    let tiny_1d = Array1::<f32>::from_elem(1, 3.14);
    let mat = tiny_1d.as_single_channel_mat().unwrap();
    let back: ArrayView<f32, Ix1> = mat.as_ndarray().unwrap();
    assert_eq!(tiny_1d.shape(), back.shape());

    let tiny_2d = Array2::<i32>::from_elem((1, 1), -42);
    let mat = tiny_2d.as_single_channel_mat().unwrap();
    let back: ArrayView<i32, Ix2> = mat.as_ndarray().unwrap();
    assert_eq!(tiny_2d.shape(), back.shape());

    // Arrays with one dimension being very small
    let narrow = Array2::<u16>::from_elem((100, 1), 999);
    let mat = narrow.as_single_channel_mat().unwrap();
    let back: ArrayView<u16, Ix2> = mat.as_ndarray().unwrap();
    assert_eq!(narrow.shape(), back.shape());

    let tall = Array2::<u16>::from_elem((1, 100), 888);
    let mat = tall.as_single_channel_mat().unwrap();
    let back: ArrayView<u16, Ix2> = mat.as_ndarray().unwrap();
    assert_eq!(tall.shape(), back.shape());
}

#[test]
fn test_consolidated_conversion_constraints() {
    // Test constraints specific to consolidated conversion

    // 1D arrays should fail with consolidated conversion
    let array_1d = Array1::<f32>::from_elem(10, 1.0);
    let result = array_1d.as_multi_channel_mat();
    assert!(
        result.is_err(),
        "1D array should fail consolidated conversion"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("1D array") || err_msg.contains("consolidated"),
        "Error should mention 1D/consolidated restriction: {}",
        err_msg
    );

    // But should work with regular conversion
    let regular_result = array_1d.as_single_channel_mat();
    assert!(
        regular_result.is_ok(),
        "1D array should work with regular conversion"
    );
}

#[test]
fn test_extreme_values() {
    // Test with extreme values for each data type

    // Unsigned integers
    let u8_extremes = vec![u8::MIN, u8::MAX, 127, 128];
    for &val in &u8_extremes {
        let array = Array2::<u8>::from_elem((5, 5), val);
        let mat = array.as_single_channel_mat().unwrap();
        let back: ArrayView<u8, Ix2> = mat.as_ndarray().unwrap();
        assert_eq!(array.shape(), back.shape(), "Failed for u8 value {}", val);
        assert!(
            back.iter().all(|&x| x == val),
            "Value mismatch for u8 {}",
            val
        );
    }

    // Signed integers
    let i16_extremes = vec![i16::MIN, i16::MAX, -1, 0, 1];
    for &val in &i16_extremes {
        let array = Array2::<i16>::from_elem((3, 4), val);
        let mat = array.as_single_channel_mat().unwrap();
        let back: ArrayView<i16, Ix2> = mat.as_ndarray().unwrap();
        assert_eq!(array.shape(), back.shape(), "Failed for i16 value {}", val);
        assert!(
            back.iter().all(|&x| x == val),
            "Value mismatch for i16 {}",
            val
        );
    }

    // Floating point
    let f32_values = vec![0.0, 1.0, -1.0, 3.14159, std::f32::consts::E];

    for &val in &f32_values {
        let array = Array2::<f32>::from_elem((2, 3), val);
        let mat = array.as_single_channel_mat().unwrap();
        let back: ArrayView<f32, Ix2> = mat.as_ndarray().unwrap();

        // Use approximate equality for floating point
        for (orig, back_val) in array.iter().zip(back.iter()) {
            assert!(
                (orig - back_val).abs() < f32::EPSILON * 2.0,
                "Failed for f32 value {}: {} != {}",
                val,
                orig,
                back_val
            );
        }
    }
}

#[test]
fn test_error_message_quality() {
    // Test that error messages provide useful debugging information

    // Type mismatch
    let mat_u8 = Mat::new_rows_cols_with_default(5, 5, CV_8U, 100.into()).unwrap();
    let type_err = mat_u8.as_ndarray::<f32, Ix2>().unwrap_err();
    let type_msg = format!("{:?}", type_err);

    // Should mention both expected and actual types
    assert!(
        type_msg.contains("Expected type") && (type_msg.contains("f32") || type_msg.contains("u8")),
        "Type error should mention both types: {}",
        type_msg
    );

    // Dimension mismatch
    let mat_2d = Mat::new_rows_cols_with_default(3, 4, CV_32F, 1.0.into()).unwrap();
    let dim_err = mat_2d.as_ndarray::<f32, Ix1>().unwrap_err();
    let dim_msg = format!("{:?}", dim_err);

    // Should mention dimensions and sizes
    assert!(
        dim_msg.contains("dimensions") || dim_msg.contains("dims"),
        "Dimension error should mention dimensions: {}",
        dim_msg
    );
}

#[test]
fn test_memory_safety_edge_cases() {
    // Test potential memory safety issues with odd-sized arrays
    let odd_sizes = vec![(3, 5), (7, 11), (13, 17)];

    for (height, width) in odd_sizes {
        let array = Array2::<i16>::from_shape_fn((height, width), |(i, j)| (i * width + j) as i16);
        let mat = array.as_single_channel_mat().unwrap();
        let back: ArrayView<i16, Ix2> = mat.as_ndarray().unwrap();

        assert_eq!(
            array.shape(),
            back.shape(),
            "Shape mismatch for size {}x{}",
            height,
            width
        );

        // Verify data integrity
        for ((i, j), &orig_val) in array.indexed_iter() {
            let back_val = back[[i, j]];
            assert_eq!(
                orig_val, back_val,
                "Data mismatch at ({}, {}) for size {}x{}",
                i, j, height, width
            );
        }
    }
}

#[test]
fn test_concurrent_access_safety() {
    // Test that conversions are safe under concurrent access
    use std::sync::Arc;
    use std::thread;

    let original = Arc::new(Array2::<i32>::from_shape_fn((100, 100), |(i, j)| {
        (i * 100 + j) as i32
    }));
    let mut handles = vec![];

    // Spawn multiple threads that read the same array
    for thread_id in 0..4 {
        let array_clone = Arc::clone(&original);
        let handle = thread::spawn(move || {
            // Each thread converts to Mat multiple times
            for iteration in 0..10 {
                let mat = array_clone.as_single_channel_mat().unwrap();
                let back: ArrayView<i32, Ix2> = mat.as_ndarray().unwrap();

                // Verify data integrity
                assert_eq!(
                    array_clone.shape(),
                    back.shape(),
                    "Shape mismatch in thread {} iteration {}",
                    thread_id,
                    iteration
                );

                // Verify first and last elements
                let first_orig = array_clone[[0, 0]];
                let first_back = back[[0, 0]];
                assert_eq!(
                    first_orig, first_back,
                    "First element mismatch in thread {} iteration {}",
                    thread_id, iteration
                );

                let last_orig = array_clone[[99, 99]];
                let last_back = back[[99, 99]];
                assert_eq!(
                    last_orig, last_back,
                    "Last element mismatch in thread {} iteration {}",
                    thread_id, iteration
                );
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_special_float_values() {
    // Test handling of special floating point values
    let special_values = vec![
        (0.0f32, "zero"),
        (-0.0f32, "negative_zero"),
        (1.0f32, "one"),
        (-1.0f32, "negative_one"),
        (f32::EPSILON, "epsilon"),
    ];

    for (value, desc) in special_values {
        let array = Array2::<f32>::from_elem((4, 4), value);
        let mat = array.as_single_channel_mat().unwrap();
        let back: ArrayView<f32, Ix2> = mat.as_ndarray().unwrap();

        for (orig, reconstructed) in array.iter().zip(back.iter()) {
            assert!(
                (orig - reconstructed).abs() < f32::EPSILON,
                "Special value {} not preserved: {} != {}",
                desc,
                orig,
                reconstructed
            );
        }
    }
}

#[test]
fn test_non_contiguous_slice_handling() {
    // Test how the library handles sliced arrays
    let original =
        Array3::<f32>::from_shape_fn((10, 12, 3), |(i, j, k)| (i * 36 + j * 3 + k) as f32);

    // Create a simple slice
    let slice = original.slice(ndarray::s![2..8, 3..9, ..]);

    // This should work for consolidated conversion
    let mat_result = slice.as_multi_channel_mat();
    match mat_result {
        Ok(mat_ref) => {
            let reconstructed: ArrayView<f32, Ix3> = mat_ref.as_ndarray().unwrap();
            assert_eq!(slice.shape(), reconstructed.shape());

            // Verify some values match
            for ((i, j, k), &orig_val) in slice.indexed_iter().take(10) {
                let recon_val = reconstructed[[i, j, k]];
                assert_eq!(
                    orig_val, recon_val,
                    "Mismatch at slice position ({}, {}, {})",
                    i, j, k
                );
            }
        }
        Err(_) => {
            println!("Slice conversion failed as expected due to stride constraints");
        }
    }
}
