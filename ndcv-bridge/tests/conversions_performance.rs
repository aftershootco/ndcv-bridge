//! Performance and benchmark tests for ndcv-bridge conversions
//!
//! This test suite focuses on:
//! - Performance regression detection
//! - Throughput benchmarks for different array sizes
//! - Conversion overhead measurement
//! - Basic scalability testing

use ndarray::{Array2, Array3, ArrayView, Ix2};
use ndcv_bridge::conversions::{MatAsNd, NdAsMat};
use opencv::core::{CV_8U, CV_16U, CV_32F, CV_64F, Mat};

use std::time::{Duration, Instant};

/// Performance measurement result
#[derive(Debug, Clone)]
struct PerfResult {
    operation: String,
    mean_duration: Duration,
    min_duration: Duration,
    max_duration: Duration,
    throughput_mb_per_sec: f64,
}

fn measure_performance<F, T>(
    operation_name: &str,
    data_size_bytes: usize,
    iterations: usize,
    mut operation: F,
) -> PerfResult
where
    F: FnMut() -> T,
{
    let mut durations = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..3 {
        let _ = operation();
    }

    for _ in 0..iterations {
        let start = Instant::now();
        let _ = operation();
        let duration = start.elapsed();
        durations.push(duration);
    }

    let total_duration: Duration = durations.iter().sum();
    let mean_duration = total_duration / iterations as u32;
    let min_duration = *durations.iter().min().unwrap();
    let max_duration = *durations.iter().max().unwrap();

    let throughput_mb_per_sec = if mean_duration.as_nanos() > 0 {
        (data_size_bytes as f64) / (1024.0 * 1024.0) / mean_duration.as_secs_f64()
    } else {
        f64::INFINITY
    };

    PerfResult {
        operation: operation_name.to_string(),
        mean_duration,
        min_duration,
        max_duration,
        throughput_mb_per_sec,
    }
}

#[test]
fn test_2d_array_conversion_performance() {
    let sizes = vec![(64, 64, "small"), (256, 256, "medium"), (512, 512, "large")];

    for (height, width, size_desc) in sizes {
        println!(
            "=== Performance test for {} 2D arrays ({}x{}) ===",
            size_desc, height, width
        );

        let data_size = height * width * std::mem::size_of::<u8>();
        let iterations = if data_size > 256 * 256 { 10 } else { 50 };

        let array = Array2::<u8>::from_shape_fn((height, width), |(i, j)| ((i + j) % 256) as u8);

        // Test ndarray -> Mat conversion
        let to_mat_result = measure_performance("ndarray_to_mat", data_size, iterations, || {
            array.as_single_channel_mat().unwrap()
        });

        // Test Mat -> ndarray conversion
        let mat = array.as_single_channel_mat().unwrap();
        let from_mat_result = measure_performance("mat_to_ndarray", data_size, iterations, || {
            mat.as_ndarray::<u8, Ix2>().unwrap()
        });

        // Print results
        println!(
            "  To Mat   - Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
            to_mat_result.mean_duration, to_mat_result.throughput_mb_per_sec
        );
        println!(
            "  From Mat - Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
            from_mat_result.mean_duration, from_mat_result.throughput_mb_per_sec
        );

        // Performance sanity checks (very generous limits)
        assert!(
            to_mat_result.mean_duration.as_millis() < 2000,
            "Conversion to Mat took too long: {:?}",
            to_mat_result.mean_duration
        );
        assert!(
            from_mat_result.mean_duration.as_millis() < 2000,
            "Conversion from Mat took too long: {:?}",
            from_mat_result.mean_duration
        );

        // Expect reasonable throughput (at least 50 MB/s for medium+ sizes)
        if data_size > 64 * 64 {
            assert!(
                to_mat_result.throughput_mb_per_sec > 50.0,
                "Throughput too low for to_mat: {:.1} MB/s",
                to_mat_result.throughput_mb_per_sec
            );
            assert!(
                from_mat_result.throughput_mb_per_sec > 50.0,
                "Throughput too low for from_mat: {:.1} MB/s",
                from_mat_result.throughput_mb_per_sec
            );
        }
    }
}

#[test]
fn test_3d_array_conversion_performance() {
    let configurations = vec![
        (64, 64, 3, "small_rgb"),
        (256, 256, 3, "medium_rgb"),
        (128, 128, 4, "medium_rgba"),
    ];

    for (height, width, channels, desc) in configurations {
        println!(
            "=== Performance test for {} 3D arrays ({}x{}x{}) ===",
            desc, height, width, channels
        );

        let data_size = height * width * channels * std::mem::size_of::<u8>();
        let iterations = if data_size > 128 * 128 * 4 { 10 } else { 20 };

        let array = Array3::<u8>::from_shape_fn((height, width, channels), |(i, j, k)| {
            ((i + j + k) % 256) as u8
        });

        // Test consolidated conversion (channels as OpenCV channels)
        let to_mat_consolidated =
            measure_performance("ndarray_to_mat_consolidated", data_size, iterations, || {
                array.as_multi_channel_mat().unwrap()
            });

        // Test regular conversion (channels as separate dimension)
        let to_mat_regular =
            measure_performance("ndarray_to_mat_regular", data_size, iterations, || {
                array.as_single_channel_mat().unwrap()
            });

        println!(
            "  Consolidated - Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
            to_mat_consolidated.mean_duration, to_mat_consolidated.throughput_mb_per_sec
        );
        println!(
            "  Regular      - Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
            to_mat_regular.mean_duration, to_mat_regular.throughput_mb_per_sec
        );

        // Performance checks
        assert!(to_mat_consolidated.mean_duration.as_millis() < 2000);
        assert!(to_mat_regular.mean_duration.as_millis() < 2000);
    }
}

#[test]
fn test_data_type_conversion_performance() {
    let data_types = vec![
        (CV_8U, "u8", 1),
        (CV_16U, "u16", 2),
        (CV_32F, "f32", 4),
        (CV_64F, "f64", 8),
    ];

    let array_size = (256, 256);

    for (cv_type, type_name, type_size) in data_types {
        println!("=== Performance test for {} data type ===", type_name);

        let data_size = array_size.0 * array_size.1 * type_size;
        let iterations = 20;

        let mat = Mat::new_rows_cols_with_default(
            array_size.0 as i32,
            array_size.1 as i32,
            cv_type,
            100.into(),
        )
        .unwrap();

        match cv_type {
            CV_8U => {
                let result =
                    measure_performance("mat_to_ndarray_u8", data_size, iterations, || {
                        mat.as_ndarray::<u8, ndarray::Ix2>().unwrap()
                    });
                println!(
                    "  Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
                    result.mean_duration, result.throughput_mb_per_sec
                );
            }
            CV_16U => {
                let result =
                    measure_performance("mat_to_ndarray_u16", data_size, iterations, || {
                        mat.as_ndarray::<u16, ndarray::Ix2>().unwrap()
                    });
                println!(
                    "  Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
                    result.mean_duration, result.throughput_mb_per_sec
                );
            }
            CV_32F => {
                let result =
                    measure_performance("mat_to_ndarray_f32", data_size, iterations, || {
                        mat.as_ndarray::<f32, ndarray::Ix2>().unwrap()
                    });
                println!(
                    "  Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
                    result.mean_duration, result.throughput_mb_per_sec
                );
            }
            CV_64F => {
                let result =
                    measure_performance("mat_to_ndarray_f64", data_size, iterations, || {
                        mat.as_ndarray::<f64, ndarray::Ix2>().unwrap()
                    });
                println!(
                    "  Mean: {:>8.2?}, Throughput: {:>8.1} MB/s",
                    result.mean_duration, result.throughput_mb_per_sec
                );
            }
            _ => panic!("Unsupported type in benchmark"),
        }
    }
}

#[test]
fn test_conversion_overhead_analysis() {
    // Compare conversion overhead vs raw memory operations
    let size = (256, 256);
    let data_size = size.0 * size.1 * std::mem::size_of::<u8>();
    let iterations = 100;

    let array = Array2::<u8>::zeros(size);

    // Measure raw memory copy time
    let raw_copy_time = {
        let mut durations = Vec::new();
        let src = vec![0u8; data_size];

        for _ in 0..iterations {
            let start = Instant::now();
            let _dst = src.clone();
            durations.push(start.elapsed());
        }

        durations.iter().sum::<Duration>() / iterations as u32
    };

    // Measure conversion time
    let conversion_time = {
        let mut durations = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();
            let _mat = array.as_single_channel_mat().unwrap();
            durations.push(start.elapsed());
        }

        durations.iter().sum::<Duration>() / iterations as u32
    };

    let overhead_ratio = conversion_time.as_nanos() as f64 / raw_copy_time.as_nanos() as f64;

    println!("Overhead Analysis:");
    println!("  Raw copy time: {:?}", raw_copy_time);
    println!("  Conversion time: {:?}", conversion_time);
    println!("  Overhead ratio: {:.2}x", overhead_ratio);

    // Conversion should not be more than 20x slower than raw copy (very generous)
    assert!(
        overhead_ratio < 20.0,
        "Conversion overhead too high: {:.2}x",
        overhead_ratio
    );
}

#[test]
fn test_repeated_conversion_stability() {
    // Test that performance remains stable over repeated conversions
    let array = Array2::<f32>::from_shape_fn((128, 128), |(i, j)| (i + j) as f32);
    let rounds = 100;
    let mut durations = Vec::with_capacity(rounds);

    println!("Testing conversion stability over {} rounds", rounds);

    for _round in 0..rounds {
        let start = Instant::now();
        let mat = array.as_single_channel_mat().unwrap();
        let _back: ArrayView<f32, Ix2> = mat.as_ndarray().unwrap();
        durations.push(start.elapsed());
    }

    // Check first vs last quartile performance
    let first_quartile_avg = durations[0..25].iter().sum::<Duration>() / 25;
    let last_quartile_avg = durations[75..].iter().sum::<Duration>() / 25;
    let degradation_ratio =
        last_quartile_avg.as_nanos() as f64 / first_quartile_avg.as_nanos() as f64;

    println!("Stability analysis:");
    println!("  First quartile avg: {:?}", first_quartile_avg);
    println!("  Last quartile avg: {:?}", last_quartile_avg);
    println!("  Performance ratio: {:.2}x", degradation_ratio);

    // Performance should not degrade significantly over time
    assert!(
        degradation_ratio < 2.0,
        "Performance degradation detected: {:.2}x",
        degradation_ratio
    );
}

#[test]
fn test_scalability_basic() {
    // Test basic scalability with different array sizes
    let sizes = vec![64, 128, 256, 512];
    let mut results = Vec::new();

    for &size in &sizes {
        let array = Array2::<f32>::from_shape_fn((size, size), |(i, j)| (i + j) as f32);
        let data_size = size * size * std::mem::size_of::<f32>();

        let result =
            measure_performance(&format!("scale_{}x{}", size, size), data_size, 10, || {
                array.as_single_channel_mat().unwrap()
            });

        println!(
            "Size {}x{} - Duration: {:?}, Throughput: {:.1} MB/s",
            size, size, result.mean_duration, result.throughput_mb_per_sec
        );
        results.push((size, result));
    }

    // Basic scalability check - throughput shouldn't drop too dramatically
    let base_throughput = results[0].1.throughput_mb_per_sec;
    for (size, result) in &results[1..] {
        let throughput_ratio = result.throughput_mb_per_sec / base_throughput;
        assert!(
            throughput_ratio > 0.2,
            "Throughput dropped too much at size {}: {:.2}x of baseline",
            size,
            throughput_ratio
        );
    }
}

#[test]
fn test_memory_layout_performance() {
    // Test performance with different memory layouts
    let size = (256, 256);
    let iterations = 20;

    // Standard array
    let std_array = Array2::<u8>::from_shape_fn(size, |(i, j)| ((i + j) % 256) as u8);
    let std_result = measure_performance("standard_layout", size.0 * size.1, iterations, || {
        std_array.as_single_channel_mat().unwrap()
    });

    // Transposed view
    let transposed = std_array.t();
    let transposed_result =
        measure_performance("transposed_layout", size.0 * size.1, iterations, || {
            transposed.as_single_channel_mat().unwrap()
        });

    println!("Memory layout performance:");
    println!("  Standard: {:?}", std_result.mean_duration);
    println!("  Transposed: {:?}", transposed_result.mean_duration);

    // Both should complete in reasonable time
    assert!(std_result.mean_duration.as_millis() < 1000);
    assert!(transposed_result.mean_duration.as_millis() < 1000);
}
