use divan::black_box;
use ndarray::*;
use ndcv_bridge::*;

// #[global_allocator]
// static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

// Helper function to create test images with different patterns
fn create_test_image(size: usize, pattern: &str) -> Array3<u8> {
    let mut arr = Array3::<u8>::zeros((size, size, 3));
    match pattern {
        "edges" => {
            // Create a pattern with sharp edges
            arr.slice_mut(s![size / 4..3 * size / 4, size / 4..3 * size / 4, ..])
                .fill(255);
        }
        "gradient" => {
            // Create a gradual gradient
            for i in 0..size {
                let val = (i * 255 / size) as u8;
                arr.slice_mut(s![i, .., ..]).fill(val);
            }
        }
        "checkerboard" => {
            // Create a checkerboard pattern
            for i in 0..size {
                for j in 0..size {
                    if (i / 20 + j / 20) % 2 == 0 {
                        arr[[i, j, 0]] = 255;
                        arr[[i, j, 1]] = 255;
                        arr[[i, j, 2]] = 255;
                    }
                }
            }
        }
        _ => arr.fill(255), // Default to solid white
    }
    arr
}

#[divan::bench_group]
mod sizes {
    use super::*;
    // Benchmark different image sizes
    #[divan::bench(args = [512, 1024, 2048, 4096])]
    fn bench_gaussian_sizes_u8(size: usize) {
        let arr = Array3::<u8>::ones((size, size, 3));
        let _out = black_box(
            arr.gaussian_blur((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }

    #[divan::bench(args = [512, 1024, 2048, 4096])]
    fn bench_gaussian_sizes_u8_inplace(size: usize) {
        let mut arr = Array3::<u8>::ones((size, size, 3));
        black_box(
            arr.gaussian_blur_inplace((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }

    #[divan::bench(args = [512, 1024, 2048, 4096])]
    fn bench_gaussian_sizes_f32(size: usize) {
        let arr = Array3::<f32>::ones((size, size, 3));
        let _out = black_box(
            arr.gaussian_blur((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }

    #[divan::bench(args = [512, 1024, 2048, 4096])]
    fn bench_gaussian_sizes_f32_inplace(size: usize) {
        let mut arr = Array3::<f32>::ones((size, size, 3));
        black_box(
            arr.gaussian_blur_inplace((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
}

// Benchmark different kernel sizes
#[divan::bench(args = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])]
fn bench_gaussian_kernels(kernel_size: (u8, u8)) {
    let mut arr = Array3::<u8>::ones((1000, 1000, 3));
    arr.gaussian_blur_inplace(kernel_size, 1.0, 1.0, BorderType::BorderConstant)
        .unwrap();
}

// Benchmark different sigma values
#[divan::bench(args = [0.5, 1.0, 2.0, 5.0])]
fn bench_gaussian_sigmas(sigma: f64) {
    let mut arr = Array3::<u8>::ones((1000, 1000, 3));
    arr.gaussian_blur_inplace((3, 3), sigma, sigma, BorderType::BorderConstant)
        .unwrap();
}

// Benchmark different sigma_x and sigma_y combinations
#[divan::bench(args = [(0.5, 2.0), (1.0, 1.0), (2.0, 0.5), (3.0, 1.0)])]
fn bench_gaussian_asymmetric_sigmas(sigmas: (f64, f64)) {
    let mut arr = Array3::<u8>::ones((1000, 1000, 3));
    arr.gaussian_blur_inplace((3, 3), sigmas.0, sigmas.1, BorderType::BorderConstant)
        .unwrap();
}

// Benchmark different border types
#[divan::bench]
fn bench_gaussian_border_types() -> Vec<()> {
    let border_types = [
        BorderType::BorderConstant,
        BorderType::BorderReplicate,
        BorderType::BorderReflect,
        BorderType::BorderReflect101,
    ];

    let mut arr = Array3::<u8>::ones((1000, 1000, 3));
    border_types
        .iter()
        .map(|border_type| {
            arr.gaussian_blur_inplace((3, 3), 1.0, 1.0, *border_type)
                .unwrap();
        })
        .collect()
}

// Benchmark different image patterns
#[divan::bench]
fn bench_gaussian_patterns() {
    let patterns = ["edges", "gradient", "checkerboard", "solid"];

    patterns.iter().for_each(|&pattern| {
        let mut arr = create_test_image(1000, pattern);
        arr.gaussian_blur_inplace((3, 3), 1.0, 1.0, BorderType::BorderConstant)
            .unwrap();
    })
}

#[divan::bench_group]
mod allocation {
    use super::*;
    #[divan::bench]
    fn bench_gaussian_allocation_inplace() {
        let mut arr = Array3::<f32>::ones((3840, 2160, 3));

        black_box(
            arr.gaussian_blur_inplace((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }

    #[divan::bench]
    fn bench_gaussian_allocation_allocate() {
        let arr = Array3::<f32>::ones((3840, 2160, 3));

        let _out = black_box(
            arr.gaussian_blur((3, 3), 1.0, 1.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
}

#[divan::bench_group]
mod realistic {
    use super::*;
    #[divan::bench]
    fn small_800_600_3x3() {
        let small_blur = Array3::<u8>::ones((800, 600, 3));
        let _blurred = black_box(
            small_blur
                .gaussian_blur((3, 3), 0.5, 0.5, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn small_800_600_3x3_inplace() {
        let mut small_blur = Array3::<u8>::ones((800, 600, 3));
        small_blur
            .gaussian_blur_inplace((3, 3), 0.5, 0.5, BorderType::BorderConstant)
            .unwrap();
    }
    #[divan::bench]
    fn medium_1920x1080_5x5() {
        let mut medium_blur = Array3::<u8>::ones((1920, 1080, 3));
        let _blurred = black_box(
            medium_blur
                .gaussian_blur_inplace((5, 5), 2.0, 2.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn medium_1920x1080_5x5_inplace() {
        let mut medium_blur = Array3::<u8>::ones((1920, 1080, 3));
        medium_blur
            .gaussian_blur_inplace((5, 5), 2.0, 2.0, BorderType::BorderConstant)
            .unwrap();
    }
    #[divan::bench]
    fn large_3840x2160_9x9() {
        let large_blur = Array3::<u8>::ones((3840, 2160, 3));
        let _blurred = black_box(
            large_blur
                .gaussian_blur((9, 9), 5.0, 5.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn large_3840x2160_9x9_inplace() {
        let mut large_blur = Array3::<u8>::ones((3840, 2160, 3));
        large_blur
            .gaussian_blur_inplace((9, 9), 5.0, 5.0, BorderType::BorderConstant)
            .unwrap();
    }
    #[divan::bench]
    fn small_800_600_3x3_f32() {
        let small_blur = Array3::<f32>::ones((800, 600, 3));
        let _blurred = black_box(
            small_blur
                .gaussian_blur((3, 3), 0.5, 0.5, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn small_800_600_3x3_inplace_f32() {
        let mut small_blur = Array3::<f32>::ones((800, 600, 3));
        small_blur
            .gaussian_blur_inplace((3, 3), 0.5, 0.5, BorderType::BorderConstant)
            .unwrap();
    }
    #[divan::bench]
    fn medium_1920x1080_5x5_f32() {
        let mut medium_blur = Array3::<f32>::ones((1920, 1080, 3));
        let _blurred = black_box(
            medium_blur
                .gaussian_blur_inplace((5, 5), 2.0, 2.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn medium_1920x1080_5x5_inplace_f32() {
        let mut medium_blur = Array3::<f32>::ones((1920, 1080, 3));
        medium_blur
            .gaussian_blur_inplace((5, 5), 2.0, 2.0, BorderType::BorderConstant)
            .unwrap();
    }
    #[divan::bench]
    fn large_3840x2160_9x9_f32() {
        let large_blur = Array3::<f32>::ones((3840, 2160, 3));
        let _blurred = black_box(
            large_blur
                .gaussian_blur((9, 9), 5.0, 5.0, BorderType::BorderConstant)
                .unwrap(),
        );
    }
    #[divan::bench]
    fn large_3840x2160_9x9_inplace_f32() {
        let mut large_blur = Array3::<f32>::ones((3840, 2160, 3));
        large_blur
            .gaussian_blur_inplace((9, 9), 5.0, 5.0, BorderType::BorderConstant)
            .unwrap();
    }
}
