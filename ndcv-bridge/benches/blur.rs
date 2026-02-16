use divan::black_box;
use ndarray::*;
use ndcv_bridge::*;

fn main() {
    divan::main();
}

fn create_test_image(size: usize, pattern: &str) -> Array3<u8> {
    let mut arr = Array3::<u8>::zeros((size, size, 3));
    match pattern {
        "edges" => {
            arr.slice_mut(s![size / 4..3 * size / 4, size / 4..3 * size / 4, ..])
                .fill(255);
        }
        "gradient" => {
            for i in 0..size {
                let val = (i * 255 / size) as u8;
                arr.slice_mut(s![i, .., ..]).fill(val);
            }
        }
        "checkerboard" => {
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
        _ => arr.fill(255),
    }
    arr
}

#[divan::bench_group]
mod sizes {
    use super::*;

    #[divan::bench(args = [512, 1024, 2048, 4096, 8096, 12000])]
    fn bench_blur_sizes_u8(size: usize) {
        let arr = Array3::<u8>::ones((size, size, 3));
        let _out = black_box(arr.blur((3, 3), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench(args = [512, 1024, 2048, 4096, 8096, 12000])]
    fn bench_blur_sizes_f32(size: usize) {
        let arr = Array3::<f32>::ones((size, size, 3));
        let _out = black_box(arr.blur((3, 3), BorderType::BorderConstant).unwrap());
    }
}

#[divan::bench(args = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])]
fn bench_blur_kernels(kernel_size: (i32, i32)) {
    let arr = Array3::<u8>::ones((1000, 1000, 3));
    let _out = black_box(arr.blur(kernel_size, BorderType::BorderConstant).unwrap());
}

#[divan::bench]
fn bench_blur_border_types() -> Vec<()> {
    let border_types = [
        BorderType::BorderConstant,
        BorderType::BorderReplicate,
        BorderType::BorderReflect,
        BorderType::BorderReflect101,
    ];

    let arr = Array3::<u8>::ones((1000, 1000, 3));
    border_types
        .iter()
        .map(|border_type| {
            let _out = black_box(arr.blur((3, 3), *border_type).unwrap());
        })
        .collect()
}

#[divan::bench]
fn bench_blur_patterns() {
    let patterns = ["edges", "gradient", "checkerboard", "solid"];

    patterns.iter().for_each(|&pattern| {
        let arr = create_test_image(1000, pattern);
        let _out = black_box(arr.blur((3, 3), BorderType::BorderConstant).unwrap());
    })
}

#[divan::bench_group]
mod realistic {
    use super::*;

    #[divan::bench]
    fn small_800_600_3x3() {
        let arr = Array3::<u8>::ones((800, 600, 3));
        let _blurred = black_box(arr.blur((3, 3), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn medium_1920x1080_5x5() {
        let arr = Array3::<u8>::ones((1920, 1080, 3));
        let _blurred = black_box(arr.blur((5, 5), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn large_3840x2160_9x9() {
        let arr = Array3::<u8>::ones((3840, 2160, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn xlarge_8096x4320_9x9() {
        let arr = Array3::<u8>::ones((8096, 4320, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn xxlarge_12000x6000_9x9() {
        let arr = Array3::<u8>::ones((12000, 6000, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn small_800_600_3x3_f32() {
        let arr = Array3::<f32>::ones((800, 600, 3));
        let _blurred = black_box(arr.blur((3, 3), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn medium_1920x1080_5x5_f32() {
        let arr = Array3::<f32>::ones((1920, 1080, 3));
        let _blurred = black_box(arr.blur((5, 5), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn large_3840x2160_9x9_f32() {
        let arr = Array3::<f32>::ones((3840, 2160, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn xlarge_8096x4320_9x9_f32() {
        let arr = Array3::<f32>::ones((8096, 4320, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }

    #[divan::bench]
    fn xxlarge_12000x6000_9x9_f32() {
        let arr = Array3::<f32>::ones((12000, 6000, 3));
        let _blurred = black_box(arr.blur((9, 9), BorderType::BorderConstant).unwrap());
    }
}
