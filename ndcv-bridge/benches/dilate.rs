use divan::black_box;
use ndarray::*;
use ndcv_bridge::*;

fn main() {
    divan::main();
}

fn rect_kernel(size: usize) -> Array2<u8> {
    Array2::ones((size, size))
}

/// Allocated vs in-place at 4K and 8K
#[divan::bench_group]
mod allocation {
    use super::*;

    #[divan::bench]
    fn dilate_4k_allocated_u8() {
        let arr = Array3::<u8>::ones((3840, 2160, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_4k_inplace_u8() {
        let mut arr = Array3::<u8>::ones((3840, 2160, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_4k_allocated_f32() {
        let arr = Array3::<f32>::ones((3840, 2160, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_4k_inplace_f32() {
        let mut arr = Array3::<f32>::ones((3840, 2160, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_8k_allocated_u8() {
        let arr = Array3::<u8>::ones((8192, 8192, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_8k_inplace_u8() {
        let mut arr = Array3::<u8>::ones((8192, 8192, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_8k_allocated_f32() {
        let arr = Array3::<f32>::ones((8192, 8192, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench]
    fn dilate_8k_inplace_f32() {
        let mut arr = Array3::<f32>::ones((8192, 8192, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }
}

/// Sweep image sizes
#[divan::bench_group]
mod sizes {
    use super::*;

    #[divan::bench(args = [512, 1024, 2048, 4096, 8192])]
    fn dilate_allocated_u8(size: usize) {
        let arr = Array3::<u8>::ones((size, size, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench(args = [512, 1024, 2048, 4096, 8192])]
    fn dilate_inplace_u8(size: usize) {
        let mut arr = Array3::<u8>::ones((size, size, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }

    #[divan::bench(args = [512, 1024, 2048, 4096, 8192])]
    fn dilate_allocated_f32(size: usize) {
        let arr = Array3::<f32>::ones((size, size, 3));
        let kernel = rect_kernel(3);
        let _out = black_box(arr.dilate_def(kernel.view(), 1).unwrap());
    }

    #[divan::bench(args = [512, 1024, 2048, 4096, 8192])]
    fn dilate_inplace_f32(size: usize) {
        let mut arr = Array3::<f32>::ones((size, size, 3));
        let kernel = rect_kernel(3);
        black_box(arr.dilate_def_inplace(kernel.view(), 1).unwrap());
    }
}
