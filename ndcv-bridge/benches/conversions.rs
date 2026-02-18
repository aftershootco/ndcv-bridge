use divan::black_box;
use ndcv_bridge::*;

// #[global_allocator]
// static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    divan::main();
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_512() {
    bench_mat_to_3d_ndarray(512);
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_1024() {
    bench_mat_to_3d_ndarray(1024);
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_2k() {
    bench_mat_to_3d_ndarray(2048);
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_4k() {
    bench_mat_to_3d_ndarray(4096);
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_8k() {
    bench_mat_to_3d_ndarray(8192);
}

#[divan::bench]
fn bench_3d_mat_to_ndarray_8k_ref() {
    bench_mat_to_3d_ndarray_ref(8192);
}

#[divan::bench]
fn bench_2d_mat_to_ndarray_8k_ref() {
    bench_mat_to_2d_ndarray(8192);
}

fn bench_mat_to_2d_ndarray(size: i32) -> ndarray::Array2<u8> {
    let mat =
        opencv::core::Mat::new_nd_with_default(&[size, size], opencv::core::CV_8UC1, (200).into())
            .expect("failed");
    let ndarray: ndarray::Array2<u8> = mat.as_ndarray().expect("failed").to_owned();
    ndarray
}

fn bench_mat_to_3d_ndarray(size: i32) -> ndarray::Array3<u8> {
    let mat = opencv::core::Mat::new_nd_with_default(
        &[size, size],
        opencv::core::CV_8UC3,
        (200, 100, 10).into(),
    )
    .expect("failed");
    // ndarray::Array3::<u8>::from_mat(black_box(mat)).expect("failed")
    let ndarray: ndarray::Array3<u8> = mat.as_ndarray().expect("failed").to_owned();
    ndarray
}

fn bench_mat_to_3d_ndarray_ref(size: i32) {
    let mut mat = opencv::core::Mat::new_nd_with_default(
        &[size, size],
        opencv::core::CV_8UC3,
        (200, 100, 10).into(),
    )
    .expect("failed");
    let array: ndarray::ArrayView3<u8> = black_box(&mut mat).as_ndarray().expect("failed");
    let _ = black_box(array);
}
