#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::Array1;
use ndcv_bridge::percentile::Percentile;

#[derive(Arbitrary, Debug)]
struct PercentileInput {
    /// Raw bytes reinterpreted as i32 values for the array
    data: Vec<i32>,
    /// Percentile value components (used to construct f64 in [0, 1) range and beyond)
    percentile_numerator: u32,
    percentile_denominator: u32,
}

fuzz_target!(|input: PercentileInput| {
    if input.data.is_empty() || input.data.len() > 10_000 {
        return;
    }

    let arr = Array1::from_vec(input.data);

    // Construct various percentile values to test edge cases
    let denom = input.percentile_denominator.max(1) as f64;
    let qth = input.percentile_numerator as f64 / denom;

    let result = arr.percentile(qth);

    // Valid range is [0, 1) -- the code checks !(0..1).contains(&qth)
    if (0.0..1.0).contains(&qth) {
        // Should succeed for non-empty arrays
        assert!(result.is_ok(), "Expected Ok for qth={qth}, got {result:?}");
        let val = result.unwrap();
        assert!(val.is_finite(), "Percentile returned non-finite: {val}");
    } else {
        // Out of range should fail
        assert!(result.is_err(), "Expected Err for out-of-range qth={qth}");
    }

    // Also test exact boundary values
    let _ = arr.percentile(0.0);
    let _ = arr.percentile(0.5);
    let _ = arr.percentile(0.999_999_999);
    let _ = arr.percentile(1.0);
    let _ = arr.percentile(-0.001);
    let _ = arr.percentile(f64::NAN);
    let _ = arr.percentile(f64::INFINITY);
    let _ = arr.percentile(f64::NEG_INFINITY);
});
