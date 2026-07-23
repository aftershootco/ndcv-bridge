use ndarray::{ArrayBase, Ix1};
use num::cast::AsPrimitive;

#[derive(Debug, thiserror::Error)]
pub enum PercentileError {
    #[error("Input array is empty")]
    EmptyInput,
    #[error("Qth percentile must be between 0 and 1")]
    InvalidPercentile,
}

pub trait Percentile {
    fn percentile(&self, qth_percentile: f64) -> Result<f64, PercentileError>;
}

impl<T: std::cmp::Ord + Clone + AsPrimitive<f64>, S: ndarray::Data<Elem = T>> Percentile
    for ArrayBase<S, Ix1>
{
    fn percentile(&self, qth_percentile: f64) -> Result<f64, PercentileError> {
        if self.is_empty() {
            return Err(PercentileError::EmptyInput);
        }

        if !(0_f64..1_f64).contains(&qth_percentile) {
            return Err(PercentileError::InvalidPercentile);
        }

        let mut standard_array = self.as_standard_layout();
        let raw_data = standard_array
            .as_slice_mut()
            .expect("An array in standard layout will always return its inner slice");

        raw_data.sort();

        let actual_index = qth_percentile * (raw_data.len() - 1) as f64;

        let lower_index = (actual_index.floor() as usize).clamp(0, raw_data.len() - 1);
        let upper_index = (actual_index.ceil() as usize).clamp(0, raw_data.len() - 1);

        if lower_index == upper_index {
            Ok(raw_data[lower_index].as_())
        } else {
            let weight = actual_index - lower_index as f64;
            Ok(raw_data[lower_index].as_() * (1.0 - weight) + raw_data[upper_index].as_() * weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_input_errors() {
        let arr = ndarray::Array1::<i32>::zeros(0);
        assert!(arr.percentile(0.5).is_err());
    }

    #[test]
    fn out_of_range_quantile_errors() {
        let arr = array![0, 1, 2];
        // 1.0 is excluded (range is [0, 1)) and negatives are rejected.
        assert!(arr.percentile(1.0).is_err());
        assert!(arr.percentile(-0.5).is_err());
        assert!(arr.percentile(2.0).is_err());
        // The valid boundary still works.
        assert!(arr.percentile(0.0).is_ok());
    }

    #[test]
    fn interpolates_between_neighbours() {
        // len 2, q=0.5 -> actual_index = 0.5, lower=0 upper=1 weight=0.5
        // -> 0*0.5 + 10*0.5 = 5.0
        let arr = array![0, 10];
        assert!((arr.percentile(0.5).unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_zero_returns_minimum() {
        let arr = array![3, 7, 9];
        assert!((arr.percentile(0.0).unwrap() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn quantile_weight_favours_upper_neighbour() {
        // len 2, q=0.9 -> actual_index = 0.9 -> 0*0.1 + 10*0.9 = 9.0
        let arr = array![0, 10];
        assert!((arr.percentile(0.9).unwrap() - 9.0).abs() < 1e-9);
    }

    #[test]
    fn interpolates_between_nonzero_neighbours() {
        // len 3, q=0.75 -> actual_index = 1.5, lower=1 upper=2 weight=0.5
        // -> raw[1]*(1-0.5) + raw[2]*0.5 = 10*0.5 + 20*0.5 = 15.0
        // Uses a nonzero lower neighbour so the `raw[lower] * (1 - weight)`
        // term is exercised (catches operator mutations there).
        let arr = array![0, 10, 20];
        assert!((arr.percentile(0.75).unwrap() - 15.0).abs() < 1e-9);
    }

    #[test]
    fn exact_index_hits_that_element() {
        // len 3, q=0.5 -> actual_index = 1.0 -> lower==upper==1 -> element 1
        let arr = array![1, 2, 9];
        assert!((arr.percentile(0.5).unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn sorts_unordered_input() {
        // Same data as interpolates_between_neighbours but reversed; must sort first.
        let arr = array![10, 0];
        assert!((arr.percentile(0.5).unwrap() - 5.0).abs() < 1e-9);
    }
}

// fn percentile(data: &Array1<f64>, p: f64) -> f64 {
//     if data.len() == 0 {
//         return 0.0;
//     }
//
//     let mut sorted_data = data.to_vec();
//     sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
//
//     let index = (p / 100.0) * (sorted_data.len() - 1) as f64;
//     let lower = index.floor() as usize;
//     let upper = index.ceil() as usize;
//
//     if lower == upper {
//         sorted_data[lower] as f64
//     } else {
//         let weight = index - lower as f64;
//         sorted_data[lower] as f64 * (1.0 - weight) + sorted_data[upper] as f64 * weight
//     }
// }
