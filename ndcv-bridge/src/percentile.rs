use error_stack::*;
use ndarray::{ArrayBase, Ix1};
use num::cast::AsPrimitive;

use crate::NdCvError;

pub trait Percentile {
    fn percentile(&self, qth_percentile: f64) -> Result<f64, NdCvError>;
}

impl<T: std::cmp::Ord + Clone + AsPrimitive<f64>, S: ndarray::Data<Elem = T>> Percentile
    for ArrayBase<S, Ix1>
{
    fn percentile(&self, qth_percentile: f64) -> Result<f64, NdCvError> {
        if self.is_empty() {
            return Err(error_stack::Report::new(NdCvError).attach_printable("Empty Input"));
        }

        if !(0_f64..1_f64).contains(&qth_percentile) {
            return Err(error_stack::Report::new(NdCvError)
                .attach_printable("Qth percentile must be between 0 and 1"));
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
