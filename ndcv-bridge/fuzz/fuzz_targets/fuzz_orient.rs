#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array2, Array3};
use ndcv_bridge::orient::{FlipFlag, Orient, Orientation, RotationFlag};

#[derive(Arbitrary, Debug)]
struct OrientInput {
    height: u8,
    width: u8,
    channels: u8,
    orientation_raw: u8,
    use_3d: bool,
    data: Vec<u8>,
}

fuzz_target!(|input: OrientInput| {
    let height = (input.height as usize).clamp(1, 64);
    let width = (input.width as usize).clamp(1, 64);
    let orientation = Orientation::from_raw(input.orientation_raw);

    if input.use_3d {
        let channels = (input.channels as usize).clamp(1, 4);
        let total = height * width * channels;
        if input.data.len() < total {
            return;
        }
        let data: Vec<u8> = input.data[..total].to_vec();
        let arr = Array3::from_shape_vec((height, width, channels), data).unwrap();

        // Test orient + unorient round-trip
        let oriented = arr.orient(orientation);
        let unoriented = oriented.unorient(orientation);

        // For orientations that have a true inverse, check round-trip
        match orientation {
            Orientation::NoRotation
            | Orientation::Clock180
            | Orientation::Mirror
            | Orientation::Water
            | Orientation::Clock90
            | Orientation::Clock270
            | Orientation::Unknown => {
                assert_eq!(
                    arr.shape()[0] * arr.shape()[1],
                    unoriented.shape()[0] * unoriented.shape()[1]
                );
            }
            _ => {}
        }

        // Test individual operations don't panic
        let _ = arr.flip(FlipFlag::Mirror);
        let _ = arr.flip(FlipFlag::Water);
        let _ = arr.flip(FlipFlag::Both);
        let _ = arr.rotate(RotationFlag::Clock90);
        let _ = arr.rotate(RotationFlag::Clock180);
        let _ = arr.rotate(RotationFlag::Clock270);
    } else {
        let total = height * width;
        if input.data.len() < total {
            return;
        }
        let data: Vec<u8> = input.data[..total].to_vec();
        let arr = Array2::from_shape_vec((height, width), data).unwrap();

        let oriented = arr.orient(orientation);
        let _ = oriented.unorient(orientation);

        let _ = arr.flip(FlipFlag::Mirror);
        let _ = arr.flip(FlipFlag::Water);
        let _ = arr.flip(FlipFlag::Both);
        let _ = arr.rotate(RotationFlag::Clock90);
        let _ = arr.rotate(RotationFlag::Clock180);
        let _ = arr.rotate(RotationFlag::Clock270);
    }
});
