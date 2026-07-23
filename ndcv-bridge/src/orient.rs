use ndarray::{Array, ArrayBase};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orientation {
    #[default]
    NoRotation,
    Mirror,
    Clock180,
    Water,
    MirrorClock270,
    Clock90,
    MirrorClock90,
    Clock270,
    Unknown,
}

impl Orientation {
    pub fn inverse(&self) -> Self {
        match self {
            Self::Clock90 => Self::Clock270,
            Self::Clock270 => Self::Clock90,
            _ => *self,
        }
    }
}

impl Orientation {
    pub fn from_raw(flip: u8) -> Self {
        match flip {
            1 => Orientation::NoRotation,
            2 => Orientation::Mirror,
            3 => Orientation::Clock180,
            4 => Orientation::Water,
            5 => Orientation::MirrorClock270,
            6 => Orientation::Clock90,
            7 => Orientation::MirrorClock90,
            8 => Orientation::Clock270,
            _ => Orientation::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationFlag {
    Clock90,
    Clock180,
    Clock270,
}

impl RotationFlag {
    pub fn neg(&self) -> Self {
        match self {
            RotationFlag::Clock90 => RotationFlag::Clock270,
            RotationFlag::Clock180 => RotationFlag::Clock180,
            RotationFlag::Clock270 => RotationFlag::Clock90,
        }
    }

    pub fn to_orientation(&self) -> Orientation {
        match self {
            RotationFlag::Clock90 => Orientation::Clock90,
            RotationFlag::Clock180 => Orientation::Clock180,
            RotationFlag::Clock270 => Orientation::Clock270,
        }
    }
}

#[derive(Clone, Copy)]
pub enum FlipFlag {
    Mirror,
    Water,
    Both,
}

pub trait Orient<T: bytemuck::Pod, D: ndarray::Dimension> {
    fn flip(&self, flip: FlipFlag) -> Array<T, D>;
    fn rotate(&self, rotation: RotationFlag) -> Array<T, D>;
    fn owned(&self) -> Array<T, D>;

    fn unorient(&self, orientation: Orientation) -> Array<T, D>
    where
        Array<T, D>: Orient<T, D>,
        Self: ToOwned<Owned = Array<T, D>>,
    {
        let inverse_orientation = orientation.inverse();
        self.orient(inverse_orientation)

        // match orientation {
        //     Orientation::NoRotation | Orientation::Unknown => self.to_owned(),
        //     Orientation::Mirror => self.flip(FlipFlag::Mirror).to_owned(),
        //     Orientation::Clock180 => self.rotate(RotationFlag::Clock180),
        //     Orientation::Water => self.flip(FlipFlag::Water).to_owned(),
        //     Orientation::MirrorClock270 => self
        //         .rotate(RotationFlag::Clock90)
        //         .flip(FlipFlag::Mirror)
        //         .to_owned(),
        //     Orientation::Clock90 => self.rotate(RotationFlag::Clock270),
        //     Orientation::MirrorClock90 => self
        //         .rotate(RotationFlag::Clock270)
        //         .flip(FlipFlag::Mirror)
        //         .to_owned(),
        //     Orientation::Clock270 => self.rotate(RotationFlag::Clock90),
        // }
    }

    fn orient(&self, orientation: Orientation) -> Array<T, D>
    where
        Array<T, D>: Orient<T, D>,
    {
        match orientation {
            Orientation::NoRotation | Orientation::Unknown => self.owned(),
            Orientation::Mirror => self.flip(FlipFlag::Mirror).to_owned(),
            Orientation::Clock180 => self.rotate(RotationFlag::Clock180),
            Orientation::Water => self.flip(FlipFlag::Water).to_owned(),
            Orientation::MirrorClock270 => self
                .flip(FlipFlag::Mirror)
                .rotate(RotationFlag::Clock270)
                .to_owned(),
            Orientation::Clock90 => self.rotate(RotationFlag::Clock90),
            Orientation::MirrorClock90 => self
                .flip(FlipFlag::Mirror)
                .rotate(RotationFlag::Clock90)
                .to_owned(),
            Orientation::Clock270 => self.rotate(RotationFlag::Clock270),
        }
        .as_standard_layout()
        .to_owned()
    }
}

impl<T: bytemuck::Pod + Copy, S: ndarray::Data<Elem = T>> Orient<T, ndarray::Ix3>
    for ArrayBase<S, ndarray::Ix3>
{
    fn flip(&self, flip: FlipFlag) -> Array<T, ndarray::Ix3> {
        match flip {
            FlipFlag::Mirror => self.slice(ndarray::s![.., ..;-1, ..]),
            FlipFlag::Water => self.slice(ndarray::s![..;-1, .., ..]),
            FlipFlag::Both => self.slice(ndarray::s![..;-1, ..;-1, ..]),
        }
        .as_standard_layout()
        .to_owned()
    }

    fn owned(&self) -> Array<T, ndarray::Ix3> {
        self.to_owned()
    }

    fn rotate(&self, rotation: RotationFlag) -> Array<T, ndarray::Ix3> {
        match rotation {
            RotationFlag::Clock90 => self
                .view()
                .permuted_axes([1, 0, 2])
                .flip(FlipFlag::Mirror)
                .to_owned(),
            RotationFlag::Clock180 => self.flip(FlipFlag::Both).to_owned(),
            RotationFlag::Clock270 => self
                .view()
                .permuted_axes([1, 0, 2])
                .flip(FlipFlag::Water)
                .to_owned(),
        }
    }
}

impl<T: bytemuck::Pod + Copy, S: ndarray::Data<Elem = T>> Orient<T, ndarray::Ix2>
    for ArrayBase<S, ndarray::Ix2>
{
    fn flip(&self, flip: FlipFlag) -> Array<T, ndarray::Ix2> {
        match flip {
            FlipFlag::Mirror => self.slice(ndarray::s![.., ..;-1,]),
            FlipFlag::Water => self.slice(ndarray::s![..;-1, ..,]),
            FlipFlag::Both => self.slice(ndarray::s![..;-1, ..;-1,]),
        }
        .as_standard_layout()
        .to_owned()
    }

    fn owned(&self) -> Array<T, ndarray::Ix2> {
        self.to_owned()
    }

    fn rotate(&self, rotation: RotationFlag) -> Array<T, ndarray::Ix2> {
        match rotation {
            RotationFlag::Clock90 => self.t().flip(FlipFlag::Mirror).to_owned(),
            RotationFlag::Clock180 => self.flip(FlipFlag::Both).to_owned(),
            RotationFlag::Clock270 => self.t().flip(FlipFlag::Water).to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn orientation_inverse_swaps_only_the_90s() {
        assert_eq!(Orientation::Clock90.inverse(), Orientation::Clock270);
        assert_eq!(Orientation::Clock270.inverse(), Orientation::Clock90);
        // Every other orientation is its own inverse.
        for o in [
            Orientation::NoRotation,
            Orientation::Mirror,
            Orientation::Clock180,
            Orientation::Water,
            Orientation::MirrorClock270,
            Orientation::MirrorClock90,
            Orientation::Unknown,
        ] {
            assert_eq!(o.inverse(), o);
        }
    }

    #[test]
    fn orientation_from_raw_maps_each_exif_value() {
        assert_eq!(Orientation::from_raw(1), Orientation::NoRotation);
        assert_eq!(Orientation::from_raw(2), Orientation::Mirror);
        assert_eq!(Orientation::from_raw(3), Orientation::Clock180);
        assert_eq!(Orientation::from_raw(4), Orientation::Water);
        assert_eq!(Orientation::from_raw(5), Orientation::MirrorClock270);
        assert_eq!(Orientation::from_raw(6), Orientation::Clock90);
        assert_eq!(Orientation::from_raw(7), Orientation::MirrorClock90);
        assert_eq!(Orientation::from_raw(8), Orientation::Clock270);
        // Anything outside 1..=8 is Unknown.
        assert_eq!(Orientation::from_raw(0), Orientation::Unknown);
        assert_eq!(Orientation::from_raw(9), Orientation::Unknown);
        assert_eq!(Orientation::from_raw(255), Orientation::Unknown);
    }

    #[test]
    fn rotation_flag_neg_reverses_direction() {
        assert_eq!(RotationFlag::Clock90.neg(), RotationFlag::Clock270);
        assert_eq!(RotationFlag::Clock180.neg(), RotationFlag::Clock180);
        assert_eq!(RotationFlag::Clock270.neg(), RotationFlag::Clock90);
    }

    #[test]
    fn rotation_flag_to_orientation() {
        assert_eq!(RotationFlag::Clock90.to_orientation(), Orientation::Clock90);
        assert_eq!(
            RotationFlag::Clock180.to_orientation(),
            Orientation::Clock180
        );
        assert_eq!(
            RotationFlag::Clock270.to_orientation(),
            Orientation::Clock270
        );
    }

    // A 2x3 array whose values encode (row, col) as row*10 + col, so a rotation
    // is only correct if every element lands in the right place.
    fn grid() -> Array2<i32> {
        array![[0, 1, 2], [10, 11, 12]]
    }

    #[test]
    fn rotate_clock90_then_clock270_is_identity() {
        let g = grid();
        let there = g.rotate(RotationFlag::Clock90);
        let back = there.rotate(RotationFlag::Clock270);
        assert_eq!(back, g);
    }

    #[test]
    fn rotate_clock90_places_bottom_left_at_top_left() {
        // Clockwise 90: the bottom-left element (10) becomes the new top-left.
        let g = grid();
        let r = g.rotate(RotationFlag::Clock90);
        assert_eq!(r.dim(), (3, 2));
        assert_eq!(r[[0, 0]], 10);
        assert_eq!(r[[0, 1]], 0);
        assert_eq!(r[[2, 1]], 2);
    }

    #[test]
    fn flip_mirror_reverses_columns() {
        let g = grid();
        let m = g.flip(FlipFlag::Mirror);
        assert_eq!(m, array![[2, 1, 0], [12, 11, 10]]);
    }

    #[test]
    fn flip_water_reverses_rows() {
        let g = grid();
        let w = g.flip(FlipFlag::Water);
        assert_eq!(w, array![[10, 11, 12], [0, 1, 2]]);
    }

    #[test]
    fn orient_unorient_round_trips() {
        let g = grid();
        for o in [
            Orientation::NoRotation,
            Orientation::Mirror,
            Orientation::Clock180,
            Orientation::Water,
            Orientation::Clock90,
            Orientation::Clock270,
        ] {
            let oriented = g.orient(o);
            let restored = oriented.unorient(o);
            assert_eq!(restored, g, "round trip failed for {o:?}");
        }
    }
}
