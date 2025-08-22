use ndarray::{Array, ArrayBase, ArrayView};

#[derive(Clone, Copy)]
pub enum Orientation {
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
