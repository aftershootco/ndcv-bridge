use crate::{NdAsImage, NdAsImageMut, conversions::MatAsNd, prelude_::*};

pub(crate) mod seal {
    pub trait ConnectedComponentOutput: Sized + Copy + bytemuck::Pod + num::Zero {
        fn as_cv_type() -> i32 {
            crate::type_depth::<Self>()
        }
    }
    impl ConnectedComponentOutput for i32 {}
    impl ConnectedComponentOutput for u16 {}
}

pub trait NdCvConnectedComponents<T> {
    fn connected_components<O: seal::ConnectedComponentOutput>(
        &self,
        connectivity: Connectivity,
    ) -> Result<ndarray::Array2<O>, NdCvError>;
    fn connected_components_with_stats<O: seal::ConnectedComponentOutput>(
        &self,
        connectivity: Connectivity,
    ) -> Result<ConnectedComponentStats<O>, NdCvError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Connectivity {
    Four = 4,
    #[default]
    Eight = 8,
}

#[derive(Debug, Clone)]
pub struct ConnectedComponentStats<O: seal::ConnectedComponentOutput> {
    pub num_labels: i32,
    pub labels: ndarray::Array2<O>,
    pub stats: ndarray::Array2<i32>,
    pub centroids: ndarray::Array2<f64>,
}

// use crate::conversions::NdCvConversionRef;
impl<T: bytemuck::Pod, S: ndarray::Data<Elem = T>> NdCvConnectedComponents<T>
    for ndarray::ArrayBase<S, ndarray::Ix2>
where
    ndarray::Array2<T>: NdAsImage<T, ndarray::Ix2>,
{
    fn connected_components<O: seal::ConnectedComponentOutput>(
        &self,
        connectivity: Connectivity,
    ) -> Result<ndarray::Array2<O>, NdCvError> {
        let mat = self.as_image_mat()?;
        let mut labels = ndarray::Array2::<O>::zeros(self.dim());
        let mut cv_labels = labels.as_image_mat_mut()?;
        opencv::imgproc::connected_components(
            mat.as_ref(),
            cv_labels.as_mut(),
            connectivity as i32,
            O::as_cv_type(),
        )
        .change_context(NdCvError)?;
        Ok(labels)
    }

    fn connected_components_with_stats<O: seal::ConnectedComponentOutput>(
        &self,
        connectivity: Connectivity,
    ) -> Result<ConnectedComponentStats<O>, NdCvError> {
        let mut labels = ndarray::Array2::<O>::zeros(self.dim());
        let mut stats = opencv::core::Mat::default();
        let mut centroids = opencv::core::Mat::default();
        let num_labels = opencv::imgproc::connected_components_with_stats(
            self.as_image_mat()?.as_ref(),
            labels.as_image_mat_mut()?.as_mut(),
            &mut stats,
            &mut centroids,
            connectivity as i32,
            O::as_cv_type(),
        )
        .change_context(NdCvError)?;
        let stats = stats.as_ndarray()?.to_owned();
        let centroids = centroids.as_ndarray()?.to_owned();
        Ok(ConnectedComponentStats {
            labels,
            stats,
            centroids,
            num_labels,
        })
    }
}

// #[test]
// fn test_connected_components() {
//     use opencv::core::MatTrait as _;
//     let mat = opencv::core::Mat::new_nd_with_default(&[10, 10], opencv::core::CV_8UC1, 0.into())
//         .expect("failed");
//     let roi1 = opencv::core::Rect::new(2, 2, 2, 2);
//     let roi2 = opencv::core::Rect::new(6, 6, 3, 3);
//     let mut mat1 = opencv::core::Mat::roi(&mat, roi1).expect("failed");
//     mat1.set_scalar(1.into()).expect("failed");
//     let mut mat2 = opencv::core::Mat::roi(&mat, roi2).expect("failed");
//     mat2.set_scalar(1.into()).expect("failed");

//     let array2: ndarray::ArrayView2<u8> = mat.as_ndarray().expect("failed");
//     let output = array2
//         .connected_components::<u16>(Connectivity::Four)
//         .expect("failed");
//     let expected = {
//         let mut expected = ndarray::Array2::zeros((10, 10));
//         expected.slice_mut(ndarray::s![2..4, 2..4]).fill(1);
//         expected.slice_mut(ndarray::s![6..9, 6..9]).fill(2);
//         expected
//     };

//     assert_eq!(output, expected);
// }
