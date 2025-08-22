//! # ndarray-serialize
//!
//! A Rust library for serializing and deserializing `ndarray` arrays using the SafeTensors format.
//!
//! ## Features
//! - Serialize `ndarray::ArrayView` to SafeTensors format
//! - Deserialize SafeTensors data back to `ndarray::ArrayView`
//! - Support for multiple data types (f32, f64, i8-i64, u8-u64, f16, bf16)
//! - Zero-copy deserialization when possible
//! - Metadata support
//!
//! ## Example
//! ```rust
//! use ndarray::Array2;
//! use ndarray_safetensors::{SafeArrays, SafeArrayView};
//!
//! // Create some data
//! let array = Array2::<f32>::zeros((3, 4));
//!
//! // Serialize
//! let mut safe_arrays = SafeArrays::new();
//! safe_arrays.insert_ndarray("my_tensor", array.view()).unwrap();
//! safe_arrays.insert_metadata("author", "example");
//! let bytes = safe_arrays.serialize().unwrap();
//!
//! // Deserialize
//! let view = SafeArrayView::from_bytes(&bytes).unwrap();
//! let tensor: ndarray::ArrayView2<f32> = view.tensor("my_tensor").unwrap();
//! assert_eq!(tensor.shape(), &[3, 4]);
//! ```

use safetensors::View;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};

use thiserror::Error;
/// Errors that can occur during SafeTensor operations
#[derive(Error, Debug)]
pub enum SafeTensorError {
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Invalid tensor data: Got {0} Expected: {1}")]
    InvalidTensorData(&'static str, String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Safetensor error: {0}")]
    SafeTensor(#[from] safetensors::SafeTensorError),
    #[error("ndarray::ShapeError error: {0}")]
    NdarrayShapeError(#[from] ndarray::ShapeError),
}

type Result<T, E = SafeTensorError> = core::result::Result<T, E>;

use safetensors::tensor::SafeTensors;

/// A view into SafeTensors data that provides access to ndarray tensors
///
/// # Example
/// ```rust
/// use ndarray::Array2;
/// use ndarray_safetensors::{SafeArrays, SafeArrayView};
///
/// let array = Array2::<f32>::ones((2, 3));
/// let mut safe_arrays = SafeArrays::new();
/// safe_arrays.insert_ndarray("data", array.view()).unwrap();
/// let bytes = safe_arrays.serialize().unwrap();
///
/// let view = SafeArrayView::from_bytes(&bytes).unwrap();
/// let tensor: ndarray::ArrayView2<f32> = view.tensor("data").unwrap();
/// ```
#[derive(Debug)]
pub struct SafeArraysView<'a> {
    pub tensors: SafeTensors<'a>,
}

impl<'a> SafeArraysView<'a> {
    fn new(tensors: SafeTensors<'a>) -> Self {
        Self { tensors }
    }

    /// Create a SafeArrayView from serialized bytes
    pub fn from_bytes(bytes: &'a [u8]) -> Result<SafeArraysView<'a>> {
        let tensors = SafeTensors::deserialize(bytes)?;
        Ok(Self::new(tensors))
    }

    /// Get a dynamic-dimensional tensor by name
    pub fn dynamic_tensor<T: STDtype>(&self, name: &str) -> Result<ndarray::ArrayViewD<'a, T>> {
        self.tensors
            .tensor(name)
            .map(|tensor| tensor_view_to_array_view(tensor))?
    }

    /// Get a tensor with specific dimensions by name
    ///
    /// # Example
    /// ```rust
    /// # use ndarray::Array2;
    /// # use ndarray_safetensors::{SafeArrays, SafeArrayView};
    /// # let array = Array2::<f32>::ones((2, 3));
    /// # let mut safe_arrays = SafeArrays::new();
    /// # safe_arrays.insert_ndarray("data", array.view()).unwrap();
    /// # let bytes = safe_arrays.serialize().unwrap();
    /// # let view = SafeArrayView::from_bytes(&bytes).unwrap();
    /// let tensor: ndarray::ArrayView2<f32> = view.tensor("data").unwrap();
    /// ```
    pub fn tensor<T: STDtype, Dim: ndarray::Dimension>(
        &self,
        name: &str,
    ) -> Result<ndarray::ArrayView<'a, T, Dim>> {
        Ok(self
            .tensors
            .tensor(name)
            .map(|tensor| tensor_view_to_array_view(tensor))?
            .map(|array_view| array_view.into_dimensionality::<Dim>())??)
    }

    pub fn tensor_by_index<T: STDtype, Dim: ndarray::Dimension>(
        &self,
        index: usize,
    ) -> Result<ndarray::ArrayView<'a, T, Dim>> {
        self.tensors
            .iter()
            .nth(index)
            .ok_or(SafeTensorError::TensorNotFound(format!(
                "Index {} out of bounds",
                index
            )))
            .map(|(_, tensor)| tensor_view_to_array_view(tensor))?
            .map(|array_view| array_view.into_dimensionality::<Dim>())?
            .map_err(SafeTensorError::NdarrayShapeError)
    }

    /// Get an iterator over tensor names
    pub fn names(&self) -> std::vec::IntoIter<&str> {
        self.tensors.names().into_iter()
    }

    /// Get the number of tensors
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if there are no tensors
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Trait for types that can be stored in SafeTensors
///
/// Implemented for: f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, f16, bf16
pub trait STDtype: bytemuck::Pod {
    fn dtype() -> safetensors::tensor::Dtype;
    fn size() -> usize {
        (Self::dtype().bitsize() / 8).max(1)
    }
}

macro_rules! impl_dtype {
    ($($t:ty => $dtype:expr),* $(,)?) => {
        $(
            impl STDtype for $t {
                fn dtype() -> safetensors::tensor::Dtype {
                    $dtype
                }
            }
        )*
    };
}

use safetensors::tensor::Dtype;

impl_dtype!(
    // bool => Dtype::BOOL, // idk if ndarray::ArrayD<bool> is packed
    f32 => Dtype::F32,
    f64 => Dtype::F64,
    i8 => Dtype::I8,
    i16 => Dtype::I16,
    i32 => Dtype::I32,
    i64 => Dtype::I64,
    u8 => Dtype::U8,
    u16 => Dtype::U16,
    u32 => Dtype::U32,
    u64 => Dtype::U64,
    half::f16 => Dtype::F16,
    half::bf16 => Dtype::BF16,
);

fn tensor_view_to_array_view<'a, T: STDtype>(
    tensor: safetensors::tensor::TensorView<'a>,
) -> Result<ndarray::ArrayViewD<'a, T>> {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    if T::dtype() != dtype {
        return Err(SafeTensorError::InvalidTensorData(
            core::any::type_name::<T>(),
            dtype.to_string(),
        ));
    }

    let data = tensor.data();
    let data: &[T] = bytemuck::cast_slice(data);
    let array = ndarray::ArrayViewD::from_shape(shape, data)?;
    Ok(array)
}

/// Builder for creating SafeTensors data from ndarray tensors
///
/// # Example
/// ```rust
/// use ndarray::{Array1, Array2};
/// use ndarray_safetensors::SafeArrays;
///
/// let mut safe_arrays = SafeArrays::new();
///
/// let array1 = Array1::<f32>::from_vec(vec![1.0, 2.0, 3.0]);
/// let array2 = Array2::<i32>::zeros((2, 2));
///
/// safe_arrays.insert_ndarray("vector", array1.view()).unwrap();
/// safe_arrays.insert_ndarray("matrix", array2.view()).unwrap();
/// safe_arrays.insert_metadata("version", "1.0");
///
/// let bytes = safe_arrays.serialize().unwrap();
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SafeArrays<'a> {
    pub tensors: BTreeMap<String, SafeArray<'a>>,
    pub metadata: Option<HashMap<String, String>>,
}

impl<'a, K: AsRef<str>> FromIterator<(K, SafeArray<'a>)> for SafeArrays<'a> {
    fn from_iter<T: IntoIterator<Item = (K, SafeArray<'a>)>>(iter: T) -> Self {
        let tensors = iter
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v))
            .collect();
        Self {
            tensors,
            metadata: None,
        }
    }
}

impl<'a, K: AsRef<str>, T: IntoIterator<Item = (K, SafeArray<'a>)>> From<T> for SafeArrays<'a> {
    fn from(iter: T) -> Self {
        let tensors = iter
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_owned(), v))
            .collect();
        Self {
            tensors,
            metadata: None,
        }
    }
}

impl<'a> SafeArrays<'a> {
    /// Create a SafeArrays from an iterator of (name, ndarray::ArrayView) pairs
    /// ```rust
    /// use ndarray::{Array2, Array3};
    /// use ndarray_safetensors::{SafeArrays, SafeArray};
    /// let array = Array2::<f32>::zeros((3, 4));
    /// let safe_arrays = SafeArrays::from_ndarrays(vec![
    ///     ("test_tensor", array.view()),
    ///     ("test_tensor2", array.view()),
    /// ]).unwrap();
    /// ```

    pub fn from_ndarrays<
        K: AsRef<str>,
        T: STDtype,
        D: ndarray::Dimension + 'a,
        I: IntoIterator<Item = (K, ndarray::ArrayView<'a, T, D>)>,
    >(
        iter: I,
    ) -> Result<Self> {
        let tensors = iter
            .into_iter()
            .map(|(k, v)| Ok((k.as_ref().to_owned(), SafeArray::from_ndarray(v)?)))
            .collect::<Result<BTreeMap<String, SafeArray<'a>>>>()?;
        Ok(Self {
            tensors,
            metadata: None,
        })
    }
}

// impl<'a, K: AsRef<str>, T: IntoIterator<Item = (K, SafeArray<'a>)>> From<T> for SafeArrays<'a> {
//     fn from(iter: T) -> Self {
//         let tensors = iter
//             .into_iter()
//             .map(|(k, v)| (k.as_ref().to_owned(), v))
//             .collect();
//         Self {
//             tensors,
//             metadata: None,
//         }
//     }
// }

impl<'a> SafeArrays<'a> {
    /// Create a new empty SafeArrays builder
    pub const fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            metadata: None,
        }
    }

    /// Insert a SafeArray tensor with the given name
    pub fn insert_tensor<'b: 'a>(&mut self, name: impl AsRef<str>, tensor: SafeArray<'b>) {
        self.tensors.insert(name.as_ref().to_owned(), tensor);
    }

    /// Insert an ndarray tensor with the given name
    ///
    /// The array must be in standard layout and contiguous.
    pub fn insert_ndarray<'b: 'a, T: STDtype, D: ndarray::Dimension + 'a>(
        &mut self,
        name: impl AsRef<str>,
        array: ndarray::ArrayView<'b, T, D>,
    ) -> Result<()> {
        self.insert_tensor(name, SafeArray::from_ndarray(array)?);
        Ok(())
    }

    /// Insert metadata key-value pair
    pub fn insert_metadata(&mut self, key: impl AsRef<str>, value: impl AsRef<str>) {
        self.metadata
            .get_or_insert_default()
            .insert(key.as_ref().to_owned(), value.as_ref().to_owned());
    }

    /// Serialize all tensors and metadata to bytes
    pub fn serialize(self) -> Result<Vec<u8>> {
        let out = safetensors::serialize(self.tensors, self.metadata)
            .map_err(SafeTensorError::SafeTensor)?;
        Ok(out)
    }
}

/// A tensor that can be serialized to SafeTensors format
#[derive(Debug, Clone)]
pub struct SafeArray<'a> {
    data: Cow<'a, [u8]>,
    shape: Vec<usize>,
    dtype: safetensors::tensor::Dtype,
}

impl View for SafeArray<'_> {
    fn dtype(&self) -> safetensors::tensor::Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        self.data.clone()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl<'a> SafeArray<'a> {
    fn from_ndarray<'b: 'a, T: STDtype, D: ndarray::Dimension + 'a>(
        array: ndarray::ArrayView<'b, T, D>,
    ) -> Result<Self> {
        let shape = array.shape().to_vec();
        let dtype = T::dtype();
        if array.ndim() == 0 {
            return Err(SafeTensorError::InvalidTensorData(
                core::any::type_name::<T>(),
                "Cannot insert a scalar tensor".to_string(),
            ));
        }

        if !array.is_standard_layout() {
            return Err(SafeTensorError::InvalidTensorData(
                core::any::type_name::<T>(),
                "ArrayView is not standard layout".to_string(),
            ));
        }
        let data =
            bytemuck::cast_slice(array.to_slice().ok_or(SafeTensorError::InvalidTensorData(
                core::any::type_name::<T>(),
                "ArrayView is not contiguous".to_string(),
            ))?);
        let safe_array = SafeArray {
            data: Cow::Borrowed(data),
            shape,
            dtype,
        };
        Ok(safe_array)
    }
}

#[test]
fn test_safe_array_from_ndarray() {
    use ndarray::Array2;

    let array = Array2::<f32>::zeros((3, 4));
    let safe_array = SafeArray::from_ndarray(array.view()).unwrap();
    assert_eq!(safe_array.shape, vec![3, 4]);
    assert_eq!(safe_array.dtype, safetensors::tensor::Dtype::F32);
    assert_eq!(safe_array.data.len(), 3 * 4 * 4); // 3x4x4 bytes for f32
}

#[test]
fn test_serialize_safe_arrays() {
    use ndarray::{Array2, Array3};

    let mut safe_arrays = SafeArrays::new();
    let array = Array2::<f32>::zeros((3, 4));
    let array2 = Array3::<u16>::zeros((8, 1, 9));
    safe_arrays
        .insert_ndarray("test_tensor", array.view())
        .unwrap();
    safe_arrays
        .insert_ndarray("test_tensor2", array2.view())
        .unwrap();
    safe_arrays.insert_metadata("author", "example");

    let serialized = safe_arrays.serialize().unwrap();
    assert!(!serialized.is_empty());

    // Deserialize to check if it works
    let deserialized = SafeArraysView::from_bytes(&serialized).unwrap();
    assert_eq!(deserialized.len(), 2);
    assert_eq!(
        deserialized
            .tensor::<f32, ndarray::Ix2>("test_tensor")
            .unwrap()
            .shape(),
        &[3, 4]
    );
    assert_eq!(
        deserialized
            .tensor::<u16, ndarray::Ix3>("test_tensor2")
            .unwrap()
            .shape(),
        &[8, 1, 9]
    );
}
