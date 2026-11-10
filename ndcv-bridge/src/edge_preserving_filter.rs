//! <https://docs.rs/opencv/latest/opencv/photo/fn.edge_preserving_filter.html>
use crate::conversions::*;
use ndarray::*;
use opencv::core::MatTraitConst;

#[derive(Debug, thiserror::Error)]
pub enum EdgePreservingFilterError {
    #[error("Conversion error: {0}")]
    ConversionError(#[from] crate::conversions::ConversionError),
    #[error("OpenCV error: {0}")]
    OpenCvError(#[from] opencv::Error),
    #[error("expected a 3-channel image, got {0} channels")]
    ChannelMismatch(usize),
    #[error("OpenCV wrote to a buffer of its own instead of the output array")]
    OutputReallocated,
}

/// Which of Gastal and Oliveira's domain-transform filters to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum EdgePreservingFilterKind {
    /// `cv::RECURS_FILTER` (`flags=1`), cv2's default. Faster.
    #[default]
    Recursive = opencv::photo::RECURS_FILTER,
    /// `cv::NORMCONV_FILTER` (`flags=2`). Smooths a little more evenly.
    NormalizedConvolution = opencv::photo::NORMCONV_FILTER,
}

/// OpenCV's `edgePreservingFilter` from the `photo` module: a domain-transform
/// smoother that flattens texture while keeping strong colour edges.
///
/// Only 3-channel `u8` images are accepted, which is what the OpenCV
/// implementation is written for (it always writes `CV_8UC3`). Channels are
/// not filtered independently: an edge is measured across all three at once.
pub trait NdCvEdgePreservingFilter {
    /// - `kind`: [`EdgePreservingFilterKind`], cv2's `flags`.
    /// - `sigma_s`: spatial extent, in pixels. OpenCV documents `0..=200`.
    /// - `sigma_r`: how large a colour difference counts as an edge, on the
    ///   image scaled to `[0, 1]`. OpenCV documents `0..=1`; smaller keeps
    ///   more edges.
    ///
    /// An image without 3 channels is
    /// [`EdgePreservingFilterError::ChannelMismatch`].
    fn edge_preserving_filter(
        &self,
        kind: EdgePreservingFilterKind,
        sigma_s: f32,
        sigma_r: f32,
    ) -> Result<Array3<u8>, EdgePreservingFilterError>;

    /// cv2's defaults: `flags=1` (recursive), `sigma_s=60`, `sigma_r=0.4`.
    fn edge_preserving_filter_def(&self) -> Result<Array3<u8>, EdgePreservingFilterError> {
        self.edge_preserving_filter(EdgePreservingFilterKind::Recursive, 60.0, 0.4)
    }
}

impl<S: ndarray::Data<Elem = u8>> NdCvEdgePreservingFilter for ArrayBase<S, Ix3> {
    fn edge_preserving_filter(
        &self,
        kind: EdgePreservingFilterKind,
        sigma_s: f32,
        sigma_r: f32,
    ) -> Result<Array3<u8>, EdgePreservingFilterError> {
        let channels = self.len_of(Axis(2));
        if channels != 3 {
            return Err(EdgePreservingFilterError::ChannelMismatch(channels));
        }
        let mut dst = Array3::zeros(self.raw_dim());
        // OpenCV asserts on empty input; an empty image filters to itself.
        if self.is_empty() {
            return Ok(dst);
        }
        // The Mat conversion keeps only the outer stride, so input that is not
        // in standard layout is copied first. Contiguous input is borrowed, but
        // ndarray leaves the stride of a length-1 axis arbitrary (0 for a
        // one-row slice), which OpenCV can reject; a view over the same slice
        // has the default strides, without a copy.
        let src = self.as_standard_layout();
        let src = match src.as_slice() {
            Some(data) => {
                ArrayView3::from_shape(self.raw_dim(), data).map_err(ConversionError::from)?
            }
            None => src.view(),
        };
        let cv_src = src.as_image_mat()?;
        let dst_data = dst.as_ptr();
        let mut cv_dst = dst.as_image_mat_mut()?;
        opencv::photo::edge_preserving_filter(
            &*cv_src,
            &mut *cv_dst,
            kind as i32,
            sigma_s,
            sigma_r,
        )?;
        // Had OpenCV wanted a different size or type it would have written into
        // a new buffer and left `dst` all zeros.
        if cv_dst.data() != dst_data {
            return Err(EdgePreservingFilterError::OutputReallocated);
        }
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(h: usize, w: usize, c: usize) -> Array3<u8> {
        Array3::from_shape_fn((h, w, c), |(y, x, ch)| {
            u8::try_from((y * 37 + x * 11 + ch * 71 + (x * y) % 13) % 256).unwrap()
        })
    }

    fn assert_close(res: &Array3<u8>, sum: i64, pixels: &[((usize, usize), [u8; 3])]) {
        // Values from cv2 on OpenCV 5.0. Allow a level of rounding difference
        // between OpenCV versions.
        for &((y, x), want) in pixels {
            for (c, &w) in want.iter().enumerate() {
                let got = res[[y, x, c]];
                assert!(got.abs_diff(w) <= 1, "({y}, {x}, {c}): got {got}, want {w}");
            }
        }
        let got: i64 = res.iter().map(|&v| i64::from(v)).sum();
        let slack = i64::try_from(res.len() / 10).unwrap();
        assert!((got - sum).abs() < slack, "sum {got}, want {sum}");
    }

    #[test]
    fn test_edge_preserving_recursive_matches_cv2() {
        // cv2.edgePreservingFilter(t, flags=1, sigma_s=40, sigma_r=0.2), the
        // call face/neck smoothening makes.
        let t = texture(40, 40, 3);
        let res = t
            .edge_preserving_filter(EdgePreservingFilterKind::Recursive, 40.0, 0.2)
            .unwrap();
        assert_close(
            &res,
            609_036,
            &[
                ((20, 20), [207, 22, 93]),
                ((0, 0), [6, 77, 148]),
                ((39, 17), [90, 161, 232]),
            ],
        );
    }

    #[test]
    fn test_edge_preserving_normconv_matches_cv2() {
        let t = texture(40, 40, 3);
        let res = t
            .edge_preserving_filter(EdgePreservingFilterKind::NormalizedConvolution, 40.0, 0.2)
            .unwrap();
        assert_close(
            &res,
            611_603,
            &[
                ((20, 20), [214, 29, 100]),
                ((0, 0), [14, 85, 156]),
                ((39, 17), [89, 160, 230]),
            ],
        );
    }

    #[test]
    fn test_edge_preserving_def_matches_cv2_defaults() {
        let t = texture(40, 40, 3);
        let res = t.edge_preserving_filter_def().unwrap();
        assert_close(&res, 600_798, &[((20, 20), [213, 33, 101])]);
    }

    #[test]
    fn test_edge_preserving_keeps_a_step_edge() {
        let mut img = Array3::<u8>::zeros((20, 20, 3));
        img.slice_mut(s![..10, .., ..]).fill(220);
        let res = img
            .edge_preserving_filter(EdgePreservingFilterKind::Recursive, 40.0, 0.2)
            .unwrap();
        assert!(
            res[[9, 10, 0]] >= 210,
            "above the edge: {}",
            res[[9, 10, 0]]
        );
        assert!(
            res[[10, 10, 0]] <= 10,
            "below the edge: {}",
            res[[10, 10, 0]]
        );
    }

    #[test]
    fn test_edge_preserving_non_standard_layout() {
        let t = texture(10, 12, 3);
        let flipped = t.slice(s![.., ..;-1, ..]);
        let res = flipped
            .edge_preserving_filter(EdgePreservingFilterKind::Recursive, 20.0, 0.3)
            .unwrap();
        let expected = flipped
            .to_owned()
            .edge_preserving_filter(EdgePreservingFilterKind::Recursive, 20.0, 0.3)
            .unwrap();
        assert_eq!(res, expected);

        let row = t.slice(s![3..4, .., ..]);
        let res = row.edge_preserving_filter_def().unwrap();
        assert_eq!(res.shape(), &[1, 12, 3]);
    }

    #[test]
    fn test_edge_preserving_rejects_other_channel_counts() {
        for c in [1, 4] {
            let img = texture(8, 8, c);
            assert!(matches!(
                img.edge_preserving_filter_def(),
                Err(EdgePreservingFilterError::ChannelMismatch(n)) if n == c
            ));
        }
    }

    #[test]
    fn test_edge_preserving_empty() {
        let empty = Array3::<u8>::zeros((0, 0, 3));
        assert_eq!(
            empty.edge_preserving_filter_def().unwrap().shape(),
            &[0, 0, 3]
        );
    }
}
