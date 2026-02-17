use bounding_box::nalgebra::Point2;
use error_stack::ResultExt;
use ndarray::{Array3, ArrayView2, Axis, s};

use super::traits::{Paste, PasteConfig, PasteError};
use super::{AnchoredPos, Bounds, NormalisedPos};

#[derive(Debug, Clone, Copy)]
pub struct ChannelPaster<'a> {
    to_paste: ArrayView2<'a, u8>,
    channel_idx: usize,
    position: AnchoredPos,
}

impl<'a> ChannelPaster<'a> {
    pub fn new(to_paste: ArrayView2<'a, u8>) -> Self {
        Self {
            to_paste: to_paste,
            channel_idx: 4,
            position: AnchoredPos::default(),
        }
    }

    /// On which channel of Array3, to paste the array on.
    /// If channel is out of bounds, a channel will be appeneded to the array
    /// with default values of 255_u8.
    ///
    /// # Example
    /// ```ignore
    /// // Paste at 4th channel
    /// let mask = ndarray::Array2::ones((100, 100)) * 255_u8;
    /// ChannelPaster::new(mask.view()).with_channel_idx(3);
    /// ```
    pub fn with_channel_idx(mut self, channel_idx: usize) -> Self {
        self.channel_idx = channel_idx;
        self
    }

    /// Where should the center of the mask be on the image
    ///
    /// # Example
    /// ```ignore
    /// let mask = ndarray::Array2::ones((100, 100)) * 255_u8;
    /// ChannelPaster::new(mask.view()).with_center_position(NormalisedPos::mid_point());
    /// ```
    pub fn with_position(mut self, pos: AnchoredPos) -> Self {
        self.position = pos;
        self
    }
}

impl<'a, 'b> Paste<ChannelPaster<'b>> for &'a mut Array3<u8>
where
    'a: 'b,
{
    type Out = &'a mut Array3<u8>;

    fn paste(self, other: ChannelPaster) -> error_stack::Result<Self::Out, PasteError> {
        let ChannelPaster {
            to_paste,
            channel_idx,
            position,
        } = other;

        let channel = channel_idx + 1;
        let (mh, mw) = to_paste.dim();
        let (ih, iw, ic) = self.dim();

        let c = if channel > ic {
            let filled_3d_arrray = Array3::ones((ih as usize, iw as usize, 1)) * 255_u8;
            *self = ndarray::concatenate(Axis(2), &[self.view(), filled_3d_arrray.view()])
                .change_context(PasteError)?;
            // TODO: sus
            // .as_standard_layout()
            // .to_owned();

            self.dim().2
        } else {
            channel
        };

        let img_bounds = Bounds::from_dim(ih, iw);
        let mask_bounds = Bounds::from_dim(mh, mw);

        let mask_translation = position.get_top_left_pos(img_bounds, mask_bounds);

        let mask_translated = mask_bounds.translate(mask_translation.coords);

        let img_bounds = img_bounds.intersection(mask_translated);

        if let Some(img_bounds) = img_bounds {
            let mask_bounds = img_bounds.translate(-mask_translation.coords);

            // safe to cast to usize because img_bounds lies in the 1st quadrant, which means any
            // intersection with it will also have positive coords
            let mut cropped_src = self.slice_mut(s![
                img_bounds.h_min() as usize..img_bounds.h_max() as usize,
                img_bounds.w_min() as usize..img_bounds.w_max() as usize,
                c
            ]);

            // safe to cast to mask bounds (which were originally in 1st quadrant) were first translated by mask_translation,
            // any intersection with it will also be atleast be moved by this translation,
            // so reversing this translation will result in positive coords
            let cropped_mask = to_paste.slice(s![
                mask_bounds.h_min() as usize..mask_bounds.h_max() as usize,
                mask_bounds.w_min() as usize..mask_bounds.w_max() as usize
            ]);

            // TODO: no need now
            assert_eq!(cropped_src.dim(), cropped_mask.dim());

            cropped_src.assign(&cropped_mask);
        }

        Ok(self)
    }
}
