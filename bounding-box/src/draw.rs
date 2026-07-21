use crate::{roi::RoiMut, *};
pub use color::Rgba8;

use ndarray::{Array1, Array3, ArrayViewMut3};

pub trait Draw<T> {
    fn draw(&mut self, item: &T, color: color::Rgba8, thickness: usize);
}

impl Draw<Aabb2<usize>> for Array3<u8> {
    fn draw(&mut self, item: &Aabb2<usize>, color: color::Rgba8, thickness: usize) {
        item.draw(self, color, thickness)
    }
}

pub trait Drawable<Canvas> {
    fn draw(&self, canvas: &mut Canvas, color: color::Rgba8, thickness: usize);
}

/// Implementing Drawable for `Aabb2` with `Array3<u8>` as the canvas type
/// Assuming `Array3<u8>` is a 3D array representing an image with RGB/RGBA channels
impl Drawable<ArrayViewMut3<'_, u8>> for Aabb2<usize> {
    fn draw(&self, canvas: &mut ArrayViewMut3<u8>, color: color::Rgba8, thickness: usize) {
        use itertools::Itertools;
        // let (height, width, channels) = canvas.dim();
        let color = Array1::from_vec(vec![color.r, color.g, color.b, color.a]);
        self.corners()
            .iter()
            .zip(self.padding_uniform(thickness).corners())
            .cycle()
            .take(5)
            .tuple_windows()
            .for_each(|((a, b), (c, d))| {
                let bbox = Aabb2::from_vertices([*a, b, *c, d]).expect("Invalid bounding box");
                use crate::roi::RoiMut;
                let mut out = canvas.roi_mut(bbox).expect("Failed to get ROI");
                out.lanes_mut(ndarray::Axis(2))
                    .into_iter()
                    .for_each(|mut pixel| {
                        pixel.assign(&color);
                    });
            });
    }
}

impl Drawable<Array3<u8>> for Aabb2<usize> {
    fn draw(&self, canvas: &mut Array3<u8>, color: color::Rgba8, thickness: usize) {
        let color = Array1::from_vec(vec![color.r, color.g, color.b, color.a]);
        let pixel_size = canvas.dim().2;
        let color = color.slice(ndarray::s![..pixel_size]);
        let [x1y1, x2y1, x2y2, x1y2] = self.corners();
        let top = Aabb2::from_x1y1x2y2(x1y1.x, x1y1.y, x2y1.x, x2y1.y + thickness);
        let bottom = Aabb2::from_x1y1x2y2(x1y2.x, x1y2.y, x2y2.x, x2y2.y + thickness);
        let left = Aabb2::from_x1y1x2y2(x1y1.x, x1y1.y, x1y2.x + thickness, x1y2.y);
        let right = Aabb2::from_x1y1x2y2(x2y1.x, x2y1.y, x2y2.x + thickness, x2y2.y + thickness);
        let canvas_bbox = Aabb2::from_x1y1x2y2(0, 0, canvas.dim().1 - 1, canvas.dim().0 - 1);
        let lines = [top, bottom, left, right].map(|bbox| bbox.clamp(canvas_bbox));
        lines.into_iter().flatten().for_each(|line| {
            canvas
                .roi_mut(line)
                .map(|mut line| {
                    line.lanes_mut(ndarray::Axis(2))
                        .into_iter()
                        .for_each(|mut pixel| {
                            pixel.assign(&color);
                        })
                })
                .inspect_err(|_e| {
                    #[cfg(feature = "tracing")]
                    tracing::error!("{_e}")
                })
                .ok();
        });
    }
}

#[cfg(test)]
mod draw_tests {
    use super::*;
    use ndarray::{Array3, ArrayViewMut3};

    fn red() -> Rgba8 {
        Rgba8 {
            r: 255,
            g: 10,
            b: 20,
            a: 255,
        }
    }

    #[test]
    fn draw_box_colors_borders_and_leaves_interior() {
        let mut canvas = Array3::<u8>::zeros((10, 10, 4));
        let bbox = Aabb2::from_x1y1x2y2(2usize, 2, 6, 6);
        // Uses `Draw for Array3<u8>`, which delegates to `Drawable for Array3<u8>`.
        canvas.draw(&bbox, red(), 1);

        // Something was drawn.
        assert!(canvas.iter().any(|&v| v != 0));
        // Each of the four edges is coloured (channel 0 = red = 255).
        assert_eq!(canvas[[2, 4, 0]], 255, "top edge");
        assert_eq!(canvas[[6, 4, 0]], 255, "bottom edge");
        assert_eq!(canvas[[4, 2, 0]], 255, "left edge");
        assert_eq!(canvas[[4, 6, 0]], 255, "right edge");
        // Bottom-right corner: only covered when the right edge extends the full
        // height (y2 = x2y2.y + thickness), pinning that offset.
        assert_eq!(canvas[[6, 6, 0]], 255, "right edge full height");
        // A coloured pixel carries the whole colour, not just red.
        assert_eq!(canvas[[2, 4, 1]], 10);
        // The interior is untouched.
        assert_eq!(canvas[[4, 4, 0]], 0, "interior");
    }

    #[test]
    fn draw_on_view_mut_writes_pixels() {
        let mut canvas = Array3::<u8>::zeros((20, 20, 4));
        let bbox = Aabb2::from_x1y1x2y2(5usize, 5, 15, 15);
        {
            let mut view = canvas.view_mut();
            // Exercise `Drawable for ArrayViewMut3<u8>` specifically.
            <Aabb2<usize> as Drawable<ArrayViewMut3<u8>>>::draw(&bbox, &mut view, red(), 4);
        }
        assert!(canvas.iter().any(|&v| v != 0));
    }
}
