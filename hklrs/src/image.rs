use crate::Error;

/// A whole image, row-major over its (slow, fast) shape: a raw frame, a
/// mask, a dark rate or flat field. It states its shape, so wherever it
/// meets another shape the two shapes are compared, and a transposed one of
/// the same pixel count is refused.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    shape: (usize, usize),
    data: Vec<T>,
}

impl<T> Image<T> {
    /// `data` row-major over `shape`, `shape.0 · shape.1` values.
    pub fn new(shape: (usize, usize), data: Vec<T>) -> Result<Self, Error> {
        if shape.0.checked_mul(shape.1) != Some(data.len()) {
            return Err(Error::PixelCount {
                what: "image",
                len: data.len(),
                expected: shape.0.saturating_mul(shape.1),
            });
        }
        Ok(Self { shape, data })
    }

    /// `f(slow, fast)` at every pixel of `shape`.
    pub fn from_fn(shape: (usize, usize), mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(shape.0.saturating_mul(shape.1));
        for slow in 0..shape.0 {
            for fast in 0..shape.1 {
                data.push(f(slow, fast));
            }
        }
        Self { shape, data }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// The values, writable in place; their number stays the shape's.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Refuses another shape than `expected`, naming the image `what`.
    pub(crate) fn check(&self, what: &'static str, expected: (usize, usize)) -> Result<(), Error> {
        if self.shape == expected {
            Ok(())
        } else {
            Err(Error::ShapeMismatch {
                what,
                shape: self.shape,
                expected,
            })
        }
    }
}

impl<T: Copy> Image<T> {
    /// The value at pixel (slow, fast); panics outside the shape.
    pub fn get(&self, slow: usize, fast: usize) -> T {
        assert!(slow < self.shape.0 && fast < self.shape.1);
        self.data[slow * self.shape.1 + fast]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_refuses_another_pixel_count_and_an_overflowing_shape() {
        assert_eq!(
            Image::new((2, 3), vec![0.0; 5]),
            Err(Error::PixelCount {
                what: "image",
                len: 5,
                expected: 6
            })
        );
        assert_eq!(
            Image::new((usize::MAX, 2), vec![false; 2]),
            Err(Error::PixelCount {
                what: "image",
                len: 2,
                expected: usize::MAX
            })
        );
        let image = Image::new((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            (image.shape(), image.data()),
            ((2, 3), &[1, 2, 3, 4, 5, 6][..])
        );
    }

    #[test]
    fn from_fn_is_row_major() {
        let image = Image::from_fn((2, 3), |s, f| 10 * s + f);
        assert_eq!(
            image,
            Image::new((2, 3), vec![0, 1, 2, 10, 11, 12]).unwrap()
        );
        assert_eq!(image.get(1, 2), 12);
    }

    /// (0, 3) is the fourth value of a (2, 3) image, but outside its shape.
    #[test]
    #[should_panic]
    fn get_outside_the_shape_panics() {
        Image::from_fn((2, 3), |s, f| 10 * s + f).get(0, 3);
    }

    #[test]
    fn check_refuses_a_transposed_shape() {
        let image = Image::from_fn((2, 3), |_, _| false);
        assert_eq!(image.check("mask", (2, 3)), Ok(()));
        assert_eq!(
            image.check("mask", (3, 2)),
            Err(Error::ShapeMismatch {
                what: "mask",
                shape: (2, 3),
                expected: (3, 2)
            })
        );
    }
}
