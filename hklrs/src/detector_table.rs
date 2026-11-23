//! The binoculars detector table (binoculars-ng at hkl 7be4f79,
//! `hkl-binoculars-detectors-2d*`) as data. A row holds module size, gap,
//! pitch and flips; positions, masks and the [`PixelDetector`] are generated
//! from it.
//!
//! The frame is binoculars' own: x is the beam-parallel axis (0 for every
//! detector except Cirpad), pixel (slow, fast) is image row, column, and
//! columns run along -y, rows along ±z, from a pixel corner at the origin.
//! Metres.

use nalgebra::{Point3, Rotation3, Vector3};

use crate::{Error, Image, Module, PixelAxis, PixelDetector};

/// Detector ids; the discriminants are the binoculars `HklBinocularsDetectorEnum`
/// values, so a stored id keeps its meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Detector {
    ImxpadS140 = 0,
    XpadFlatCorrected = 1,
    ImxpadS70 = 2,
    Eiger1M = 3,
    Ufxc = 4,
    Merlin = 5,
    MerlinMedipix3RxQuad = 6,
    MerlinMedipix3RxQuad512 = 7,
    Cirpad = 8,
    RigakuXspa1M = 9,
    PilatusCdte2M = 10,
}

impl Detector {
    pub const ALL: [Detector; 11] = [
        Self::ImxpadS140,
        Self::XpadFlatCorrected,
        Self::ImxpadS70,
        Self::Eiger1M,
        Self::Ufxc,
        Self::Merlin,
        Self::MerlinMedipix3RxQuad,
        Self::MerlinMedipix3RxQuad512,
        Self::Cirpad,
        Self::RigakuXspa1M,
        Self::PilatusCdte2M,
    ];
}

impl TryFrom<u32> for Detector {
    type Error = Error;

    fn try_from(id: u32) -> Result<Self, Error> {
        Self::ALL
            .get(id as usize)
            .copied()
            .ok_or(Error::UnknownDetector {
                id,
                count: Self::ALL.len(),
            })
    }
}

/// A band of masked columns (`vertical`) or rows starting at `origin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strip {
    pub vertical: bool,
    pub origin: usize,
    pub thickness: usize,
}

const fn vert(origin: usize, thickness: usize) -> Strip {
    Strip {
        vertical: true,
        origin,
        thickness,
    }
}

const fn horz(origin: usize, thickness: usize) -> Strip {
    Strip {
        vertical: false,
        origin,
        thickness,
    }
}

/// Modules of `module` pixels (slow, fast) separated by `gap` pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tiling {
    pub module: (usize, usize),
    pub gap: (usize, usize),
    /// The image holds the gap pixels. Otherwise the gap is only a jump in
    /// the pixel positions and module `k` starts at image index `module * k`.
    pub gap_in_data: bool,
    pub gap_masked: bool,
    /// Pixels masked along every module edge.
    pub border: usize,
    /// Pixels masked on the module centre column; 0 for none.
    pub centre_strip: usize,
}

impl Tiling {
    /// Image index where module `k` starts along an axis of module size
    /// `module` and gap `gap`.
    fn origin(&self, k: usize, module: usize, gap: usize) -> usize {
        (module + if self.gap_in_data { gap } else { 0 }) * k
    }

    /// Image indices of the modules starting inside `len`.
    fn origins(&self, len: usize, module: usize, gap: usize) -> impl Iterator<Item = usize> {
        let t = *self;
        (0..)
            .map(move |k| t.origin(k, module, gap))
            .take_while(move |&o| o < len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layout {
    /// Square pixels on a regular grid; a [`Tiling`] with the gap in the data
    /// is also regular.
    Uniform,
    /// imXpad chips of `chip` pixels (slow, fast): the two pixels either side
    /// of an interior chip boundary are 2.5 pitches wide, as pyFAI has them
    /// (binoculars places them otherwise, BIN-12).
    Chips { chip: (usize, usize) },
    /// 20 imXpad S70 modules (120 x 560) placed on a circle.
    Cirpad,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    pub name: &'static str,
    /// (slow, fast) = (height, width).
    pub shape: (usize, usize),
    pub pitch: f64,
    pub flip_z: bool,
    pub layout: Layout,
    pub tiling: Option<Tiling>,
    /// Masked bands that follow from no module structure.
    pub strips: &'static [Strip],
}

const fn tiling(
    module: (usize, usize),
    gap: (usize, usize),
    gap_in_data: bool,
    gap_masked: bool,
    border: usize,
    centre_strip: usize,
) -> Option<Tiling> {
    Some(Tiling {
        module,
        gap,
        gap_in_data,
        gap_masked,
        border,
        centre_strip,
    })
}

const fn row(name: &'static str, shape: (usize, usize), pitch: f64, flip_z: bool) -> Row {
    Row {
        name,
        shape,
        pitch,
        flip_z,
        layout: Layout::Uniform,
        tiling: None,
        strips: &[],
    }
}

/// Indexed by the [`Detector`] discriminant.
static ROWS: [Row; 11] = [
    Row {
        layout: Layout::Chips { chip: (120, 80) },
        ..row("ImXpadS140", (240, 560), 130e-6, false)
    },
    Row {
        // Seven bad row bands of 30 starting at 118 + 148 k.
        strips: &[
            horz(118, 30),
            horz(266, 30),
            horz(414, 30),
            horz(562, 30),
            horz(710, 30),
            horz(858, 30),
            horz(1006, 30),
        ],
        ..row("XpadFlatCorrected", (1154, 576), 130e-6, false)
    },
    Row {
        layout: Layout::Chips { chip: (120, 80) },
        ..row("ImXpadS70", (120, 560), 130e-6, false)
    },
    Row {
        tiling: tiling((514, 1030), (37, 10), true, true, 0, 0),
        ..row("Eiger1M", (1065, 1030), 75e-6, true)
    },
    row("Ufxc", (256, 257), 75e-6, false),
    row("Merlin", (256, 256), 55e-6, false),
    Row {
        tiling: tiling((256, 256), (3, 3), true, true, 0, 0),
        ..row("MerlinMedipix3RXQuad", (515, 515), 55e-6, false)
    },
    Row {
        tiling: tiling((256, 256), (3, 3), false, false, 2, 0),
        ..row("MerlinMedipix3RXQuad512", (512, 512), 55e-6, false)
    },
    Row {
        layout: Layout::Cirpad,
        ..row("Cirpad", (2400, 560), 130e-6, false)
    },
    Row {
        // A 5 pixel ring and the 70 rows of the central module gap.
        strips: &[
            vert(0, 5),
            vert(1029, 5),
            horz(0, 5),
            horz(1099, 5),
            horz(517, 70),
        ],
        ..row("RigakuXspa1M", (1104, 1034), 76e-6, true)
    },
    Row {
        tiling: tiling((195, 487), (17, 7), true, true, 0, 3),
        ..row("PilatusCdTe2M", (1679, 1475), 172e-6, true)
    },
];

/// Cirpad module placement: (x translation, rotation about y in degrees),
/// as the `CIRPAD_PARAMETERS` table writes them. Rotation about x is 90.
static CIRPAD_MODULES: [(f64, f64); 20] = [
    (0.64907 - 5.0e-3, 0.0000 + 0.012000),
    (0.64907 - 4.0e-3, -6.7400 + 0.010000),
    (0.64907 - 5.0e-3, -13.480 - 0.000000),
    (0.64907 - 5.1e-3, -20.220 - 0.000000),
    (0.64907 - 5.6e-3, -26.960 - 0.000000),
    (0.64907 - 5.3e-3, -33.700 - 0.050000),
    (0.64907 - 5.5e-3, -40.440 - 0.080000),
    (0.64907 - 5.8e-3, -47.180 - 0.000000),
    (0.64907 - 5.6e-3, -53.920 - 0.070000),
    (0.64907 - 5.8e-3, -60.660 - 0.000000),
    (0.64907 - 6.2e-3, -67.400 - 0.000000),
    (0.64907 - 5.5e-3, -74.140 - 0.000000),
    (0.64907 - 5.5e-3, -80.880 - 0.000000),
    (0.64907 - 5.7e-3, -87.620 - 0.060000),
    (0.64907 - 5.0e-3, -94.360 - 0.000000),
    (0.64907 - 5.0e-3, -101.10 + 0.050000),
    (0.64907 - 5.0e-3, -107.84 - 0.000000),
    (0.64907 - 4.0e-3, -114.58 - 0.000000),
    (0.64907 - 4.0e-3, -121.32 + 0.010000),
    (0.64907 - 4.0e-3, -128.06 + 0.010000),
];

const CIRPAD_MODULE_SHAPE: (usize, usize) = (120, 560);
const CIRPAD_CHIP: (usize, usize) = (120, 80);

/// pyFAI's imXpad pixel edges along `n` pixels in chips of `chip`
/// (`ImXPadS10._calc_pixels_size`, `calc_pixels_edges`), summed in order as
/// numpy's cumsum does, so the centres equal pyFAI's to the bit.
fn chip_edges(n: usize, chip: usize, p: f64) -> Vec<f64> {
    let wide = |i: usize| (1..n / chip).any(|k| i == k * chip - 1 || i == k * chip);
    let widths = (0..n).map(|i| p * if wide(i) { 2.5 } else { 1.0 });
    std::iter::once(0.0)
        .chain(widths.scan(0.0, |e, w| {
            *e += w;
            Some(*e)
        }))
        .collect()
}

impl Detector {
    pub fn row(self) -> &'static Row {
        &ROWS[self as usize]
    }

    pub fn name(self) -> &'static str {
        self.row().name
    }

    /// (slow, fast).
    pub fn shape(self) -> (usize, usize) {
        self.row().shape
    }

    /// The masked pixels, of [`shape`](Self::shape).
    pub fn mask(self) -> Image<bool> {
        let row = self.row();
        let (h, w) = row.shape;
        let mut image = Image::from_fn((h, w), |_, _| false);
        let mask = image.data_mut();
        for s in self.strips() {
            let extent = if s.vertical { w } else { h };
            // Out-of-range bands are dropped, as binoculars does.
            if s.origin >= extent || s.origin + s.thickness > extent {
                continue;
            }
            for k in s.origin..s.origin + s.thickness {
                if s.vertical {
                    (0..h).for_each(|j| mask[j * w + k] = true);
                } else {
                    mask[k * w..(k + 1) * w].fill(true);
                }
            }
        }
        image
    }

    /// Every masked band the row implies, from its chips, its tiling and its
    /// own strips.
    fn strips(self) -> Vec<Strip> {
        let row = self.row();
        let (h, w) = row.shape;
        let mut out = Vec::new();
        if let Layout::Chips { chip } = row.layout {
            // The two pixels either side of every chip boundary.
            for k in 1..w / chip.1 {
                out.extend([vert(k * chip.1 - 1, 1), vert(k * chip.1, 1)]);
            }
            for k in 1..h / chip.0 {
                out.extend([horz(k * chip.0 - 1, 1), horz(k * chip.0, 1)]);
            }
        }
        if let Some(t) = row.tiling {
            let cols = || t.origins(w, t.module.1, t.gap.1);
            let rows = || t.origins(h, t.module.0, t.gap.0);
            if t.gap_in_data && t.gap_masked {
                out.extend(cols().skip(1).map(|o| vert(o - t.gap.1, t.gap.1)));
                out.extend(rows().skip(1).map(|o| horz(o - t.gap.0, t.gap.0)));
            }
            if t.border > 0 {
                for o in cols() {
                    out.extend([vert(o, t.border), vert(o + t.module.1 - t.border, t.border)]);
                }
                for o in rows() {
                    out.extend([horz(o, t.border), horz(o + t.module.0 - t.border, t.border)]);
                }
            }
            if t.centre_strip > 0 {
                out.extend(cols().map(|o| vert(o + t.module.1 / 2 - 1, t.centre_strip)));
            }
        }
        out.extend_from_slice(row.strips);
        out
    }

    /// The detector as a [`PixelDetector`] with its mask.
    pub fn pixel_detector(self) -> PixelDetector {
        self.modules()
            .and_then(|m| PixelDetector::from_modules(self.shape(), m))
            .and_then(|d| d.with_mask(self.mask()))
            .expect("every table row is a valid detector")
    }

    fn modules(self) -> Result<Vec<Module>, Error> {
        let row = self.row();
        let (h, w) = row.shape;
        let p = row.pitch;
        let z = if row.flip_z {
            -Vector3::z_axis()
        } else {
            Vector3::z_axis()
        };
        let y = -Vector3::y_axis();
        let uniform = |dir, n| PixelAxis::uniform(dir, n, p);
        let chips = |dir, n, chip| PixelAxis::edges(dir, chip_edges(n, chip, p));
        match (row.layout, row.tiling) {
            (Layout::Cirpad, _) => {
                let (mh, mw) = CIRPAD_MODULE_SHAPE;
                let (ch, cw) = CIRPAD_CHIP;
                CIRPAD_MODULES
                    .iter()
                    .enumerate()
                    .map(|(m, &(tx, ry))| {
                        // An imXpad S70 translated along x, then rotated z-y-x
                        // by (90 deg, ry, 0).
                        let rot =
                            Rotation3::from_euler_angles(90f64.to_radians(), ry.to_radians(), 0.0);
                        Module::new(
                            (m * mh, 0),
                            rot * Point3::new(tx, 0.0, 0.0),
                            chips(rot * z, mh, ch)?,
                            chips(rot * y, mw, cw)?,
                        )
                    })
                    .collect()
            }
            (Layout::Chips { chip }, _) => Ok(vec![Module::new(
                (0, 0),
                Point3::origin(),
                chips(z, h, chip.0)?,
                chips(y, w, chip.1)?,
            )?]),
            (Layout::Uniform, Some(t)) if !t.gap_in_data => {
                let ((mh, mw), (gh, gw)) = (t.module, t.gap);
                (0..h / mh)
                    .flat_map(|a| (0..w / mw).map(move |b| (a, b)))
                    .map(|(a, b)| {
                        // The gap is not in the image, only between the
                        // modules' positions.
                        let corner = z.into_inner() * ((a * (mh + gh)) as f64 * p)
                            + y.into_inner() * ((b * (mw + gw)) as f64 * p);
                        Module::new(
                            (a * mh, b * mw),
                            Point3::from(corner),
                            uniform(z, mh)?,
                            uniform(y, mw)?,
                        )
                    })
                    .collect()
            }
            (Layout::Uniform, _) => Ok(vec![Module::new(
                (0, 0),
                Point3::origin(),
                uniform(z, h)?,
                uniform(y, w)?,
            )?]),
        }
    }
}
