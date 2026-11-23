//! pyFAI 2026.9.0's detector classes as `Detector.factory` finds them by a
//! PONI file's detector name; golden/pyfai_detectors.py generates the table.

#[rustfmt::skip]
mod table;

use crate::Image;

/// A pyFAI detector class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Class {
    pub(super) name: &'static str,
    /// The constructor's arguments, the only `Detector_config` entries the
    /// factory passes on.
    pub(super) args: &'static [&'static str],
    /// pyFAI's `force_pixel`: a version 1 file's two pixel sizes build this
    /// class rather than the generic detector.
    pub(super) force_pixel: bool,
    pub(super) layout: Layout,
}

/// The class's default instance; (slow, fast) pairs, pixels in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Grid {
    pub(super) pixel: (f64, f64),
    pub(super) shape: (usize, usize),
    pub(super) max_shape: (usize, usize),
    pub(super) binning: (u64, u64),
    /// pyFAI's orientation number.
    pub(super) orientation: u32,
    pub(super) module_size: Option<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Layout {
    /// pyFAI's generic `Detector`, which the file lays out.
    Generic,
    /// The pixels of the generic detector of the instance's pitch, shape
    /// and orientation, with the class's mask; the grid is the default
    /// instance's.
    Uniform(Grid, Pitch, Option<Rule>),
    /// The ImXPad chips, the pixels of the grid's shape and orientation at
    /// the chips' edges.
    Edges(Grid, Pitch, Chips),
    /// Why hklrs has no model of the class.
    Unsupported(&'static str),
}

/// How the class's constructor takes the pitch (pixel1, pixel2) a file
/// gives, by the class that defines `__init__`; a pitch the file omits is
/// the default instance's, the constructor's default argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Pitch {
    /// The constructor takes none, and the factory drops a given one.
    Fixed,
    /// `Detector.__init__`: as given, a 0 or null pitch none.
    AsGiven,
    /// `Detector.__init__` with `force_pixel`: a 0 or null pitch is `size`;
    /// one over `size` bins its axis by the ratio, rounded, and is kept.
    Rounded { size: (f64, f64) },
    /// `Apex2`, `Pixium`: [`Pitch::Rounded`], then a given pitch other than
    /// `unit` on either axis bins both by the ratio to `unit`, rounded.
    Rebinned { size: (f64, f64), unit: (f64, f64) },
    /// `Perkin`: [`Pitch::Rounded`], then a given pitch other than `size` on
    /// either axis bins both by twice the ratio to `unit`, truncated; `size`
    /// itself gives the default instance.
    Perkin { size: (f64, f64), unit: (f64, f64) },
    /// `_Rayonix`: as given, binned by the last binning of the table whose
    /// pitch it is exactly; the binning setter takes the table's pitch of a
    /// binning, else binning 1's divided by the binning.
    Table(&'static [(u64, f64)]),
    /// `Mar345`: as given, `max_shape` the plate's `max_shape` of pitch
    /// `unit` scaled by `unit` over the pitch and truncated.
    Plate {
        max_shape: (usize, usize),
        unit: f64,
    },
}

/// pyFAI's instance as its constructor and the factory's binning key leave
/// it; Python's integers as i64, an unset pitch None.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Built {
    pub(super) pixel: [Option<f64>; 2],
    pub(super) shape: [i64; 2],
    pub(super) max_shape: [i64; 2],
    pub(super) binning: [i64; 2],
}

/// Why hklrs refuses pyFAI's instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Fault {
    /// pyFAI leaves the pitch of this axis unset.
    NoPitch(usize),
    /// The pitch, binning and shape disagree as the class's do not.
    Inconsistent,
}

// Python's float to int conversions; a value beyond i64 saturates, which
// only changes how far out of range a refused instance is.

/// `round(x)`, ties to even; None where it raises.
fn py_round(x: f64) -> Option<i64> {
    x.is_finite().then(|| x.round_ties_even() as i64)
}

/// `int(x)`, truncated; None where it raises.
fn py_int(x: f64) -> Option<i64> {
    x.is_finite().then(|| x.trunc() as i64)
}

/// `a // b`, rounded down; None where it raises.
fn floor_div(a: i64, b: i64) -> Option<i64> {
    let q = a.checked_div(b)?;
    Some(if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    })
}

/// Relative tolerance of a pitch to the class's own: a file's decimal
/// pitch is a rounding of the product pyFAI computes.
const PITCH_REL: f64 = 1e-9;

impl Pitch {
    /// pyFAI's instance of the class of default instance `grid`: its
    /// constructor given `pixel` (0 for a 0 or null), then the factory's
    /// `binning` by the class's setter; None where pyFAI raises.
    pub(super) fn build(
        self,
        grid: &Grid,
        pixel: [Option<f64>; 2],
        binning: Option<(u64, u64)>,
    ) -> Option<Built> {
        let own = [grid.pixel.0, grid.pixel.1];
        let raw = [0, 1].map(|k| pixel[k].unwrap_or(own[k]));
        let max = [grid.max_shape.0 as i64, grid.max_shape.1 as i64];
        let mut b = Built {
            // `float(p) if p else None`
            pixel: raw.map(|p| (p != 0.0).then_some(p)),
            shape: max,
            max_shape: max,
            binning: [1, 1],
        };
        match self {
            Pitch::Fixed => {
                b = Built {
                    pixel: own.map(Some),
                    shape: [grid.shape.0 as i64, grid.shape.1 as i64],
                    max_shape: max,
                    binning: [grid.binning.0 as i64, grid.binning.1 as i64],
                }
            }
            Pitch::AsGiven => {}
            Pitch::Rounded { size } => b.rounded(size)?,
            Pitch::Rebinned { size, unit } => {
                b.rounded(size)?;
                let unit = [unit.0, unit.1];
                if raw != unit {
                    b.rebin([0, 1].map(|k| py_round(raw[k] / unit[k])))?;
                }
            }
            Pitch::Perkin { size, unit } => {
                b.rounded(size)?;
                if raw != [size.0, size.1] {
                    let unit = [unit.0, unit.1];
                    b.rebin([0, 1].map(|k| py_int(2.0 * raw[k] / unit[k])))?;
                } else {
                    b.shape = [grid.shape.0 as i64, grid.shape.1 as i64];
                    b.binning = [grid.binning.0 as i64, grid.binning.1 as i64];
                }
            }
            Pitch::Table(table) => {
                b.rebin(raw.map(|p| {
                    let found = table.iter().rev().find(|&&(_, q)| q == p);
                    Some(found.map_or(1, |&(n, _)| n as i64))
                }))?;
            }
            Pitch::Plate { max_shape, unit } => {
                for (k, n) in [max_shape.0, max_shape.1].into_iter().enumerate() {
                    b.max_shape[k] = py_int(n as f64 * unit / b.pixel[k]?)?;
                }
                b.shape = b.max_shape;
            }
        }
        match binning {
            Some((b0, b1)) if [b0 as i64, b1 as i64] != b.binning => {
                b.bin([b0 as i64, b1 as i64], self)?
            }
            _ => {}
        }
        Some(b)
    }
}

impl Built {
    /// `Detector.__init__`'s `force_pixel` branch.
    fn rounded(&mut self, size: (f64, f64)) -> Option<()> {
        for (k, size) in [(0, size.0), (1, size.1)] {
            match self.pixel[k] {
                None => self.pixel[k] = Some(size),
                Some(p) if p / size > 1.0 => {
                    self.binning[k] = py_round(p / size)?;
                    self.shape[k] = floor_div(self.max_shape[k], self.binning[k])?;
                }
                Some(_) => {}
            }
        }
        Some(())
    }

    /// Both axes binned `binning`, `max_shape` floor divided.
    fn rebin(&mut self, binning: [Option<i64>; 2]) -> Option<()> {
        self.binning = [binning[0]?, binning[1]?];
        for k in 0..2 {
            self.shape[k] = floor_div(self.max_shape[k], self.binning[k])?;
        }
        Some(())
    }

    /// The `binning` setter for a binning other than the instance's.
    fn bin(&mut self, binning: [i64; 2], pitch: Pitch) -> Option<()> {
        for k in 0..2 {
            let p = match pitch {
                Pitch::Table(table) => match table.iter().find(|&&(n, _)| n as i64 == binning[k]) {
                    Some(&(_, p))
                        if binning
                            .iter()
                            .all(|&b| table.iter().any(|&(n, _)| n as i64 == b)) =>
                    {
                        p
                    }
                    _ => {
                        let one = table.iter().find(|&&(n, _)| n == 1)?.1;
                        (binning[k] != 0).then(|| one / binning[k] as f64)?
                    }
                },
                _ => {
                    let ratio = (self.binning[k] != 0)
                        .then(|| binning[k] as f64 / self.binning[k] as f64)?;
                    self.pixel[k]? * ratio
                }
            };
            self.pixel[k] = Some(p);
        }
        self.rebin(binning.map(Some))
    }

    /// The grid of this instance of the class of default instance `grid`,
    /// where its pitch, binning and shape agree as the class's do: the
    /// class's pitch at the binning (a Rayonix's table, the default's
    /// scaled by the binning, any pitch of a Mar345 whose plate is at most
    /// the class's `max_shape`). `edges`:
    /// ImXPad's chip edges stay the default instance's whatever the pitch
    /// and binning, so only its binning agrees.
    pub(super) fn grid(&self, grid: &Grid, pitch: Pitch, edges: bool) -> Result<Grid, Fault> {
        let px = [0, 1].map(|k| self.pixel[k].ok_or(Fault::NoPitch(k)));
        let px = [px[0]?, px[1]?];
        let own = [
            (grid.pixel.0, grid.binning.0 as i64),
            (grid.pixel.1, grid.binning.1 as i64),
        ];
        let agrees = (0..2).all(|k| {
            let b = self.binning[k];
            let want = match pitch {
                Pitch::Table(table) => table.iter().find(|&&(n, _)| n as i64 == b).map(|&(_, p)| p),
                // any pitch, the plate's max_shape scaled to it
                Pitch::Plate { max_shape, .. } => {
                    let plate = [max_shape.0, max_shape.1][k] as i64;
                    (1..=plate).contains(&self.max_shape[k]).then_some(px[k])
                }
                _ => Some(own[k].0 * b as f64 / own[k].1 as f64),
            };
            b >= 1
                && px[k].is_finite()
                && px[k] > 0.0
                && want.is_some_and(|w| (px[k] - w).abs() <= PITCH_REL * w)
                && (!edges || b == own[k].1)
        });
        if !agrees {
            return Err(Fault::Inconsistent);
        }
        let n = |v: i64| usize::try_from(v).map_err(|_| Fault::Inconsistent);
        Ok(Grid {
            pixel: (px[0], px[1]),
            shape: (n(self.shape[0])?, n(self.shape[1])?),
            max_shape: (n(self.max_shape[0])?, n(self.max_shape[1])?),
            binning: (self.binning[0] as u64, self.binning[1] as u64),
            ..*grid
        })
    }
}

/// The class's `calc_mask`, in image indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Rule {
    /// `_Dectris`, `_Lambda` and `Maxipix`: the `gap` rows and columns after
    /// each module; a `strip` masks columns [lo, hi) about each module's
    /// middle column too.
    Gaps {
        module: (usize, usize),
        gap: (usize, usize),
        strip: Option<(isize, isize)>,
    },
    /// `Mar345`, `Rayonix133`, `RayonixSx165`: outside the disc of radius
    /// half the slow shape.
    Disc,
    /// `Lambda9M`: nothing masked, though over `max_shape`, which a
    /// binning must divide.
    Empty,
}

/// pyFAI's `ImXPadS10`: chips of `module` pixels, the pixels either side
/// of an interior chip boundary `border` pitches wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Chips {
    pub(super) module: (usize, usize),
    pub(super) border: f64,
}

/// The value `key` maps to in a sorted table.
fn find(map: &[(&str, &'static str)], key: &str) -> Option<&'static str> {
    map.binary_search_by_key(&key, |&(k, _)| k)
        .ok()
        .map(|i| map[i].1)
}

impl Class {
    /// The class `Detector.factory` returns for `name`, or None where it
    /// raises: an attribute of `pyFAI.detectors` by its exact name, else a
    /// registry key by the lower-case name, as given or with spaces as `_`.
    pub(super) fn resolve(name: &str) -> Option<&'static Class> {
        let lower = name.to_lowercase();
        let class = find(table::ATTRIBUTES, name)
            .or_else(|| find(table::REGISTRY, &lower))
            .or_else(|| find(table::REGISTRY, &lower.replace(' ', "_")))?;
        Some(Self::named(class))
    }

    pub(super) fn generic() -> &'static Class {
        Self::named("Detector")
    }

    fn named(name: &str) -> &'static Class {
        let i = table::CLASSES
            .binary_search_by_key(&name, |c| c.name)
            .expect("the table maps names to its own classes");
        &table::CLASSES[i]
    }
}

/// `line[i + lo : i + hi] = True` for i in `range(first, len, step)`, cut at
/// the end as a slice is.
fn mark(line: &mut [bool], first: usize, step: usize, (lo, hi): (isize, isize)) {
    let n = line.len();
    for i in (first..n).step_by(step) {
        let end = i.saturating_add_signed(hi).min(n);
        line[i.saturating_add_signed(lo).min(end)..end].fill(true);
    }
}

/// The pixels in a marked row or a marked column.
fn cross(rows: &[bool], cols: &[bool]) -> Image<bool> {
    Image::from_fn((rows.len(), cols.len()), |s, f| rows[s] || cols[f])
}

impl Chips {
    /// `calc_pixels_edges` along pyFAI's axes 1 and 2 before the
    /// orientation reverses them: from 0, one more edge than the grid's
    /// `max_shape` has pixels.
    pub(super) fn edges(self, grid: &Grid) -> [Vec<f64>; 2] {
        let axis = |n: usize, m: usize, pitch: f64| {
            let mut size = vec![1.0; n];
            for i in 1..n / m {
                size[i * m - 1] = self.border;
                size[i * m] = self.border;
            }
            // numpy.cumsum of pitch * size, in order
            let mut edges = vec![0.0];
            edges.extend(size.iter().scan(0.0, |e, s| {
                *e += pitch * s;
                Some(*e)
            }));
            edges
        };
        [
            axis(grid.max_shape.0, self.module.0, grid.pixel.0),
            axis(grid.max_shape.1, self.module.1, grid.pixel.1),
        ]
    }

    /// `calc_mask` over `shape`: the pixels either side of each chip
    /// boundary and the outer border.
    pub(super) fn mask(self, (n0, n1): (usize, usize)) -> Image<bool> {
        let chips = |n: usize, m: usize| {
            let mut line = vec![false; n];
            for i in 1..n / m {
                line[i * m - 1] = true;
                line[i * m] = true;
            }
            line[0] = true;
            line[n - 1] = true;
            line
        };
        cross(&chips(n0, self.module.0), &chips(n1, self.module.1))
    }
}

impl Rule {
    /// `Detector.mask` of `grid`: the disc over the shape, the others over
    /// `max_shape` and, for a binned shape, a pixel masked if any in its
    /// binning block is; None where pyFAI's `rebin` raises, the binning not
    /// dividing `max_shape`.
    pub(super) fn mask(self, grid: &Grid) -> Option<Image<bool>> {
        if self == Rule::Disc || grid.shape == grid.max_shape {
            return Some(self.drawn(grid.shape));
        }
        let ((n0, n1), (b0, b1)) = (grid.max_shape, grid.binning);
        let (b0, b1) = (b0 as usize, b1 as usize);
        if n0 % b0 != 0 || n1 % b1 != 0 {
            return None;
        }
        let full = self.drawn(grid.max_shape);
        let m1 = n1 / b1;
        let mut image = Image::from_fn((n0 / b0, m1), |_, _| false);
        let out = image.data_mut();
        for (k, _) in full.data().iter().enumerate().filter(|&(_, &m)| m) {
            out[(k / n1 / b0) * m1 + k % n1 / b1] = true;
        }
        Some(image)
    }

    /// `calc_mask` over `shape`.
    fn drawn(self, (n0, n1): (usize, usize)) -> Image<bool> {
        match self {
            Rule::Gaps { module, gap, strip } => {
                let (mut rows, mut cols) = (vec![false; n0], vec![false; n1]);
                let step = (module.0 + gap.0, module.1 + gap.1);
                mark(&mut rows, module.0, step.0, (0, gap.0 as isize));
                mark(&mut cols, module.1, step.1, (0, gap.1 as isize));
                if let Some(strip) = strip {
                    mark(&mut cols, module.1 / 2, step.1, strip);
                }
                cross(&rows, &cols)
            }
            Rule::Disc => {
                let (c0, c1) = ((n0 / 2) as f64, (n1 / 2) as f64);
                Image::from_fn((n0, n1), |s, f| {
                    let x = s as f64 + 0.5 - c0;
                    let y = f as f64 + 0.5 - c1;
                    x * x + y * y > c0 * c0
                })
            }
            Rule::Empty => Image::from_fn((n0, n1), |_, _| false),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::super::PoniFile;
    use super::*;

    /// The geometry lines of the files the golden reads.
    const GEOMETRY: &str = "Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\nRot3: 0.3\n\
                            Wavelength: 1e-10\n";

    fn golden() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../golden/pyfai_detectors.json"
        );
        serde_json::from_str(&std::fs::read_to_string(path).expect(path)).unwrap()
    }

    fn pair<T: TryFrom<i64>>(v: &Value) -> (T, T)
    where
        T::Error: std::fmt::Debug,
    {
        let n = |v: &Value| T::try_from(v.as_i64().unwrap()).unwrap();
        (n(&v[0]), n(&v[1]))
    }

    fn rule(v: &Value) -> Option<Rule> {
        match v["rule"].as_str()? {
            "gaps" => Some(Rule::Gaps {
                module: pair(&v["module"]),
                gap: pair(&v["gap"]),
                strip: (!v["strip"].is_null()).then(|| pair(&v["strip"])),
            }),
            "disc" => Some(Rule::Disc),
            "empty" => Some(Rule::Empty),
            other => panic!("rule {other}"),
        }
    }

    /// A pitch pair from its repr.
    fn pitch(v: &Value) -> (f64, f64) {
        let p = |k: usize| v[k].as_str().unwrap().parse().unwrap();
        (p(0), p(1))
    }

    /// The default instance as the golden gives it.
    fn grid(c: &Value) -> Grid {
        Grid {
            pixel: pitch(&c["pixel"]),
            shape: pair(&c["shape"]),
            max_shape: pair(&c["max_shape"]),
            binning: pair(&c["binning"]),
            orientation: c["orientation"].as_u64().unwrap() as u32,
            module_size: (!c["module_size"].is_null()).then(|| pair(&c["module_size"])),
        }
    }

    fn floats(v: &Value) -> (f64, f64) {
        (v[0].as_f64().unwrap(), v[1].as_f64().unwrap())
    }

    /// The pitch rule as the golden gives it.
    fn pitch_rule(p: &Value) -> Pitch {
        match p["rule"].as_str().unwrap() {
            "fixed" => Pitch::Fixed,
            "as given" => Pitch::AsGiven,
            "rounded" => Pitch::Rounded {
                size: floats(&p["size"]),
            },
            "rebinned" => Pitch::Rebinned {
                size: floats(&p["size"]),
                unit: floats(&p["unit"]),
            },
            "perkin" => Pitch::Perkin {
                size: floats(&p["size"]),
                unit: floats(&p["unit"]),
            },
            "table" => {
                let table: Vec<(u64, f64)> = p["table"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| (e[0].as_u64().unwrap(), e[1].as_f64().unwrap()))
                    .collect();
                Pitch::Table(Box::leak(table.into_boxed_slice()))
            }
            "plate" => Pitch::Plate {
                max_shape: pair(&p["max_shape"]),
                unit: p["unit"].as_f64().unwrap(),
            },
            other => panic!("pitch rule {other}"),
        }
    }

    fn sorted(keys: impl Iterator<Item = &'static str>) -> bool {
        keys.collect::<Vec<_>>().windows(2).all(|w| w[0] < w[1])
    }

    #[test]
    fn table_is_the_golden() {
        let doc = golden();
        let classes = doc["classes"].as_array().unwrap();
        assert_eq!(classes.len(), table::CLASSES.len());
        assert!(sorted(table::CLASSES.iter().map(|c| c.name)));
        for (c, t) in classes.iter().zip(table::CLASSES) {
            assert_eq!(t.name, c["class"]);
            let args: Vec<_> = c["args"].as_array().unwrap().iter().collect();
            assert_eq!(t.args, args.as_slice(), "{}", t.name);
            assert_eq!(t.force_pixel, c["force_pixel"], "{}", t.name);
            match (c["layout"].as_str().unwrap(), t.layout) {
                ("generic", Layout::Generic) => {}
                ("uniform", Layout::Uniform(g, p, r)) => assert_eq!(
                    (g, p, r),
                    (grid(c), pitch_rule(&c["pitch"]), rule(&c["mask"])),
                    "{}",
                    t.name
                ),
                ("edges", Layout::Edges(g, p, chips)) => {
                    assert_eq!(p, pitch_rule(&c["pitch"]), "{}", t.name);
                    assert_eq!(c["mask"]["rule"], "chips", "{}", t.name);
                    let want = Chips {
                        module: pair(&c["mask"]["module"]),
                        border: c["border"].as_f64().unwrap(),
                    };
                    assert_eq!((g, chips), (grid(c), want), "{}", t.name)
                }
                (why, Layout::Unsupported(reason)) => assert_eq!(why, reason, "{}", t.name),
                (why, layout) => panic!("{}: {why} vs {layout:?}", t.name),
            }
        }
        for (map, want) in [
            (table::ATTRIBUTES, &doc["attributes"]),
            (table::REGISTRY, &doc["registry"]),
        ] {
            let want = want.as_object().unwrap();
            assert!(sorted(map.iter().map(|&(k, _)| k)));
            assert_eq!(map.len(), want.len());
            for &(k, v) in map {
                assert_eq!(want[k], v);
                assert_eq!(Class::named(v).name, v);
            }
        }
    }

    #[test]
    fn names_resolve_as_the_factory_resolves_them() {
        let doc = golden();
        let probes = doc["probes"].as_object().unwrap();
        for (name, class) in probes {
            assert_eq!(
                Class::resolve(name).map(|c| c.name),
                class.as_str(),
                "{name:?}"
            );
        }
        let unknown = probes.values().filter(|c| c.is_null()).count();
        assert!(unknown > 0 && unknown < probes.len());
    }

    /// count, sum and sum of squares (mod 2^64) of the masked row-major
    /// indices.
    fn fingerprint(mask: &[bool]) -> [u64; 3] {
        mask.iter()
            .enumerate()
            .filter(|&(_, &m)| m)
            .fold([0u64; 3], |[n, s, q], (k, _)| {
                let k = k as u64;
                [n + 1, s.wrapping_add(k), q.wrapping_add(k.wrapping_mul(k))]
            })
    }

    #[test]
    fn masks_are_pyfai_calc_mask() {
        let doc = golden();
        let mut ruled = 0;
        for (c, t) in doc["classes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(table::CLASSES)
        {
            let (grid, mask) = match t.layout {
                Layout::Uniform(grid, _, rule) => (grid, rule.map(|r| r.mask(&grid).unwrap())),
                Layout::Edges(grid, _, chips) => (grid, Some(chips.mask(grid.shape))),
                _ => continue,
            };
            let ruled_here = mask.is_some();
            let mask = mask.unwrap_or_else(|| Image::from_fn(grid.shape, |_, _| false));
            assert_eq!(mask.shape(), grid.shape, "{}", t.name);
            let want: Vec<u64> = c["fingerprint"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(fingerprint(mask.data()).as_slice(), want, "{}", t.name);
            ruled += usize::from(ruled_here);
        }
        assert!(ruled > 0);
    }

    /// A pitch from its repr, None for pyFAI's None.
    fn given(v: &Value) -> Option<f64> {
        v.as_str().map(|p| p.parse().unwrap())
    }

    fn ints(v: &Value) -> [i64; 2] {
        [v[0].as_i64().unwrap(), v[1].as_i64().unwrap()]
    }

    /// The instance pyFAI's `Geometry.load` builds from each file the golden
    /// reads, and from each it writes of a binned instance: [`Pitch::build`]
    /// gives its pitch, shape, max_shape and binning, or raises where pyFAI
    /// raises, and the mask of [`Built::grid`], where the instance agrees
    /// with its class, is pyFAI's `Detector.mask` or raises where it does.
    #[test]
    fn reads_build_pyfai_s_instance() {
        let doc = golden();
        let (mut raises, mut built, mut masked, mut mask_raises, mut generic) = (0, 0, 0, 0, 0);
        for (c, t) in doc["classes"]
            .as_array()
            .unwrap()
            .iter()
            .zip(table::CLASSES)
        {
            let (grid, pitch, edges) = match t.layout {
                Layout::Uniform(grid, pitch, _) => (grid, pitch, false),
                Layout::Edges(grid, pitch, _) => (grid, pitch, true),
                _ => continue,
            };
            let trips = c["round_trip"].as_array().unwrap().iter();
            let texts = c["reads"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| (format!("{}{}", e["head"].as_str().unwrap(), GEOMETRY), e))
                .chain(trips.map(|e| (e["poni"].as_str().unwrap().to_string(), e)));
            for (text, e) in texts {
                let at = format!("{}: {text:?}", t.name);
                let f = PoniFile::parse(&text).expect(&at);
                let class = f.class().unwrap();
                let facts = if e["read"].is_object() { &e["read"] } else { e };
                if class.name != t.name {
                    // version 1 of one pitch, or of a class without force_pixel
                    assert_eq!(
                        (class.name, &facts["class"]),
                        ("Detector", &Value::from("Detector")),
                        "{at}"
                    );
                    generic += 1;
                    continue;
                }
                let b = f.built(class, &grid, pitch);
                if e["raises"].is_string() {
                    assert_eq!(b, None, "{at}");
                    raises += 1;
                    continue;
                }
                let b = b.expect(&at);
                assert_eq!(facts["class"], t.name, "{at}");
                assert_eq!(
                    b.pixel,
                    [given(&facts["pixel"][0]), given(&facts["pixel"][1])],
                    "{at}"
                );
                assert_eq!(
                    (b.shape, b.max_shape, b.binning),
                    (
                        ints(&facts["shape"]),
                        ints(&facts["max_shape"]),
                        ints(&facts["binning"])
                    ),
                    "{at}"
                );
                built += 1;
                let Ok(g) = b.grid(&grid, pitch, edges) else {
                    continue;
                };
                if g.shape.0 == 0 || g.shape.1 == 0 {
                    // the core refuses an empty image; pyFAI's mask is not read
                    assert!(
                        !e["mask"].is_array() && !e["mask_error"].is_string(),
                        "{at}"
                    );
                    continue;
                }
                // pyFAI's Detector.mask of every instance that agrees
                let mask = match t.layout {
                    Layout::Uniform(_, _, rule) => match rule {
                        Some(r) => r.mask(&g),
                        None => Some(Image::from_fn(g.shape, |_, _| false)),
                    },
                    Layout::Edges(_, _, chips) => Some(chips.mask(g.shape)),
                    _ => unreachable!(),
                };
                match (mask, e["mask"].as_array()) {
                    (None, None) => {
                        assert!(e["mask_error"].is_string(), "{at}");
                        mask_raises += 1;
                    }
                    (Some(m), Some(want)) => {
                        assert_eq!(m.shape(), g.shape, "{at}");
                        let want: Vec<u64> = want.iter().map(|v| v.as_u64().unwrap()).collect();
                        assert_eq!(fingerprint(m.data()).as_slice(), want, "{at}");
                        masked += 1;
                    }
                    (mask, _) => panic!("{at}: {:?} vs {}", mask.map(|m| m.shape()), e),
                }
            }
        }
        assert!(raises > 0 && built > 0 && masked > 0 && mask_raises > 0 && generic > 0);
    }
}
