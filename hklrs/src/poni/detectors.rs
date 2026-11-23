//! pyFAI 2026.9.0's detector classes as `Detector.factory` finds them by a
//! PONI file's detector name; golden/pyfai_detectors.py generates the table.

#[rustfmt::skip]
mod table;

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
    /// The pixels of the generic detector of the grid's pitch, shape and
    /// orientation, with the class's mask.
    Uniform(Grid, Option<Rule>),
    /// The ImXPad chips, the pixels of the grid's shape and orientation at
    /// the chips' edges.
    Edges(Grid, Chips),
    /// Why hklrs has no model of the class.
    Unsupported(&'static str),
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

/// Row-major, the pixels in a marked row or a marked column.
fn cross(rows: &[bool], cols: &[bool]) -> Vec<bool> {
    let n1 = cols.len();
    (0..rows.len() * n1)
        .map(|k| rows[k / n1] || cols[k % n1])
        .collect()
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

    /// `calc_mask`, row-major over `shape`: the pixels either side of each
    /// chip boundary and the outer border.
    pub(super) fn mask(self, (n0, n1): (usize, usize)) -> Vec<bool> {
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
    /// The masked pixels, row-major over `shape`.
    pub(super) fn mask(self, (n0, n1): (usize, usize)) -> Vec<bool> {
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
                (0..n0 * n1)
                    .map(|k| {
                        let x = (k / n1) as f64 + 0.5 - c0;
                        let y = (k % n1) as f64 + 0.5 - c1;
                        x * x + y * y > c0 * c0
                    })
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

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
            other => panic!("rule {other}"),
        }
    }

    /// The default instance as the golden gives it, the pixels from their
    /// repr.
    fn grid(c: &Value) -> Grid {
        let pixel = |k: usize| c["pixel"][k].as_str().unwrap().parse().unwrap();
        Grid {
            pixel: (pixel(0), pixel(1)),
            shape: pair(&c["shape"]),
            max_shape: pair(&c["max_shape"]),
            binning: pair(&c["binning"]),
            orientation: c["orientation"].as_u64().unwrap() as u32,
            module_size: (!c["module_size"].is_null()).then(|| pair(&c["module_size"])),
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
                ("uniform", Layout::Uniform(g, r)) => {
                    assert_eq!((g, r), (grid(c), rule(&c["mask"])), "{}", t.name)
                }
                ("edges", Layout::Edges(g, chips)) => {
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
                Layout::Uniform(grid, rule) => (grid, rule.map(|r| r.mask(grid.shape))),
                Layout::Edges(grid, chips) => (grid, Some(chips.mask(grid.shape))),
                _ => continue,
            };
            let (n0, n1) = grid.shape;
            let ruled_here = mask.is_some();
            let mask = mask.unwrap_or(vec![false; n0 * n1]);
            assert_eq!(mask.len(), n0 * n1);
            let want: Vec<u64> = c["fingerprint"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(fingerprint(&mask).as_slice(), want, "{}", t.name);
            ruled += usize::from(ruled_here);
        }
        assert!(ruled > 0);
    }
}
