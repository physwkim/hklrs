//! Python binoculars' cubes of the Pt(111) scans 63-92 of Zenodo deposit
//! 10.5281/zenodo.6787572 (CC-BY-4.0), rebuilt here: the scans read as
//! picca/binoculars 93956b2's input sixs:flyscanuhv2 reads them, its
//! detector, and the cube compared with binoculars' voxel by voxel. `DATA`
//! names the deposit, its raw scans zip extracted into `raw/`.
//!
//! The input configuration is the cubes' own: centralpixel 205,177 (fast,
//! slow), sdd 1.170 m, detrot 90°, attenuation_coefficient 1.87.
//!
//! - Lattice: bin k is centred on k·res in both, `Binning::new(Linear,
//!   res, -res/2)` here, and the steps are the same double.
//! - Extent: binoculars sizes each axis from ⌊min/res⌋ to ⌈max/res⌉ over
//!   every pixel of every frame it projects, masked or of weight 0
//!   (space.py:47-57, :766-769), so its axes have planes no pixel is binned
//!   in; `Bounds` spans only the pixels binned. A convention, not a defect:
//!   those planes are checked empty and binoculars' extent is restated.
//! - Contributions: exact.
//! - Photons: binoculars' estimator, Σ raw·1.87^att with weight 1 per
//!   pixel, is fed as each frame's raw counts. Exact where no attenuated
//!   frame reaches the voxel: integer terms, sums below 2⁵³. Elsewhere
//!   binoculars scales in single precision (PYBIN-1), each attenuated term
//!   off by at most (2⁻²³ + 2⁻⁴⁸) of itself; repeating its arithmetic
//!   leaves the order of the double sums, 2nε of the voxel for n
//!   contributions.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hklrs::frame_map::{FrameMap, MappedFrame};
use hklrs::grid::{Binning, Bounds, CoordKind, Projected};
use hklrs::normalise::{FrameScalars, NormalisedFrame, PixelFields, Region};
use hklrs::presets::preset;
use hklrs::projection::Projection;
use hklrs::{Detector, Image, PixelDetector, Wavelength, sixs};
use rayon::prelude::*;
use rust_hdf5::{H5Dataset, H5File, ReadNumeric};

pub const DATA: &str = "HKLRS_ZENODO_6787572";
const SCANS: Range<u32> = 63..93;
const COEFFICIENT: f64 = 1.87;
/// binoculars' WRONG_ATTENUATION (sixs.py:73), the scans' "unknown".
const UNKNOWN: f32 = -100.0;
/// Frames mapped in parallel before they are added in order.
const BATCH: usize = 192;

pub struct Scan {
    path: PathBuf,
    wavelength: Wavelength,
    /// mu, omega, delta, gamma, degrees.
    readbacks: Vec<[f64; 4]>,
    attenuation: Vec<f32>,
}

impl Scan {
    fn open(path: PathBuf) -> Self {
        let file = h5(&path);
        let read = |name: &str| -> Vec<f64> {
            let ds = file.dataset(name).unwrap();
            ds.read_numeric_as().unwrap()
        };
        let axes = ["mu", "omega", "delta", "gamma"].map(|a| read(&format!("com/scan_data/{a}")));
        // binoculars takes the NXmonochromator's lambda (sixs.py:190-201)
        let wavelength = Wavelength::new(read("com/SIXS/i14-c-c02-op-mono/lambda")[0]).unwrap();
        let readbacks = (0..axes[0].len())
            .map(|i| axes.each_ref().map(|v| v[i]))
            .collect();
        let attenuation = file
            .dataset("com/scan_data/attenuation")
            .unwrap()
            .read_numeric_as()
            .unwrap();
        Self {
            path,
            wavelength,
            readbacks,
            attenuation,
        }
    }

    /// Frame `i`'s attenuation. The scans record it two points late, as
    /// binoculars reads it (sixs.py:888) and as the beamline's corrected
    /// ROI shows (roi0c = roi0·1.87^att[i+2]); None where it is unknown or
    /// past the scan's end, the frames binoculars weighs 0 (sixs.py:870-871,
    /// :925-926).
    fn attenuation(&self, i: usize) -> Option<f64> {
        let a = *self.attenuation.get(i + 2)?;
        (a.is_finite() && a != UNKNOWN).then_some(f64::from(a))
    }

    pub fn read<T: ReadNumeric>(&self, name: &str) -> Vec<T> {
        h5(&self.path)
            .dataset(name)
            .unwrap()
            .read_numeric_as()
            .unwrap()
    }

    /// The detector images, (frame, row, column).
    fn images(&self) -> H5Dataset {
        h5(&self.path).dataset("com/scan_data/xpad_image").unwrap()
    }

    fn frames(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        let n = self.readbacks.len();
        (0..n).step_by(BATCH).map(move |s| s..(s + BATCH).min(n))
    }
}

fn h5(path: &Path) -> H5File {
    H5File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The deposit `DATA` names.
pub fn deposit() -> PathBuf {
    PathBuf::from(
        std::env::var_os(DATA)
            .unwrap_or_else(|| panic!("{DATA} must name the deposit, raw/ extracted")),
    )
}

/// Scans 63-92 of the deposit's raw data.
pub fn scans(deposit: &Path) -> Vec<Scan> {
    let raw = deposit.join("raw/20190919_SIXS_beamline_SXRD_raw_data");
    SCANS
        .map(|n| Scan::open(raw.join(format!("align_ascan_omega_{n:05}.nxs"))))
        .collect()
}

/// binoculars' detector (sixs.py:698-702, :957-968): pyFAI 0.19's
/// ImXPadS140 mask, the pixels either side of every chip boundary and the
/// outer frame (_imxpad.py:126-142), or the beamline's maskmatrix. The .npy
/// is not in the deposit; every scan carries it as the config publisher's
/// `mask`, whose `mask_file` attribute names it.
fn detector(scan: &Scan) -> PixelDetector {
    let maskmatrix: Vec<i32> = scan.read("com/SIXS/i14-c-c00-ex-config-publisher/mask");
    let table = Detector::ImxpadS140;
    let (h, w) = table.shape();
    let chips = table.mask();
    let pyfai = Image::from_fn((h, w), |s, f| {
        chips.get(s, f) || s == 0 || s == h - 1 || f == 0 || f == w - 1
    });
    let count = |m: &Image<bool>| m.data().iter().filter(|&&m| m).count();
    assert_eq!(count(&pyfai), 5544);
    let mask = Image::from_fn((h, w), |s, f| pyfai.get(s, f) || maskmatrix[s * w + f] != 0);
    assert_eq!(count(&mask), 6445);
    let det = table.pixel_detector().with_mask(mask).unwrap();
    sixs(&det, (177, 205), 1.170, 90f64.to_radians()).unwrap()
}

/// (min, max) of each coordinate over every pixel, masked or not.
fn span(m: &MappedFrame) -> [(f64, f64); 3] {
    let mut out = [(f64::INFINITY, f64::NEG_INFINITY); 3];
    for (o, p) in out.iter_mut().zip(m.points().planes()) {
        *o = p
            .data()
            .iter()
            .fold(*o, |(lo, hi), &v| (lo.min(v), hi.max(v)));
    }
    out
}

/// The unmasked pixels holding each coordinate's minimum and maximum. The
/// bin rule is monotonic, so `Bounds` takes the frame's span from these
/// alone, and they can be picked in parallel; pass 2 adding every pixel
/// with none outside checks that.
fn extremes(m: &MappedFrame) -> (Projected, NormalisedFrame) {
    let (planes, pixels) = (m.points().planes(), m.pixels().data());
    let mut picked = Vec::new();
    for p in planes {
        let v = p.data();
        let unmasked = (0..v.len()).filter(|&i| pixels[i].is_ok());
        picked.extend(unmasked.clone().min_by(|&a, &b| v[a].total_cmp(&v[b])));
        picked.extend(unmasked.max_by(|&a, &b| v[a].total_cmp(&v[b])));
    }
    let shape = (1, picked.len());
    let pick = |p: &Image<f64>| Image::new(shape, picked.iter().map(|&i| p.data()[i]).collect());
    let points = planes.iter().map(pick).collect::<Result<_, _>>().unwrap();
    (
        Projected::new(m.points().coords().to_vec(), points).unwrap(),
        Image::new(shape, picked.iter().map(|&i| pixels[i]).collect()).unwrap(),
    )
}

#[derive(Debug, Default)]
struct Tally {
    voxels: u64,
    nonempty: u64,
    /// Reference voxels outside our axes that hold anything.
    stray: u64,
    contributions: u64,
    /// Voxels reached by an attenuated frame.
    attenuated: u64,
    photons: u64,
    /// max |Δphotons| / attenuated photons.
    scaled: f64,
    /// max |reference − ours with binoculars' single precision| / photons.
    residual: f64,
    reference: (f64, f64),
    ours: (f64, f64),
}

/// Builds the cube of `projection` at `resolution` from every frame of
/// `scans` and compares it with binoculars' cube `reference`, whose axes
/// are `names`.
pub fn compare(
    scans: &[Scan],
    projection: Projection,
    resolution: [f64; 3],
    reference: &Path,
    names: [&str; 3],
) {
    let detector = detector(&scans[0]);
    let (h, w) = detector.shape();

    let geometry = preset("ZAXIS").unwrap();
    let axes: Vec<&str> = geometry.axes().iter().map(|a| a.name()).collect();
    assert_eq!(axes, ["mu", "omega", "delta", "gamma"]);
    let binnings: Vec<Binning> = resolution
        .iter()
        .map(|&r| Binning::new(CoordKind::Linear, r, -r / 2.0).unwrap())
        .collect();
    let map = FrameMap::new(
        geometry,
        projection,
        &detector,
        &Region::new(0..h, 0..w, (1, 1)),
        PixelFields::new(),
        binnings.clone(),
    )
    .unwrap();
    let one = FrameScalars::new();

    // pass 1: bounds, and binoculars' extent over every frame
    let clock = Instant::now();
    let zero = Image::from_fn((h, w), |_, _| 0.0);
    let mut bounds = Bounds::new(binnings.clone());
    let mut attenuated_bounds = Bounds::new(binnings.clone());
    let mut extent = [(f64::INFINITY, f64::NEG_INFINITY); 3];
    let mut used = 0;
    for scan in scans {
        for frames in scan.frames() {
            let mapped: Vec<_> = frames
                .clone()
                .into_par_iter()
                .map(|i| {
                    let m = map
                        .map(&scan.readbacks[i], scan.wavelength, &zero, &one)
                        .unwrap();
                    (span(&m), extremes(&m))
                })
                .collect();
            for (i, (s, (points, pixels))) in frames.zip(&mapped) {
                for (e, s) in extent.iter_mut().zip(s) {
                    *e = (e.0.min(s.0), e.1.max(s.1));
                }
                let Some(a) = scan.attenuation(i) else {
                    continue;
                };
                used += 1;
                bounds.include(points, pixels).unwrap();
                if a != 0.0 {
                    attenuated_bounds.include(points, pixels).unwrap();
                }
            }
        }
    }
    println!("pass 1: {used} frames binned, {:.0?}", clock.elapsed());

    // pass 2: ours, and the attenuated frames ours and in binoculars'
    // single precision
    let clock = Instant::now();
    let mut cube = bounds.cube().unwrap();
    let mut attenuated = attenuated_bounds.clone().cube().unwrap();
    let mut single = attenuated_bounds.cube().unwrap();
    for scan in scans {
        assert_eq!(scan.images().shape(), [scan.readbacks.len(), h, w]);
        for frames in scan.frames() {
            // an H5File is not Sync: each rayon job reads through its own
            let mapped: Vec<(MappedFrame, Option<MappedFrame>)> = frames
                .into_par_iter()
                .map_init(
                    || scan.images(),
                    |images, i| {
                        let a = scan.attenuation(i)?;
                        let image: Vec<u16> = images
                            .read_numeric_slice_as(&[i, 0, 0], &[1, h, w])
                            .unwrap();
                        let frame = |counts: Vec<f64>| {
                            let raw = Image::new((h, w), counts).unwrap();
                            map.map(&scan.readbacks[i], scan.wavelength, &raw, &one)
                                .unwrap()
                        };
                        let scale = COEFFICIENT.powf(a);
                        let ours = frame(image.iter().map(|&r| f64::from(r) * scale).collect());
                        // PYBIN-1: uint16 → float32, *= 1.87**att (sixs.py:917, :922)
                        let theirs = (a != 0.0).then(|| {
                            let scale = scale as f32;
                            frame(
                                image
                                    .iter()
                                    .map(|&r| f64::from(f32::from(r) * scale))
                                    .collect(),
                            )
                        });
                        Some((ours, theirs))
                    },
                )
                .flatten()
                .collect();
            let all: Vec<_> = mapped
                .iter()
                .map(|(ours, _)| (ours.points(), ours.pixels()))
                .collect();
            let (ours, theirs): (Vec<_>, Vec<_>) = mapped
                .iter()
                .filter_map(|(ours, theirs)| {
                    let theirs = theirs.as_ref()?;
                    Some((
                        (ours.points(), ours.pixels()),
                        (theirs.points(), theirs.pixels()),
                    ))
                })
                .unzip();
            assert_eq!(cube.add_frames(&all).unwrap(), 0);
            assert_eq!(attenuated.add_frames(&ours).unwrap(), 0);
            assert_eq!(single.add_frames(&theirs).unwrap(), 0);
        }
    }
    println!("pass 2: {:.0?}", clock.elapsed());

    let reference = h5(reference);
    // index, min, max, res, imin, imax (space.py Axis)
    let theirs = names.map(|n| -> Vec<f64> {
        let ds = reference.dataset(&format!("binoculars/axes/{n}")).unwrap();
        ds.read_numeric_as().unwrap()
    });
    let mut origin = [0i64; 3];
    for (d, (ours, theirs)) in cube.axes().iter().zip(&theirs).enumerate() {
        let (res, imin, imax) = (theirs[3], theirs[4] as i64, theirs[5] as i64);
        assert_eq!(res, resolution[d], "{d}");
        assert_eq!(ours.binning(), binnings[d], "{d}");
        let last = ours.first() + ours.size() as i64 - 1;
        println!(
            "axis {d}: binoculars {imin}..={imax}, hklrs {}..={last}",
            ours.first()
        );
        assert!(imin <= ours.first() && last <= imax, "{d}");
        let (lo, hi) = extent[d];
        assert_eq!(
            (imin, imax),
            ((lo / res).floor() as i64, (hi / res).ceil() as i64),
            "{d}"
        );
        origin[d] = imin;
    }

    let clock = Instant::now();
    let counts = reference.dataset("binoculars/counts").unwrap();
    let contributions = reference.dataset("binoculars/contributions").unwrap();
    let shape = counts.shape();
    assert_eq!(shape, contributions.shape());
    let (ny, nz) = (shape[1], shape[2]);
    let single_term = 2f64.powi(-23) + 2f64.powi(-48);
    let mut t = Tally::default();
    // slabs of the datasets' chunk height along the first axis
    let height = counts.chunk_dims().map_or(1, |c| c[0]);
    for x0 in (0..shape[0]).step_by(height) {
        let n = height.min(shape[0] - x0);
        let read = |ds: &H5Dataset| -> Vec<f64> {
            ds.read_numeric_slice_as(&[x0, 0, 0], &[n, ny, nz]).unwrap()
        };
        let (p_ref, c_ref) = (read(&counts), read(&contributions));
        for (k, (&p, &c)) in p_ref.iter().zip(&c_ref).enumerate() {
            t.voxels += 1;
            t.reference.0 += p;
            t.reference.1 += c;
            t.nonempty += u64::from(c != 0.0);
            let g = [
                origin[0] + (x0 + k / (ny * nz)) as i64,
                origin[1] + (k / nz % ny) as i64,
                origin[2] + (k % nz) as i64,
            ];
            let Some(o) = cube.offset(&g) else {
                t.stray += u64::from(c != 0.0 || p != 0.0);
                continue;
            };
            let (ours, n) = (cube.photons()[o], cube.contributions()[o]);
            t.ours.0 += ours;
            t.ours.1 += n as f64;
            if n as f64 != c {
                if t.contributions < 10 {
                    println!("contributions {g:?}: binoculars {c}, hklrs {n}");
                }
                t.contributions += 1;
            }
            let reached = attenuated
                .offset(&g)
                .filter(|&a| attenuated.contributions()[a] > 0);
            let fine = match reached {
                None => ours == p,
                Some(a) => {
                    t.attenuated += 1;
                    let (ap, sp) = (attenuated.photons()[a], single.photons()[a]);
                    let order = 2.0 * c * f64::EPSILON * p;
                    let residual = (p - (ours - ap + sp)).abs();
                    t.scaled = t.scaled.max((p - ours).abs() / ap);
                    t.residual = t.residual.max(residual / p);
                    (p - ours).abs() <= single_term * ap + order && residual <= order
                }
            };
            if !fine {
                if t.photons < 10 {
                    println!("photons {g:?}: binoculars {p}, hklrs {ours}");
                }
                t.photons += 1;
            }
        }
    }
    println!("compare: {:.0?}\n{t:#?}", clock.elapsed());
    assert_eq!(
        (t.stray, t.contributions, t.photons),
        (0, 0, 0),
        "stray, contribution and photon mismatches"
    );
}
