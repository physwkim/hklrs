//! Detectors set up by xrayutilities' `QConversion.init_area` and
//! `init_linear`, as pixel detectors in the detector-arm frame. The beam is
//! xrayutilities' `r_i`: the centre channel lies along it at all axes zero.
//!
//! Tilts keep the sense xrayutilities actually applies: its `rotation_arb`
//! turns left-handed (XU-3) and the importer reproduces that turn, so `tilt`
//! and `tiltazimuth` values fitted in xrayutilities import unchanged. A
//! positive tilt turns a linear row toward the beam and an area chip's
//! `tiltazimuth` direction toward `dir1 × dir2`.

use std::str::FromStr;

use nalgebra::{Unit, UnitQuaternion, Vector3};

use crate::{Deg, Error, PARALLEL_SIN, PixelDetector, Rad};

/// A direction string as xrayutilities writes it: `x+`, `z-`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Direction {
    axis: usize,
    negative: bool,
}

impl FromStr for Direction {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self, Error> {
        let err = || Error::XuDirection {
            text: text.to_owned(),
        };
        let &[axis, sign] = text.as_bytes() else {
            return Err(err());
        };
        let axis = match axis {
            b'x' => 0,
            b'y' => 1,
            b'z' => 2,
            _ => return Err(err()),
        };
        let negative = match sign {
            b'+' => false,
            b'-' => true,
            _ => return Err(err()),
        };
        Ok(Self { axis, negative })
    }
}

impl Direction {
    pub fn vector(self) -> Unit<Vector3<f64>> {
        let mut v = Vector3::zeros();
        v[self.axis] = if self.negative { -1.0 } else { 1.0 };
        Unit::new_unchecked(v)
    }
}

fn positive(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::NotPositive { what, value })
    }
}

fn finite(what: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::NotFinite { what, value })
    }
}

/// `init_area` parameters. Pixel (j1, j2) is slow index j1 along `dir1`, fast
/// index j2 along `dir2`, as in the arrays `QConversion.area` returns.
#[derive(Clone, Debug, PartialEq)]
pub struct Area {
    pub dir1: Direction,
    pub dir2: Direction,
    pub cch1: f64,
    pub cch2: f64,
    pub nch1: usize,
    pub nch2: usize,
    pub distance: f64,
    pub pwidth1: f64,
    pub pwidth2: f64,
    pub tilt: Deg,
    pub tiltazimuth: Deg,
    pub detrot: Deg,
    /// Binning applied after the roi; a binned pixel is the centre of the raw
    /// block it averages.
    pub nav: [usize; 2],
    /// `[start1, end1, start2, end2]`, half-open; `None` is the whole chip.
    pub roi: Option<[usize; 4]>,
}

impl Area {
    /// No tilt, detector rotation, binning or roi.
    pub fn new(
        dir1: Direction,
        dir2: Direction,
        cch: [f64; 2],
        nch: [usize; 2],
        distance: f64,
        pwidth: [f64; 2],
    ) -> Self {
        Self {
            dir1,
            dir2,
            cch1: cch[0],
            cch2: cch[1],
            nch1: nch[0],
            nch2: nch[1],
            distance,
            pwidth1: pwidth[0],
            pwidth2: pwidth[1],
            tilt: Deg(0.0),
            tiltazimuth: Deg(0.0),
            detrot: Deg(0.0),
            nav: [1, 1],
            roi: None,
        }
    }

    pub fn detector(&self, beam: Unit<Vector3<f64>>) -> Result<PixelDetector, Error> {
        positive("xu distance", self.distance)?;
        positive("xu pwidth1", self.pwidth1)?;
        positive("xu pwidth2", self.pwidth2)?;
        finite("xu cch1", self.cch1)?;
        finite("xu cch2", self.cch2)?;
        finite("xu tilt", self.tilt.0)?;
        finite("xu tiltazimuth", self.tiltazimuth.0)?;
        finite("xu detrot", self.detrot.0)?;
        if self.dir1.axis == self.dir2.axis {
            return Err(Error::ParallelDirections {
                what: "area detector directions",
            });
        }
        let (p1, p2) = (
            self.dir1.vector().into_inner(),
            self.dir2.vector().into_inner(),
        );
        let az = Rad::from(self.tiltazimuth).0;
        let toward = p1 * az.cos() + p2 * az.sin();
        // xu turns about −sin(az)·p1 + cos(az)·p2 with the left-handed
        // rotation_arb (XU-3); a right-handed turn about the negated axis is
        // the same rotation: it tilts `toward` onto the normal p1 × p2.
        let tilt_axis = Unit::new_normalize(toward.cross(&p1.cross(&p2)));
        let turn = UnitQuaternion::from_axis_angle(&beam, Rad::from(self.detrot).0)
            * UnitQuaternion::from_axis_angle(&tilt_axis, Rad::from(self.tilt).0);
        let e_slow = turn * (p1 * self.pwidth1);
        let e_fast = turn * (p2 * self.pwidth2);
        let [r0, r1, r2, r3] = self.roi.unwrap_or([0, self.nch1, 0, self.nch2]);
        PixelDetector::new(
            (self.nch1, self.nch2),
            beam.into_inner() * self.distance - e_slow * self.cch1 - e_fast * self.cch2,
            e_slow,
            e_fast,
        )?
        .roi(r0..r1, r2..r3)?
        .bin(self.nav[0], self.nav[1])
    }
}

/// `init_linear` parameters, as a detector of one row.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub dir: Direction,
    pub cch: f64,
    pub nch: usize,
    pub distance: f64,
    pub pwidth: f64,
    pub tilt: Deg,
    /// Binning applied after the roi; a binned pixel is the centre of the raw
    /// block it averages.
    pub nav: usize,
    /// `[start, end]`, half-open; `None` is every channel.
    pub roi: Option<[usize; 2]>,
}

impl Linear {
    /// No tilt, binning or roi.
    pub fn new(dir: Direction, cch: f64, nch: usize, distance: f64, pwidth: f64) -> Self {
        Self {
            dir,
            cch,
            nch,
            distance,
            pwidth,
            tilt: Deg(0.0),
            nav: 1,
            roi: None,
        }
    }

    pub fn detector(&self, beam: Unit<Vector3<f64>>) -> Result<PixelDetector, Error> {
        positive("xu distance", self.distance)?;
        positive("xu pwidth", self.pwidth)?;
        finite("xu cch", self.cch)?;
        finite("xu tilt", self.tilt.0)?;
        let along = self.dir.vector().into_inner();
        let step = along * self.pwidth;
        let tilt = Rad::from(self.tilt).0;
        let e_fast = if tilt == 0.0 {
            step
        } else {
            // xu turns left-handed (XU-3) about r_i × dir, i.e. right-handed
            // about dir × r_i: the channel row tilts toward the beam.
            let normal = along.cross(&beam);
            if normal.norm() < PARALLEL_SIN {
                return Err(Error::ParallelDirections {
                    what: "the tilted linear detector direction and the beam",
                });
            }
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(normal), tilt) * step
        };
        let [r0, r1] = self.roi.unwrap_or([0, self.nch]);
        PixelDetector::new(
            (1, self.nch),
            beam.into_inner() * self.distance - e_fast * self.cch,
            Vector3::zeros(),
            e_fast,
        )?
        .roi(0..1, r0..r1)?
        .bin(1, self.nav)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use super::*;

    fn dir(text: &str) -> Direction {
        text.parse().unwrap()
    }

    fn y() -> Unit<Vector3<f64>> {
        Vector3::y_axis()
    }

    fn close(a: Point3<f64>, b: Point3<f64>, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    #[test]
    fn direction_accepts_exactly_axis_and_sign() {
        for (text, v) in [
            ("x+", [1.0, 0.0, 0.0]),
            ("x-", [-1.0, 0.0, 0.0]),
            ("y+", [0.0, 1.0, 0.0]),
            ("y-", [0.0, -1.0, 0.0]),
            ("z+", [0.0, 0.0, 1.0]),
            ("z-", [0.0, 0.0, -1.0]),
        ] {
            assert_eq!(dir(text).vector().into_inner(), Vector3::from(v));
        }
        for text in ["", "x", "X+", "x+ ", "w+", "x*", "+x", "xx+"] {
            assert_eq!(
                text.parse::<Direction>(),
                Err(Error::XuDirection { text: text.into() })
            );
        }
    }

    #[test]
    fn area_centre_channel_is_on_the_beam_and_steps_follow_the_directions() {
        let det = Area::new(dir("z+"), dir("x-"), [4.0, 6.0], [9, 13], 0.5, [1e-4, 2e-4])
            .detector(y())
            .unwrap();
        assert_eq!(det.shape(), (9, 13));
        let c = det.pixel(4, 6).unwrap();
        assert!(close(c, Point3::new(0.0, 0.5, 0.0), 1e-16));
        assert!(close(
            det.pixel(5, 6).unwrap(),
            c + Vector3::new(0.0, 0.0, 1e-4),
            1e-16
        ));
        assert!(close(
            det.pixel(4, 7).unwrap(),
            c + Vector3::new(-2e-4, 0.0, 0.0),
            1e-16
        ));
    }

    #[test]
    fn area_tilt_turns_the_azimuth_direction_onto_the_plane_normal() {
        // dir1 × dir2 = z × x = y; azimuth 0 tilts dir1 toward it.
        let area = Area {
            tilt: Deg(30.0),
            ..Area::new(dir("z+"), dir("x+"), [4.0, 6.0], [9, 13], 0.5, [1e-4, 1e-4])
        };
        let det = area.detector(y()).unwrap();
        let (s, c) = 30f64.to_radians().sin_cos();
        let step = det.pixel(5, 6).unwrap() - det.pixel(4, 6).unwrap();
        assert!((step - Vector3::new(0.0, s, c) * 1e-4).norm() <= 1e-15);
        let fast = det.pixel(4, 7).unwrap() - det.pixel(4, 6).unwrap();
        assert!((fast - Vector3::new(1e-4, 0.0, 0.0)).norm() <= 1e-15);
    }

    #[test]
    fn area_detrot_turns_the_chip_about_the_beam() {
        let area = Area {
            detrot: Deg(90.0),
            ..Area::new(dir("z+"), dir("x+"), [4.0, 6.0], [9, 13], 0.5, [1e-4, 1e-4])
        };
        let det = area.detector(y()).unwrap();
        // right-handed about y takes z to x
        let step = det.pixel(5, 6).unwrap() - det.pixel(4, 6).unwrap();
        assert!((step - Vector3::new(1e-4, 0.0, 0.0)).norm() <= 1e-15);
    }

    #[test]
    fn area_nav_centres_each_binned_pixel_on_its_roi_block() {
        let base = Area::new(dir("z+"), dir("x+"), [4.0, 6.0], [9, 13], 0.5, [1e-4, 2e-4]);
        let full = base.detector(y()).unwrap();
        let det = Area {
            nav: [2, 3],
            roi: Some([1, 9, 2, 13]),
            ..base
        }
        .detector(y())
        .unwrap();
        // 11 fast pixels hold three whole blocks of 3; the remainder is dropped
        assert_eq!(det.shape(), (4, 3));
        for k1 in 0..4 {
            for k2 in 0..3 {
                let mean = (0..2)
                    .flat_map(|a| (0..3).map(move |b| (1 + 2 * k1 + a, 2 + 3 * k2 + b)))
                    .map(|(s, f)| full.pixel(s, f).unwrap().coords)
                    .sum::<Vector3<f64>>()
                    / 6.0;
                assert!(close(det.pixel(k1, k2).unwrap(), Point3::from(mean), 1e-16));
            }
        }
    }

    #[test]
    fn area_rejects_parallel_directions_and_bad_numbers() {
        let base = Area::new(dir("z+"), dir("x+"), [4.0, 6.0], [9, 13], 0.5, [1e-4, 1e-4]);
        let parallel = Area {
            dir2: dir("z-"),
            ..base.clone()
        };
        assert!(matches!(
            parallel.detector(y()),
            Err(Error::ParallelDirections { .. })
        ));
        let bad = [
            Area {
                distance: 0.0,
                ..base.clone()
            },
            Area {
                pwidth2: -1e-4,
                ..base.clone()
            },
        ];
        for area in bad {
            assert!(matches!(area.detector(y()), Err(Error::NotPositive { .. })));
        }
        let nan = [
            Area {
                cch1: f64::NAN,
                ..base.clone()
            },
            Area {
                tilt: Deg(f64::INFINITY),
                ..base.clone()
            },
            Area {
                detrot: Deg(f64::NAN),
                ..base.clone()
            },
        ];
        for area in nan {
            assert!(matches!(area.detector(y()), Err(Error::NotFinite { .. })));
        }
        let roi = Area {
            roi: Some([0, 10, 0, 13]),
            ..base.clone()
        };
        assert!(matches!(roi.detector(y()), Err(Error::Roi { .. })));
        let nav = Area {
            nav: [1, 0],
            ..base
        };
        assert!(matches!(nav.detector(y()), Err(Error::Bin { .. })));
    }

    #[test]
    fn linear_is_one_row_tilted_toward_the_beam() {
        let det = Linear {
            tilt: Deg(30.0),
            ..Linear::new(dir("z+"), 4.0, 9, 1.0, 1e-4)
        }
        .detector(y())
        .unwrap();
        assert_eq!(det.shape(), (1, 9));
        assert!(close(
            det.pixel(0, 4).unwrap(),
            Point3::new(0.0, 1.0, 0.0),
            1e-16
        ));
        let (s, c) = 30f64.to_radians().sin_cos();
        let step = det.pixel(0, 5).unwrap() - det.pixel(0, 4).unwrap();
        assert!((step - Vector3::new(0.0, s, c) * 1e-4).norm() <= 1e-15);
    }

    #[test]
    fn linear_along_the_beam_has_no_tilt_axis() {
        let along = Linear::new(dir("y+"), 4.0, 9, 1.0, 1e-4);
        assert!(along.detector(y()).is_ok());
        let tilted = Linear {
            tilt: Deg(30.0),
            ..along
        };
        assert!(matches!(
            tilted.detector(y()),
            Err(Error::ParallelDirections { .. })
        ));
    }

    #[test]
    fn linear_nav_centres_each_binned_channel_on_its_roi_block() {
        let base = Linear::new(dir("x-"), 4.0, 9, 1.0, 1e-4);
        let full = base.detector(y()).unwrap();
        let det = Linear {
            nav: 3,
            roi: Some([1, 9]),
            ..base
        }
        .detector(y())
        .unwrap();
        assert_eq!(det.shape(), (1, 2));
        for k in 0..2 {
            assert!(close(
                det.pixel(0, k).unwrap(),
                full.pixel(0, 2 + 3 * k).unwrap(),
                1e-16
            ));
        }
    }

    /// splitmix64, so the property cases are fixed without a dependency.
    struct Cases(u64);

    impl Cases {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[test]
    fn point_detector_is_the_centre_pixel_of_an_area_detector() {
        const DIRS: [&str; 6] = ["x+", "x-", "y+", "y-", "z+", "z-"];
        let mut cases = Cases(0x5eed);
        for _ in 0..500 {
            let beam = Unit::new_normalize(Vector3::new(
                cases.range(-1.0, 1.0),
                cases.range(-1.0, 1.0),
                cases.range(-1.0, 1.0),
            ));
            let dir1 = dir(DIRS[cases.below(6)]);
            let dir2 = loop {
                let d = dir(DIRS[cases.below(6)]);
                if d.axis != dir1.axis {
                    break d;
                }
            };
            let nch = [1 + cases.below(40), 1 + cases.below(40)];
            let c = [cases.below(nch[0]), cases.below(nch[1])];
            let lo = [cases.below(c[0] + 1), cases.below(c[1] + 1)];
            let hi = [
                c[0] + 1 + cases.below(nch[0] - c[0]),
                c[1] + 1 + cases.below(nch[1] - c[1]),
            ];
            let distance = cases.range(0.05, 3.0);
            let area = Area {
                tilt: Deg(cases.range(-80.0, 80.0)),
                tiltazimuth: Deg(cases.range(-360.0, 360.0)),
                detrot: Deg(cases.range(-360.0, 360.0)),
                roi: Some([lo[0], hi[0], lo[1], hi[1]]),
                ..Area::new(
                    dir1,
                    dir2,
                    [c[0] as f64, c[1] as f64],
                    nch,
                    distance,
                    [cases.range(1e-5, 1e-3), cases.range(1e-5, 1e-3)],
                )
            };
            let centre = area
                .detector(beam)
                .unwrap()
                .pixel(c[0] - lo[0], c[1] - lo[1])
                .unwrap();
            let point = PixelDetector::point(beam, distance)
                .unwrap()
                .pixel(0, 0)
                .unwrap();
            assert!(close(centre, point, 1e-14), "{area:?} {beam:?}");
        }
    }
}
