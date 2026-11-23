//! [`Cube`] files in binoculars-ng 7be4f79's layout
//! (hkl-binoculars-io-hdf5.c), so its tools open hklrs cubes and hklrs
//! opens theirs.
//!
//! Group `/binoculars` holds `config`, one fixed-length string; group
//! `axes`, one dataset per axis named after it, six doubles
//! `[index, first·step, last·step, step, first, last]`; and `counts` and
//! `contributions`, one value per bin, shaped like the axes, last axis
//! fastest. binoculars reads index, step, first and last, and centres bin
//! k on k·step, which is anchor −step/2; no other anchor has a place in
//! the file.
//!
//! hklrs writes counts as f64 and contributions as u64 (binoculars writes
//! f32 and u32 and has HDF5 convert on read), and writes axes of one bin,
//! which binoculars drops (BIN-14). Σweight goes beside them in `weight`,
//! f64 and shaped like counts; a file without it, as binoculars writes,
//! reads with Σweight = contributions, binoculars' meaning of
//! photons/contributions. New data goes in attributes or in datasets of
//! `/binoculars` beside these, which binoculars ignores, never in `axes`,
//! whose every link binoculars reads as an axis.

use std::path::Path;

use rust_hdf5::{DatatypeMessage, H5Dataset, H5File, Hdf5Error};

use super::{Axis, Binning, Cube};
use crate::Error;

const ROOT: &str = "binoculars";
const AXES: &str = "binoculars/axes";
/// Σweight, which binoculars' files lack.
const WEIGHT: &str = "weight";

fn h5(e: Hdf5Error) -> Error {
    Error::Hdf5 {
        reason: e.to_string(),
    }
}

fn layout(reason: String) -> Error {
    Error::CubeFile { reason }
}

/// binoculars' bins of width `step`: bin k centred on k·step.
fn binoculars(step: f64) -> Result<Binning, Error> {
    Binning::new(step, -step / 2.0)
}

/// Bin indices are stored as doubles; beyond 2^53 they would not read
/// back as the same integer.
const EXACT: f64 = (1u64 << f64::MANTISSA_DIGITS) as f64;

fn exact(k: i64, name: &str) -> Result<f64, Error> {
    let x = k as f64;
    if x.abs() <= EXACT {
        Ok(x)
    } else {
        Err(layout(format!(
            "axis {name:?}: bin {k} has no exact double"
        )))
    }
}

/// Writes `cube` and `config` to a new file at `path`, replacing any file
/// there. The empty cube has no axes to write and is refused, as are axis
/// names HDF5 cannot use as a link name and axes binned other than
/// binoculars' way, which its tools would show shifted.
pub fn write(path: impl AsRef<Path>, cube: &Cube, config: &str) -> Result<(), Error> {
    if cube.is_empty() {
        return Err(layout("the empty cube has no axes".to_owned()));
    }
    for a in &cube.axes {
        if a.name.is_empty() || a.name == "." || a.name.contains('/') {
            return Err(layout(format!("axis name {:?} is not a link name", a.name)));
        }
        let (step, anchor) = (a.binning.step, a.binning.anchor);
        if a.binning != binoculars(step)? {
            return Err(Error::CubeFileAnchor {
                axis: a.name.clone(),
                step,
                anchor,
            });
        }
    }
    let file = H5File::create(path).map_err(h5)?;
    let root = file.create_group(ROOT).map_err(h5)?;

    // NUL-terminated, so an empty config is still a string of width 1
    let mut text = config.as_bytes().to_vec();
    text.push(0);
    let width = u32::try_from(text.len())
        .map_err(|_| layout(format!("config of {} bytes is too long", config.len())))?;
    root.new_dataset::<u8>()
        .shape([1])
        .datatype(DatatypeMessage::fixed_string(width))
        .create("config")
        .map_err(h5)?
        .write_raw_bytes(&text)
        .map_err(h5)?;

    let axes = root.create_group("axes").map_err(h5)?;
    for (i, a) in cube.axes.iter().enumerate() {
        let step = a.binning.step;
        let (first, last) = (exact(a.first, &a.name)?, exact(a.last(), &a.name)?);
        let ds = axes
            .new_dataset::<f64>()
            .shape([6])
            .create(&a.name)
            .map_err(h5)?;
        ds.write_raw(&[i as f64, first * step, last * step, step, first, last])
            .map_err(h5)?;
    }

    let shape: Vec<usize> = cube.axes.iter().map(|a| a.size).collect();
    root.new_dataset::<f64>()
        .shape(&shape)
        .create("counts")
        .map_err(h5)?
        .write_raw(&cube.photons)
        .map_err(h5)?;
    root.new_dataset::<u64>()
        .shape(&shape)
        .create("contributions")
        .map_err(h5)?
        .write_raw(&cube.contributions)
        .map_err(h5)?;
    root.new_dataset::<f64>()
        .shape(&shape)
        .create(WEIGHT)
        .map_err(h5)?
        .write_raw(&cube.weight)
        .map_err(h5)?;
    file.close().map_err(h5)
}

/// Reads the cube and config at `path`. Every link in `axes` must be an
/// axis of six values whose index column numbers them 0..n once each, and
/// counts, contributions and any weight must have the axes' shape. Every
/// axis gets binoculars' bins, anchored at −step/2.
pub fn read(path: impl AsRef<Path>) -> Result<(Cube, String), Error> {
    let file = H5File::open(path).map_err(h5)?;
    let root = file.root_group().group(ROOT).map_err(h5)?;
    let names = root
        .group("axes")
        .and_then(|g| g.link_names())
        .map_err(h5)?;
    let n = names.len();
    if n == 0 {
        return Err(layout("the axes group is empty".to_owned()));
    }
    let mut slots: Vec<Option<Axis>> = vec![None; n];
    for name in &names {
        let ds = file.dataset(&format!("{AXES}/{name}")).map_err(h5)?;
        let (index, axis) = axis(&ds, name)?;
        let slot = slots
            .get_mut(index)
            .ok_or_else(|| layout(format!("axis {name:?}: index {index} is not below {n}")))?;
        if slot.replace(axis).is_some() {
            return Err(layout(format!("axis index {index} appears twice")));
        }
    }
    // n distinct indices below n fill every slot
    let axes: Vec<Axis> = slots.into_iter().flatten().collect();

    let shape: Vec<usize> = axes.iter().map(|a| a.size).collect();
    let bins = |name: &str| -> Result<H5Dataset, Error> {
        let ds = file.dataset(&format!("{ROOT}/{name}")).map_err(h5)?;
        if ds.shape() == shape {
            Ok(ds)
        } else {
            Err(layout(format!(
                "{name} has shape {:?}, the axes {shape:?}",
                ds.shape()
            )))
        }
    };
    let photons = bins("counts")?.read_numeric_as::<f64>().map_err(h5)?;
    let contributions = bins("contributions")?
        .read_numeric_as::<u64>()
        .map_err(h5)?;
    let weight = if root.link_names().map_err(h5)?.iter().any(|n| n == WEIGHT) {
        bins(WEIGHT)?.read_numeric_as::<f64>().map_err(h5)?
    } else {
        contributions.iter().map(|&c| c as f64).collect()
    };

    let config = file
        .dataset(&format!("{ROOT}/config"))
        .and_then(|d| d.read_strings())
        .map_err(h5)?;
    let [config] = <[String; 1]>::try_from(config)
        .map_err(|v| layout(format!("config holds {} strings, not 1", v.len())))?;
    Ok((
        Cube {
            axes,
            photons,
            weight,
            contributions,
        },
        config,
    ))
}

/// The index column and the axis of one dataset of `axes`.
fn axis(ds: &H5Dataset, name: &str) -> Result<(usize, Axis), Error> {
    let values = ds.read_numeric_as::<f64>().map_err(h5)?;
    let [index, _, _, step, first, last] = <[f64; 6]>::try_from(values)
        .map_err(|v| layout(format!("axis {name:?} holds {} values, not 6", v.len())))?;
    let int = |x: f64, what: &str| {
        if x.fract() == 0.0 && x.abs() <= EXACT {
            Ok(x as i64)
        } else {
            Err(layout(format!(
                "axis {name:?}: {what} {x} is not an integer"
            )))
        }
    };
    let index = usize::try_from(int(index, "index")?)
        .map_err(|_| layout(format!("axis {name:?}: index {index} is negative")))?;
    let (first, last) = (int(first, "first bin")?, int(last, "last bin")?);
    if first > last {
        return Err(layout(format!(
            "axis {name:?}: first bin {first} is past last bin {last}"
        )));
    }
    Ok((
        index,
        Axis::spanning(name, binoculars(step)?, (first, last))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A file of this test process; nextest runs each test in its own.
    fn path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hklrs-{}-{name}.h5", std::process::id()))
    }

    fn cube(axes: Vec<Axis>) -> Cube {
        let mut c = Cube::zeros(axes).unwrap();
        let bins = c.photons.iter_mut().zip(&mut c.weight);
        for (i, ((p, w), n)) in bins.zip(&mut c.contributions).enumerate() {
            // none fits binoculars' f32 and u32, and Σweight is not contributions
            *p = 0.1 * i as f64;
            *w = 0.7 * i as f64;
            *n = (1 << 40) + i as u64;
        }
        c
    }

    fn spanning(name: &str, step: f64, anchor: f64, bins: (i64, i64)) -> Axis {
        Axis::spanning(name, Binning::new(step, anchor).unwrap(), bins).unwrap()
    }

    /// binoculars drops one-bin axes (BIN-14) and cannot save an empty
    /// config (BIN-17).
    #[test]
    fn round_trip_keeps_one_bin_axes_and_full_precision() {
        let c = cube(vec![
            spanning("h", 0.5, -0.25, (-3, 1)),
            spanning("k", 0.25, -0.125, (7, 7)),
            spanning("l", 2.0, -1.0, (0, 2)),
        ]);
        for config in ["", "[input]\ntype = \"x\"\n"] {
            let p = path("round_trip");
            write(&p, &c, config).unwrap();
            assert_eq!(read(&p).unwrap(), (c.clone(), config.to_owned()));
        }
    }

    #[test]
    fn write_refuses_the_empty_cube_and_names_that_are_not_links() {
        let p = path("refused");
        let refused = |c: &Cube| matches!(write(&p, c, ""), Err(Error::CubeFile { .. }));
        assert!(refused(&Cube::default()));
        for name in ["", ".", "q/x"] {
            assert!(
                refused(&cube(vec![spanning(name, 1.0, -0.5, (0, 1))])),
                "{name:?}"
            );
        }
    }

    /// Anchor step/2 is binoculars' lattice with every index one off.
    #[test]
    fn write_refuses_anchors_binoculars_would_shift() {
        let p = path("shifted");
        for anchor in [0.0, 0.25, -0.2, -0.75] {
            let c = cube(vec![
                spanning("qx", 0.5, -0.25, (0, 1)),
                spanning("qy", 0.5, anchor, (0, 1)),
            ]);
            match write(&p, &c, "") {
                Err(Error::CubeFileAnchor {
                    axis,
                    step,
                    anchor: a,
                }) => {
                    assert_eq!((axis.as_str(), step, a), ("qy", 0.5, anchor))
                }
                other => panic!("{anchor}: {other:?}"),
            }
            assert!(!p.exists(), "{anchor}: refused before the file");
        }
    }

    /// A file as binoculars writes it: f32 counts, u32 contributions, no
    /// anchor, with the given axis rows and bin shape.
    fn binoculars(name: &str, rows: &[(&str, &[f64])], shape: &[usize]) -> PathBuf {
        file(name, rows, shape, None)
    }

    /// [`binoculars`]' file, with a weight of shape `weight` when given.
    fn file(
        name: &str,
        rows: &[(&str, &[f64])],
        shape: &[usize],
        weight: Option<&[usize]>,
    ) -> PathBuf {
        let p = path(name);
        let f = H5File::create(&p).unwrap();
        let root = f.create_group(ROOT).unwrap();
        root.new_dataset::<u8>()
            .shape([1])
            .datatype(DatatypeMessage::fixed_string(4))
            .create("config")
            .unwrap()
            .write_raw_bytes(b"cfg\0")
            .unwrap();
        let axes = root.create_group("axes").unwrap();
        for (axis, row) in rows {
            axes.new_dataset::<f64>()
                .shape([row.len()])
                .create(axis)
                .unwrap()
                .write_raw(row)
                .unwrap();
        }
        let n = shape.iter().product();
        root.new_dataset::<f32>()
            .shape(shape)
            .create("counts")
            .unwrap()
            .write_raw(&vec![1.5f32; n])
            .unwrap();
        root.new_dataset::<u32>()
            .shape(shape)
            .create("contributions")
            .unwrap()
            .write_raw(&vec![3u32; n])
            .unwrap();
        if let Some(shape) = weight {
            root.new_dataset::<f64>()
                .shape(shape)
                .create(WEIGHT)
                .unwrap()
                .write_raw(&vec![0.25; shape.iter().product()])
                .unwrap();
        }
        f.close().unwrap();
        p
    }

    #[test]
    fn binoculars_axes_are_anchored_half_a_step_below_zero() {
        let p = binoculars(
            "binoculars",
            &[
                ("qy", &[1.0, -0.5, 0.0, 0.5, -1.0, 0.0]),
                ("qx", &[0.0, 1.0, 2.0, 0.5, 2.0, 4.0]),
            ],
            &[3, 2],
        );
        let want = Cube {
            axes: vec![
                spanning("qx", 0.5, -0.25, (2, 4)),
                spanning("qy", 0.5, -0.25, (-1, 0)),
            ],
            photons: vec![1.5; 6],
            weight: vec![3.0; 6],
            contributions: vec![3; 6],
        };
        assert_eq!(read(&p).unwrap(), (want, "cfg".to_owned()));
    }

    /// Σweight is read where the file has it, and must have the axes' shape.
    #[test]
    fn weight_is_read_where_the_file_has_one() {
        let row: &[f64] = &[0.0, 0.0, 1.0, 0.5, 0.0, 2.0];
        let (cube, _) = read(file("weight", &[("qx", row)], &[3], Some(&[3]))).unwrap();
        assert_eq!(
            (cube.weight(), cube.contributions()),
            (&[0.25; 3][..], &[3; 3][..])
        );
        match read(file("weight_shape", &[("qx", row)], &[3], Some(&[2]))) {
            Err(Error::CubeFile { reason }) => {
                assert!(reason.contains("weight has shape [2]"), "{reason}")
            }
            other => panic!("{other:?}"),
        }
    }

    /// binoculars' loader crashes on an empty axes group or a bad index
    /// (BIN-15) and reads an axis of other than 6 values past its row or
    /// the dataset (BIN-16).
    #[test]
    fn read_refuses_axes_binoculars_would_misread() {
        type Case<'a> = (&'a str, &'a [(&'a str, &'a [f64])], &'a [usize], &'a str);
        let row = [0.0, 0.0, 1.0, 0.5, 0.0, 2.0];
        let seven = [0.0, 0.0, 1.0, 0.5, 0.0, 2.0, 9.0];
        let cases: [Case; 8] = [
            ("no_axes", &[], &[3], "axes group is empty"),
            ("five", &[("qx", &row[..5])], &[3], "holds 5 values"),
            ("seven", &[("qx", &seven)], &[3], "holds 7 values"),
            (
                "twice",
                &[("qx", &row), ("qy", &row)],
                &[3, 3],
                "appears twice",
            ),
            (
                "past_n",
                &[("qx", &[1.0, 0.0, 1.0, 0.5, 0.0, 2.0])],
                &[3],
                "not below 1",
            ),
            (
                "half",
                &[("qx", &[0.0, 0.0, 1.0, 0.5, 0.5, 2.0])],
                &[3],
                "0.5 is not an",
            ),
            (
                "reversed",
                &[("qx", &[0.0, 0.0, 1.0, 0.5, 2.0, 0.0])],
                &[3],
                "is past last",
            ),
            ("shape", &[("qx", &row)], &[4], "shape [4], the axes [3]"),
        ];
        for (name, rows, shape, why) in cases {
            match read(binoculars(name, rows, shape)) {
                Err(Error::CubeFile { reason }) => {
                    assert!(reason.contains(why), "{name}: {reason}")
                }
                other => panic!("{name}: {other:?}"),
            }
        }
    }
}
