//! hklrs's `Projection::Hkl` against Python binoculars on the Pt(111) scans
//! ([`pt111`]): the cube `derived/Pt111-3_HR_hkl_63-92.hdf5`, which is not
//! in the deposit. Python binoculars 93956b2 wrote it from the deposit's
//! scans with libhkl 7be4f79, pyFAI 0.19.0 and numpy 1.23, by
//! `derived/run_hkl_cube.sh` and `derived/Pt111-3_HR_63-92_hkl.cfg`: the
//! deposit cube's input with the scans' mask as maskmatrix, projection
//! sixs:hklprojection, resolution 0.0015, 0.0015, 0.005, no sample
//! overrides.
//!
//! ```text
//! HKLRS_ZENODO_6787572=<deposit> cargo nextest run -p hklrs --features hdf5 \
//!     --run-ignored only --no-capture -E 'binary(binoculars_pt111_hkl)'
//! ```

mod pt111;

use hklrs::nalgebra::Matrix3;
use hklrs::projection::Projection;
use hklrs::{Deg, Lattice, Orientation, Sample};

/// The NXdiffractometer binoculars' get_sample reads, the first one
/// get_nxclass finds (sixs.py:119-165); diff-uhv records the same sample.
const DIFFRACTOMETER: &str = "com/SIXS/i14-c-cx2-ex-cm-uhv";

#[test]
#[ignore = "needs the Zenodo 10.5281/zenodo.6787572 deposit in HKLRS_ZENODO_6787572"]
fn hkl_cube_matches_python_binoculars() {
    let deposit = pt111::deposit();
    let scans = pt111::scans(&deposit);
    // binoculars builds each scan's sample from its own file: one sample
    // stands for all only if every scan records the same
    let recorded = |scan: &pt111::Scan| {
        ["A", "B", "C", "alpha", "beta", "gamma", "Ux", "Uy", "Uz"]
            .map(|k| scan.read::<f64>(&format!("{DIFFRACTOMETER}/{k}"))[0])
    };
    let values = recorded(&scans[0]);
    for scan in &scans[1..] {
        assert_eq!(recorded(scan), values);
    }
    // libhkl takes the angles in degrees (USER units) and U = Rx·Ry·Rz
    let [a, b, c, alpha, beta, gamma, ux, uy, uz] = values;
    let rad = |d: f64| Deg(d).into();
    let sample = Sample::new(
        "Pt111",
        Lattice::new(a, b, c, rad(alpha), rad(beta), rad(gamma)).unwrap(),
        Orientation::from_ux_uy_uz(rad(ux), rad(uy), rad(uz)).unwrap(),
    );
    // the UB libhkl computed from those values on the beamline
    let file = Matrix3::from_row_slice(&scans[0].read::<f64>(&format!("{DIFFRACTOMETER}/UB")));
    let ub = sample.ub();
    assert!(
        (ub - file).amax() <= 4.0 * f64::EPSILON * file.amax(),
        "UB {ub} recorded {file}"
    );
    pt111::compare(
        &scans,
        Projection::Hkl(sample),
        [0.0015, 0.0015, 0.005],
        &deposit.join("derived/Pt111-3_HR_hkl_63-92.hdf5"),
        ["H", "K", "L"],
    );
}
