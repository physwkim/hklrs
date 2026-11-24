//! hklrs against Python binoculars on a recorded scan ([`pt111`]): the
//! Pt(111) cube `voxel_map/Pt111-3_HR_63-92.hdf5` of the deposit, which
//! Python binoculars wrote in October 2020 with projection
//! sixs:qxqyqzprojection, resolution 0.002, 0.002, 0.005 Å⁻¹ and the
//! default VERTICAL surface, under pyFAI 0.19.0 and numpy 1.
//!
//! ```text
//! HKLRS_ZENODO_6787572=<deposit> cargo nextest run -p hklrs --features hdf5 \
//!     --run-ignored only --no-capture -E 'binary(binoculars_pt111)'
//! ```

mod pt111;

use hklrs::nalgebra::{Matrix3, Rotation3};
use hklrs::projection::Projection;

#[test]
#[ignore = "needs the Zenodo 10.5281/zenodo.6787572 deposit in HKLRS_ZENODO_6787572"]
fn cube_matches_python_binoculars() {
    let deposit = pt111::deposit();
    // binoculars' UB for a VERTICAL surface (sixs.py:376-377)
    let vertical = Rotation3::from_matrix_unchecked(Matrix3::new(
        1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, //
        0.0, -1.0, 0.0,
    ));
    pt111::compare(
        &pt111::scans(&deposit),
        Projection::QSurface { frame: vertical },
        [0.002, 0.002, 0.005],
        &deposit.join("voxel_map/Pt111-3_HR_63-92.hdf5"),
        ["Qx", "Qy", "Qz"],
    );
}
