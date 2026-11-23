//! The calibration result types: what is a `Fit`, what is an error.

use hklrs::calib::lm::{Failure, Options, solve};
use hklrs::calib::{
    CalFrame, CalibError, Fit, FrameKind, Param, ParamId, Peak, Peak1, Reject, Stop,
};
use hklrs::nalgebra::Vector3;

const T: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
const Y: [f64; 8] = [1.1, 3.9, 7.2, 9.8, 13.1, 16.0, 18.8, 22.2];

fn line(x: &[f64]) -> Vec<f64> {
    T.iter().zip(Y).map(|(t, y)| x[0] + x[1] * t - y).collect()
}

const IDS: [ParamId; 2] = [ParamId::Cch1, ParamId::Pwidth1];

#[test]
fn converged_run_gives_sigma_by_id_and_a_scaled_covariance() {
    let sol = solve(line, &[0.0, 0.0], &[1.0, 1.0], &Options::default()).unwrap();
    let fit = Fit::from_solution("det", &IDS, &sol, vec![(3, Reject::NoSignal)], 1e-12).unwrap();
    assert_eq!(*fit.value(), "det");
    assert_eq!(fit.dof(), 6);
    assert_eq!(fit.rejected(), &[(3, Reject::NoSignal)]);
    assert!(matches!(
        fit.stop(),
        Stop::Gradient | Stop::Cost | Stop::Param
    ));
    // s² (XᵀX)⁻¹ by hand: Σt = 28, Σt² = 140, n = 8.
    let s2 = 2.0 * sol.cost / 6.0;
    let det = 8.0 * 140.0 - 28.0 * 28.0;
    assert!((fit.sigma(ParamId::Cch1).unwrap() - (s2 * 140.0 / det).sqrt()).abs() < 1e-9);
    assert!((fit.sigma(ParamId::Pwidth1).unwrap() - (s2 * 8.0 / det).sqrt()).abs() < 1e-9);
    assert_eq!(fit.sigma(ParamId::Distance), None);
    assert!((fit.rms() - (2.0 * sol.cost / 8.0).sqrt()).abs() < 1e-15);
    assert_eq!(fit.free()[0].id, ParamId::Cch1);
}

#[test]
fn an_unconverged_run_is_an_error_holding_id_value_pairs() {
    let rosen = |x: &[f64]| vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]];
    let opts = Options {
        max_iter: 2,
        ..Options::default()
    };
    let failed = solve(rosen, &[-1.2, 1.0], &[1.0, 1.0], &opts).unwrap_err();
    let best_x = failed.x.clone();
    let CalibError::NotConverged {
        reason,
        best,
        iters,
        ..
    } = CalibError::from_failed(&IDS, failed)
    else {
        panic!("not NotConverged");
    };
    assert_eq!(reason, Failure::MaxIter);
    assert_eq!(iters, 2);
    assert_eq!(
        best,
        vec![(ParamId::Cch1, best_x[0]), (ParamId::Pwidth1, best_x[1])]
    );
}

#[test]
fn bad_solver_input_is_a_bad_problem() {
    let failed = solve(line, &[], &[], &Options::default()).unwrap_err();
    assert_eq!(CalibError::from_failed(&[], failed), CalibError::BadProblem);
}

#[test]
fn unidentifiable_parameters_are_named_by_id_and_no_fit_exists() {
    // Distance enters only as the sum with Pwidth1: the pair is degenerate.
    let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] + x[1] - 5.0, x[0] + x[1] - 4.0];
    let ids = [ParamId::Pwidth1, ParamId::Distance];
    let sol = solve(f, &[0.0, 0.0], &[1.0, 1.0], &Options::default()).unwrap();
    assert_eq!(
        Fit::from_solution((), &ids, &sol, vec![], 1e-12).unwrap_err(),
        CalibError::Unidentifiable {
            params: vec![ParamId::Pwidth1, ParamId::Distance]
        }
    );
}

#[test]
fn as_many_residuals_as_parameters_leaves_no_degrees_of_freedom() {
    let f = |x: &[f64]| vec![x[0] - 1.0, x[1] - 2.0];
    let sol = solve(f, &[0.0, 0.0], &[1.0, 1.0], &Options::default()).unwrap();
    assert_eq!(
        Fit::from_solution((), &IDS, &sol, vec![], 1e-12).unwrap_err(),
        CalibError::TooFewFrames { have: 2, need: 3 }
    );
}

#[test]
fn param_start_must_be_finite() {
    assert!(Param::free(1.0).check(ParamId::Tilt).is_ok());
    assert_eq!(
        Param::fixed(f64::NAN)
            .check(ParamId::Tilt)
            .map_err(|e| matches!(
                e,
                CalibError::BadStart {
                    id: ParamId::Tilt,
                    ..
                }
            )),
        Err(true)
    );
    assert!(!Param::fixed(2.0).free);
}

#[test]
fn frames_are_validated_where_they_are_made() {
    let peak = Peak {
        pos: [10.0, 20.0],
        sigma: [0.1, 0.1],
    };
    let ok = CalFrame::new(0, vec![0.1, 0.2], peak, FrameKind::PrimaryBeam).unwrap();
    assert_eq!(ok.angles(), &[0.1, 0.2]);
    let bad = |what: &str, r: Result<CalFrame, CalibError>| match r {
        Err(CalibError::BadFrame { index: 7, what: w }) => assert_eq!(w, what),
        other => panic!("{other:?}"),
    };
    bad(
        "angle is not finite",
        CalFrame::new(7, vec![f64::NAN], peak, FrameKind::PrimaryBeam),
    );
    bad(
        "peak position is not finite",
        CalFrame::new(
            7,
            vec![0.0],
            Peak {
                pos: [f64::NAN, 0.0],
                ..peak
            },
            FrameKind::PrimaryBeam,
        ),
    );
    bad(
        "peak sigma is not positive",
        CalFrame::new(
            7,
            vec![0.0],
            Peak {
                sigma: [0.0, 1.0],
                ..peak
            },
            FrameKind::PrimaryBeam,
        ),
    );
    bad(
        "peak sigma is infinite on both axes",
        CalFrame::new(
            7,
            vec![0.0],
            Peak {
                sigma: [f64::INFINITY; 2],
                ..peak
            },
            FrameKind::PrimaryBeam,
        ),
    );
    bad(
        "hkl is not finite",
        CalFrame::new(
            7,
            vec![0.0],
            peak,
            FrameKind::Reflection(Vector3::new(0.0, f64::NAN, 4.0)),
        ),
    );
}

#[test]
fn a_row_peak_carries_no_slow_axis_information() {
    let p: Peak = Peak1 {
        pos: 123.4,
        sigma: 0.05,
    }
    .into();
    assert_eq!(p.pos, [0.0, 123.4]);
    assert!(p.sigma[0].is_infinite() && p.sigma[1] == 0.05);
    assert!(CalFrame::new(0, vec![0.0], p, FrameKind::PrimaryBeam).is_ok());
}
