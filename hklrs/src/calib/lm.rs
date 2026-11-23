//! Levenberg–Marquardt least squares with typed stops.
//!
//! Residuals are minimised in parameters divided by a caller-given scale, so
//! every internal quantity is O(1) whatever the units (pixels, metres,
//! degrees). The Jacobian is a central difference. A run either returns a
//! [`Solution`] with the reason it stopped, or a [`Failed`] with the reason
//! it did not; there is no result that is a start value in disguise.

use nalgebra::{Cholesky, DMatrix, DVector, SymmetricEigen};

/// Damping ceiling, relative to the scaled normal matrix diagonal.
const MU_MAX: f64 = 1e16;

/// Null-direction components below this share are not named unidentifiable.
const NULL_SHARE: f64 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    pub max_iter: usize,
    /// Converged when the largest component of `Jᵀr` (scaled) is at most this.
    pub gtol: f64,
    /// Converged when an accepted step lowers the cost by at most this share.
    pub ftol: f64,
    /// Converged when an accepted step is shorter than this share of |u|.
    pub xtol: f64,
    /// Relative central-difference step in scaled parameters.
    pub fd_step: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1e-10,
            ftol: 1e-12,
            xtol: 1e-10,
            fd_step: 1e-6,
        }
    }
}

/// Why a run counts as converged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Gradient,
    Cost,
    Param,
}

/// Why a run did not converge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The iteration cap was reached while still improving.
    MaxIter,
    /// No damping gave a step that lowers the cost.
    StalledStep,
    /// The residuals were NaN or infinite where they had to be finite.
    NonFinite,
    /// Empty or non-finite start, non-positive scale, or fewer residuals
    /// than parameters.
    BadInput,
}

/// A run that did not converge; `x` and `cost` are the best point reached.
#[derive(Clone, Debug, PartialEq)]
pub struct Failed {
    pub reason: Failure,
    pub x: Vec<f64>,
    pub cost: f64,
    pub iters: usize,
}

/// A converged run.
#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Vec<f64>,
    /// `½ Σ r²`.
    pub cost: f64,
    pub iters: usize,
    pub stop: Stop,
    /// Number of residuals.
    pub m: usize,
    scale: Vec<f64>,
    /// `JᵀJ` at `x`, in scaled parameters.
    jtj: DMatrix<f64>,
}

impl Solution {
    /// `(JᵀJ)⁻¹` in parameter units, without the residual-variance factor.
    /// `Err` names the parameters (indices into `x`) that no combination of
    /// the data pins down: an eigenvalue of the scaled `JᵀJ` at most
    /// `rank_tol` of the largest, with the parameters carrying that null
    /// direction. A parameter the residuals do not depend on is named alone.
    pub fn covariance(&self, rank_tol: f64) -> Result<DMatrix<f64>, Vec<usize>> {
        let n = self.x.len();
        let eig = SymmetricEigen::new(self.jtj.clone());
        let lmax = eig.eigenvalues.max();
        let floor = rank_tol * lmax;
        let mut named = vec![false; n];
        let mut any = false;
        for (k, &l) in eig.eigenvalues.iter().enumerate() {
            if l.is_nan() || l <= floor {
                any = true;
                for (j, flag) in named.iter_mut().enumerate() {
                    *flag |= eig.eigenvectors[(j, k)].powi(2) >= NULL_SHARE;
                }
            }
        }
        if any {
            return Err((0..n).filter(|&j| named[j]).collect());
        }
        let mut inv = DMatrix::zeros(n, n);
        for k in 0..n {
            let v = eig.eigenvectors.column(k);
            inv += (v * v.transpose()) / eig.eigenvalues[k];
        }
        for i in 0..n {
            for j in 0..n {
                inv[(i, j)] *= self.scale[i] * self.scale[j];
            }
        }
        Ok(inv)
    }
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn jacobian<F: FnMut(&[f64]) -> Vec<f64>>(
    f: &mut F,
    u: &DVector<f64>,
    scale: &[f64],
    m: usize,
    fd_step: f64,
) -> Option<DMatrix<f64>> {
    let n = u.len();
    let mut jac = DMatrix::zeros(m, n);
    let mut x: Vec<f64> = (0..n).map(|i| u[i] * scale[i]).collect();
    for j in 0..n {
        let h = fd_step * u[j].abs().max(1.0);
        let (up, um) = (u[j] + h, u[j] - h);
        let xj = x[j];
        x[j] = up * scale[j];
        let rp = f(&x);
        x[j] = um * scale[j];
        let rm = f(&x);
        x[j] = xj;
        if rp.len() != m || rm.len() != m || !all_finite(&rp) || !all_finite(&rm) {
            return None;
        }
        let span = up - um;
        for i in 0..m {
            jac[(i, j)] = (rp[i] - rm[i]) / span;
        }
    }
    Some(jac)
}

/// Minimise `½ Σ f(x)²` from `x0`; `scale[i]` is the typical size of `x[i]`.
pub fn solve<F: FnMut(&[f64]) -> Vec<f64>>(
    mut f: F,
    x0: &[f64],
    scale: &[f64],
    opts: &Options,
) -> Result<Solution, Failed> {
    let n = x0.len();
    let fail = |reason, x: Vec<f64>, cost, iters| {
        Err(Failed {
            reason,
            x,
            cost,
            iters,
        })
    };
    if n == 0
        || scale.len() != n
        || !all_finite(x0)
        || scale.iter().any(|s| !(s.is_finite() && *s > 0.0))
    {
        return fail(Failure::BadInput, x0.to_vec(), f64::NAN, 0);
    }
    let x_of = |u: &DVector<f64>| -> Vec<f64> { (0..n).map(|i| u[i] * scale[i]).collect() };
    let mut u = DVector::from_fn(n, |i, _| x0[i] / scale[i]);
    let r0 = f(x0);
    if r0.len() < n {
        return fail(Failure::BadInput, x0.to_vec(), f64::NAN, 0);
    }
    if !all_finite(&r0) {
        return fail(Failure::NonFinite, x0.to_vec(), f64::NAN, 0);
    }
    let m = r0.len();
    let mut r = DVector::from_vec(r0);
    let mut cost = 0.5 * r.norm_squared();
    let (mut mu, mut nu) = (f64::NAN, 2.0);
    let done = |u: &DVector<f64>, cost, iters, stop, jtj| Solution {
        x: x_of(u),
        cost,
        iters,
        stop,
        m,
        scale: scale.to_vec(),
        jtj,
    };
    for iter in 0..opts.max_iter {
        let Some(jac) = jacobian(&mut f, &u, scale, m, opts.fd_step) else {
            return fail(Failure::NonFinite, x_of(&u), cost, iter);
        };
        let a = jac.tr_mul(&jac);
        let g = jac.tr_mul(&r);
        if g.amax() <= opts.gtol {
            return Ok(done(&u, cost, iter, Stop::Gradient, a));
        }
        let maxdiag = a.diagonal().max();
        if mu.is_nan() {
            mu = 1e-3;
        }
        loop {
            if mu > MU_MAX {
                return fail(Failure::StalledStep, x_of(&u), cost, iter);
            }
            let dvec = DVector::from_fn(n, |j, _| a[(j, j)].max(1e-8 * maxdiag));
            let mut damped = a.clone();
            for j in 0..n {
                damped[(j, j)] += mu * dvec[j];
            }
            let Some(chol) = Cholesky::new(damped) else {
                mu *= nu;
                nu *= 2.0;
                continue;
            };
            let delta = chol.solve(&(-&g));
            let trial_u = &u + &delta;
            let tiny = delta.norm() <= opts.xtol * (u.norm() + opts.xtol);
            let trial = f(&x_of(&trial_u));
            let trial_cost = if trial.len() == m && all_finite(&trial) {
                0.5 * trial.iter().map(|v| v * v).sum::<f64>()
            } else {
                f64::NAN
            };
            if trial_cost < cost {
                let pred = 0.5 * delta.dot(&(mu * dvec.component_mul(&delta) - &g));
                let rho = (cost - trial_cost) / pred.max(f64::MIN_POSITIVE);
                let small_gain = cost - trial_cost <= opts.ftol * trial_cost;
                u = trial_u;
                r = DVector::from_vec(trial);
                cost = trial_cost;
                mu *= (1.0 / 3.0_f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
                nu = 2.0;
                if tiny || small_gain {
                    let Some(jac) = jacobian(&mut f, &u, scale, m, opts.fd_step) else {
                        return fail(Failure::NonFinite, x_of(&u), cost, iter + 1);
                    };
                    let stop = if tiny { Stop::Param } else { Stop::Cost };
                    return Ok(done(&u, cost, iter + 1, stop, jac.tr_mul(&jac)));
                }
                break;
            }
            // A step too short to move the cost is the floor of the problem,
            // not a stall; a short step that made the cost worse is damping
            // run away.
            if tiny && trial_cost <= cost * (1.0 + opts.ftol) {
                return Ok(done(&u, cost, iter, Stop::Param, a));
            }
            mu *= nu;
            nu *= 2.0;
        }
    }
    fail(Failure::MaxIter, x_of(&u), cost, opts.max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    const Y: [f64; 8] = [1.1, 3.9, 7.2, 9.8, 13.1, 16.0, 18.8, 22.2];

    fn line(x: &[f64]) -> Vec<f64> {
        T.iter().zip(Y).map(|(t, y)| x[0] + x[1] * t - y).collect()
    }

    fn rosenbrock(x: &[f64]) -> Vec<f64> {
        vec![10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]]
    }

    #[test]
    fn line_matches_normal_equations_and_covariance() {
        let sol = solve(line, &[0.0, 0.0], &[1.0, 1.0], &Options::default()).unwrap();
        let n = T.len() as f64;
        let (st, st2) = (T.iter().sum::<f64>(), T.iter().map(|t| t * t).sum::<f64>());
        let sy = Y.iter().sum::<f64>();
        let sty = T.iter().zip(Y).map(|(t, y)| t * y).sum::<f64>();
        let det = n * st2 - st * st;
        let (a, b) = ((st2 * sy - st * sty) / det, (n * sty - st * sy) / det);
        assert!((sol.x[0] - a).abs() < 1e-9 && (sol.x[1] - b).abs() < 1e-9);
        let cov = sol.covariance(1e-12).unwrap();
        let want = [[st2 / det, -st / det], [-st / det, n / det]];
        for i in 0..2 {
            for j in 0..2 {
                assert!((cov[(i, j)] - want[i][j]).abs() < 1e-8 * want[i][j].abs().max(1.0));
            }
        }
        assert_eq!(sol.m, 8);
    }

    #[test]
    fn scale_makes_units_irrelevant() {
        // a ~ 2e6, b ~ 3e-6: unscaled, the two columns differ by 12 decades.
        let f = |x: &[f64]| -> Vec<f64> {
            T.iter()
                .map(|t| x[0] * 1e-6 * t + x[1] * 1e6 - (2.0 * t + 3.0))
                .collect()
        };
        let sol = solve(f, &[1.0e6, 1.0e-6], &[1e6, 1e-6], &Options::default()).unwrap();
        assert!((sol.x[0] - 2.0e6).abs() < 1e-3, "{:?}", sol.x);
        assert!((sol.x[1] - 3.0e-6).abs() < 1e-12, "{:?}", sol.x);
    }

    #[test]
    fn rosenbrock_reaches_its_minimum() {
        let sol = solve(rosenbrock, &[-1.2, 1.0], &[1.0, 1.0], &Options::default()).unwrap();
        assert!((sol.x[0] - 1.0).abs() < 1e-8 && (sol.x[1] - 1.0).abs() < 1e-8);
        assert!(sol.cost < 1e-16);
    }

    #[test]
    fn iteration_cap_is_a_failure_with_the_best_point() {
        let opts = Options {
            max_iter: 2,
            ..Options::default()
        };
        let err = solve(rosenbrock, &[-1.2, 1.0], &[1.0, 1.0], &opts).unwrap_err();
        assert_eq!(err.reason, Failure::MaxIter);
        assert_eq!(err.x.len(), 2);
        assert!(err.cost < 0.5 * (rosenbrock(&[-1.2, 1.0]).iter().map(|v| v * v).sum::<f64>()));
    }

    #[test]
    fn non_finite_start_fails() {
        let err = solve(|_| vec![f64::NAN, 0.0], &[0.0], &[1.0], &Options::default()).unwrap_err();
        assert_eq!(err.reason, Failure::NonFinite);
    }

    #[test]
    fn bad_input_fails() {
        let o = Options::default();
        assert_eq!(
            solve(line, &[0.0, 0.0], &[1.0, 0.0], &o)
                .unwrap_err()
                .reason,
            Failure::BadInput
        );
        assert_eq!(
            solve(line, &[], &[], &o).unwrap_err().reason,
            Failure::BadInput
        );
        assert_eq!(
            solve(|_| vec![1.0], &[0.0, 0.0], &[1.0, 1.0], &o)
                .unwrap_err()
                .reason,
            Failure::BadInput
        );
    }

    #[test]
    fn a_cost_that_never_improves_is_a_stall() {
        // Each call returns a larger residual than the one before, so no
        // trial can beat the stored cost.
        let mut calls = 0.0;
        let f = move |x: &[f64]| {
            calls += 1.0;
            vec![(1.0 + 0.1 * calls) * (x[0] - 1.0 + 2.0)]
        };
        let err = solve(f, &[0.0], &[1.0], &Options::default()).unwrap_err();
        assert_eq!(err.reason, Failure::StalledStep);
    }

    #[test]
    fn unused_parameter_is_named_alone() {
        let f = |x: &[f64]| vec![x[0] - 1.0, x[0] - 2.0];
        let sol = solve(f, &[0.0, 5.0], &[1.0, 1.0], &Options::default()).unwrap();
        assert_eq!(sol.covariance(1e-12).unwrap_err(), vec![1]);
    }

    #[test]
    fn collinear_parameters_are_named_together() {
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] + x[1] - 5.0];
        let sol = solve(f, &[0.0, 0.0], &[1.0, 1.0], &Options::default()).unwrap();
        assert_eq!(sol.covariance(1e-12).unwrap_err(), vec![0, 1]);
    }
}
