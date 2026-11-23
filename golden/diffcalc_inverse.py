"""Golden inverse (hkl -> angles) from diffcalc-core 059ec8a (v0.4.0-9), You (1999) closed form.
Run: <py2 python> -I diffcalc_inverse.py [--workers N]

Per case a UB (lattice + U), a wavelength, a target hkl and three constraints are fed to
HklCalculation.get_position. Every returned solution (degrees, diffcalc's virtual angles) or the error text
is stored. Everything that judges a solution is independent numpy: the rotation matrices are rebuilt from the
axis table that is itself derived numerically from diffcalc.hkl.geometry.get_rotation_matrices, the hkl round
trip and every constraint residual are computed here, and an independent scipy multi-start solve of
3 hkl equations + 3 constraints gives the reference solution set. Missing / extra counts compare the two.

Conventions this script derives (stored under "axes"): lab frame x,y,z; incident beam along +y; sample
rotation Z = MU ETA CHI PHI (lab = Z * phi-frame), detector rotation D = NU DELTA; Q_lab = (D - 1) k
in units 2 pi/lambda; hkl = UB^-1 Z^T Q_lab.
Virtual angles (You 1999, derived from eq. 18, 20, 24-28; see virtual()): alpha = asin(-n_y),
naz = atan2(n_x, n_z) for n = Z n_phi/|n_phi|; sin(beta) = n . k_out; psi = atan2((n x Q)_y,
n_y - (n.Q) Q_y) with unit Q; sin(betain) = -s . k_in, sin(betaout) = s . k_out for the surface normal s;
qaz = atan2(Q_x, Q_z). bisect = the chi axis MU ETA y lies in the plane of the beam and the detector
azimuth direction (sin qaz, 0, cos qaz); omega = angle of that axis from y in that plane, minus theta.
"""
import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from diffcalc.hkl.calc import HklCalculation
from diffcalc.hkl.constraints import Constraints
from diffcalc.hkl.geometry import Position, get_rotation_matrices
from diffcalc.ub.calc import UBCalculation
from diffcalc.util import DiffcalcException

SEED = 20260301
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffcalc_inverse.json")
ANGLES = ("mu", "delta", "nu", "eta", "chi", "phi")
TOL_HKL = 1e-6  # hkl units
TOL_CON = 1e-5  # degrees
TOL_MATCH = 1e-3  # degrees, solution identity
DET = ("delta", "nu", "qaz", "naz")
REF = ("a_eq_b", "alpha", "beta", "psi", "bin_eq_bout", "betain", "betaout")
SAMP1 = ("mu", "eta", "chi", "phi")
VOID = ("a_eq_b", "bin_eq_bout", "bisect")
DIRECT = ANGLES
Y = np.array([0.0, 1.0, 0.0])


def wrap(a):
    """Residue of a (degrees) modulo 360 in [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- axis table, from diffcalc's matrices
def axis_table():
    tab = {}
    for i, name in enumerate(ANGLES):
        th = 0.3
        pos = Position(**{name: math.degrees(th)})
        r = get_rotation_matrices(pos)[i]
        skew = (r - r.T) / (2 * math.sin(th))
        ax = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        # right-handed about ax for positive angle; cross-check with a second angle
        pos2 = Position(**{name: math.degrees(1.1)})
        r2 = get_rotation_matrices(pos2)[i]
        assert np.allclose(r2, rodrigues(ax, 1.1), atol=1e-12), name
        tab[name] = [round(float(v), 12) + 0.0 for v in ax]
    return tab


def rodrigues(ax, ang):
    ax = np.asarray(ax, float)
    ax = ax / np.linalg.norm(ax)
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def beam_direction():
    """Unit vector k with Q_lab proportional to D k - k, found from diffcalc's own get_hkl."""
    ub = UBCalculation()
    ub.set_lattice("c", "Cubic", 2 * math.pi)  # B = identity
    ub.set_u(np.eye(3))
    hc = HklCalculation(ub, Constraints())
    pos = Position(mu=0, delta=17.0, nu=9.0, eta=0, chi=0, phi=0)
    hkl = np.array(hc.get_hkl(pos, 2 * math.pi))  # q in units of 2 pi / lambda = 1
    m = get_rotation_matrices(pos)
    d = m[2] @ m[1]
    best = None
    for k in itertools.chain(np.eye(3), -np.eye(3)):
        err = np.linalg.norm(hkl - (d @ k - k))
        if best is None or err < best[0]:
            best = (err, k)
    assert best[0] < 1e-12
    return [float(v) + 0.0 for v in best[1]]


class Frame:
    """Rotations rebuilt from the recorded axis table (independent of diffcalc's matrices)."""

    def __init__(self, tab, beam):
        self.tab = {k: np.array(v) for k, v in tab.items()}
        self.beam = np.array(beam)

    def rot(self, name, deg):
        return rodrigues(self.tab[name], math.radians(deg))

    def zmat(self, p):
        return self.rot("mu", p["mu"]) @ self.rot("eta", p["eta"]) @ self.rot("chi", p["chi"]) @ self.rot("phi", p["phi"])

    def dmat(self, p):
        return self.rot("nu", p["nu"]) @ self.rot("delta", p["delta"])

    def qlab(self, p):
        """Q_lab in units of 2 pi / lambda."""
        return self.dmat(p) @ self.beam - self.beam

    def hkl(self, p, ub, wl):
        return np.linalg.solve(ub, self.zmat(p).T @ self.qlab(p) * (2 * math.pi / wl))

    def virtual(self, p, n_phi, s_phi):
        """alpha, beta, psi, qaz, naz, betain, betaout (deg), theta (deg), bisect/omega helpers."""
        z, d = self.zmat(p), self.dmat(p)
        kin = self.beam
        kout = d @ kin
        q = kout - kin
        qn = np.linalg.norm(q)
        out = {}
        if qn < 1e-12:
            return None
        qh = q / qn
        theta = math.asin(min(1.0, qn / 2))
        n = z @ (n_phi / np.linalg.norm(n_phi))
        s = z @ (s_phi / np.linalg.norm(s_phi))
        out["theta"] = math.degrees(theta)
        out["qaz"] = math.degrees(math.atan2(qh[0], qh[2]))
        out["alpha"] = math.degrees(math.asin(max(-1, min(1, -n[1]))))
        out["naz"] = math.degrees(math.atan2(n[0], n[2])) if abs(n[1]) < 1 - 1e-9 else float("nan")
        out["beta"] = math.degrees(math.asin(max(-1, min(1, float(n @ kout)))))
        tau_c = float(n @ qh)
        out["psi"] = (math.degrees(math.atan2(float(np.cross(n, qh)[1]), n[1] - tau_c * qh[1]))
                      if abs(1 - tau_c * tau_c) > 1e-14 else float("nan"))
        out["betain"] = math.degrees(math.asin(max(-1, min(1, float(-s @ kin)))))
        out["betaout"] = math.degrees(math.asin(max(-1, min(1, float(s @ kout)))))
        # chi axis in the lab: MU ETA y
        a = self.rot("mu", p["mu"]) @ self.rot("eta", p["eta"]) @ Y
        eq = np.array([math.sin(math.radians(out["qaz"])), 0.0, math.cos(math.radians(out["qaz"]))])
        out["bisect"] = math.degrees(math.asin(max(-1, min(1, float(a @ np.cross(Y, eq))))))
        out["omega"] = math.degrees(math.atan2(float(a @ eq), float(a @ Y))) - out["theta"]
        return out

    def residual(self, name, val, p, vir):
        """Constraint residual in degrees (0 when satisfied); val is the requested value."""
        if name in ANGLES:
            return wrap(p[name] - val)
        if vir is None:
            return float("nan")
        if name == "a_eq_b":
            return wrap(vir["alpha"] - vir["beta"])
        if name == "bin_eq_bout":
            return wrap(vir["betain"] - vir["betaout"])
        if name == "bisect":
            return vir["bisect"]
        return wrap(vir[name] - val)


# ---------------------------------------------------------------- diffcalc setup
LATTICES = {
    "cubic": ("Cubic", dict(a=3.905)),
    "tetragonal": ("Tetragonal", dict(a=4.1, c=6.3)),
    "hexagonal": ("Hexagonal", dict(a=3.2, c=5.2)),
    "orthorhombic": ("Orthorhombic", dict(a=4.1, b=5.3, c=6.7)),
    "monoclinic": ("Monoclinic", dict(a=5.1, b=6.2, c=7.3, beta=101.5)),
    "triclinic": ("Triclinic", dict(a=4.1, b=5.3, c=6.7, alpha=84.0, beta=97.0, gamma=76.0)),
}


def cell_B(name):
    """Independent B matrix (2 pi included) for the lattice, Busing-Levy cartesian setting."""
    sys_, kw = LATTICES[name]
    if sys_ == "Cubic":
        a = b = c = kw["a"]; al = be = ga = 90.0
    elif sys_ == "Tetragonal":
        a = b = kw["a"]; c = kw["c"]; al = be = ga = 90.0
    elif sys_ == "Hexagonal":
        a = b = kw["a"]; c = kw["c"]; al = be = 90.0; ga = 120.0
    elif sys_ == "Orthorhombic":
        a, b, c = kw["a"], kw["b"], kw["c"]; al = be = ga = 90.0
    elif sys_ == "Monoclinic":
        a, b, c = kw["a"], kw["b"], kw["c"]; al = ga = 90.0; be = kw["beta"]
    else:
        a, b, c, al, be, ga = (kw[k] for k in ("a", "b", "c", "alpha", "beta", "gamma"))
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    sa, sb, sg = (math.sin(math.radians(x)) for x in (al, be, ga))
    v = a * b * c * math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    a1, a2, a3 = b * c * sa / v, a * c * sb / v, a * b * sg / v
    b1 = math.acos((cb * cg - ca) / (sb * sg))
    b2 = math.acos((ca * cg - cb) / (sa * sg))
    b3 = math.acos((ca * cb - cg) / (sa * sb))
    return 2 * math.pi * np.array([
        [a1, a2 * math.cos(b3), a3 * math.cos(b2)],
        [0, a2 * math.sin(b3), -a3 * math.sin(b2) * ca],
        [0, 0, 1 / c]])


def make_ubcalc(cell, u, n_ref, surf):
    ub = UBCalculation()
    sys_, kw = LATTICES[cell]
    ub.set_lattice("g", sys_, *kw.values())
    ub.set_u(np.array(u))
    if n_ref["rlv"]:
        ub.n_hkl = tuple(n_ref["vec"])
    else:
        ub.n_phi = tuple(n_ref["vec"])
    if surf["rlv"]:
        ub.surf_nhkl = tuple(surf["vec"])
    else:
        ub.surf_nphi = tuple(surf["vec"])
    return ub


# ---------------------------------------------------------------- independent multi-start solve
def circ_dist(a, b):
    d = np.abs(wrap(np.asarray(a) - np.asarray(b)))
    return float(np.max(d))


def numeric_solve(F, ub, wl, hkl, cons, n_phi, s_phi, seed):
    """All isolated solutions of 3 hkl equations + 3 constraints from a deterministic Sobol start set."""
    direct = {k: v for k, v in cons.items() if k in ANGLES}
    free = [a for a in ANGLES if a not in direct]
    virt = {k: v for k, v in cons.items() if k not in ANGLES}
    nfree = len(free)
    assert nfree == 3 + len(virt), (cons, nfree)
    target = np.array(hkl, float)

    def pos(x):
        p = dict(direct)
        p.update({a: math.degrees(v) for a, v in zip(free, x)})
        return p

    def fun(x):
        p = pos(x)
        r = list(F.hkl(p, ub, wl) - target)
        if virt:
            vir = F.virtual(p, n_phi, s_phi)
            for name, val in virt.items():
                if vir is None or (name not in VOID and math.isnan(vir[name])):
                    r.append(1e3)
                elif name == "bisect":
                    a = F.rot("mu", p["mu"]) @ F.rot("eta", p["eta"]) @ Y
                    eq = np.array([math.sin(math.radians(vir["qaz"])), 0.0, math.cos(math.radians(vir["qaz"]))])
                    r.append(float(a @ np.cross(Y, eq)))
                else:
                    r.append(math.radians(F.residual(name, val, p, vir)))
        return np.array(r)

    n = {3: 1500, 4: 3000, 5: 5000, 6: 8000}[nfree]
    starts = (qmc.Sobol(d=nfree, scramble=True, seed=seed).random(n) * 2 - 1) * math.pi
    kept = []  # (angles array in degrees over ANGLES order, singular flag)
    for x0 in starts:
        try:
            s = least_squares(fun, x0, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200)
        except Exception:
            continue
        if not np.all(np.isfinite(s.fun)) or np.max(np.abs(s.fun)) > 1e-10:
            continue
        p = pos(s.x)
        vec = np.array([wrap(p[a]) for a in ANGLES])
        if any(circ_dist(vec, k[0]) < 1e-4 for k in kept):
            continue
        jac = np.empty((len(s.fun), nfree))
        for j in range(nfree):
            dx = np.zeros(nfree); dx[j] = 1e-6
            jac[:, j] = (fun(s.x + dx) - fun(s.x - dx)) / 2e-6
        sv = np.linalg.svd(jac, compute_uv=False)
        sing = bool(sv[-1] < 1e-5 * sv[0])
        kept.append((vec, sing))
        if sum(1 for k in kept if k[1]) > 30:
            break
    return dict(
        n_starts=int(n), n_free=nfree,
        solutions=[dict(zip(ANGLES, [float(v) for v in k[0]])) for k in kept if not k[1]],
        family_samples=[dict(zip(ANGLES, [float(v) for v in k[0]])) for k in kept if k[1]][:6],
        family=any(k[1] for k in kept),
    )


# ---------------------------------------------------------------- one case
def cons_in_diffcalc(cons):
    return {k: (True if v is True else float(v)) for k, v in cons.items()}


def run_diffcalc(spec, ub):
    import io, contextlib
    buf = io.StringIO()
    out = dict(status="ok", error=None, stdout=None, solutions=[])
    with contextlib.redirect_stdout(buf):
        try:
            con = Constraints(cons_in_diffcalc(spec["constraints"]))
        except Exception as e:  # constraint object refuses the set
            out.update(status="constraints_error", error=f"{type(e).__name__}: {e}".replace("\n", " | "))
            out["stdout"] = buf.getvalue() or None
            return out, None
        hc = HklCalculation(ub, con)
        try:
            res = hc.get_position(*spec["hkl"], spec["wavelength"])
        except Exception as e:
            out.update(status="error", error=f"{type(e).__name__}: {e}".strip().replace("\n", " | "))
            res = []
    out["stdout"] = buf.getvalue() or None
    for pos, vir in res:
        out["solutions"].append(dict(angles={a: float(getattr(pos, a)) for a in ANGLES},
                                     virtual={k: float(v) for k, v in vir.items()}))
    return out, hc


def run_case(args):
    spec, tab, beam = args
    F = Frame(tab, beam)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ub = make_ubcalc(spec["cell"], spec["U"], spec["n_ref"], spec["surface"])
    ubm = np.array(ub.UB)
    n_phi = np.array(ub.n_phi)[:, 0]
    s_phi = np.array(ub.surf_nphi)[:, 0]
    wl = spec["wavelength"]
    dc, _ = run_diffcalc(spec, ub)
    cons = spec["constraints"]
    sols = []
    for s in dc["solutions"]:
        p = s["angles"]
        h = F.hkl(p, ubm, wl)
        vir = F.virtual(p, n_phi, s_phi)
        res = {k: F.residual(k, v, p, vir) for k, v in cons.items()}
        miss = float(np.max(np.abs(h - np.array(spec["hkl"]))))
        valid = miss <= TOL_HKL and all(abs(r) <= TOL_CON for r in res.values())
        sols.append(dict(s, hkl_roundtrip=[float(v) for v in h], hkl_miss=miss,
                         constraint_residual_deg={k: float(v) for k, v in res.items()},
                         valid=bool(valid),
                         outside_pm180=[a for a in ANGLES if abs(p[a]) > 180.0 + 1e-9]))
    dc["solutions"] = sols
    num = None
    missing = extra_not_in_numeric = None
    if spec.get("numeric", True):
        num = numeric_solve(F, ubm, wl, spec["hkl"], {k: v for k, v in cons.items()}, n_phi, s_phi, spec["seed"])
        valid_vecs = [np.array([wrap(s["angles"][a]) for a in ANGLES]) for s in sols if s["valid"]]
        miss_list = [n for n in num["solutions"]
                     if not any(circ_dist([n[a] for a in ANGLES], v) <= TOL_MATCH for v in valid_vecs)]
        missing = len(miss_list)
        num["missing_solutions"] = miss_list
        def in_num(v):
            allv = num["solutions"] + num["family_samples"]
            return any(circ_dist([n[a] for a in ANGLES], v) <= TOL_MATCH for n in allv)
        extra_not_in_numeric = sum(1 for v in valid_vecs if not in_num(v)) if not num["family"] else None
    uniq = []
    dup = 0
    for s in sols:
        v = np.array([wrap(s["angles"][a]) for a in ANGLES])
        if any(circ_dist(v, u) <= TOL_MATCH for u in uniq):
            dup += 1
        else:
            uniq.append(v)
    out = dict(spec, UB=ubm.tolist(), n_phi=n_phi.tolist(), surf_nphi=s_phi.tolist(), diffcalc=dc, numeric=num,
               counts=dict(returned=len(sols), valid=sum(s["valid"] for s in sols),
                           invalid=sum(not s["valid"] for s in sols), duplicates=dup,
                           outside_pm180=sum(bool(s["outside_pm180"]) for s in sols),
                           missing=missing, valid_not_in_numeric=extra_not_in_numeric,
                           numeric_isolated=None if num is None else len(num["solutions"])))
    return out


# ---------------------------------------------------------------- defect tags (ids in upstream-defects.md)
def tag_defects(c):
    """DC ids a case exercises; the golden's reference set is the numeric one, never diffcalc's."""
    ids = []
    dc, cons, n = c["diffcalc"], c["constraints"], c["counts"]
    err = dc["error"] or ""
    num = c["numeric"]["solutions"] if c["numeric"] else []
    if "naz" in cons and "No code yet" in err and num:
        ids.append("DC-1")
    if "naz" in cons and any(k in cons for k in ("betain", "betaout", "bin_eq_bout")) and \
            n["returned"] == 0 and "pseudo-angle" in err and num:
        ids.append("DC-2")
    if "bisect" in cons and any(abs(x["constraint_residual_deg"].get("bisect", 0)) > TOL_CON
                                for x in dc["solutions"]):
        ids.append("DC-3")
    if "eta" in cons and "phi" in cons and "angles calculated for hkl" in err and num:
        ids.append("DC-4")
    if err.startswith("ZeroDivisionError"):
        ids.append("DC-5")
    if "bisect" not in cons and n["invalid"]:
        ids.append("DC-6")
    return ids


# ---------------------------------------------------------------- case generation
WAVELENGTHS = (0.62, 1.0, 1.5406, 2.4)
CELLS = tuple(LATTICES)
N_REFS = (dict(rlv=False, vec=[0, 0, 1]), dict(rlv=True, vec=[1, 0, 0]), dict(rlv=True, vec=[0, 1, 0]),
          dict(rlv=False, vec=[1, 0, 0]), dict(rlv=True, vec=[0, 0, 1]))
SURFS = (dict(rlv=False, vec=[0, 0, 1]), dict(rlv=True, vec=[0, 0, 1]), dict(rlv=True, vec=[1, 0, 1]))
SAMP2_DET = (("chi", "phi"), ("mu", "eta"), ("mu", "phi"), ("mu", "chi"), ("eta", "phi"), ("eta", "chi"),
             ("bisect", "mu"), ("bisect", "eta"), ("bisect", "omega"))
SAMP2_REF = (("chi", "phi"), ("chi", "eta"), ("chi", "mu"), ("mu", "eta"), ("mu", "phi"), ("eta", "phi"))
SAMP3 = (("eta", "chi", "phi"), ("mu", "chi", "phi"), ("mu", "eta", "phi"), ("mu", "eta", "chi"))


def rand_u(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


class Gen:
    def __init__(self, tab, beam):
        self.F = Frame(tab, beam)
        self.rng = np.random.default_rng(SEED)
        self.cases = []
        self.k = 0

    def base(self, cat, cell=None, wl=None, u=None, n_ref=None, surf=None, tag=""):
        r = self.rng
        self.k += 1
        return dict(id=f"c{self.k:03d}", category=cat, tag=tag, seed=SEED + self.k,
                    cell=cell or CELLS[(self.k) % len(CELLS)],
                    wavelength=wl or WAVELENGTHS[r.integers(len(WAVELENGTHS))],
                    U=u if u is not None else rand_u(r),
                    n_ref=n_ref or N_REFS[r.integers(len(N_REFS))],
                    surface=surf or SURFS[r.integers(len(SURFS))],
                    dc_ids=[])

    def true_position(self, spec, want_bisect=False):
        r = self.rng
        p = dict(mu=r.uniform(-180, 180), delta=r.uniform(8, 100), nu=r.uniform(-40, 60),
                 eta=r.uniform(-180, 180), chi=r.uniform(-180, 180), phi=r.uniform(-180, 180))
        return p

    def realise(self, spec, cons_names, p, bisect_theta_off=None):
        """Fill hkl and constraint values of spec from position p (values read with the independent maths)."""
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ub = make_ubcalc(spec["cell"], spec["U"], spec["n_ref"], spec["surface"])
        ubm = np.array(ub.UB)
        n_phi = np.array(ub.n_phi)[:, 0]
        s_phi = np.array(ub.surf_nphi)[:, 0]
        if "bisect" in cons_names:
            vir0 = self.F.virtual(p, n_phi, s_phi)
            th = math.radians(vir0["theta"])
            qaz = math.radians(vir0["qaz"])
            off = bisect_theta_off if bisect_theta_off is not None else self.rng.uniform(-40, 40)
            thp = th + math.radians(off)
            p["eta"] = math.degrees(math.asin(math.sin(thp) * math.sin(qaz)))
            p["mu"] = math.degrees(math.atan(math.tan(thp) * math.cos(qaz)))
            if self.rng.random() < 0.5:  # the other branch of the same condition
                p["eta"] = 180 - p["eta"]
                p["mu"] = p["mu"] + 180
        vir = self.F.virtual(p, n_phi, s_phi)
        cons = {}
        for n in cons_names:
            if n in VOID:
                cons[n] = True
            elif n in ANGLES:
                cons[n] = round(float(wrap(p[n])), 6)
            else:
                cons[n] = round(float(vir[n]), 6)
        # hkl at the (rounded) position values the constraints carry
        spec["hkl"] = [round(float(v), 6) for v in self.F.hkl(p, ubm, spec["wavelength"])]
        spec["constraints"] = cons
        spec["true_position"] = {k: round(float(wrap(v)), 6) for k, v in p.items()}
        return spec

    def add(self, spec):
        self.cases.append(spec)

    def build(self):
        r = self.rng
        # A. detector + reference + one sample angle: every combination
        for det, ref, sa in itertools.product(DET, REF, SAMP1):
            s = self.base("det+ref+samp", tag=f"{det}+{ref}+{sa}")
            self.add(self.realise(s, (det, ref, sa), self.true_position(s)))
        # B. detector + two sample angles: documented combinations
        for rep in range(2):
            for det in DET:
                for pair in SAMP2_DET:
                    s = self.base("det+2samp", tag=f"{det}+{'+'.join(pair)}")
                    names = (det,) + pair
                    self.add(self.realise(s, names, self.true_position(s)))
        # C. reference + two sample angles
        for rep in range(2):
            for ref in REF:
                for pair in SAMP2_REF:
                    s = self.base("ref+2samp", tag=f"{ref}+{'+'.join(pair)}")
                    self.add(self.realise(s, (ref,) + pair, self.true_position(s)))
        # D. three sample angles
        for rep in range(len(CELLS)):
            for trio in SAMP3:
                s = self.base("3samp", tag="+".join(trio), cell=CELLS[rep])
                self.add(self.realise(s, trio, self.true_position(s)))
        self.build_unreachable()
        self.build_degenerate()
        self.build_unsupported()
        return self.cases

    def build_unreachable(self):
        """hkl beyond the Ewald sphere for one case per constraint family; constraint values from a real position."""
        fam = [("delta", "alpha", "chi"), ("nu", "a_eq_b", "phi"), ("qaz", "beta", "eta"), ("naz", "psi", "mu"),
               ("delta", "chi", "phi"), ("nu", "mu", "eta"), ("qaz", "mu", "phi"), ("delta", "bisect", "omega"),
               ("alpha", "chi", "phi"), ("psi", "mu", "eta"), ("betain", "eta", "phi"),
               ("chi", "phi", "mu"), ("eta", "chi", "phi")]
        for names in fam:
            s = self.base("unreachable", tag="+".join(names))
            s = self.realise(s, names, self.true_position(s))
            # scale so that |Q| exceeds 2 k: pick hkl along the same direction, twice the limit
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                ub = make_ubcalc(s["cell"], s["U"], s["n_ref"], s["surface"])
            d = np.array(s["hkl"], float)
            qlen = np.linalg.norm(np.array(ub.UB) @ d)
            qmax = 4 * math.pi / s["wavelength"]
            s["hkl"] = [round(float(v), 6) for v in d * (1.6 * qmax / qlen)]
            s["tag"] += " |Q|=1.6*qmax"
            self.add(s)

    def deg_case(self, tag, cons, hkl, cell="cubic", u=None, n_ref=None, surf=None, wl=1.0, cat="degenerate"):
        s = self.base(cat, cell=cell, wl=wl, u=u if u is not None else np.eye(3).tolist(),
                      n_ref=n_ref or dict(rlv=False, vec=[0, 0, 1]), surf=surf or dict(rlv=False, vec=[0, 0, 1]), tag=tag)
        s["hkl"] = hkl
        s["constraints"] = cons
        self.add(s)

    def build_degenerate(self):
        a = 3.905
        wl = 1.0
        I3 = np.eye(3).tolist()
        # chi = 90, eta = 0: phi || mu; horizontal four-circle (delta=0)
        self.deg_case("chi=90 eta=0 delta=0 (phi||mu)", dict(chi=90, eta=0, delta=0), [1, 0, 0], wl=wl)
        self.deg_case("chi=90 eta=0 delta=0 (phi||mu), (110)", dict(chi=90, eta=0, delta=0), [1, 1, 0], wl=wl)
        self.deg_case("chi=90 eta=0 qaz=90", dict(chi=90, eta=0, qaz=90), [1, 0, 1], wl=wl)
        self.deg_case("chi=90 eta=0 nu=0", dict(chi=90, eta=0, nu=0), [1, 1, 1], wl=wl)
        self.deg_case("chi=90 eta=0 phi=0", dict(chi=90, eta=0, phi=0), [1, 1, 1], wl=wl)
        self.deg_case("chi=90 phi=0 a_eq_b", dict(chi=90, phi=0, a_eq_b=True), [1, 1, 1], wl=wl)
        self.deg_case("chi=90 eta=0 psi=0", dict(chi=90, eta=0, psi=0), [1, 1, 1], wl=wl)
        self.deg_case("chi=90 eta=0 mu=0 (3 samp)", dict(chi=90, eta=0, mu=0), [1, 1, 0], wl=wl)
        self.deg_case("eta=0 chi=0 delta=0 (phi||eta)", dict(chi=0, eta=0, delta=0), [1, 0, 0], wl=wl)
        self.deg_case("eta=0 chi=0 nu=0", dict(chi=0, eta=0, nu=0), [1, 0, 0], wl=wl)
        self.deg_case("mu=0 chi=0 nu=0 (phi||eta)", dict(mu=0, chi=0, nu=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 chi=0 delta=0", dict(mu=0, chi=0, delta=0), [0, 1, 1], wl=wl)
        self.deg_case("mu=0 chi=0 phi=0 (3 samp)", dict(mu=0, chi=0, phi=0), [0, 1, 1], wl=wl)
        self.deg_case("chi=0 phi=0 nu=0", dict(chi=0, phi=0, nu=0), [0, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 nu=0 (vertical 4C)", dict(mu=0, eta=0, nu=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 delta=0 (horizontal 4C)", dict(mu=0, eta=0, delta=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 qaz=0", dict(mu=0, eta=0, qaz=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 a_eq_b", dict(mu=0, eta=0, a_eq_b=True), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 psi=0", dict(mu=0, eta=0, psi=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 phi=0 (3 samp)", dict(mu=0, eta=0, phi=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 phi=0 eta=0 (3 samp)", dict(mu=0, eta=0, phi=0), [1, 1, 1], wl=wl)
        self.deg_case("mu=0 eta=0 chi=0 (3 samp)", dict(mu=0, eta=0, chi=0), [1, 0, 1], wl=wl)
        self.deg_case("mu=0 eta=0 chi=90 (3 samp)", dict(mu=0, eta=0, chi=90), [1, 0, 1], wl=wl)
        self.deg_case("eta=0 chi=0 phi=0 (3 samp)", dict(chi=0, eta=0, phi=0), [1, 0, 1], wl=wl)
        # reference parallel to Q
        for refc in (dict(psi=30), dict(a_eq_b=True), dict(alpha=10), dict(beta=10)):
            for sc in (dict(chi=20, phi=30), dict(mu=10, eta=20), dict(eta=10), dict(chi=25), dict(mu=0), dict(phi=0)):
                self.deg_case(f"n || Q: {refc} + {sc}", {**refc, **sc, **({"delta": 40} if len(sc) == 1 else {})},
                              [0, 0, 1], wl=wl, cat="degenerate-ref-parallel-Q")
        self.deg_case("n || Q: psi=30 + qaz=0 + chi=20", dict(psi=30, delta=30, chi=20), [0, 0, 1], wl=wl,
                      cat="degenerate-ref-parallel-Q")
        self.deg_case("n=(1 0 0)hkl || Q=(1 0 0): a_eq_b + mu + eta", dict(a_eq_b=True, mu=10, eta=20), [1, 0, 0],
                      n_ref=dict(rlv=True, vec=[1, 0, 0]), wl=wl, cat="degenerate-ref-parallel-Q")
        self.deg_case("n=(1 0 0)hkl || Q=(1 0 0): psi=0 + delta + chi", dict(psi=0, delta=40, chi=20), [1, 0, 0],
                      n_ref=dict(rlv=True, vec=[1, 0, 0]), wl=wl, cat="degenerate-ref-parallel-Q")
        # detector degeneracies
        self.deg_case("delta=90", dict(delta=90, alpha=0, chi=20), [1, 0, 0], wl=2 * a / 2.0 * 0 + 2.7, cat="degenerate-detector")
        self.deg_case("delta=90 qaz free", dict(delta=90, mu=0, eta=0), [1, 0, 0], wl=2.7, cat="degenerate-detector")
        self.deg_case("nu=90", dict(nu=90, mu=0, eta=0), [1, 0, 1], wl=2.0, cat="degenerate-detector")
        self.deg_case("delta=0 nu=0 impossible: hkl=(0 0 0)", dict(delta=0, mu=0, eta=0), [0, 0, 0], wl=1.0, cat="degenerate-detector")
        self.deg_case("backscatter |Q|=2k", dict(delta=0, mu=0, eta=0), [2, 0, 0], wl=2 * a / 2.0 / 1.0 * 1.0,
                      cat="degenerate-detector")
        self.deg_case("tiny Q", dict(delta=0.01, mu=0, eta=0), [0.0005, 0, 0], wl=1.0, cat="degenerate-detector")
        self.deg_case("qaz=0 hkl on beam axis plane", dict(qaz=0, mu=0, eta=0), [0, 0, 1], wl=1.0, cat="degenerate-detector")
        self.deg_case("naz = qaz (tau=0 path)", dict(naz=0, mu=0, alpha=5), [0, 0, 1], wl=1.0, cat="degenerate-detector")
        # sample = surface normal conventions
        self.deg_case("betain=betaout surface normal || Q", dict(bin_eq_bout=True, mu=0, eta=0), [0, 0, 1],
                      surf=dict(rlv=True, vec=[0, 0, 1]), wl=1.0, cat="degenerate-ref-parallel-Q")
        self.deg_case("bisect eta=0 delta=0 horizontal", dict(bisect=True, eta=0, delta=0), [1, 0, 1], wl=1.0)
        self.deg_case("bisect mu=0 nu=0 vertical", dict(bisect=True, mu=0, nu=0), [1, 0, 1], wl=1.0)
        self.deg_case("bisect omega=0 delta=0", dict(bisect=True, omega=0, delta=0), [1, 0, 1], wl=1.0)
        self.deg_case("bisect omega=0 nu=0", dict(bisect=True, omega=0, nu=0), [1, 0, 1], wl=1.0)
        self.deg_case("bisect omega=0 nu=0 hkl=(0 0 1)", dict(bisect=True, omega=0, nu=0), [0, 0, 1], wl=1.0)
        self.deg_case("bisect omega=0 qaz=0", dict(bisect=True, omega=0, qaz=0), [0, 0, 1], wl=1.0)
        self.deg_case("bisect omega=0 qaz=90", dict(bisect=True, omega=0, qaz=90), [1, 0, 1], wl=1.0)

    def build_unsupported(self):
        """Constraint sets the docs or the class do not list; the golden records what diffcalc says."""
        combos = [
            ("two detector constraints", dict(delta=30, nu=10, chi=10), False),
            ("two reference constraints", dict(alpha=3, beta=3, chi=10), False),
            ("under-constrained (2)", dict(delta=30, chi=10), False),
            ("3 samp with bisect", dict(bisect=True, mu=10, eta=20), False),
            ("3 samp with omega", dict(omega=3, mu=10, eta=20), False),
            ("det + 3 samp", dict(delta=30, mu=10, eta=20, chi=30), False),
            ("samp(1)+bisect+det", dict(delta=30, bisect=True, chi=20), False),
            ("omega without bisect + det", dict(delta=30, omega=3, chi=20), False),
            ("ref + bisect + samp", dict(alpha=3, bisect=True, mu=20), False),
            ("ref + omega + bisect", dict(alpha=3, bisect=True, omega=0), False),
        ]
        for tag, cons, num in combos:
            s = self.base("unsupported", cell="cubic", wl=1.0, tag=tag)
            s["hkl"] = [1, 0, 1]
            s["constraints"] = cons
            s["numeric"] = False
            self.add(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--retag", action="store_true", help="re-apply tag_defects to the existing json")
    a = ap.parse_args()
    if a.retag:
        doc = json.load(open(OUT))
        for c in doc["cases"]:
            c["dc_ids"] = tag_defects(c)
        json.dump(doc, open(OUT, "w"), indent=1)
        print({i: sum(i in c["dc_ids"] for c in doc["cases"]) for i in ("DC-1", "DC-2", "DC-3", "DC-4", "DC-5", "DC-6")})
        return
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        tab = axis_table()
        beam = beam_direction()
    gen = Gen(tab, beam)
    with contextlib.redirect_stdout(io.StringIO()):
        cases = gen.build()
    if a.only:
        cases = [c for c in cases if c["id"] in a.only.split(",")]
    if a.limit:
        cases = cases[: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_case, [(c, tab, beam) for c in cases], chunksize=1))
    for c in results:
        c["dc_ids"] = tag_defects(c)
    doc = dict(
        diffcalc_core="059ec8a (v0.4.0-9)", seed=SEED,
        units=dict(angle="deg", wavelength="angstrom", hkl="r.l.u."),
        tolerances=dict(hkl_miss=TOL_HKL, constraint_deg=TOL_CON, solution_match_deg=TOL_MATCH),
        axes=dict(axis_vector_positive_is_right_handed=tab, beam_direction=beam,
                  composition="Z = MU ETA CHI PHI (phi-frame to lab); D = NU DELTA; Q_lab = (D - 1) beam, units 2 pi/lambda;"
                              " hkl = UB^-1 Z^T Q_lab",
                  angle_order=list(ANGLES)),
        cases=results)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
    tot = {k: sum((c["counts"][k] or 0) for c in results) for k in ("returned", "valid", "invalid", "duplicates",
                                                                   "outside_pm180", "missing", "valid_not_in_numeric")}
    print(len(results), "cases", tot)


if __name__ == "__main__":
    main()
