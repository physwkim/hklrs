"""Golden inverse (hkl -> angles) from libhkl 5.1.9 (gi Hkl 5.0).
Run: python -I hkl_inverse.py

Per (geometry, mode, parameter set) the "hkl" engine is put in the mode, the
parameters are set, modes with an init step (psi_constant*, constant_incidence)
are initialised at the start geometry, then pseudo_axis_values_set(target) is
called from the start axes. Every returned solution is recorded in hkl's order
(canonical deg) with the hkl hkl's own get reports for it, an independent numpy
forward hkl, and the residuals of the mode's own constraint.
The C solver restarts with libc rand(); srand() is seeded per call through
ctypes so the output is reproducible.
Mode residual sources (C pin 7be4f79, hkl/): hkl-engine-e4c.c, -e6c.c, -k4c.c,
-k6c.c, -zaxis.c, hkl-pseudoaxis-common-hkl.c (_double_diffraction,
_psi_constant_*); the ESRF BM28 and ID01 PSIC modes only hold axes. Only
hkl's own GError is recorded as a case error; a failure of the recording
itself raises.
"""
import ctypes
import json
import math
import os
import random

import gi
import numpy as np

gi.require_version("Hkl", "5.0")
from gi.repository import GLib, Hkl  # noqa: E402

SEED = 20260301
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_inverse.json")
LIBC = ctypes.CDLL(None)
LATTICE = dict(a=4.1, b=5.3, c=6.7, alpha=90.0, beta=90.0, gamma=90.0)
WL = 1.54
N_TARGETS = 40
N_UNREACH = 4
TOL = 1e-6
GEOMETRIES = ["E4CV", "E4CH", "E6C", "K4CV", "K6C", "ESRF BM28 PSIC", "ZAXIS",
              "ESRF ID01 PSIC"]
PARAM_MODES = ("double_diffraction", "psi_constant", "constant_incidence")
UNIT = {"Degree": ("rot", 1.0), "Milli Radian": ("rot", math.pi / 180.0 * 1000.0),
        "Milli Meter": ("trans", 1.0), "Meter": ("trans", 1e-3)}
ALPHA = math.radians(50.0)  # kappa angle of the 4-circle kappa geometries


def mat(m):
    return [[m.get(i, j) for j in range(3)] for i in range(3)]


def wrap(a, period=360.0):
    """Residue of a modulo period, in (-period/2, period/2]."""
    r = math.fmod(a, period)
    if r > period / 2:
        r -= period
    elif r <= -period / 2:
        r += period
    return r


def rs_rd(geo, smp, det):
    return (np.array(mat(geo.sample_rotation_get(smp).to_matrix())),
            np.array(mat(geo.detector_rotation_get(det).to_matrix())))


class Ctx:
    def __init__(self, gname, rng):
        self.gname = gname
        self.fac = Hkl.factories()[gname]
        self.geo = self.fac.create_new_geometry()
        self.geo.wavelength_set(WL, USER)
        self.names = list(self.geo.axis_names_get())
        self.units = [self.geo.axis_get(n).user_unit_get() for n in self.names]
        self.det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.smp = Hkl.Sample.new("inv")
        self.smp.lattice_set(Hkl.Lattice.new(*(LATTICE[k] if k in "abc" else math.radians(LATTICE[k])
                                               for k in ("a", "b", "c", "alpha", "beta", "gamma"))))
        e = [rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)]
        self.u_euler = e
        self.smp.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in e)))
        self.el = self.fac.create_new_engine_list()
        self.el.init(self.geo, self.det, self.smp)
        self.eng = self.el.engine_get_by_name("hkl")
        engines = [e.name_get() for e in self.el.engines_get()]
        self.psi_eng = self.el.engine_get_by_name("psi") if "psi" in engines else None
        self.ub = np.array(mat(self.smp.UB_get()))
        self.ranges = {n: [v / UNIT[u][1] for v in self.geo.axis_get(n).min_max_get(USER)]
                       for n, u in zip(self.names, self.units)}

    def to_user(self, canon):
        return [v * UNIT[u][1] for v, u in zip(canon, self.units)]

    def to_canon(self, user):
        return [v / UNIT[u][1] for v, u in zip(user, self.units)]

    def set_axes(self, canon):
        self.geo.axis_values_set(self.to_user(canon), USER)

    def axes(self):
        return self.to_canon(list(self.geo.axis_values_get(USER)))

    def forward_numpy(self, geo):
        rs, rd = rs_rd(geo, self.smp, self.det)
        k = 2 * math.pi / WL
        q = rs.T @ (k * (rd @ np.array([1.0, 0, 0]) - np.array([1.0, 0, 0])))
        return np.linalg.solve(self.ub, q).tolist()

    def forward_hkl(self, canon):
        """hkl as hkl's own engine get reports it for these axes."""
        self.set_axes(canon)
        self.el.get()
        return [float(x) for x in self.eng.pseudo_axis_values_get(USER)]


def named(ctx, canon):
    return dict(zip(ctx.names, canon))


def mode_params(ctx, rng, mode, random_set):
    """Parameter vector (user units) for the current mode; defaults when not random_set."""
    names = list(ctx.eng.parameters_names_get())
    vals = list(ctx.eng.parameters_values_get(USER))
    if not random_set:
        return vals
    out = []
    for n, v in zip(names, vals):
        if n in ("h2", "k2", "l2", "x", "y", "z"):
            out.append(float(rng.choice([-2, -1, 0, 1, 2])))
        elif n == "psi":
            out.append(rng.uniform(-180, 180))
        elif n == "incidence":
            out.append(rng.uniform(0, 30))
        elif n == "azimuth":
            out.append(rng.uniform(0, 180))
        else:
            out.append(v)
    # keep the reference vectors non-null
    for trio in (("h2", "k2", "l2"), ("x", "y", "z")):
        idx = [names.index(t) for t in trio if t in names]
        if idx and not any(out[i] for i in idx):
            out[idx[0]] = 1.0
    return out


def enter_mode(ctx, mode, params, start):
    """Mode, parameters, init at the start geometry, then parameters again (psi is overwritten by init)."""
    ctx.set_axes(start)
    ctx.eng.current_mode_set(mode)
    names = list(ctx.eng.parameters_names_get())
    ctx.eng.parameters_values_set(params, USER)
    ctx.eng.initialized_set(True)
    ctx.eng.parameters_values_set(params, USER)
    ctx.set_axes(start)
    return names


# ---- mode constraints -------------------------------------------------------
def kappa_derived(a):
    """omega, chi, phi of the kappa geometries from (komega, kappa, kphi) in deg; C: hkl-engine-k4c.c:77."""
    ko, ka, kp = (math.radians(x) for x in a)
    p = math.atan(math.tan(ka / 2) * math.cos(ALPHA))
    omega = math.degrees(ko + p - math.pi / 2)
    chi = math.degrees(2 * math.asin(math.sin(ka / 2) * math.sin(ALPHA)))
    phi = math.degrees(kp + p + math.pi / 2)
    return omega, chi, phi


def psi_of(ctx, h2, geo):
    """psi from hkl's read-only "psi" engine, whose reference is h2k2l2, for geometry geo."""
    if ctx.psi_eng is None:
        return None
    ctx.geo.axis_values_set(list(geo.axis_values_get(USER)), USER)
    ctx.el.get()
    cur = ctx.psi_eng.current_mode_get()
    if ctx.psi_eng.parameters_names_get():
        ctx.psi_eng.parameters_values_set([float(x) for x in h2], USER)
    ctx.el.get()
    return float(ctx.psi_eng.pseudo_axis_values_get(USER)[0])


def incidence_of(ctx, geo, n, init_geo):
    """incidence, azimuth (deg) as _constant_incidence_func computes them (k6c.c:322, pin 7be4f79)."""
    rs, _ = rs_rd(geo, ctx.smp, ctx.det)
    rs0, _ = rs_rd(init_geo, ctx.smp, ctx.det)
    nv = rs @ rs0.T @ np.array(n, float)
    ki = np.array([1.0, 0, 0])

    def ang(a, b):
        return math.acos(max(-1.0, min(1.0, a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))
    inc = ang(nv, ki) - math.pi / 2
    proj = nv - (nv @ ki) * ki
    return math.degrees(inc), math.degrees(ang(proj, np.array([0.0, 1, 0])))


def constraints(ctx, mode, par, start, sol, geo_sol, init_geo, writable):
    """Residuals of the mode's own constraint (deg or reciprocal units).

    Axes absent from the mode's writable list must keep their start value
    (hkl-pseudoaxis-auto.c: only axes_w are solved). Mode formulas cite the C
    functions (pin 7be4f79).
    """
    g = ctx.gname
    n = dict(zip(ctx.names, sol))
    s = dict(zip(ctx.names, start))
    r = {f"fixed:{a}": wrap(n[a] - s[a]) for a in ctx.names if a not in writable}
    kappa = g in ("K4CV", "K6C")
    if kappa:
        om, chi, phi = kappa_derived([n["komega"], n["kappa"], n["kphi"]])
    else:
        om, chi, phi = (n.get(a) for a in ("omega", "chi", "phi"))
    pv = dict(zip(ctx.eng.parameters_names_get(), par))
    horiz = mode.endswith("horizontal")
    tth = n["tth"] if g in ("E4CV", "E4CH", "K4CV") else n["delta"]
    if mode in ("bissector", "bissector_vertical"):
        r["tth-2*omega"] = wrap(tth - 2 * om)  # e4c.c:38, e6c.c:61, k4c.c:68, k6c.c:204
    if mode == "reflectivity":
        r["mu-gamma"] = n["mu"] - n["gamma"]  # zaxis.c:60, unwrapped as hkl solves it
    if mode == "bissector_horizontal":
        r["omega mod 180"] = wrap(om, 180.0)  # e6c.c:50, k6c.c:70
        r["gamma-2*mu"] = wrap(n["gamma"] - 2 * n["mu"])  # e6c.c:51, k6c.c:80
    if kappa:
        # f1/f2 branches differ by omega+-180, phi+-180, chi sign (k4c.c:127-285)
        if mode in ("constant_omega", "constant_omega_vertical"):
            r["omega-omega0 mod 180"] = wrap(om - pv["omega"], 180.0)
        if mode in ("constant_chi", "constant_chi_vertical"):
            r["|chi|-|chi0|"] = abs(chi) - abs(pv["chi"])
        if mode in ("constant_phi", "constant_phi_vertical"):
            r["phi-phi0 mod 180"] = wrap(phi - pv["phi"], 180.0)
        if mode == "constant_phi_horizontal":
            r["omega mod 180"] = wrap(om, 180.0)
            r["phi mod 180"] = wrap(phi, 180.0)  # k6c.c:119-135: phi=0, parameter unused
        if mode == "constant_kphi_horizontal":
            r["omega mod 180"] = wrap(om, 180.0)
        if mode == "double_diffraction_horizontal":
            r["omega mod 180"] = wrap(om, 180.0)  # k6c.c:270
    if mode.startswith("double_diffraction"):
        rs, _ = rs_rd(geo_sol, ctx.smp, ctx.det)
        k = 2 * math.pi / WL
        v = rs @ ctx.ub @ np.array([pv["h2"], pv["k2"], pv["l2"]], float) + np.array([k, 0, 0])
        r["|R.UB.h2+ki|^2-|ki|^2"] = float(v @ v - k * k)  # common-hkl.c:512 (f[3])
    if mode.startswith("psi_constant"):
        ps = psi_of(ctx, [pv["h2"], pv["k2"], pv["l2"]], geo_sol)
        if ps is not None:
            r["psi-psi0"] = wrap(ps - pv["psi"])  # common-hkl.c:567
    if mode == "constant_incidence":
        inc, az = incidence_of(ctx, geo_sol, [pv["x"], pv["y"], pv["z"]], init_geo)
        r["incidence-incidence0"] = wrap(inc - pv["incidence"])  # k6c.c:322
        r["azimuth-azimuth0"] = wrap(az - pv["azimuth"])
    return r


# ---- one case ---------------------------------------------------------------
def one_case(ctx, mode, params, target, start, unreachable, c_seed):
    case = {"geometry": ctx.gname, "mode": mode, "wavelength": WL,
            "parameter_names": list(ctx.eng.parameters_names_get()), "parameters": params,
            "UB": ctx.ub.tolist(), "axis_ranges": ctx.ranges,
            "start_axes": named(ctx, start), "target_hkl": list(target),
            "unreachable_by_q": unreachable, "c_srand": c_seed}
    try:
        enter_mode(ctx, mode, params, start)
        init_geo = ctx.geo.copy()
        case["writable_axes"] = list(ctx.eng.axis_names_get(Hkl.EngineAxisNamesGet.WRITE))
        LIBC.srand(c_seed)
        sols = ctx.eng.pseudo_axis_values_set([float(x) for x in target], USER)
    except GLib.Error as e:  # recorded, never dropped
        case["error"] = f"GError: {e}"
        case["n_solutions"] = 0
        case["solutions"] = []
        return case
    items = sols.items() if sols is not None else []
    case["n_solutions"] = len(items)
    out = []
    for it in items:
        gs = it.geometry_get().copy()
        canon = ctx.to_canon(list(gs.axis_values_get(USER)))
        ent = {"axes": named(ctx, canon)}
        ent["hkl_engine"] = ctx.forward_hkl(canon)
        ent["hkl_numpy"] = ctx.forward_numpy(gs)
        ent["hkl_miss"] = float(np.abs(np.array(ent["hkl_engine"]) - np.array(target)).max())
        ent["constraint_resid"] = constraints(ctx, mode, params, start, canon, gs, init_geo, case["writable_axes"])
        out.append(ent)
    case["solutions"] = out
    return case


def fixed_values(ctx, rng, writable):
    """Value of every axis the mode does not solve: 0 for mu/gamma/delta/omega, seeded +-60 otherwise."""
    return {n: (0.0 if n in ("mu", "gamma", "delta", "omega") else rng.uniform(-60, 60))
            for n in ctx.names if n not in writable}


def make_start(ctx, rng, writable, fixed, prev, near):
    out = []
    for n in ctx.names:
        if n in fixed:
            out.append(fixed[n])
        elif near and prev is not None:
            out.append(prev[n] + rng.uniform(-5, 5))
        else:
            lo, hi = ctx.ranges[n]
            out.append(rng.uniform(lo, hi))
    return out


def constructed_target(ctx, rng, mode, params, writable, fixed):
    """hkl of a geometry that satisfies the mode's fixed-axis / constant-angle constraint (reachable by construction)."""
    pv = dict(zip(ctx.eng.parameters_names_get(), params))
    for _ in range(100):
        a = {n: fixed[n] if n in fixed else rng.uniform(-180, 180) for n in ctx.names}
        if ctx.gname in ("K4CV", "K6C") and "kappa" not in fixed:
            ka = rng.uniform(-150, 150) if mode.find("chi") < 0 else 0.0
            a["kappa"] = ka
            p = math.degrees(math.atan(math.tan(math.radians(ka) / 2) * math.cos(ALPHA)))
            if mode.startswith("constant_omega") and "komega" not in fixed:
                a["komega"] = pv["omega"] + 90 - p
            if mode.startswith("constant_phi") and "kphi" not in fixed:
                a["kphi"] = pv["phi"] - 90 - p
        h = ctx.forward_hkl([a[n] for n in ctx.names])
        if max(abs(x) for x in h) > 1e-3 and not any(math.isnan(x) for x in h):
            return [float(x) for x in h], a
    return None, None


def targets_for(ctx, rng, mode, params, writable, fixed):
    bmax = 4 * math.pi / WL
    b = 2 * math.pi * np.array([1 / LATTICE["a"], 1 / LATTICE["b"], 1 / LATTICE["c"]])
    out = []
    while len(out) < N_TARGETS - N_UNREACH:
        if rng.random() < 0.5:
            h, built = constructed_target(ctx, rng, mode, params, writable, fixed)
            kind = "constructed"
        else:
            h = [rng.randint(-3, 3) for _ in range(3)] if rng.random() < 0.7 else \
                [round(rng.uniform(-2.5, 2.5), 3) for _ in range(3)]
            kind, built = "random", None
        if h is None or not any(h) or np.linalg.norm(b * np.array(h)) > 0.95 * bmax:
            continue
        out.append((h, False, kind, built))
    while len(out) < N_TARGETS:
        h = [rng.choice([-1, 1]) * rng.randint(5, 9) for _ in range(3)]
        if np.linalg.norm(b * np.array(h)) > 1.1 * bmax:
            out.append((h, True, "unreachable_q", None))
    rng.shuffle(out)
    return out


def main():
    cases, geoms, summary = [], {}, {}
    for gname in GEOMETRIES:
        ctx0 = Ctx(gname, random.Random(f"{SEED}-{gname}-U"))
        geoms[gname] = {"axes": ctx0.names, "units": ctx0.units, "ranges": ctx0.ranges,
                        "U_euler_deg": ctx0.u_euler, "UB": ctx0.ub.tolist(),
                        "lattice": dict(LATTICE), "wavelength": WL,
                        "modes": list(ctx0.eng.modes_names_get())}
        for mode in ctx0.eng.modes_names_get():
            sets = [("defaults", False)]
            if mode.startswith(PARAM_MODES):
                sets.append(("random", True))
            for label, rand_p in sets:
                rng = random.Random(f"{SEED}-{gname}-{mode}-{label}")
                ctx = Ctx(gname, random.Random(f"{SEED}-{gname}-U"))  # same U for every mode
                ctx.eng.current_mode_set(mode)
                params = mode_params(ctx, rng, mode, rand_p)
                writable = list(ctx.eng.axis_names_get(Hkl.EngineAxisNamesGet.WRITE))
                fixed = fixed_values(ctx, rng, writable)
                prev = None
                for h, unr, kind, built in targets_for(ctx, rng, mode, params, writable, fixed):
                    near = rng.random() < 0.5
                    start = make_start(ctx, rng, writable, fixed, prev, near)
                    c = one_case(ctx, mode, params, h, start, unr, rng.randint(1, 2**30))
                    c["param_set"] = label
                    c["target_kind"] = kind
                    c["constructed_from_axes"] = built
                    c["start_kind"] = "near" if (near and prev is not None) else "far"
                    cases.append(c)
                    if c["solutions"]:
                        prev = c["solutions"][0]["axes"]
                    summary.setdefault((gname, mode, label), []).append(c)
    doc = {"hkl_version": Hkl.VERSION, "seed": SEED,
           "units": {"angle": "deg", "length": "mm", "wavelength": "angstrom", "hkl": "r.l.u."},
           "tolerance_hkl_miss": TOL, "geometries": geoms, "cases": cases}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, "cases", len(cases))
    miss, viol, bad_unreach = [], [], []
    for (g, m, lab), cs in summary.items():
        ns = [c["n_solutions"] for c in cs]
        errs = [c for c in cs if "error" in c]
        reach = [c for c in cs if not c["unreachable_by_q"]]
        rfail = [c for c in reach if c["n_solutions"] == 0]
        print(f"{g:5s} {m:32s} {lab:8s} cases {len(cs):3d} sols min/max {min(ns)}/{max(ns)} "
              f"mean {sum(ns)/len(ns):.2f} err {len(errs)} reachable-fail {len(rfail)}")
        for c in cs:
            for s in c["solutions"]:
                if s["hkl_miss"] > TOL:
                    miss.append((g, m, lab, s["hkl_miss"]))
                for k, v in s["constraint_resid"].items():
                    if abs(v) > 1e-5:
                        viol.append((g, m, lab, k, v))
            if c["unreachable_by_q"] and c["n_solutions"]:
                bad_unreach.append((g, m, lab))
    print("hkl misses >1e-6:", len(miss), "constraint violations >1e-6:", len(viol),
          "unreachable-Q cases that returned solutions:", len(bad_unreach))


main()
