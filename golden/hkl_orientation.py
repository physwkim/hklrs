"""Golden Busing-Levy orientation from libhkl (gi Hkl 5.0) and hklpy2.
Run: python -I hkl_orientation.py

Per case: set a seeded random U in hkl, solve two reachable non-parallel
reflections with hkl's own "hkl" engine (default mode), then rebuild a fresh
sample with the same lattice, add both reflections at those angles and call
compute_UB_busing_levy. UB_reference is an independent numpy Busing-Levy with
the correct B; UB_hkl12_model is the same with libhkl's B[1][2] of HKL-12.
"""
import json
import math
import os
import random

import gi
import numpy as np

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

SEED = 20260201
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_orientation.json")
SAMPLES = {
    "cubic": dict(a=5.431020511, b=5.431020511, c=5.431020511, alpha=90.0, beta=90.0, gamma=90.0),
    "hexagonal": dict(a=3.0, b=3.0, c=5.0, alpha=90.0, beta=90.0, gamma=120.0),
    "monoclinic": dict(a=4.1, b=5.3, c=6.7, alpha=90.0, beta=97.0, gamma=90.0),
    "triclinic": dict(a=4.1, b=5.3, c=6.7, alpha=82.0, beta=97.0, gamma=101.0),
}
WAVELENGTHS = [1.54, 0.71]
MAX_TRIES = 200
TOL_EXPLAIN = 1e-6  # UB entries are O(1) Å^-1

# user unit -> (kind, factor from canonical deg|mm to the user unit)
UNIT = {"Degree": ("rot", 1.0), "Milli Radian": ("rot", math.pi / 180.0 * 1000.0),
        "Milli Meter": ("trans", 1.0), "Meter": ("trans", 1e-3)}
X = np.array([1.0, 0.0, 0.0])


def mat(m):
    return [[m.get(i, j) for j in range(3)] for i in range(3)]


def vec(v):
    return None if v is None else [float(x) for x in v.data]


def b_matrix(s):
    """Correct Busing-Levy B with 2 pi: Cholesky of the reciprocal metric."""
    a, b, c = s["a"], s["b"], s["c"]
    al, be, ga = (math.radians(s[k]) for k in ("alpha", "beta", "gamma"))
    g = np.array([[a * a, a * b * math.cos(ga), a * c * math.cos(be)],
                  [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
                  [a * c * math.cos(be), b * c * math.cos(al), c * c]])
    return 2 * math.pi * np.linalg.cholesky(np.linalg.inv(g)).T


def b_matrix_hkl12(s):
    """B as libhkl 5.1.9 builds it: B[1][2] uses cos(alpha*) (HKL-12)."""
    b = b_matrix(s)
    al, be, ga = (math.radians(s[k]) for k in ("alpha", "beta", "gamma"))
    cas = (math.cos(be) * math.cos(ga) - math.cos(al)) / (math.sin(be) * math.sin(ga))
    b[1, 2] = 2 * math.pi * cas / (s["c"] * math.sin(al))
    return b


def triad(a, b):
    t1 = a / np.linalg.norm(a)
    t3 = np.cross(a, b)
    t3 /= np.linalg.norm(t3)
    return np.column_stack([t1, np.cross(t3, t1), t3])


def busing_levy(b, hkls, qs):
    """U from two reflections (Busing & Levy 1967): h1 exact, h2 in plane."""
    tc = triad(b @ hkls[0], b @ hkls[1])
    tp = triad(qs[0], qs[1])
    u = tp @ tc.T
    return u, u @ b


def q_sample(fac, wl, user_vals):
    """q in the sample frame from hkl's own holder rotations, kf = R_det x."""
    geo = fac.create_new_geometry()
    geo.wavelength_set(wl, USER)
    geo.axis_values_set(user_vals, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    smp = Hkl.Sample.new("ref")
    rs = np.array(mat(geo.sample_rotation_get(smp).to_matrix()))
    rd = np.array(mat(geo.detector_rotation_get(det).to_matrix()))
    return rs.T @ (2 * math.pi / wl * (rd @ X - X))


def lattice(s):
    return Hkl.Lattice.new(s["a"], s["b"], s["c"], math.radians(s["alpha"]),
                           math.radians(s["beta"]), math.radians(s["gamma"]))


def reflection_record(h, user_vals, canon, names, wl):
    return {"hkl": list(h), "axes": dict(zip(names, user_vals)),
            "axes_canonical_deg_mm": dict(zip(names, canon)), "wavelength": wl}


def solve(eng, h):
    """First solution of hkl's own engine for h, or None when unreachable."""
    try:
        sols = eng.pseudo_axis_values_set([float(x) for x in h], USER)
    except Exception:
        return None
    items = sols.items() if sols is not None else []
    return items[0].geometry_get().copy() if items else None


def one_case(fac, gname, sname, s, wl, rng, units, names):
    case = {"geometry": gname, "sample": sname, "lattice": dict(s), "wavelength": wl}
    ux, uy, uz = rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)
    case["U_euler_deg"] = [ux, uy, uz]
    try:
        geo = fac.create_new_geometry()
        geo.wavelength_set(wl, USER)
        det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        sample = Hkl.Sample.new("true")
        sample.lattice_set(lattice(s))
        sample.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in (ux, uy, uz))))
        case["U_true"] = mat(sample.U_get())
        el = fac.create_new_engine_list()
        el.init(geo, det, sample)
        eng = el.engine_get_by_name("hkl")
        if eng is None:
            raise RuntimeError("no 'hkl' engine in this geometry")
        case["mode"] = eng.current_mode_get()
        b = b_matrix(s)
        kmax = 2 * 2 * math.pi / wl  # |Q| <= 2k
        picked = []
        for _ in range(MAX_TRIES):
            h = [rng.randint(-4, 4) for _ in range(3)]
            hc = b @ np.array(h, float)
            if not any(h) or np.linalg.norm(hc) > 0.9 * kmax:
                continue
            if picked:
                h1c = b @ np.array(picked[0][0], float)
                sin = np.linalg.norm(np.cross(h1c, hc)) / (np.linalg.norm(h1c) * np.linalg.norm(hc))
                if sin < 0.3:
                    continue
            g = solve(eng, h)
            if g is None:
                continue
            picked.append((h, g))
            if len(picked) == 2:
                break
        if len(picked) < 2:
            raise RuntimeError(f"no two reachable reflections in {MAX_TRIES} tries "
                               f"(found {len(picked)})")
        refl, qs = [], []
        s2 = Hkl.Sample.new("fit")
        s2.lattice_set(lattice(s))
        hr = []
        for h, g in picked:
            user = list(g.axis_values_get(USER))
            canon = [v / UNIT[u][1] for v, u in zip(user, units)]
            refl.append(reflection_record(h, user, canon, names, wl))
            qs.append(q_sample(fac, wl, user))
            hr.append(s2.add_reflection(g, det, *[float(x) for x in h]))
        case["reflections"] = refl
        s2.compute_UB_busing_levy(hr[0], hr[1])
        case["U"] = mat(s2.U_get())
        case["UB"] = mat(s2.UB_get())
        hk = [np.array(r["hkl"], float) for r in refl]
        u_ref, ub_ref = busing_levy(b, hk, qs)
        u_w, ub_w = busing_levy(b_matrix_hkl12(s), hk, qs)
        case["U_reference"] = u_ref.tolist()
        case["UB_reference"] = ub_ref.tolist()
        case["UB_hkl12_model"] = ub_w.tolist()
        ub = np.array(case["UB"])
        d_ref = float(np.abs(ub - ub_ref).max())
        d_w = float(np.abs(ub - ub_w).max())
        case["UB_vs_reference_maxabs"] = d_ref
        case["UB_vs_hkl12_model_maxabs"] = d_w
        case["U_vs_U_true_maxabs"] = float(np.abs(np.array(case["U"]) - np.array(case["U_true"])).max())
        case["differs_from_reference"] = d_ref > TOL_EXPLAIN
        case["explained_by_HKL12"] = bool(d_ref > TOL_EXPLAIN and d_w <= TOL_EXPLAIN)
    except Exception as e:  # recorded, never dropped
        case["error"] = f"{type(e).__name__}: {e}"
    return case


def hklpy2_case():
    from hklpy2.blocks.lattice import SI_LATTICE_PARAMETER
    from hklpy2.diffract import creator

    refl = [("(400)", (4, 0, 0), {"tth": 69.0966, "omega": -145.451, "chi": 0, "phi": 0}),
            ("(040)", (0, 4, 0), {"tth": 69.0966, "omega": -145.451, "chi": 90, "phi": 0})]
    wl = 1.54
    f = creator()
    f.add_sample("silicon", SI_LATTICE_PARAMETER)
    for name, h, reals in refl:
        f.add_reflection(h, reals, wavelength=wl, name=name)
    f.core.calc_UB(*f.sample.reflections.order)
    fac = Hkl.factories()["E4CV"]
    names = list(fac.create_new_geometry().axis_names_get())
    s = dict(SAMPLES["cubic"])
    s.update(a=SI_LATTICE_PARAMETER, b=SI_LATTICE_PARAMETER, c=SI_LATTICE_PARAMETER)
    qs = [q_sample(fac, wl, [reals[n] for n in names]) for _, _, reals in refl]
    u_ref, ub_ref = busing_levy(b_matrix(s), [np.array(h, float) for _, h, _ in refl], qs)
    ub = np.array(f.sample.UB)
    return {"source": "hklpy2/src/hklpy2/tests/test_diffract.py:758-790 (test_orientation)",
            "solver": f.core.solver.name, "geometry": f.core.solver.geometry,
            "mode": f.core.mode, "lattice": s, "wavelength": wl,
            "reflections": [{"name": n, "hkl": list(h), "axes": reals, "wavelength": wl}
                            for n, h, reals in refl],
            "U": np.array(f.sample.U).tolist(), "UB": ub.tolist(),
            "U_reference": u_ref.tolist(), "UB_reference": ub_ref.tolist(),
            "UB_vs_reference_maxabs": float(np.abs(ub - ub_ref).max())}


def main():
    # self-test of the HKL-12 model: both B agree exactly when beta = gamma = 90
    for n in ("cubic", "hexagonal", "monoclinic"):
        assert np.allclose(b_matrix(SAMPLES[n]), b_matrix_hkl12(SAMPLES[n]), atol=1e-12), n
    assert not np.allclose(b_matrix(SAMPLES["triclinic"]), b_matrix_hkl12(SAMPLES["triclinic"]), atol=1e-6)
    factories = Hkl.factories()
    geometries, cases = {}, []
    for gname in factories.keys():
        fac = factories[gname]
        geo = fac.create_new_geometry()
        names = list(geo.axis_names_get())
        units = [geo.axis_get(n).user_unit_get() for n in names]
        geometries[gname] = {"axes": [{"name": n, "axis_v": vec(geo.axis_get(n).axis_v_get()), "unit": u}
                                      for n, u in zip(names, units)]}
        for sname, s in SAMPLES.items():
            for wl in WAVELENGTHS:
                rng = random.Random(f"{SEED}-{gname}-{sname}-{wl}")
                case = one_case(fac, gname, sname, s, wl, rng, units, names)
                if "mode" in case or "error" in case:
                    cases.append(case)
    doc = {"hkl_version": Hkl.VERSION, "seed": SEED,
           "units": {"angle": "deg", "length": "mm", "wavelength": "angstrom", "UB": "1/angstrom, with 2 pi"},
           "tolerance_explain": TOL_EXPLAIN, "geometries": geometries, "cases": cases,
           "hklpy2": hklpy2_case()}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    ok = [c for c in cases if "error" not in c]
    print("wrote", OUT, "cases", len(cases), "ok", len(ok), "errors", len(cases) - len(ok))
    for c in cases:
        if "error" in c:
            print("ERR", c["geometry"], c["sample"], c["wavelength"], c["error"])
    for sname in SAMPLES:
        g = [c for c in ok if c["sample"] == sname]
        d = [c for c in g if c["differs_from_reference"]]
        ex = [c for c in d if c["explained_by_HKL12"]]
        print(f"{sname}: ok {len(g)} differ {len(d)} explained_by_HKL12 {len(ex)} "
              f"max|UB-ref| {max([c['UB_vs_reference_maxabs'] for c in g] or [0]):.3g} "
              f"max|U-Utrue| {max([c['U_vs_U_true_maxabs'] for c in g] or [0]):.3g}")
    print("hklpy2 |UB-ref|", doc["hklpy2"]["UB_vs_reference_maxabs"])


main()
