"""hklrs Busing-Levy orientation against libhkl 5.1.9 (gi, live) and hklpy2.

libhkl's B is wrong where c(beta) c(gamma) != c(alpha) (1 - s(beta) s(gamma))
(HKL-12), so there its UB is expected to diverge; hklrs is then checked against
a numpy Busing-Levy with the correct B.
"""
import math
import random

import numpy as np
import pytest

import _ref
import hklrs
from test_hkl import UNIT, USER, Hkl, hkl_case  # noqa: F401  (Hkl: gi imported there)

CELLS = {
    "cubic": (5.431020511, 5.431020511, 5.431020511, 90.0, 90.0, 90.0),
    "hexagonal": (3.0, 3.0, 5.0, 90.0, 90.0, 120.0),
    "monoclinic": (4.1, 5.3, 6.7, 90.0, 97.0, 90.0),
    "triclinic": (4.1, 5.3, 6.7, 82.0, 97.0, 101.0),
}
WAVELENGTHS = (1.54, 0.71)
SEEDS = (0, 1)
MAX_TRIES = 200
X = _ref.X


def hkl12_holds(lat):
    al, be, ga = (math.radians(x) for x in lat[3:])
    lhs = math.cos(be) * math.cos(ga)
    rhs = math.cos(al) * (1 - math.sin(be) * math.sin(ga))
    return abs(lhs - rhs) < 1e-12


def has_hkl_engine(preset, fac):
    geo = fac.create_new_geometry()
    el = fac.create_new_engine_list()
    el.init(geo, Hkl.Detector.factory_new(Hkl.DetectorType(0)), Hkl.Sample.new("p"))
    return "hkl" in [e.name_get() for e in el.engines_get()]


def lattice_hkl(lat):
    return Hkl.Lattice.new(lat[0], lat[1], lat[2], *(math.radians(x) for x in lat[3:]))


def q_sample_hkl(fac, wl, user):
    """Q in the sample frame from libhkl's holder rotations, kf = R_det x."""
    geo = fac.create_new_geometry()
    geo.wavelength_set(wl, USER)
    geo.axis_values_set(user, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    rs = np.array(_mat(geo.sample_rotation_get(Hkl.Sample.new("r")).to_matrix()))
    rd = np.array(_mat(geo.detector_rotation_get(det).to_matrix()))
    return rs.T @ (2 * math.pi / wl * (rd @ X - X))


def _mat(m):
    return [[m.get(i, j) for j in range(3)] for i in range(3)]


def solve(eng, h):
    try:
        sols = eng.pseudo_axis_values_set([float(x) for x in h], USER)
    except Exception:
        return None
    items = sols.items() if sols is not None else []
    return items[0].geometry_get().copy() if items else None


def two_reflections(preset, cell, wl, seed):
    """libhkl's two solved reflections and its Busing-Levy UB, or a skip."""
    lat = CELLS[cell]
    factories = Hkl.factories()
    if preset not in factories:
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}; "
                    "its parity is the Rust golden hklrs/tests/hkl_golden.rs "
                    "(hkl 7be4f79 driver)")
    fac = factories[preset]
    if not has_hkl_engine(preset, fac):
        pytest.skip(f"{preset} has no 'hkl' engine in libhkl {Hkl.VERSION}, "
                    "so there is nothing to solve reflections with")
    rng = random.Random(f"bl-{preset}-{cell}-{wl}-{seed}")
    geo = fac.create_new_geometry()
    geo.wavelength_set(wl, USER)
    names = list(geo.axis_names_get())
    units = [geo.axis_get(n).user_unit_get() for n in names]
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    true = Hkl.Sample.new("true")
    true.lattice_set(lattice_hkl(lat))
    true.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in (
        rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)))))
    el = fac.create_new_engine_list()
    el.init(geo, det, true)
    eng = el.engine_get_by_name("hkl")
    mode = eng.current_mode_get()
    b = _ref.b_matrix(*lat)
    picked = []
    for _ in range(MAX_TRIES):
        h = [rng.randint(-4, 4) for _ in range(3)]
        hc = b @ np.array(h, float)
        if not any(h) or np.linalg.norm(hc) > 0.9 * 2 * 2 * math.pi / wl:
            continue
        if picked:
            h1 = b @ np.array(picked[0][0], float)
            if np.linalg.norm(np.cross(h1, hc)) < 0.3 * np.linalg.norm(h1) * np.linalg.norm(hc):
                continue
        g = solve(eng, h)
        if g is not None:
            picked.append((h, g))
        if len(picked) == 2:
            break
    if len(picked) < 2:
        pytest.skip(f"{preset} in its default mode {mode}: no two reachable reflections "
                    f"for a random U in {MAX_TRIES} tries (found {len(picked)})")
    fit = Hkl.Sample.new("fit")
    fit.lattice_set(lattice_hkl(lat))
    refl, hr = [], []
    for h, g in picked:
        user = list(g.axis_values_get(USER))
        refl.append((h, [v / UNIT[u][1] for v, u in zip(user, units)], user))
        hr.append(fit.add_reflection(g, det, *[float(x) for x in h]))
    fit.compute_UB_busing_levy(hr[0], hr[1])
    return fac, refl, np.array(_mat(fit.UB_get()))


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_busing_levy_against_libhkl(preset, cell):
    lat = CELLS[cell]
    n = 0
    for wl in WAVELENGTHS:
        for seed in SEEDS:
            fac, refl, ub_h = two_reflections(preset, cell, wl, seed)
            geo = hklrs.Geometry.preset(preset)
            r = [hklrs.Reflection(geo, h, canon, wl) for h, canon, _ in refl]
            qs = [q_sample_hkl(fac, wl, user) for _, _, user in refl]
            for ri, q in zip(r, qs):
                np.testing.assert_allclose(ri.q_sample, q, atol=1e-12)
            smp = hklrs.Sample.busing_levy("bl", hklrs.Lattice(*lat), r[0], r[1])
            ub = np.array(smp.ub())
            _, ub_ref = _ref.busing_levy_ub(_ref.b_matrix(*lat), [x[0] for x in refl], qs)
            np.testing.assert_allclose(ub, ub_ref, atol=1e-12,
                                       err_msg=f"{preset} {cell} wl={wl} seed={seed} vs numpy")
            diff = np.abs(ub - ub_h).max()
            if hkl12_holds(lat):
                assert diff < 1e-12, f"{preset} {cell} wl={wl} seed={seed}: |UB-UB_hkl|={diff}"
            else:
                assert diff > 1e-4, f"{preset} {cell}: HKL-12 divergence expected, got {diff}"
            n += 1
    assert n == len(WAVELENGTHS) * len(SEEDS)


def test_busing_levy_hklpy2_si_400_040():
    """hklpy2/src/hklpy2/tests/test_diffract.py:758-790, run live."""
    pytest.importorskip("hklpy2")
    from hklpy2.blocks.lattice import SI_LATTICE_PARAMETER
    from hklpy2.diffract import creator

    wl = 1.54
    f = creator()
    f.add_sample("silicon", SI_LATTICE_PARAMETER)
    reals = [{"tth": 69.0966, "omega": -145.451, "chi": 0, "phi": 0},
             {"tth": 69.0966, "omega": -145.451, "chi": 90, "phi": 0}]
    for name, h, rl in zip(("(400)", "(040)"), ((4, 0, 0), (0, 4, 0)), reals):
        f.add_reflection(h, rl, wavelength=wl, name=name)
    f.core.calc_UB("(400)", "(040)")
    geo = hklrs.Geometry.preset("E4CV")
    a = SI_LATTICE_PARAMETER
    r = [hklrs.Reflection(geo, h, [rl[n] for n in geo.axis_names], wl)
         for h, rl in zip(((4, 0, 0), (0, 4, 0)), reals)]
    smp = hklrs.Sample.busing_levy("si", hklrs.Lattice(a, a, a, 90, 90, 90), r[0], r[1])
    np.testing.assert_allclose(smp.ub(), np.array(f.sample.UB), atol=1e-11)


def test_parallel_reflections_raise():
    geo = hklrs.Geometry.preset("E4CV")
    lat = hklrs.Lattice(5.0, 5.0, 5.0, 90, 90, 90)
    ang = [10.0, 20.0, 30.0, 40.0]
    # parallel hkl
    a = hklrs.Reflection(geo, (1, 0, 0), ang, 1.54)
    b = hklrs.Reflection(geo, (2, 0, 0), [15.0, 20.0, 30.0, 40.0], 1.54)
    with pytest.raises(ValueError):
        hklrs.Sample.busing_levy("p", lat, a, b)
    # parallel measured Q (same angles), independent hkl
    c = hklrs.Reflection(geo, (0, 1, 0), ang, 1.54)
    with pytest.raises(ValueError):
        hklrs.Sample.busing_levy("p", lat, a, c)
