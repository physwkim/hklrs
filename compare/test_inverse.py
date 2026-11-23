"""hklrs E4C inverse against libhkl 5.1.9 (gi, live), E4CV and E4CH, every mode.

Per mode and seed: random axes that satisfy the mode give a reachable hkl;
both sides solve it from a start that moves only the axes the mode writes.
Asserted: every libhkl solution is in the hklrs set (1e-6 rad), every hklrs
solution round-trips through hklrs.forward, |Q| > 4 pi / lambda raises
Unreachable. libhkl's solution count is never asserted (hklrs may find more).
"""
import ctypes
import math
import random

import numpy as np
import pytest

import hklrs
from test_hkl import USER, Hkl, mat

LIBC = ctypes.CDLL(None)
LATTICE = (5.431020511, 5.431020511, 5.431020511, 90.0, 90.0, 90.0)
WL = 1.54
N_CASES = 30
TOL_RAD = 1e-6
# libhkl stops its GSL solve at ~1e-8 in hkl; where the system is ill-conditioned
# (chi near 0 or 90) that moves an axis by up to ~5e-6 rad from the exact root
# hklrs returns. A libhkl solution further than TOL_RAD from every hklrs
# solution is accepted within LOOSE_RAD only when its own hkl residual exceeds
# EXACT_HKL (hklrs's round trip is 1e-15).
LOOSE_RAD = 1e-4
EXACT_HKL = 1e-9
MODES = ("bissector", "constant_omega", "constant_chi", "constant_phi",
         "double_diffraction", "psi_constant")
PRESETS = ("E4CV", "E4CH")
WRITTEN = {"bissector": 4, "double_diffraction": 4, "psi_constant": 4,
           "constant_omega": 3, "constant_chi": 3, "constant_phi": 3}
HELD = {"constant_omega": 0, "constant_chi": 1, "constant_phi": 2}


def wrap(d):
    """Degrees to (-180, 180]."""
    return 180.0 - ((180.0 - d) % 360.0)


class Rig:
    def __init__(self, preset, seed):
        fac = Hkl.factories()[preset]
        self.geo = fac.create_new_geometry()
        self.geo.wavelength_set(WL, USER)
        self.names = list(self.geo.axis_names_get())
        assert self.names == ["omega", "chi", "phi", "tth"]
        assert {self.geo.axis_get(n).user_unit_get() for n in self.names} == {"Degree"}
        det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.smp = Hkl.Sample.new("inv")
        self.smp.lattice_set(Hkl.Lattice.new(*LATTICE[:3], *(math.radians(x) for x in LATTICE[3:])))
        rng = random.Random(f"inv-{preset}-{seed}")
        self.smp.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in (
            rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)))))
        self.el = fac.create_new_engine_list()  # owns the engine: keep it alive
        self.el.init(self.geo, det, self.smp)
        self.eng = self.el.engine_get_by_name("hkl")
        self.g = hklrs.Geometry.preset(preset)
        self.lat = hklrs.Lattice(*LATTICE)
        self.sample = hklrs.Sample("inv", self.lat, mat(self.smp.U_get()).tolist())
        np.testing.assert_allclose(self.sample.ub(), mat(self.smp.UB_get()), atol=1e-12)

    def set_axes(self, v):
        self.geo.axis_values_set([float(x) for x in v], USER)

    def solve_hkl(self, mode, params, start, target, c_seed):
        """libhkl's solutions as lists of degrees (hkl's user unit is degree)."""
        self.set_axes(start)
        self.eng.current_mode_set(mode)
        if params:
            self.eng.parameters_values_set(params, USER)
        self.eng.initialized_set(True)  # psi_constant's init overwrites psi
        if params:
            self.eng.parameters_values_set(params, USER)
        self.set_axes(start)
        LIBC.srand(c_seed)
        try:
            sols = self.eng.pseudo_axis_values_set([float(x) for x in target], USER)
        except Exception:
            return []
        items = sols.items() if sols is not None else []
        return [list(i.geometry_get().axis_values_get(USER)) for i in items]


def forward_hkl(rig, axes):
    return np.array(hklrs.forward(rig.g, rig.sample, axes, WL)["hkl"])


def case_inputs(rig, mode, rng):
    """(hklrs mode, hkl params, start, target) with a target reachable under the mode."""
    a = [rng.uniform(-180, 180), rng.uniform(-180, 180), rng.uniform(-180, 180),
         rng.uniform(5, 150)]
    if mode == "bissector":
        a[0] = a[3] / 2
    target = forward_hkl(rig, a)
    hkl2 = [float(rng.choice([-2, -1, 1, 2])) for _ in range(3)]
    kw, params = {}, None
    if mode == "double_diffraction":
        kw, params = {"hkl2": hkl2}, hkl2
    elif mode == "psi_constant":
        psi = hklrs.psi(rig.g, rig.sample, hkl2, a)
        kw, params = {"reference": hkl2, "psi": psi}, hkl2 + [psi]
    start = list(a)
    for i in range(4):
        if mode not in HELD or HELD[mode] != i:
            start[i] += rng.uniform(-20, 20)
    return hklrs.Mode(rig.g, mode, **kw), params, start, target


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("preset", PRESETS)
def test_inverse_against_libhkl(preset, mode):
    if preset not in Hkl.factories():
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}")
    rig = Rig(preset, mode)
    if mode not in rig.eng.modes_names_get():
        pytest.skip(f"libhkl {Hkl.VERSION} {preset} has no mode {mode}")
    rng = random.Random(f"cases-{preset}-{mode}")
    both = hkl_sols = 0
    for n in range(N_CASES):
        m, params, start, target = case_inputs(rig, mode, rng)
        theirs = rig.solve_hkl(mode, params, start, target, 1000 + n)
        try:
            ours = hklrs.solve(m, rig.sample, target, WL, start)
        except hklrs.NoSolution:
            ours = []
        for s in ours:  # round trip, and the held axis is untouched
            np.testing.assert_allclose(forward_hkl(rig, s), target, atol=1e-9)
            if mode in HELD:
                assert s[HELD[mode]] == pytest.approx(start[HELD[mode]], abs=1e-12)
        for t in theirs:
            miss = min(max(abs(math.radians(wrap(x - y))) for x, y in zip(t, s))
                       for s in ours) if ours else math.inf
            if miss >= TOL_RAD:
                resid = np.abs(forward_hkl(rig, t) - target).max()
                assert miss < LOOSE_RAD and resid > EXACT_HKL, (
                    f"{preset} {mode} case {n}: libhkl solution {t} is not in the hklrs "
                    f"set {ours} (miss {miss} rad, libhkl hkl residual {resid})")
        hkl_sols += bool(theirs)
        both += bool(theirs) and bool(ours)
    assert hkl_sols > 0, f"libhkl found no solution in {N_CASES} reachable cases"
    assert both == hkl_sols


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("preset", PRESETS)
def test_unreachable_raises(preset, mode):
    rig = Rig(preset, mode)
    m, _, start, _ = case_inputs(rig, mode, random.Random(1))
    q_max = 4 * math.pi / WL
    for h in ((10, 10, 10), (0, 0, 8), (-7, 3, 5)):
        assert np.linalg.norm(np.array(rig.sample.ub()) @ h) > q_max
        with pytest.raises(hklrs.Unreachable) as e:
            hklrs.solve(m, rig.sample, h, WL, start)
        assert e.value.q > e.value.q_max == pytest.approx(q_max)
        assert isinstance(e.value, hklrs.InverseError) and isinstance(e.value, ValueError)
        with pytest.raises(hklrs.Unreachable):
            hklrs.pick(m, rig.sample, h, WL, start)
