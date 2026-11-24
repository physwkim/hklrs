"""hklrs Geometry.set_limits and set_cut_point against hklpy2 1.0.2.dev5's own
LimitsConstraint (live, hkl_soleil on libhkl 5.1.9), E4CV and E6C.

Per mode and seed: random axes that satisfy the mode give a reachable hkl;
cut points are drawn away from every solution value, then limits between
solution values on the written axes. Asserted: each returned rotation lies in
[cut, cut + 360) where hklpy2's apply_cut leaves it, hklrs keeps a solution
exactly when hklpy2's constraints accept it, and hklpy2 keeps and drops the
same libhkl solutions. Held axes start inside their window with no limits.

Both limit held axes too; hklrs checks one at its current value, unwrapped,
as the motor stands, where hklpy2 wraps it to its cut point first. So both
return nothing for a held axis outside its limits, and they differ on purpose
for one outside its window that its limits admit only once wrapped: hklpy2
keeps every solution, hklrs none. Modes taking parameters are left out: the
filter does not read the equations.
"""
import functools
import math
import random

import numpy as np
import pytest

import hklrs
from test_hkl import Hkl, mat

pytest.importorskip("hklpy2")
from hklpy2.diffract import creator  # noqa: E402

LATTICE = (5.431, 5.431, 5.431, 90.0, 90.0, 90.0)
WL = 1.54
SEEDS = range(6)
# deg; libhkl's raw solutions are within 1e-4 deg of hklrs's
TOL = 1e-3
# deg between any solution value and a limit or cut point: above hklpy2's
# ENDPOINT_TOLERANCE (1e-4) plus TOL
MARGIN = 0.05
MODES = {
    "E4CV": ("bissector", "constant_omega", "constant_chi", "constant_phi"),
    "E6C": ("bissector_vertical", "constant_omega_vertical", "constant_chi_vertical",
            "constant_phi_vertical", "lifting_detector_phi", "lifting_detector_omega",
            "lifting_detector_mu", "bissector_horizontal", "constant_mu_horizontal"),
}
# sample axis at half the detector axis, the third axis at 0
BISECTORS = {"bissector": ("omega", "tth", None),
             "bissector_vertical": ("omega", "delta", None),
             "bissector_horizontal": ("mu", "gamma", "omega")}
CASES = [(p, m) for p in MODES for m in MODES[p]]


@functools.cache
def diffractometer(preset):
    f = creator(geometry=preset)
    # forward() rounds its solutions to this many decimals
    f.digits = 12
    f.beam.wavelength.put(WL)
    f.add_sample("s", *LATTICE)
    return f


def window(cut, v):
    """v in [cut, cut + 360)."""
    return cut + (v - cut) % 360.0


def turn(d):
    """Degrees to [-180, 180)."""
    return (d + 180.0) % 360.0 - 180.0


def match(p, sols):
    """Index of the one solution within TOL of p on every axis, else None."""
    hits = [i for i, s in enumerate(sols) if np.abs(np.subtract(s, p)).max() < TOL]
    assert len(hits) <= 1, (p, hits)
    return hits[0] if hits else None


def cuts_away(c, first):
    """Cut points away from every value an axis takes in `first`, set with no
    limits; a held axis's window holds its current value, which both return."""
    cuts = {}
    for i, n in enumerate(c.names):
        values = [s[i] for s in first]
        lo, hi = (-360, 180) if i in c.written else (c.start[i] - 360, c.start[i])
        while True:
            cut = c.rng.uniform(lo, hi)
            if all(abs(turn(cut - v)) > MARGIN for v in values):
                break
        cuts[n] = cut
        c.set(n, (-math.inf, math.inf), cut)
    return cuts


class Case:
    def __init__(self, preset, mode, seed):
        self.rng = random.Random(f"constraints-{preset}-{mode}-{seed}")
        self.f = f = diffractometer(preset)
        self.g = hklrs.Geometry.preset(preset)
        self.names = self.g.axis_names
        assert list(f.real_axis_names) == self.names
        u = mat(Hkl.Matrix.new_euler(*(math.radians(self.rng.uniform(-180, 180))
                                       for _ in range(3)))).tolist()
        # Sample.U alone is lost: Core.update_solver pushes the stale UB after it
        f.sample.U = u
        f.sample.UB = hklrs.Sample("s", hklrs.Lattice(*LATTICE), u).ub()
        f.core.mode = mode
        f.core.update_solver()
        self.sample = hklrs.Sample("s", hklrs.Lattice(*LATTICE), f.core.solver.U)
        np.testing.assert_allclose(self.sample.ub(), f.core.solver.UB, rtol=0, atol=1e-11)
        self.mode = hklrs.Mode(self.g, mode)
        self.written = [self.names.index(n) for n in f.core.solver_written_axis_names]

        a = {n: self.rng.uniform(-180, 180) for n in self.names}
        if mode in BISECTORS:
            sample, det, zero = BISECTORS[mode]
            a[det] = self.rng.uniform(5, 150)
            a[sample] = a[det] / 2
            if zero:
                a[zero] = 0.0
        self.axes = [a[n] for n in self.names]
        self.hkl = hklrs.forward(self.g, self.sample, self.axes, WL)["hkl"]
        # hklpy2's simulated motors refuse positions outside [-180, 180]
        self.start = [turn(v + self.rng.uniform(-20, 20)) if i in self.written else v
                      for i, v in enumerate(self.axes)]

    def set(self, name, limits, cut=None):
        """Both sides' limits and, when given, cut point of axis `name`."""
        c = self.f.core.constraints[name]
        c.limits = limits
        self.g.set_limits(name, *limits)
        assert self.g.limits(name) == pytest.approx(limits, rel=1e-15, abs=0)
        if cut is not None:
            c.cut_point = cut
            self.g.set_cut_point(name, cut)
            assert self.g.cut_point(name) == pytest.approx(cut, rel=0, abs=1e-12)

    def hklrs(self):
        try:
            return hklrs.solve(self.mode, self.sample, self.hkl, WL, self.start)
        except hklrs.NoSolutionInLimits:
            return []

    def hklpy2(self):
        self.f.move_reals(dict(zip(self.names, self.start)))
        return [list(p) for p in self.f.core.forward(dict(zip("hkl", self.hkl)))]

    def accepted(self, sol):
        """hklpy2's own filter on one solution: apply_cut, then valid."""
        cs = self.f.core.constraints
        return cs.valid(**{n: cs[n].apply_cut(v) for n, v in zip(self.names, sol)})


@pytest.mark.parametrize("preset,mode", CASES)
def test_kept_and_dropped_as_hklpy2(preset, mode):
    if preset not in Hkl.factories():
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}")
    split = found = 0
    for seed in SEEDS:
        c = Case(preset, mode, seed)
        for n in c.names:
            c.set(n, (-math.inf, math.inf), -180.0)
        first = c.hklrs()
        assert first, "no solution through the axes the hkl came from"
        cuts = cuts_away(c, first)
        everything = c.hklrs()
        assert len(everything) == len(first)
        for s in everything:
            for n, v in zip(c.names, s):
                assert cuts[n] <= v < cuts[n] + 360, (n, v, cuts[n])
                assert c.f.core.constraints[n].apply_cut(v) == pytest.approx(v, rel=0, abs=1e-9)
        assert match([window(cuts[n], v) for n, v in zip(c.names, c.axes)], everything) is not None
        raw = c.hklpy2()
        found += bool(raw)
        assert all(match(p, everything) is not None for p in raw), (raw, everything)

        # limits between solution values on one or two written axes
        for i in c.rng.sample(c.written, c.rng.choice([1, 2])):
            values = sorted(s[i] for s in everything)
            ends = [-math.inf] + [(a + b) / 2 for a, b in zip(values, values[1:])
                                  if b - a > 2 * MARGIN] + [math.inf]
            if len(ends) > 2:
                lo, hi = sorted(c.rng.sample(range(len(ends)), 2))
                if (lo, hi) == (0, len(ends) - 1):
                    hi -= 1
                c.set(c.names[i], (ends[lo], ends[hi]))
        kept = c.hklrs()
        assert kept == [s for s in everything if c.accepted(s)]
        split += 0 < len(kept) < len(everything)
        assert_hklpy2_filters_alike(c, raw, everything, kept)

        # limits no solution meets
        i = c.rng.choice(c.written)
        c.set(c.names[i], (-math.inf, min(s[i] for s in everything) - MARGIN))
        with pytest.raises(hklrs.NoSolutionInLimits) as e:
            hklrs.solve(c.mode, c.sample, c.hkl, WL, c.start)
        assert e.value.solutions == everything
        assert not any(c.accepted(s) for s in everything)
        assert c.hklpy2() == []
    assert split, "no seed kept some solutions and dropped others"
    assert found, "libhkl found no solution for any seed"


def assert_hklpy2_filters_alike(c, raw, everything, kept):
    """hklpy2's filter accepts exactly the libhkl solutions whose hklrs match
    hklrs keeps, and its forward returns those. libhkl's solutions move by
    about 1e-6 deg from one forward to the next, hence the matching."""
    for p in raw:
        assert c.accepted(p) == (everything[match(p, everything)] in kept), p
    by_hklpy2 = [match(p, kept) for p in c.hklpy2()]
    assert None not in by_hklpy2
    assert sorted(by_hklpy2) == sorted(match(p, kept) for p in raw if c.accepted(p))


def held_case(preset, mode, seed):
    """A Case with no limits and cut points away from its solutions, those
    solutions on both sides, and one held axis."""
    if preset not in Hkl.factories():
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}")
    c = Case(preset, mode, seed)
    held = [i for i in range(len(c.names)) if i not in c.written]
    if not held:
        pytest.skip(f"{mode} holds no axis")
    for n in c.names:
        c.set(n, (-math.inf, math.inf), -180.0)
    first = c.hklrs()
    assert first, "no solution through the axes the hkl came from"
    cuts_away(c, first)
    everything = c.hklrs()
    assert len(everything) == len(first)
    i = c.rng.choice(held)
    # held exactly; the binding's degrees go through radians and back
    assert all(s[i] == math.degrees(math.radians(c.start[i])) for s in everything)
    return c, everything, c.hklpy2(), i


@pytest.mark.parametrize("preset,mode", CASES)
def test_held_axis_outside_its_limits_has_no_solution(preset, mode):
    """The held axis's current value is in its window, below its limits."""
    found = 0
    for seed in SEEDS:
        c, everything, raw, i = held_case(preset, mode, seed)
        found += bool(raw)
        v = c.start[i]
        c.set(c.names[i], (v + MARGIN, v + 10.0))
        with pytest.raises(hklrs.NoSolutionInLimits) as e:
            hklrs.solve(c.mode, c.sample, c.hkl, WL, c.start)
        assert e.value.solutions == everything
        assert c.hklpy2() == []
    assert found, "libhkl found no solution for any seed"


@pytest.mark.parametrize("preset,mode", CASES)
def test_held_axis_in_its_limits_only_once_wrapped_differs_on_purpose(preset, mode):
    """The held axis stands at v below its window [cut, cut + 360), limited
    to v + 360 +- 1. hklpy2 wraps v to v + 360 and keeps every solution there;
    hklrs neither wraps nor admits v, and raises with the same solutions as
    without the limits."""
    found = 0
    for seed in SEEDS:
        c, everything, raw, i = held_case(preset, mode, seed)
        found += bool(raw)
        v = c.start[i]
        c.set(c.names[i], (v + 359.0, v + 361.0), c.rng.uniform(v + 1.0, v + 359.0))
        with pytest.raises(hklrs.NoSolutionInLimits) as e:
            hklrs.solve(c.mode, c.sample, c.hkl, WL, c.start)
        assert e.value.solutions == everything
        kept = c.hklpy2()
        assert len(kept) == len(raw)
        for p in kept:
            assert p[i] == pytest.approx(v + 360.0, rel=0, abs=1e-9)
            assert match(p[:i] + [v] + p[i + 1:], everything) is not None, p
    assert found, "libhkl found no solution for any seed"
