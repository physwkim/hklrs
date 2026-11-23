"""hklrs.pseudo against libhkl 5.1.9 (gi, live): q, q2, tth2, qper_qpar,
incidence, emergence over every preset the installed libhkl has them for.

Not compared: the incidence azimuth. incidence and emergence both register
the pseudo axis "azimuth", hkl shares one parameter between them and the
engine computed last wins, so libhkl's incidence azimuth is not incidence's
(hklrs/tests/hkl_pseudo.rs checks it against an independent value).
Points with 2 theta < TTH_MIN are skipped (hkl_vector_angle's acos, HKL-11).
"""
import math
import random

import numpy as np
import pytest

import hklrs
from test_hkl import UNIT, USER, Hkl

ENGINES = ("q", "q2", "tth2", "qper_qpar", "incidence", "emergence")
NORMALS = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 2.0, -3.0), (0.0, 3.0, 4.0))
WL = 1.54
N_RANDOM = 12
TTH_MIN = 0.1   # degrees
TOL = 1e-7      # degrees, q in 1/Angstrom


def wrap(d):
    return 180.0 - ((180.0 - d) % 360.0)


def rig(preset):
    if preset not in Hkl.factories():
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}")
    fac = Hkl.factories()[preset]
    geo = fac.create_new_geometry()
    geo.wavelength_set(WL, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    smp = Hkl.Sample.new("pseudo")
    el = fac.create_new_engine_list()
    el.init(geo, det, smp)
    engines = {e.name_get(): e for e in el.engines_get() if e.name_get() in ENGINES}
    return geo, smp, el, engines


def configs(preset, geo):
    rng = random.Random(f"pseudo-{preset}")
    names = list(geo.axis_names_get())
    units = [geo.axis_get(n).user_unit_get() for n in names]
    for _ in range(N_RANDOM):
        yield units, [rng.uniform(-0.05, 0.05) if UNIT[u][0] == "trans"
                      else rng.uniform(-180.0, 180.0) for u in units]


@pytest.mark.parametrize("engine", ENGINES)
def test_pseudo_against_libhkl(preset, engine):
    geo, smp, el, engines = rig(preset)
    if engine not in engines:
        pytest.skip(f"{preset} has no {engine} engine in libhkl {Hkl.VERSION}")
    eng = engines[engine]
    g = hklrs.Geometry.preset(preset)
    rng_n = NORMALS if engine in ("qper_qpar", "incidence", "emergence") else (None,)
    n = 0
    for units, canon in configs(preset, geo):
        f = hklrs.forward(g, hklrs.Sample("s", hklrs.Lattice(5, 5, 5, 90, 90, 90)), canon, WL)
        if f["tth"] < TTH_MIN:
            continue
        geo.axis_values_set([c * UNIT[u][1] for c, u in zip(canon, units)], USER)
        for surf in rng_n:
            if surf is not None:
                eng.parameters_values_set(list(surf), USER)
            assert el.get()
            theirs = [float(x) for x in eng.pseudo_axis_values_get(USER)]
            ours = hklrs.pseudo(g, engine, canon, WL, surface=surf)
            # q is hkl's signed q; angles in degrees (alpha, azimuth wrap)
            for i, (a, b) in enumerate(zip(ours, theirs)):
                if engine in ("incidence", "emergence") and i == 1:
                    if engine == "incidence":
                        continue
                    if a is None:
                        assert abs(b) < TOL
                        continue
                    assert abs(wrap(a - b)) < TOL, (preset, engine, canon, surf, a, b)
                    continue
                angle = (engine == "q2" and i == 1) or engine == "tth2" or \
                    (engine in ("incidence", "emergence") and i == 0)
                d = wrap(a - b) if angle else a - b
                assert abs(d) < TOL, (preset, engine, canon, surf, i, a, b)
            n += 1
    assert n > 0
