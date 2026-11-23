"""hklrs against hklpy2's th_tth solver: golden/th_tth.json and live.

th_tth has no orientation; its q depends on tth alone, theta is passed
through. hklrs is driven as E4CV with omega = th, chi = phi = 0.
"""
import json
import math
import os

import numpy as np
import pytest

import hklrs

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden", "th_tth.json")
with open(GOLDEN) as f:
    DOC = json.load(f)
WL = DOC["wavelength"]
A = DOC["lattice_a"]
GEO = hklrs.Geometry.preset("E4CV")
SAMPLE = hklrs.Sample("cubic", hklrs.Lattice(A, A, A, 90, 90, 90))


def hklrs_th_tth(th, tth, wl=WL):
    assert GEO.axis_names == ["omega", "chi", "phi", "tth"]
    return hklrs.forward(GEO, SAMPLE, [th, 0.0, 0.0, tth], wl)


def q_numpy(tth, wl=WL):
    return 4 * math.pi / wl * math.sin(math.radians(tth) / 2)


CASES = [pytest.param(c, id=f"th{c['theta']}-tth{c['tth']}") for c in DOC["cases"]]


@pytest.mark.parametrize("c", CASES)
def test_golden_q_and_hkl_norm(c):
    r = hklrs_th_tth(c["theta"], c["tth"])
    assert r["q"] == pytest.approx(c["q"], rel=1e-12, abs=1e-14)
    assert r["q"] == pytest.approx(q_numpy(c["tth"]), rel=1e-12, abs=1e-14)
    assert np.linalg.norm(r["hkl"]) == pytest.approx(c["hkl_norm"], rel=1e-12, abs=1e-14)
    assert r["tth"] == pytest.approx(c["tth"], abs=1e-12)


@pytest.mark.parametrize("c", CASES)
def test_golden_forward_roundtrip(c):
    fw = c["forward"]  # th = asin(q wl / 4 pi), tth = 2 th
    assert fw["tth"] == pytest.approx(2 * math.degrees(math.asin(c["q"] * WL / 4 / math.pi)), abs=1e-10)
    assert fw["th"] == pytest.approx(fw["tth"] / 2, abs=1e-12)
    r = hklrs_th_tth(fw["th"], fw["tth"])
    assert r["q"] == pytest.approx(c["q"], rel=1e-12)
    assert r["tth"] == pytest.approx(fw["tth"], abs=1e-10)


def solver():
    pytest.importorskip("hklpy2")
    from hklpy2.backends.th_tth_q import TH_TTH_Q_GEOMETRY, ThTthSolver

    s = ThTthSolver(TH_TTH_Q_GEOMETRY)
    s.wavelength = WL
    return s


@pytest.mark.parametrize("c", CASES)
def test_live_inverse_and_forward(c):
    s = solver()
    q = s.inverse({"th": c["theta"], "tth": c["tth"]})["q"]
    r = hklrs_th_tth(c["theta"], c["tth"])
    assert r["q"] == pytest.approx(q, rel=1e-12, abs=1e-14)
    fw = s.forward({"q": q})[0]
    r2 = hklrs_th_tth(fw["th"], fw["tth"])
    assert r2["q"] == pytest.approx(q, rel=1e-12, abs=1e-14)
    assert r2["tth"] == pytest.approx(fw["tth"], abs=1e-10)


@pytest.mark.parametrize("wl", [0.71, 1.0, 2.5])
def test_live_other_wavelengths(wl):
    s = solver()
    s.wavelength = wl
    for tth in (1.0, 17.0, 60.0, 120.0, 179.0):
        q = s.inverse({"th": tth / 2, "tth": tth})["q"]
        assert hklrs_th_tth(tth / 2, tth, wl)["q"] == pytest.approx(q, rel=1e-12)
