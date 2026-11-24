"""hklrs.frame_map's "qsurface" against Python binoculars 93956b2's
sixs:qxqyqzprojection, the upstream code run: binoculars_qxqyqz.py runs it
in a subprocess under the binhkl env, whose Python 3.8, pyFAI 0.19 and
libhkl 7be4f79 typelib this suite's interpreter lacks. HKLRS_BINHKL names
that env (its prefix) and HKLRS_BINOCULARS the binoculars tree; the tests
skip without them.

binoculars' surface frame is the UB it substitutes: Rx(-90 deg) for a
VERTICAL surface, I for a HORIZONTAL one, times Rz(-omega_offset) or
Rz(-mu_offset) (sixs.py:374-390). Its R is the sample holder alone
(hkl-geometry.c:891-895 at 7be4f79), so the sample's U, given as the Pt(111)
scans record it, moves nothing. Each case is a short omega scan binned into
one cube on a grid laid on binoculars' coordinates.
Run with the py2 interpreter: python -I -m pytest compare
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import hklrs
from hklrs import frame_map as fm

ENV = os.environ.get("HKLRS_BINHKL")
TREE = os.environ.get("HKLRS_BINOCULARS")
pytestmark = pytest.mark.skipif(
    not (ENV and TREE), reason="HKLRS_BINHKL and HKLRS_BINOCULARS name binhkl and binoculars"
)

DX = 2.0**-8
# Q_sample per pixel: some 20 ulp of |Q| < 6 1/A
Q_TOL = 2e-14
# no coordinate this close to a bin edge, in bins: far above Q_TOL / DX, so
# both sides bin every pixel alike
MARGIN = 1e-9
PT111 = dict(A=2.77, B=2.77, C=6.789, alpha=90.0, beta=90.0, gamma=120.0,
             Ux=-90.2943398, Uy=-0.90294791, Uz=-18.05830816)
SIXS = dict(centralpixel="205, 177", sdd="1.170", detrot="90.0")


def case(base, surface=None, offset=None, wavelength=1.12712685, **source):
    """`base` mu, omega, delta, gamma (degrees) and two omega steps."""
    frames = [[base[0], base[1] + 0.25 * i, *base[2:]] for i in range(3)]
    source = dict(SIXS, **source)
    if surface:
        source["surface_orientation"] = surface
    projection = {"resolution": "0.01"}
    if offset:
        projection.update(offset)
    return dict(sample=PT111, wavelength=wavelength, frames=frames, input=source,
                projection=projection)


CASES = {
    "vertical": case([0.3, 20.0, 10.0, 15.0]),
    "vertical-omega_offset": case([1.2, 60.0, 35.0, 5.0], offset={"omega_offset": "7.5"}),
    "horizontal-mu_offset": case([4.0, 0.5, 25.0, 20.0], "horizontal", {"mu_offset": "-12"}),
    "horizontal-detrot0": case([-2.0, 110.0, 50.0, -10.0], "horizontal", wavelength=0.7,
                               centralpixel="100, 60", sdd="0.8", detrot="0"),
}


@pytest.fixture(scope="module")
def binoculars(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("binoculars")
    cases, out = tmp / "cases.json", tmp / "q.npz"
    cases.write_text(json.dumps(list(CASES.values())))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binoculars_qxqyqz.py")
    env = dict(GI_TYPELIB_PATH=os.path.join(ENV, "lib", "girepository-1.0"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PATH=os.environ["PATH"])
    run = subprocess.run([os.path.join(ENV, "bin", "python"), "-I", "-B", script, TREE,
                          str(cases), str(out)], cwd=tmp, env=env, capture_output=True,
                         text=True)
    sys.stderr.write(run.stderr)
    assert run.returncode == 0, run.stderr
    with np.load(out) as q:
        return {name: q[f"q{i}"] for i, name in enumerate(CASES)}


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def surface_frame(c):
    """binoculars' UB, whose columns are its Qx, Qy, Qz in the sample frame."""
    vertical = c["input"].get("surface_orientation", "vertical") == "vertical"
    ub = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]) if vertical else np.eye(3)
    offset = c["projection"].get("omega_offset" if vertical else "mu_offset")
    return ub if offset is None else ub @ rz(-math.radians(float(offset)))


def grid(q):
    """Per component (xmin, n): xmin a multiple of DX two bins below the
    data, n bins reaching two above; bin k is centred on xmin + k DX."""
    out = []
    for x in np.moveaxis(q, -1, 0):
        xmin = (np.floor(x.min() / DX) - 2) * DX
        n = int(np.ceil((x.max() - xmin) / DX)) + 3
        fr = (x - xmin) / DX + 0.5
        assert np.abs(fr - np.round(fr)).min() > MARGIN, "a pixel on a bin edge"
        out.append((xmin, n))
    return out


def detector(c):
    x, y = (int(v) for v in c["input"]["centralpixel"].split(","))
    det = hklrs.detector("ImXpadS140")
    return hklrs.sixs(det, (y, x), float(c["input"]["sdd"]), float(c["input"]["detrot"]))


@pytest.mark.parametrize("name", CASES)
def test_qsurface_is_binoculars_qxqyqz(binoculars, name):
    c = CASES[name]
    q_b = binoculars[name]
    geo = hklrs.Geometry.preset("ZAXIS")
    assert geo.axis_names == ["mu", "omega", "delta", "gamma"]
    det = detector(c)
    frame = surface_frame(c)
    wl = c["wavelength"]
    assert q_b.shape == (len(c["frames"]), 3) + det.shape

    for ang, q in zip(c["frames"], q_b):
        sample = np.einsum("ij,jkl->ikl", frame, q)
        np.testing.assert_allclose(np.stack(hklrs.q_map(geo, ang, wl, det)), sample,
                                   rtol=0, atol=Q_TOL, err_msg=f"Q_sample at {ang}")

    q_last = np.moveaxis(q_b, 1, -1)
    g3 = grid(q_last)
    m = fm.FrameMap(geo, det, [(DX, xmin - DX / 2) for xmin, _ in g3], "qsurface",
                    frame=frame.tolist())
    cube = fm.Cube()
    rng = np.random.default_rng(20261124)
    raws = [rng.integers(0, 1000, det.shape).astype(float) for _ in c["frames"]]
    for ang, raw in zip(c["frames"], raws):
        m.add(cube, ang, wl, raw)

    axes = cube.axes
    assert [a[0] for a in axes] == ["qx_surface", "qy_surface", "qz_surface"]
    keep = np.ones(det.shape, bool) if det.mask is None else ~det.mask
    bins = [np.rint((q_last[..., k] - xmin) / DX).astype(int)[:, keep] for k, (xmin, _) in
            enumerate(g3)]
    shape = tuple(n for _, n in g3)
    flat = np.ravel_multi_index(bins, shape).ravel()
    weights = np.stack([raw[keep] for raw in raws]).ravel()
    contributions = np.bincount(flat, minlength=math.prod(shape)).reshape(shape)
    photons = np.bincount(flat, weights, minlength=math.prod(shape)).reshape(shape)
    for (_, step, anchor, first, size), (xmin, n) in zip(axes, g3):
        assert (step, anchor) == (DX, xmin - DX / 2)
        assert 0 <= first and first + size <= n
    at = tuple(slice(a[3], a[3] + a[4]) for a in axes)
    assert int(cube.contributions.sum()) == int(keep.sum()) * len(raws) == contributions.sum()
    np.testing.assert_array_equal(cube.contributions, contributions[at])
    np.testing.assert_array_equal(cube.photons, photons[at])
