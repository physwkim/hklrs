"""hklrs.frame_map against xrayutilities Ang2Q.area plus Gridder3D.

A few frames of a sample-axis step go through FrameMap (qsample) into one
Cube and, on the xu side, through qc.area and a Gridder3D that keeps its
data across calls. Counts are integers, so sums are exact on both sides;
masked pixels are NaN for xu, whose gridder skips them. Each frame's count
time is 2 s and nothing else scales it, so a pixel's weight is 2:
contributions are xu's _gnorm, photons its _gdata and intensity half its
normalised data. xu's grid of n nodes from
xmin to xmax is the lattice of step (xmax - xmin) / (n - 1) anchored half a
step below xmin, so global bin k is node k. Binned regions are not compared:
xu places a Nav pixel on its block's first pixel (test_xu, XU-13).
Run with the py2 interpreter: python -I -m pytest compare
"""
import numpy as np
import pytest

import hklrs
from hklrs import frame_map as fm
from hklrs import normalise as hn
from test_xu import (  # noqa: F401  (fixtures: name, kappa_axis_in_xu_frame)
    AREA_BASE,
    Q_TOL,
    WL,
    XU_CIRCLES,
    angles,
    hklrs_area,
    kappa_axis_in_xu_frame,
    name,
    params,
    xu_area,
    xu_vectors,
)

xu = pytest.importorskip("xrayutilities")

DX = 2.0**-10  # exact, so xu's (xmax - xmin) / (n - 1) is DX to the bit
FRAMES = 5
STEP = 0.02  # degrees on the innermost sample axis per frame
COUNT_TIME = 2.0  # a power of two, so halving xu's mean is exact
ROIS = {"whole": None, "roi": (1, 8, 3, 12)}


def frame_angles(name):
    base = angles(name, 1, "frame_map")[0]
    inner = len(XU_CIRCLES[name][0]) - 1
    out = []
    for i in range(FRAMES):
        ang = list(base)
        ang[inner] += STEP * i
        out.append(ang)
    return out


def grid(q):
    """Per component (xmin, n): xmin a multiple of DX two nodes below the
    data, n nodes reaching two above, no point within 1e-6 bin of an edge."""
    out = []
    for c in range(3):
        x = q[..., c]
        xmin = (np.floor(x.min() / DX) - 2) * DX
        n = int(np.ceil((x.max() - xmin) / DX)) + 3
        fr = (x - xmin) / DX + 0.5
        assert np.all(np.abs(fr - np.round(fr)) > 1e-6), "pixel on a bin edge: floor and rint differ"
        out.append((xmin, n))
    return out


@pytest.mark.parametrize("roi", sorted(ROIS))
def test_frames_into_a_cube_are_xu_area_gridded(name, roi):
    r = ROIS[roi]
    base = params(AREA_BASE, {})
    shape = tuple(base["nch"])
    rng = np.random.default_rng(20261123)
    raws = [rng.integers(0, 1000, shape).astype(float) for _ in range(FRAMES)]
    mask = rng.random(shape) < 0.1
    cut = (slice(r[0], r[1]), slice(r[2], r[3])) if r else (slice(None), slice(None))

    qc = xu_area(name, params(base, dict(roi=r)) if r else base)
    det = hklrs_area(base).with_mask(mask)
    geo = hklrs.Geometry.preset(name)
    frames = frame_angles(name)
    qs = []
    for ang in frames:
        q_x = xu_vectors(qc.area(*ang))
        q_h = np.stack(hklrs.q_map(geo, ang, WL, det), axis=-1)[cut]
        np.testing.assert_allclose(q_h, q_x, rtol=0, atol=Q_TOL, err_msg=f"Q at {ang}")
        qs.append(q_x)
    g3 = grid(np.stack(qs))

    region = dict(region=hn.Region(r[:2], r[2:], (1, 1))) if r else {}
    m = fm.FrameMap(geo, det, [(DX, xmin - DX / 2) for xmin, _ in g3], "qsample", **region)
    scalars = hn.FrameScalars(count_time=COUNT_TIME)
    cube = fm.Cube()
    g = xu.Gridder3D(*(n for _, n in g3))
    g.KeepData(True)
    g.dataRange(*(v for xmin, n in g3 for v in (xmin, xmin + (n - 1) * DX)), fixed=True)
    for ang, q_x, raw in zip(frames, qs, raws):
        m.add(cube, ang, WL, raw, scalars)
        data = np.where(mask, np.nan, raw)[cut]
        g(q_x[..., 0], q_x[..., 1], q_x[..., 2], data)

    axes = cube.axes
    assert [a[0] for a in axes] == ["qx", "qy", "qz"]
    for (_, step, anchor, first, size), (xmin, n) in zip(axes, g3):
        assert (step, anchor) == (DX, xmin - DX / 2)
        assert 0 <= first and first + size <= n
    at = tuple(slice(a[3], a[3] + a[4]) for a in axes)

    hit = cube.contributions > 0
    unmasked = int((~mask[cut]).sum()) * FRAMES
    assert int(g._gnorm.sum()) == unmasked == int(cube.contributions.sum())
    np.testing.assert_array_equal(cube.contributions, g._gnorm[at])
    assert np.all(np.isfinite(cube.photons)) and np.all(np.isfinite(g._gdata[at]))
    np.testing.assert_array_equal(cube.photons, g._gdata[at])
    np.testing.assert_array_equal(cube.weight, COUNT_TIME * cube.contributions)
    assert np.all(np.isnan(cube.intensity[~hit]))
    assert np.all(np.isfinite(g.data[at][hit]))
    np.testing.assert_array_equal(cube.intensity[hit], g.data[at][hit] / COUNT_TIME)


def test_bad_frame_names_frame():
    geo = hklrs.Geometry.preset("E4CV")
    det = hklrs_area(params(AREA_BASE, {}))
    improper = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    with pytest.raises(ValueError, match=r"^frame is not a proper rotation"):
        fm.FrameMap(geo, det, [(DX, 0.0)] * 3, "qsurface", frame=improper)
