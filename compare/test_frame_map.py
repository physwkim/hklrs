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


def test_angle_axes_are_degrees():
    """An "angles" cube reads its axes' kind: binnings go in and axes come
    out in degrees, and its tth axis spans the pixels' 2θ from |Q|."""
    geo = hklrs.Geometry.preset("E4CV")
    det = hklrs_area(params(AREA_BASE, {}))
    ang = angles("E4CV", 1, "frame_map_angles")[0]
    q = np.linalg.norm(np.stack(hklrs.q_map(geo, ang, WL, det), axis=-1), axis=-1)
    tth = np.degrees(2 * np.arcsin(q * WL / (4 * np.pi)))
    step, anchor = 0.5, 0.25
    m = fm.FrameMap(geo, det, [(step, anchor)] * 3, "angles")
    cube = fm.Cube()
    m.add(cube, ang, WL, np.ones(tth.shape))
    axes = cube.axes
    assert [a[0] for a in axes] == ["delta_lab", "gamma_lab", "tth"]
    for _, s, a, _, _ in axes:
        assert (s, a) == (pytest.approx(step, rel=1e-15), pytest.approx(anchor, rel=1e-15))
    _, s, a, first, size = axes[2]
    assert a + first * s <= tth.min() + 1e-9
    assert tth.max() - 1e-9 < a + (first + size) * s
    assert int(cube.contributions.sum()) == tth.size


def custom_setup():
    geo = hklrs.Geometry.preset("E4CV")
    det = hklrs_area(params(AREA_BASE, {}))
    ang = angles("E4CV", 1, "frame_map_custom")[0]
    return geo, det, ang


def test_custom_identity_is_its_base():
    """A custom projection handing qsample's planes back as they came bins
    every pixel as qsample does, to the bit."""
    geo, det, ang = custom_setup()
    q = np.stack(hklrs.q_map(geo, ang, WL, det), axis=-1)
    g3 = grid(q[None])
    binnings = [(DX, xmin - DX / 2) for xmin, _ in g3]
    raw = np.arange(q[..., 0].size, dtype=float).reshape(q.shape[:2])
    axes = [("qx", "linear"), ("qy", "linear"), ("qz", "linear")]
    p = fm.Custom(lambda planes, context: planes, axes)
    cubes = [fm.Cube(), fm.Cube()]
    for cube, projection in zip(cubes, ["qsample", p]):
        fm.FrameMap(geo, det, binnings, projection).add(cube, ang, WL, raw)
    assert cubes[0].axes == cubes[1].axes
    np.testing.assert_array_equal(cubes[0].photons, cubes[1].photons)
    np.testing.assert_array_equal(cubes[0].contributions, cubes[1].contributions)


def test_custom_angles_are_degrees_both_ways():
    """An "angles" base reaches the function in degrees, the function sees
    the frame's axes and wavelength, and an axis declared an angle is
    returned, binned and read back in degrees."""
    geo, det, ang = custom_setup()
    q = np.linalg.norm(np.stack(hklrs.q_map(geo, ang, WL, det), axis=-1), axis=-1)
    tth = np.degrees(2 * np.arcsin(q * WL / (4 * np.pi)))
    seen = []

    def two_theta(planes, context):
        seen.append((planes, context.axes, context.wavelength))
        return (planes[2],)

    p = fm.Custom(two_theta, [("two_theta", "angle")], base="angles")
    step, anchor = 0.5, 0.25
    cube = fm.Cube()
    fm.FrameMap(geo, det, [(step, anchor)], p).add(cube, ang, WL, np.ones(tth.shape))
    (planes, axes, wl), = seen
    assert len(planes) == 3 and all(a.shape == tth.shape and a.dtype == np.float64 for a in planes)
    np.testing.assert_allclose(planes[2], tth, rtol=0, atol=1e-9)
    np.testing.assert_allclose(axes, ang, rtol=1e-15, atol=1e-13)
    assert wl == WL
    (name, s, a, first, size), = cube.axes
    assert name == "two_theta"
    assert (s, a) == (pytest.approx(step, rel=1e-15), pytest.approx(anchor, rel=1e-15))
    assert a + first * s <= tth.min() + 1e-9 and tth.max() - 1e-9 < a + (first + size) * s
    assert int(cube.contributions.sum()) == tth.size


class Refused(Exception):
    pass


def test_custom_function_exception_is_raised_as_it_was():
    geo, det, ang = custom_setup()
    raised = Refused("no")

    def refuse(planes, context):
        raise raised

    m = fm.FrameMap(geo, det, [(DX, 0.0)], fm.Custom(refuse, [("x", "linear")]))
    cube = fm.Cube()
    with pytest.raises(Refused) as e:
        m.add(cube, ang, WL, np.ones(det.shape))
    assert e.value is raised
    assert cube.axes == [] and int(cube.contributions.sum()) == 0


@pytest.mark.parametrize(
    "returns, error, match",
    [
        (lambda p: (p[0],), ValueError, r"^expected 2 dimensions"),
        (lambda p: (p[0][:-1], p[1][:-1]), ValueError, r"^custom projection plane shape"),
        (lambda p: (p[0], p[1].astype(np.float32)), TypeError, r"plane 1 is a 2-D float32 array"),
        (lambda p: (p[0], p[1][0]), TypeError, r"plane 1 is a 1-D float64 array"),
        (lambda p: p[0], TypeError, r"plane 0 is a 1-D float64 array"),
        (lambda p: (p[0], list(p[1])), TypeError, r"plane 1 is <class 'list'>"),
        (lambda p: (p[0], np.where(p[1] > p[1].min(), p[1], np.nan)), ValueError, r"has no bin"),
    ],
    ids=["count", "shape", "float32", "1-D", "bare", "list", "nan"],
)
def test_custom_planes_are_held_to_the_axes(returns, error, match):
    geo, det, ang = custom_setup()
    p = fm.Custom(lambda planes, context: returns(planes), [("a", "linear"), ("b", "linear")])
    m = fm.FrameMap(geo, det, [(DX, 0.0)] * 2, p)
    cube = fm.Cube()
    with pytest.raises(error, match=match):
        m.add(cube, ang, WL, np.ones(det.shape))
    assert cube.axes == []


def test_custom_setup_is_refused():
    geo, det, _ = custom_setup()
    f = lambda planes, context: planes  # noqa: E731
    with pytest.raises(ValueError, match="at least one coordinate"):
        fm.Custom(f, [])
    with pytest.raises(ValueError, match=r'"a" appears twice'):
        fm.Custom(f, [("a", "linear"), ("a", "angle")])
    with pytest.raises(ValueError, match=r'kind "degrees"'):
        fm.Custom(f, [("a", "degrees")])
    with pytest.raises(TypeError, match="not callable"):
        fm.Custom(None, [("a", "linear")])
    with pytest.raises(ValueError, match="sample= with"):
        fm.Custom(f, [("a", "linear")], base="hkl")
    p = fm.Custom(f, [("a", "linear"), ("b", "linear")])
    with pytest.raises(ValueError, match=r"^expected 2 dimensions, one per projection axis, got 3"):
        fm.FrameMap(geo, det, [(DX, 0.0)] * 3, p)
    with pytest.raises(ValueError, match="takes its base's"):
        fm.FrameMap(geo, det, [(DX, 0.0)] * 2, p, normal=[0, 0, 1])
