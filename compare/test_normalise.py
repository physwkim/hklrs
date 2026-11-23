"""hklrs.normalise against xrayutilities IntensityNormalizer, run live.

hklrs returns (counts, weight, reasons) per pixel; counts / weight is compared with xu's corrected
intensity where xu is correct, and each place xu is wrong is asserted by its upstream-defects ID."""
import functools
import re

import numpy as np
import pytest

import hklrs
from hklrs import frame_map as fm
from hklrs import normalise as nm

xu = pytest.importorskip("xrayutilities")

BEAM = [0.0, 1.0, 0.0]
RNG = np.random.default_rng(20260401)
TOL = 1e-12
BAD_FLAT, BAD_COUNT, DETECTOR = 8, 9, 7


def detector(shape):
    return hklrs.xu.Area("z+", "x+", (0.0, 0.0), shape, 0.5, (1e-3, 1e-3)).detector(BEAM)


def xu_run(raw, mon=None, time=None, absf=None, dark=None, flat=None, av_mon=None, ccd=False):
    """xu's corrected intensity of `raw` (frames first), or its exception."""
    n = raw.shape[0]
    data = np.zeros(n, dtype=[("det", "f8", raw.shape[1:]), ("mon", "f8"), ("t", "f8"), ("absf", "f8")])
    data["det"] = raw
    data["mon"] = 1.0 if mon is None else mon
    data["t"] = time if isinstance(time, np.ndarray) else 1.0
    data["absf"] = 1.0 if absf is None else absf
    if not ccd:  # xu takes a PSD's fields as 1D
        dark, flat = (None if a is None else np.ravel(a) for a in (dark, flat))
    kw = {}
    if mon is not None:
        kw["mon"] = "mon"
    if isinstance(time, np.ndarray):
        kw["time"] = "t"
    elif time is not None:
        kw["time"] = float(time)
    if absf is not None:
        kw["absfun"] = lambda d: d["absf"]
    if dark is not None:
        kw["darkfield"] = dark
    if flat is not None:
        kw["flatfield"] = flat
    if av_mon is not None:
        kw["av_mon"] = av_mon
    corr = xu.normalize.IntensityNormalizer("det", **kw)
    return np.asarray(corr(data, ccd=raw) if ccd else corr(data))


def hkl_run(raw, mon=None, time=None, absf=None, dark=None, flat=None, av_mon=None):
    """hklrs's (counts, weight, reasons) per frame, unbinned; `raw` is (frames, h, w)."""
    n, h, w = raw.shape
    ref = av_mon if av_mon is not None else (None if mon is None else float(np.mean(mon)))
    fields = nm.PixelFields((h, w), dark, flat)
    _, red = nm.Region((0, h), (0, w), (1, 1)).apply(detector((h, w)))
    norm = nm.Normaliser(fields, red)
    out = []
    for k in range(n):
        s = nm.FrameScalars(
            None if mon is None else mon[k], ref,
            time[k] if isinstance(time, np.ndarray) else time,
            None if absf is None else 1.0 / absf[k])
        out.append(norm.normalise(raw[k], s))
    return [np.stack(a) for a in zip(*out)]


def counts(*shape):
    return RNG.integers(50, 5000, size=shape).astype(float)


N, NCH, N1, N2 = 6, 9, 4, 5
MON = RNG.uniform(800, 1200, N)
T = RNG.uniform(0.5, 2.0, N)
ABS = np.array([1, 1, 3.2, 3.2, 10.5, 1.0])
DARK = RNG.uniform(0.0, 0.3, (1, NCH))
FLAT = RNG.uniform(0.7, 1.3, (1, NCH))
POINT, PSD, CCD = counts(N), counts(N, NCH), counts(N, N1, N2)

COMPARE = {
    "point_mon_time_abs": (POINT, dict(mon=MON, time=T, absf=ABS)),
    "point_mon": (POINT, dict(mon=MON)),
    "point_time_float": (POINT, dict(time=2.5)),
    "point_av_mon": (POINT, dict(mon=MON, av_mon=1000.0)),
    "point_abs_only": (POINT, dict(absf=ABS)),
    "psd_mon_time_abs": (PSD, dict(mon=MON, time=T, absf=ABS)),
    "psd_dark_flat_mon_time_abs": (PSD, dict(mon=MON, time=T, absf=ABS, dark=DARK, flat=FLAT)),
    "psd_dark_mon_time": (PSD, dict(mon=MON, time=T, dark=DARK)),
    "psd_flat_mon": (PSD, dict(mon=MON, flat=FLAT)),
    "ccd_mon_time_abs": (CCD, dict(mon=MON, time=T, absf=ABS)),
}


def as_frames(raw):
    """(frames, h, w): a point is 1x1, a PSD row 1 x nch."""
    return raw.reshape(raw.shape[0], 1, -1) if raw.ndim < 3 else raw


def value(c, w, r):
    assert not r.any() and np.isfinite(c).all() and np.isfinite(w).all()
    return c / w


@pytest.mark.parametrize("name", sorted(COMPARE))
def test_counts_over_weight_is_xus_intensity(name):
    raw, kw = COMPARE[name]
    want = xu_run(raw, ccd=raw.ndim == 3, **kw)
    got = value(*hkl_run(as_frames(raw), **kw))
    np.testing.assert_allclose(got.reshape(want.shape), want, rtol=TOL, atol=0)


def test_xu9_flat_and_dark_apply_without_a_monitor():
    psd = PSD[:1]
    np.testing.assert_array_equal(xu_run(psd, flat=FLAT[0]), psd)  # xu skips the flat field
    got = value(*hkl_run(as_frames(psd), flat=FLAT))
    np.testing.assert_allclose(got[0, 0], psd[0] / (FLAT[0] / FLAT.mean()), rtol=TOL)
    # a count time only: xu applies it and still skips dark and flat
    np.testing.assert_allclose(xu_run(psd, time=2.5, dark=DARK[0], flat=FLAT[0]), psd / 2.5, rtol=TOL)
    got = value(*hkl_run(as_frames(psd), time=2.5, dark=DARK, flat=FLAT))
    want = (psd[0] - DARK[0] * 2.5) / (2.5 * FLAT[0] / FLAT.mean())
    np.testing.assert_allclose(got[0, 0], want, rtol=TOL)


def test_xu10_ccd_without_a_monitor_normalises():
    with pytest.raises(IndexError):
        xu_run(CCD, ccd=True)
    c, w, r = hkl_run(CCD)
    assert not r.any()
    np.testing.assert_array_equal(c, CCD)
    np.testing.assert_array_equal(w, 1.0)


def test_xu11_any_callable_is_an_absorber():
    # xu: a partial has no __code__ (AttributeError), a bound method counts self (TypeError)
    ratio = functools.partial(lambda a, d: a * d["x"], 2.0)
    with pytest.raises(AttributeError):
        xu.normalize.IntensityNormalizer("det", absfun=ratio)

    class Foils:
        def at(self, d):
            return d["x"]

    with pytest.raises(TypeError):
        xu.normalize.IntensityNormalizer("det", absfun=Foils().at)
    d = {"x": 0.25}
    for f, t in [(Foils().at, 0.25), (functools.partial(lambda a, d: a * d["x"], 2.0), 0.5),
                 (lambda d: d["x"], 0.25), (np.sqrt, None)]:
        arg = d if t is not None else 0.0625
        s = nm.FrameScalars().absorber(f, arg)
        _, red = nm.Region((0, 1), (0, 1), (1, 1)).apply(detector((1, 1)))
        c, w, _ = nm.Normaliser(nm.PixelFields((1, 1)), red).normalise(np.ones((1, 1)), s)
        assert w[0, 0] == pytest.approx(t if t is not None else 0.25, rel=1e-15)
    with pytest.raises(ValueError):
        nm.FrameScalars().absorber(lambda d: 0.0, d)


def test_xu21_ccd_applies_dark_and_flat():
    dark = RNG.uniform(0.0, 0.3, (N1, N2))
    flat = RNG.uniform(0.7, 1.3, (N1, N2))
    kw = dict(mon=MON, time=T, absf=ABS)
    plain = xu_run(CCD, ccd=True, **kw)
    # xu accepts the fields and ignores them
    np.testing.assert_allclose(xu_run(CCD, dark=dark, flat=flat, ccd=True, **kw), plain, rtol=TOL)
    got = value(*hkl_run(CCD, dark=dark, flat=flat, **kw))
    assert not np.allclose(got, plain)
    t = T[:, None, None]
    want = ((CCD - dark * t) / (flat / flat.mean())) * plain / CCD
    np.testing.assert_allclose(got, want, rtol=TOL)


def test_xu22_dark_subtraction_is_signed():
    dark = np.full((1, NCH), 3000.0)
    kw = dict(mon=MON, time=T)
    sub = PSD - dark * T[:, None]
    assert (sub < 0).any()
    assert (xu_run(PSD, dark=dark[0], **kw)[sub < 0] == 0).all()  # xu clamps
    c, w, r = hkl_run(as_frames(PSD), dark=dark, **kw)
    assert not r.any()
    np.testing.assert_allclose(c[:, 0], sub, rtol=TOL)
    assert (c[:, 0][sub < 0] < 0).all()


def test_xu23_a_zero_monitor_is_an_error():
    point, mon = POINT[:3], np.array([1000.0, 0.0, 900.0])
    out = xu_run(point, mon, np.ones(3))
    assert out[1] == point[1]  # xu: the factor is inf, set to 1, raw counts returned
    with pytest.raises(ValueError, match="monitor"):
        nm.FrameScalars(0.0, 700.0)
    with pytest.raises(ValueError, match="monitor"):
        nm.FrameScalars(float("nan"), 700.0)
    with pytest.raises(ValueError):
        nm.mean_monitor([1.0, 0.0])
    with pytest.raises(ValueError, match="together"):
        nm.FrameScalars(monitor=1.0)


def test_xu24_a_dead_flat_pixel_is_masked():
    flat = np.concatenate([FLAT[0, :-1], [1e-6]])
    out = xu_run(PSD, MON, T, None, None, flat)
    assert np.isfinite(out[:, -1]).all()  # xu normalises with a gain of 1
    c, w, r = hkl_run(as_frames(PSD), mon=MON, time=T, flat=flat[None])
    assert (r[:, 0, -1] == BAD_FLAT).all() and not r[:, 0, :-1].any()
    assert np.isnan(c[:, 0, -1]).all() and np.isnan(w[:, 0, -1]).all()
    # xu averages the gains with the dead pixel counted (and negatives); hklrs over the usable ones
    good = flat[:-1]
    np.testing.assert_allclose(
        value(c[..., :-1], w[..., :-1], r[..., :-1])[:, 0],
        PSD[:, :-1] * (np.mean(MON) / (MON * T))[:, None] / (good / good.mean()), rtol=TOL)


def test_shapes_dtypes_and_nan_where_masked():
    h, w = 4, 6
    det = detector((h, w)).with_mask(np.eye(h, w, dtype=bool))
    _, red = nm.Region((0, h), (0, w), (1, 1)).apply(det)
    raw = np.ones((h, w))
    raw[0, 3] = np.inf
    c, wt, r = nm.Normaliser(nm.PixelFields((h, w)), red).normalise(raw, nm.FrameScalars())
    assert (c.dtype, wt.dtype, r.dtype) == (np.float64, np.float64, np.uint8)
    assert c.shape == wt.shape == r.shape == (h, w) == red.shape
    assert (r[np.eye(h, w, dtype=bool)] == DETECTOR).all()  # the mask wins over the count
    assert r[0, 3] == BAD_COUNT
    assert np.isnan(c[r > 0]).all() and np.isnan(wt[r > 0]).all()
    assert not np.isnan(c[r == 0]).any()


def test_precedence_with_geometric_weights():
    h, w = 1, 5
    det = detector((h, w)).with_mask(np.array([[True, False, False, False, False]]))
    binned, red = nm.Region((0, 1), (0, 5), (1, 1)).apply(det)
    geo = hklrs.Geometry.preset("E4CV")
    sa, sr = hklrs.solid_angle(geo, [0.0] * 4, detector((1, 5)))
    sr = sr.copy()
    sr[0, [0, 1]] = hklrs.Masked.EdgeOn
    flat = np.array([[1.0, 1.0, 0.0, 1.0, 1.0]])
    raw = np.array([[1.0, 1.0, 1.0, np.nan, 1.0]])
    c, wt, r = nm.Normaliser(nm.PixelFields((1, 5), flat=flat), red).normalise(
        raw, nm.FrameScalars(), [(sa, sr)])
    assert list(r[0]) == [DETECTOR, int(hklrs.Masked.EdgeOn), BAD_FLAT, BAD_COUNT, 0]
    # the weight is the solid angle times the flat gain over its mean
    np.testing.assert_allclose(wt[0, 4], sa[0, 4] / flat[0, [0, 1, 3, 4]].mean(), rtol=TOL)


def test_xu13_a_bin_holds_the_block_it_is_placed_at():
    raw_det = detector((7, 11))
    g = np.array([0.3, -2.1, 5.7])
    pos = lambda s, f: 1000.0 + g @ np.array(raw_det.pixel(s, f))
    raw = np.array([[pos(s, f) for f in range(11)] for s in range(7)])
    for slow, fast, bn in [((0, 7), (0, 11), (3, 3)), ((0, 6), (1, 10), (3, 3)),
                           ((1, 7), (4, 11), (2, 3)), ((2, 7), (5, 11), (5, 4))]:
        binned, red = nm.Region(slow, fast, bn).apply(raw_det)
        c, wt, r = nm.Normaliser(nm.PixelFields((7, 11)), red).normalise(raw, nm.FrameScalars())
        assert not r.any() and c.shape == binned.shape
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                at = 1000.0 + g @ np.array(binned.pixel(i, j))
                assert c[i, j] / (bn[0] * bn[1]) == pytest.approx(at, rel=1e-12), (bn, i, j)


def test_dark_needs_a_count_time_and_shapes_must_agree():
    _, red = nm.Region((0, 1), (0, 3), (1, 1)).apply(detector((1, 3)))
    n = nm.Normaliser(nm.PixelFields((1, 3), dark_rate=np.zeros((1, 3))), red)
    with pytest.raises(ValueError, match="count time"):
        n.normalise(np.ones((1, 3)), nm.FrameScalars())
    with pytest.raises(ValueError):
        nm.PixelFields((1, 3), flat=np.ones((1, 2)))
    n = nm.Normaliser(nm.PixelFields((1, 3)), red)
    for raw in [np.ones((1, 2)), np.ones((3, 1))]:
        msg = re.escape(f"raw frame shape {raw.shape} does not match (1, 3)")
        with pytest.raises(ValueError, match=msg):
            n.normalise(raw, nm.FrameScalars())
    with pytest.raises(ValueError, match="reason code"):
        n.normalise(np.ones((1, 3)), nm.FrameScalars(),
                    [(np.ones((1, 3)), np.full((1, 3), 99, np.uint8))])


def test_fields_of_another_raw_shape_are_refused_at_setup():
    """Where fields meet a region (Normaliser, FrameMap), not at the first frame."""
    det = detector((1, 3))
    _, red = nm.Region((0, 1), (0, 3), (1, 1)).apply(det)
    for shape in [(2, 3), (3, 1)]:
        msg = re.escape(f"pixel fields shape {shape} does not match (1, 3)")
        with pytest.raises(ValueError, match=msg):
            nm.Normaliser(nm.PixelFields(shape), red)
        with pytest.raises(ValueError, match=msg):
            fm.FrameMap(hklrs.Geometry.preset("E4CV"), det, [(0.01, 0.0)] * 3,
                        fields=nm.PixelFields(shape))


def test_a_transposed_raw_image_is_refused():
    """A frame, dark field or flat field is taken in its own numpy shape, so
    the transposed shape of the raw (2, 3) is refused, not read row-major."""
    det = detector((2, 3))
    _, red = nm.Region((0, 2), (0, 3), (1, 1)).apply(det)
    t = np.arange(6.0).reshape(3, 2)
    msg = lambda what: re.escape(f"{what} shape (3, 2) does not match (2, 3)")  # noqa: E731
    with pytest.raises(ValueError, match=msg("raw frame")):
        nm.Normaliser(nm.PixelFields((2, 3)), red).normalise(t, nm.FrameScalars())
    m = fm.FrameMap(hklrs.Geometry.preset("E4CV"), det, [(0.01, 0.0)] * 3)
    cube = fm.Cube()
    m.add(cube, [10.0, 0.0, 0.0, 20.0], 1.54, t.T.copy())
    before = cube.photons
    with pytest.raises(ValueError, match=msg("raw frame")):
        m.add(cube, [10.0, 0.0, 0.0, 20.0], 1.54, t)
    np.testing.assert_array_equal(cube.photons, before)
    with pytest.raises(ValueError, match=msg("dark field")):
        nm.PixelFields((2, 3), dark_rate=t)
    with pytest.raises(ValueError, match=msg("flat field")):
        nm.PixelFields((2, 3), flat=t + 1.0)
