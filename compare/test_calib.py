"""hklrs.calib against xrayutilities' calibration, run live.

Data is generated with xu's own Ang2Q (the pixel or channel whose |Q| is smallest, refined by a local
plane fit), so xu's model and ours see the same truth. Each place xu is wrong is asserted by its
upstream-defects ID; where xu is right, hklrs agrees with it. The hkl frames' exact pixels are read from
golden/xu_calib_hkl.json (generating them takes xu minutes), the images xu is run on are rendered here."""
import json
import os

import numpy as np
import pytest

import hklrs
from hklrs import calib as cal

xu = pytest.importorskip("xrayutilities")
from xrayutilities.analysis import sample_align as sa  # noqa: E402

xu.config.VERBOSITY = 0
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden")
N, PW, D, WL = 800, 2e-3, 1.0, 1.54
TRUTH = dict(cch1=401.3, cch2=396.7, detrot=2.0)
FREE, FIXED = cal.Param, cal.Param.fixed


def beampix(conv, *angles):
    """The pixel (fractional) where |Q| is zero: xu's own location of the direct beam."""
    q = np.array(conv.area(*angles, wl=WL, deg=True))
    i, j = np.unravel_index(np.linalg.norm(q, axis=0).argmin(), q.shape[1:])
    ii, jj = np.mgrid[-3:4, -3:4]
    A = np.c_[np.ones(49), ii.ravel(), jj.ravel()]
    sl = (slice(i - 3, i + 4), slice(j - 3, j + 4))
    cf = [np.linalg.lstsq(A, q[c][sl].ravel(), rcond=None)[0] for c in range(3)]
    G = np.array([[cf[c][1], cf[c][2]] for c in range(3)])
    d = np.linalg.lstsq(G, -np.array([cf[c][0] for c in range(3)]), rcond=None)[0]
    return i + d[0], j + d[1]


def gauss(shape, c, amp=2000.0, sig=2.0, bg=0.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return bg + amp * np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * sig ** 2))


# --- area detector, primary beam -------------------------------------------------------------------

@pytest.fixture(scope="module")
def area():
    conv = xu.QConversion([], ["z+", "y-"], [1, 0, 0])
    conv.init_area("z+", "y+", cch1=TRUTH["cch1"], cch2=TRUTH["cch2"], Nch1=N, Nch2=N, pwidth1=PW,
                   pwidth2=PW, distance=D, detrot=TRUTH["detrot"], tiltazimuth=0, tilt=0)
    g = np.linspace(-15, 15, 5)
    a1, a2 = [a.ravel() for a in np.meshgrid(g, g)]
    pix = [beampix(conv, x, y) for x, y in zip(a1, a2)]
    return dict(a1=a1, a2=a2, pix=pix, imgs=[gauss((N, N), p) for p in pix])


def geometry():
    return cal.xu_geometry([("a1", "z+"), ("a2", "y-")], beam=[1, 0, 0])


def area_frames(area, pix=None, sigma=0.01):
    pix = area["pix"] if pix is None else pix
    return [cal.CalFrame([x, y], cal.Peak(p, [sigma, sigma]), cal.FrameKind.primary_beam(), i)
            for i, (x, y, p) in enumerate(zip(area["a1"], area["a2"], pix))]


def area_start(detrot):
    return cal.AreaStart(FREE(390.0), FREE(405.0), FREE(0.0021), FREE(0.0019), FIXED(D), FIXED(WL),
                         detrot=detrot)


def hk_area(area, detrot, pix=None):
    model = cal.AreaModel(geometry(), (N, N), directions=("z+", "y+"))
    return cal.calibrate_area(model, area_frames(area, pix), [area_start(detrot)])


def xu_area(area, monkeypatch, pix, detrot_start, fix_detrot, step_beta):
    """xu's area_detector_calib on `pix` (its peak finder replaced by the exact pixels); with `step_beta`
    False the odr_fit call loses its step_beta argument (XU-6 isolated)."""
    by_id = {id(im): p for im, p in zip(area["imgs"], pix)}
    monkeypatch.setattr(sa, "_peak_position", lambda img, nw, plot=False: tuple(by_id[id(img)]))
    if not step_beta:
        odr = sa.odr_fit
        monkeypatch.setattr(sa, "odr_fit", lambda *a, **k: odr(*a, **{x: v for x, v in k.items() if x != "step_beta"}))
    res, eps = sa.area_detector_calib(
        area["a1"], area["a2"], area["imgs"], ["z+", "y-"], "x+", plot=False, wl=WL,
        start=(None, None, D, 0, 0, detrot_start, 0), fix=(False, False, True, True, True, fix_detrot, True),
        nwindow=30)
    return dict(zip("cch1 cch2 pw1 pw2 dist tiltaz tilt detrot off".split(), res)), eps


def test_area_recovers_the_truth(area):
    fit = hk_area(area, FIXED(2.0))
    p = fit.params
    for k, want in (("cch1", 401.3), ("cch2", 396.7), ("pwidth1", PW), ("pwidth2", PW)):
        assert abs(p[k] - want) <= want * 1e-4, (k, p[k], want)
    assert fit.rejected == [] and fit.stop in ("gradient", "cost", "param")
    assert fit.directions == ["z+", "y+"] and set(fit.free_ids) == {"cch1", "cch2", "pwidth1", "pwidth2"}
    assert fit.cov.shape == (4, 4) and fit.dof == 2 * 25 - 4 and set(fit.sigma) == set(fit.free_ids)


def test_xu1_xu_misses_the_truth_by_its_circle_order_hklrs_does_not(area, monkeypatch):
    # xu on the exact pixels, step_beta removed so XU-6 cannot be the cause: pwidth 2.05e-3 / 1.95e-3
    xr, _ = xu_area(area, monkeypatch, area["pix"], 2.0, True, step_beta=False)
    assert abs(xr["pw1"] - PW) > 4e-5 and abs(xr["pw2"] - PW) > 4e-5
    p = hk_area(area, FIXED(2.0)).params
    assert abs(p["pwidth1"] - PW) * 100 < abs(xr["pw1"] - PW)
    assert abs(p["pwidth2"] - PW) * 100 < abs(xr["pw2"] - PW)


def test_xu6_stock_xu_stops_at_its_start_values_and_hklrs_converges(area, monkeypatch):
    # stock xu (step_beta entry 2 is fatal for odrpack): one iteration, no error, pwidth still near its start
    xr, eps = xu_area(area, monkeypatch, area["pix"], 2.0, True, step_beta=True)
    assert abs(xr["pw1"] - 1.96e-3) < 1e-5 and eps > 1e-4
    # detrot free from 0 with step_beta removed: xu runs off (odrpack info 12), hklrs finds the 2 degrees
    xr, _ = xu_area(area, monkeypatch, area["pix"], 0.0, False, step_beta=False)
    assert abs(xr["detrot"] - 2.0) > 1.0
    fit = hk_area(area, FREE(0.0))
    assert abs(fit.params["detrot"] - 2.0) < 1e-4 and "detrot" in fit.sigma


def test_xu20_background_pulls_xu_peak_to_the_image_centre_hklrs_finds_it():
    peak = (530.0, 540.0)
    for bg in (0.1, 1.0, 5.0):
        img = gauss((N, N), peak, bg=bg)
        xu_pos = np.array(sa._peak_position(img, 30))
        assert np.hypot(*(xu_pos - peak)) > 100, bg
        got = cal.find_peak(img)
        assert np.hypot(*(np.array(got.pos) - peak)) < 1e-3, (bg, got)
    # no background: xu and hklrs agree
    img = gauss((N, N), peak)
    assert np.allclose(sa._peak_position(img, 30), cal.find_peak(img).pos, atol=0.05)


def test_find_peak_rejections_are_typed_and_masks_validated():
    with pytest.raises(cal.PeakRejected) as e:
        cal.find_peak(np.zeros((60, 60)))
    assert e.value.reason == "no_signal" and isinstance(e.value, cal.PeakFailure)
    img = gauss((60, 60), (30.0, 30.0))
    with pytest.raises(cal.PeakRejected) as e:
        cal.find_peak(img, saturation=1000.0)
    assert e.value.reason == "saturated" and img[e.value.slow, e.value.fast] >= 1000.0  # a saturated pixel, named
    with pytest.raises(cal.PeakRejected) as e:
        cal.find_peak(gauss((60, 60), (1.0, 30.0)))
    assert e.value.reason == "touches_edge"
    with pytest.raises(ValueError):
        cal.find_peak(img, mask=np.zeros((5, 5), bool))
    masked = np.zeros(img.shape, bool)
    masked[30, 30] = True
    assert abs(cal.find_peak(img, mask=masked).pos[0] - 30.0) < 0.1


# --- linear detector ---------------------------------------------------------------------------------

LN, LCCH, LPW = 1000, 300.3, 2e-4
ANGLES = np.arange(-5.0, 3.01, 0.5)


def channel(conv, ang):
    q = np.array(conv.linear(ang, wl=WL, deg=True))
    i = int(np.linalg.norm(q, axis=0).argmin())
    k = np.arange(i - 3, i + 4)
    A = np.c_[np.ones(7), k - i]
    cf = [np.linalg.lstsq(A, q[c][k], rcond=None)[0] for c in range(3)]
    g, g0 = np.array([c[1] for c in cf]), np.array([c[0] for c in cf])
    return float(i + (g @ -g0) / (g @ g))


@pytest.fixture(scope="module", params=[0.0, 3.0], ids=["tilt0", "tilt3"])
def linear(request):
    tilt = request.param
    conv = xu.QConversion([], ["x+"], [0, 1, 0])
    conv.init_linear("z+", LCCH, LN, distance=D, pixelwidth=LPW, tilt=tilt)
    pos = np.array([channel(conv, a) for a in ANGLES])
    rng = np.random.default_rng(20260102)
    ch = np.arange(LN)
    spectra = np.array([rng.poisson(2 + 3000 * np.exp(-(ch - p) ** 2 / (2 * 2.5 ** 2))).astype(float) for p in pos])
    sigma = 2.5 / np.sqrt(spectra.sum(axis=1))
    return dict(tilt=tilt, pos=pos, spectra=spectra, sigma=sigma)


def lin_model():
    return cal.LinearModel(cal.xu_geometry([("a", "x+")], beam=[0, 1, 0]), LN, direction="z+")


def lin_start(tilt):
    return cal.LinearStart(FREE(290.0), FREE(2.1e-4), FIXED(D), FIXED(WL), tilt=tilt)


def lin_frames(angles, pos, sigma):
    return [cal.CalFrame([a], cal.Peak.from_row(p, s), cal.FrameKind.primary_beam(), i)
            for i, (a, p, s) in enumerate(zip(angles, pos, sigma))]


def test_linear_matches_xu_psd_chdeg_and_recovers_the_tilt(linear):
    fr = lin_frames(ANGLES, linear["pos"], linear["sigma"])
    # tilt held at 0: the same weighted line in tan as psd_chdeg(usetilt=False)
    held = cal.calibrate_linear(lin_model(), fr, [lin_start(FIXED(0.0))])
    xr = sa.psd_chdeg(ANGLES.copy(), linear["pos"].copy(), stdev=linear["sigma"], plot=False, usetilt=False)
    assert held.params["pwidth"] == pytest.approx(-xr[0], rel=1e-9)
    assert held.params["cch"] == pytest.approx(xr[1], rel=1e-9)
    # tilt free recovers the truth from xu's channels
    fit = cal.calibrate_linear(lin_model(), fr, [lin_start(FREE(1.0))])
    assert fit.params["cch"] == pytest.approx(LCCH, rel=1e-6)
    assert fit.params["pwidth"] == pytest.approx(LPW, rel=1e-6)
    assert abs(fit.params["tilt"] - linear["tilt"]) < 1e-4
    assert "tilt" in fit.sigma and fit.value.__class__.__name__ == "Linear"


def test_linear_from_spectra_agrees_with_xu_linear_detector_calib(linear):
    peaks = [cal.find_peak_1d(s) for s in linear["spectra"]]
    fr = [cal.CalFrame([a], p, cal.FrameKind.primary_beam(), i) for i, (a, p) in enumerate(zip(ANGLES, peaks))]
    fit = cal.calibrate_linear(lin_model(), fr, [lin_start(FREE(1.0))])
    xr = sa.linear_detector_calib(ANGLES.copy(), linear["spectra"], plot=False, usetilt=True)
    # one noisy draw cannot rank two estimators; both sit within the fit's error of the truth
    assert abs(fit.params["cch"] - LCCH) < 3 * fit.sigma["cch2"]
    assert abs(xr[1] - LCCH) < 3 * fit.sigma["cch2"]
    assert abs(fit.params["tilt"] - linear["tilt"]) < 3 * fit.sigma["tilt"]


def test_linear_direction_is_searched_when_not_given(linear):
    m = cal.LinearModel(cal.xu_geometry([("a", "x+")], beam=[0, 1, 0]), LN)
    fit = cal.calibrate_linear(m, lin_frames(ANGLES, linear["pos"], linear["sigma"]), [lin_start(FREE(1.0))])
    assert fit.directions == ["z+"]


# --- hkl ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hkl():
    doc = json.load(open(os.path.join(GOLDEN, "xu_calib_hkl.json")))
    a = doc["lattice"]
    mat = xu.materials.Crystal("cubic", xu.materials.SGLattice(221, a))
    exp = xu.HXRD(mat.Q(1, 0, 0), mat.Q(0, 0, 1), en=xu.lam2en(WL))
    rng = np.random.default_rng(20260102)

    def render(scale):
        return [rng.poisson(2000.0 * (scale if f["hkl"] != [0, 0, 0] else 1.0)
                            * np.exp(-((np.mgrid[0:N, 0:N][0] - f["true_pixel"][0]) ** 2
                                       + (np.mgrid[0:N, 0:N][1] - f["true_pixel"][1]) ** 2) / 8.0)).astype(float)
                for f in doc["frames"]]
    return dict(doc=doc, mat=mat, exp=exp, strong=render(1.0), weak=render(0.05))


def hkl_model(doc):
    a = doc["lattice"]
    g = cal.xu_geometry([("a1", "z+"), ("a2", "y-")], sample=[("om", "y-")], beam=[1, 0, 0])
    sample = hklrs.Sample("cubic", hklrs.Lattice(a, a, a, 90, 90, 90))
    return cal.AreaModel(g, (N, N), sample=sample, offset_axis="a1", directions=("z+", "y+"))


def hkl_frames(doc, key="xu_pixel"):
    out = []
    for i, f in enumerate(doc["frames"]):
        kind = cal.FrameKind.primary_beam() if f["hkl"] == [0, 0, 0] else cal.FrameKind.reflection(f["hkl"])
        out.append(cal.CalFrame([f["sang"], f["angle1"], f["angle2"]],
                                cal.Peak(f[key], f["sigma_pixel"]), kind, i))
    return out


def hkl_start(doc, wavelength=FREE(1.5), tilt=FREE(0.2), azimuth=FREE(20.0)):
    area = cal.AreaStart(FREE(390.0), FREE(405.0), FREE(0.0021), FREE(0.0019), FIXED(D), wavelength,
                         detrot=FIXED(doc["truth"]["detrot"]))
    return cal.AreaHklStart(area, tilt, azimuth)


def xu_hkl(hkl, imgs, **kw):
    doc, f = hkl["doc"], hkl["doc"]["frames"]
    return sa.area_detector_calib_hkl(
        [x["sang"] for x in f], [x["angle1"] for x in f], [x["angle2"] for x in f], imgs,
        np.array([x["hkl"] for x in f]), hkl["exp"], hkl["mat"], ["z+", "y-"], "x+", plot=False, nwindow=30, **kw)


START = [None, None, D, 0, 0, 2.0, 0, 0.0, 0.0, 1.5]
FIX = (False, False, True, True, True, True, True, False, False, False)


def test_hkl_recovers_wavelength_and_sample_tilt_that_xu_leaves_at_its_start(hkl, monkeypatch):
    doc = hkl["doc"]
    fit = cal.calibrate_area_hkl(hkl_model(doc), hkl_frames(doc), [hkl_start(doc)])
    t = doc["truth"]
    for k, want in (("cch1", t["cch1"]), ("cch2", t["cch2"]), ("pwidth1", PW), ("pwidth2", PW),
                    ("wavelength", WL), ("sample_tilt", t["sampletilt"]), ("sample_azimuth", t["sampleazimuth"])):
        assert abs(fit.params[k] - want) <= 5 * fit.sigma[k], (k, fit.params[k], want, fit.sigma[k])
    assert fit.rejected == [] and fit.directions == ["z+", "y+"]
    # XU-6 at work in xu's hkl fit: a valid tuple start returns wavelength 1.5 and sample tilt ~1.66 untouched
    res = np.ravel(xu_hkl(hkl, hkl["strong"], start=tuple(START), fix=FIX)[0])
    assert res[11] == 1.5 and abs(res[9] - t["sampletilt"]) > 1.0
    assert abs(fit.params["wavelength"] - WL) * 100 < abs(res[11] - WL)


def test_xu7_default_start_tuple_and_list_start_both_crash_in_xu(hkl):
    with pytest.raises(TypeError, match="item assignment"):
        xu_hkl(hkl, hkl["strong"])  # default start tuple: sample_align.py:1692
    with pytest.raises(TypeError, match="concatenate list"):
        xu_hkl(hkl, hkl["strong"], start=list(START), fix=FIX)  # sample_align.py:1823
    # hklrs has no start tuple: the wavelength is a Param like the others, free or held
    doc = hkl["doc"]
    held = cal.calibrate_area_hkl(hkl_model(doc), hkl_frames(doc), [hkl_start(doc, wavelength=FIXED(WL))])
    assert "wavelength" not in held.sigma and held.params["wavelength"] == WL


def test_xu8_weak_zero_index_reflections_are_dropped_by_xu_and_kept_by_kind(hkl, monkeypatch):
    kept = []
    orig = sa._peak_position
    monkeypatch.setattr(sa, "_peak_position", lambda img, nw, plot=False: kept.append(1) or orig(img, nw, plot))
    with pytest.raises(TypeError):  # the list start crashes after the selection ran
        xu_hkl(hkl, hkl["weak"], start=list(START), fix=FIX)
    assert len(kept) == 9  # of 21: every (0 0 L) image has a zero index
    doc = hkl["doc"]
    fs = hkl_frames(doc)
    assert sum(f.kind.is_reflection for f in fs) == 12 and len(fs) == 21
    fit = cal.calibrate_area_hkl(hkl_model(doc), fs, [hkl_start(doc)])
    assert fit.rejected == [] and fit.dof == 2 * 21 - len(fit.free_ids)


# --- exceptions -------------------------------------------------------------------------------------

def test_every_failure_is_its_own_exception_with_typed_attributes(area):
    model = cal.AreaModel(geometry(), (N, N), directions=("z+", "y+"))
    fr = area_frames(area)

    def run(start=area_start(FIXED(2.0)), frames=fr, m=model):
        return cal.calibrate_area(m, frames, [start])

    for cls in (cal.NotConverged, cal.Unidentifiable, cal.TooFewFrames, cal.AmbiguousDirections,
                cal.BadStart, cal.BadFrame, cal.BadProblem, cal.BadModel, cal.OffsetAxisNotRotation):
        assert issubclass(cls, cal.CalibError) and issubclass(cls, ValueError)
    # a free wavelength with primary-beam frames only (XU-7): named, not arbitrary
    s = cal.AreaStart(FREE(390.0), FREE(405.0), FREE(0.0021), FREE(0.0019), FIXED(D), FREE(1.5),
                      detrot=FIXED(2.0))
    with pytest.raises(cal.Unidentifiable) as e:
        run(s)
    assert e.value.params == ["wavelength"]
    nan = cal.AreaStart(FREE(float("nan")), FREE(405.0), FREE(0.0021), FREE(0.0019), FIXED(D), FIXED(WL))
    with pytest.raises(cal.BadStart) as e:
        run(nan)
    assert e.value.param == "cch1" and np.isnan(e.value.value)
    with pytest.raises(cal.BadFrame) as e:
        cal.CalFrame([0.0, 0.0], cal.Peak([1.0, 1.0], [0.0, 1.0]), index=3)
    assert e.value.index == 3
    with pytest.raises(cal.BadFrame) as e:  # one angle where the geometry has two axes
        run(frames=[cal.CalFrame([0.0], cal.Peak([1.0, 1.0], [1.0, 1.0]), index=0)])
    assert e.value.index == 0
    with pytest.raises(cal.TooFewFrames) as e:
        run(frames=fr[:2])
    assert (e.value.have, e.value.need) == (4, 5)
    fixed = cal.AreaStart(*(FIXED(v) for v in (401.3, 396.7, PW, PW, D, WL)), detrot=FIXED(2.0))
    with pytest.raises(cal.BadProblem):
        run(fixed)
    with pytest.raises(cal.BadFrame, match="no sample"):
        run(frames=[cal.CalFrame([0.0, 0.0], cal.Peak([1.0, 1.0], [1.0, 1.0]), cal.FrameKind.reflection([0, 0, 2]))])
    with pytest.raises(cal.BadModel):
        cal.calibrate_area(cal.AreaModel(geometry(), (N, N), offset_axis="nope", directions=("z+", "y+")),
                           fr, [area_start(FIXED(2.0))])
    with pytest.raises(ValueError):
        cal.AreaModel(geometry(), (N, N), directions=("q+", "y+"))


def test_the_linear_hkl_sample_azimuth_limit_is_a_typed_error():
    # one detector circle sees only tilt*sin(azimuth) of a (0 0 L) misorientation: both free is Unidentifiable
    a = 10.86
    g = cal.xu_geometry([("a", "x+")], sample=[("om", "x+")], beam=[0, 1, 0])
    m = cal.LinearModel(g, LN, sample=hklrs.Sample("cubic", hklrs.Lattice(a, a, a, 90, 90, 90)), direction="z+")
    th = np.degrees(np.arcsin(WL * np.arange(2, 8) / (2 * a)))
    fr = [cal.CalFrame([0.0, 0.0], cal.Peak.from_row(300.0 + i, 0.05), cal.FrameKind.primary_beam(), i) for i in range(4)]
    fr += [cal.CalFrame([t, 2 * t], cal.Peak.from_row(300.0 + 20 * i, 0.05), cal.FrameKind.reflection([0, 0, 2 + i]), 4 + i)
           for i, t in enumerate(th)]
    ls = cal.LinearStart(FREE(300.0), FIXED(LPW), FIXED(D), FIXED(WL), tilt=FIXED(0.0))
    start = cal.LinearHklStart(ls, FREE(0.3), FREE(30.0))
    with pytest.raises((cal.Unidentifiable, cal.NotConverged, cal.CalibError)) as e:
        cal.calibrate_linear_hkl(m, fr, [start])
    assert isinstance(e.value, cal.CalibError)
