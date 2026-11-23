"""Golden for hkl-based area-detector calibration from xrayutilities 1.8.0
(sample_align.area_detector_calib_hkl, pin deff08f). A cubic crystal (a =
10.86 A, so (0 0 2)..(0 0 8) are reachable) on one sample circle 'y-' (xu
requires it to turn like the inner detector circle), detector circles 'z+'
(outer, angle1) and 'y-' (inner, angle2), beam x+. Primary-beam frames on a
3x3 grid plus (0 0 L) reflections at their exact Bragg setting for a sample
misoriented by (tilt, azimuth); peaks are placed with xu's own Ang2Q.area,
rendered as Gaussians (seeded Poisson noise) and found with xu's
_peak_position. Runs: XU-7 (default start tuple), XU-8 (weak reflections
dropped by the selection `numpy.all(hkl != 0)`), and the fit on strong ones.
Run: python -I xu_calib_hkl.py"""
import json
import os

import numpy as np
import xrayutilities as xu
from scipy.optimize import minimize_scalar
from xrayutilities.analysis import sample_align as sa

assert xu.__version__ == "1.8.0", xu.__version__
xu.config.VERBOSITY = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_calib_hkl.json")
N, PW, D, WL, A = 800, 2e-3, 1.0, 1.54, 10.86
TRUTH = dict(cch1=401.3, cch2=396.7, pwidth1=PW, pwidth2=PW, distance=D, tiltazimuth=0.0, tilt=0.0,
             detrot=2.0, offset=0.0, sampletilt=0.3, sampleazimuth=30.0, wavelength=WL)
AMP, SIGMA_PX, NWINDOW = 2000.0, 2.0, 30

mat = xu.materials.Crystal("cubic", xu.materials.SGLattice(221, A))
exp = xu.HXRD(mat.Q(1, 0, 0), mat.Q(0, 0, 1), en=xu.lam2en(WL))
U2 = exp._transform.matrix
sp, cp = np.sin(np.radians(TRUTH["sampleazimuth"])), np.cos(np.radians(TRUTH["sampleazimuth"]))
sc, cc = np.sin(np.radians(TRUTH["sampletilt"])), np.cos(np.radians(TRUTH["sampletilt"]))
U1 = np.array(((cp * cc, -sp, cp * sc), (sp * cc, cp, sp * sc), (-sc, 0, cc)))
UB = U1 @ U2 @ mat.B

conv = xu.QConversion(["y-"], ["z+", "y-"], [1, 0, 0])
conv.init_area("z+", "y+", cch1=TRUTH["cch1"], cch2=TRUTH["cch2"], Nch1=N, Nch2=N, pwidth1=PW,
               pwidth2=PW, distance=D, detrot=TRUTH["detrot"], tiltazimuth=0, tilt=0)


def hkl_map(sang, a1, a2):
    return np.array(conv.area(sang, a1, a2, UB=UB, wl=WL, deg=True))


def nearest(h, sang, a1, a2):
    """pixel (fractional) where the hkl map is closest to h, and the distance"""
    q = hkl_map(sang, a1, a2) - np.asarray(h, float)[:, None, None]
    m = np.linalg.norm(q, axis=0)
    i, j = np.unravel_index(m.argmin(), m.shape)
    ii, jj = np.mgrid[-3:4, -3:4]
    Am = np.c_[np.ones(49), ii.ravel(), jj.ravel()]
    sl = (slice(i - 3, i + 4), slice(j - 3, j + 4))
    cf = [np.linalg.lstsq(Am, q[c][sl].ravel(), rcond=None)[0] for c in range(3)]
    G = np.array([[cf[c][1], cf[c][2]] for c in range(3)])
    g0 = np.array([cf[c][0] for c in range(3)])
    d = np.linalg.lstsq(G, -g0, rcond=None)[0]
    return (i + d[0], j + d[1]), float(np.linalg.norm(g0 + G @ d))


def centroid_sigma(img, c, nw=25):
    c0 = [int(round(v)) for v in c]
    sl = (slice(c0[0] - nw, c0[0] + nw), slice(c0[1] - nw, c0[1] + nw))
    w = img[sl]
    bg = np.median(np.r_[w[:3].ravel(), w[-3:].ravel()])
    n = (w - bg).sum()
    yy, xx = np.mgrid[sl]
    return [float(np.sqrt((np.clip(w, 0, None) * (g - cc_) ** 2).sum()) / n)
            for g, cc_ in ((yy, c[0]), (xx, c[1]))]


setups = [(0.0, a1, a2, (0, 0, 0)) for a1 in (-10, 0, 10) for a2 in (-10, 0, 10)]
for L in (2, 4, 6, 8):
    th = np.degrees(np.arcsin(WL * L / (2 * A)))
    for a1 in (-1.5, 0.0, 1.5):
        best = minimize_scalar(lambda s: nearest((0, 0, L), s, a1, 2 * th)[1],
                               bracket=(th - 0.05, th + 0.05), tol=1e-10)
        setups.append((float(best.x), a1, 2 * th, (0, 0, L)))

yy, xx = np.mgrid[0:N, 0:N]
rng = np.random.default_rng(20260102)
frames, pos = [], []
for sang, a1, a2, h in setups:
    t, miss = nearest(h, sang, a1, a2)
    assert miss < 1e-3, (h, a1, miss)
    frames.append(dict(sang=sang, angle1=float(a1), angle2=float(a2), hkl=list(h),
                       true_pixel=[float(t[0]), float(t[1])]))
    pos.append(t)


def render(scale_reflections):
    imgs = []
    for f, t in zip(frames, pos):
        amp = AMP * (scale_reflections if f["hkl"] != [0, 0, 0] else 1.0)
        imgs.append(rng.poisson(amp * np.exp(-((yy - t[0]) ** 2 + (xx - t[1]) ** 2)
                                            / (2 * SIGMA_PX ** 2))).astype(float))
    return imgs


strong = render(1.0)
for f, img in zip(frames, strong):
    cen = sa._peak_position(img, NWINDOW)
    f["xu_pixel"] = [float(cen[0]), float(cen[1])]
    f["sigma_pixel"] = centroid_sigma(img, cen)
weak = render(0.05)

args = lambda imgs: ([f["sang"] for f in frames], [f["angle1"] for f in frames],
                     [f["angle2"] for f in frames], imgs, np.array([f["hkl"] for f in frames]),
                     exp, mat, ["z+", "y-"], "x+")


def attempt(label, imgs, **kw):
    kept = []
    orig = sa._peak_position

    def spy(img, nw, plot=False):
        kept.append(float(img.sum()))
        return orig(img, nw, plot)

    sa._peak_position = spy
    out = dict(label=label)
    try:
        r = sa.area_detector_calib_hkl(*args(imgs), plot=False, nwindow=NWINDOW, **kw)
        out["result"] = [float(x) for x in np.ravel(r[0])] if isinstance(r, tuple) else str(r)
        out["eps"] = float(r[1]) if isinstance(r, tuple) and len(r) > 1 else None
    except BaseException as e:  # noqa: BLE001 - recorded verbatim
        out["exception"] = f"{type(e).__name__}: {e}"
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        out["raised_at"] = f"{os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}"
    finally:
        sa._peak_position = orig
    out["images_given"] = len(frames)
    out["images_peak_found"] = len(kept)
    return out


FIX = (False, False, True, True, True, True, True, False, False, False)
START = [None, None, D, 0, 0, TRUTH["detrot"], 0, 0.0, 0.0, 1.5]
runs = [
    attempt("default start tuple (XU-7)", strong),
    attempt("weak reflections, start list (XU-8)", weak, start=list(START), fix=FIX),
    attempt("strong reflections, start list", strong, start=list(START), fix=FIX),
    attempt("strong reflections, start tuple", strong, start=tuple(START), fix=FIX),
]
json.dump(dict(
    program="xrayutilities 1.8.0 sample_align.area_detector_calib_hkl, scipy, numpy default_rng(20260102)",
    units=dict(angle="deg", length="m", pixel="index, centres at integers, [slow, fast]"),
    note=("sample circle 'y-' (sang), detector 'z+' (angle1), 'y-' (angle2), beam x+; detector directions "
          "z+ (slow), y+ (fast); hkl are lattice indices of a cubic a=10.86 crystal whose UB is "
          "U1(sampletilt, sampleazimuth) . U2 . B as xu builds it; reflection frames sit at their exact "
          "Bragg setting. xu runs: see runs[].exception / result"),
    wavelength=WL, shape=[N, N], lattice=A, truth=TRUTH,
    image=dict(amplitude=AMP, reflection_amplitude_weak=AMP * 0.05, sigma_px=SIGMA_PX, nwindow=NWINDOW),
    frames=frames, runs=runs), open(OUT, "w"), indent=1)
print(json.dumps(runs, indent=1))
