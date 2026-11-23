"""Golden for area-detector calibration from xrayutilities 1.8.0
(sample_align.area_detector_calib, pin deff08f) with step_beta removed from
the odr_fit call (the r13 NOSTEP path: with it odrpack stops at the start
values, XU-6). One detector circle ('z+' outer, detector arm axis), primary
beam frames over a 5x5 grid of (arm angle, outer-circle angle).
Peaks are placed with xu's own Ang2Q/pixel model (QConversion.area, the pixel
whose |Q| is smallest, refined by a local plane fit), rendered as Gaussians
(sigma 2 px, amplitude 2000, background 0, seeded Poisson noise) and found
again with xu's _peak_position. Run: python -I xu_calib_area.py"""
import json
import os

import numpy as np
import xrayutilities as xu
from xrayutilities.analysis import sample_align as sa

assert xu.__version__ == "1.8.0", xu.__version__
xu.config.VERBOSITY = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_calib_area.json")
N, PW, D, WL = 800, 2e-3, 1.0, 1.54
TRUTH = dict(cch1=401.3, cch2=396.7, pwidth1=PW, pwidth2=PW, distance=D,
             tiltazimuth=0.0, tilt=0.0, detrot=2.0, offset=0.0)
AMP, BG, SIGMA_PX, NWINDOW = 2000.0, 0.0, 2.0, 30  # xu COM has no background term: BG > 0 pulls it to the image centre
NAMES = "cch1 cch2 pwidth1 pwidth2 distance tiltazimuth tilt detrot offset".split()

conv = xu.QConversion([], ["z+", "y-"], [1, 0, 0])
conv.init_area("z+", "y+", cch1=TRUTH["cch1"], cch2=TRUTH["cch2"], Nch1=N, Nch2=N,
               pwidth1=PW, pwidth2=PW, distance=D, detrot=TRUTH["detrot"],
               tiltazimuth=0, tilt=0)


def beampix(a1, a2, area=None):
    q = np.array((area or conv.area)(a1, a2, wl=WL, deg=True))
    m = np.linalg.norm(q, axis=0)
    i, j = np.unravel_index(m.argmin(), m.shape)
    ii, jj = np.mgrid[-3:4, -3:4]
    A = np.c_[np.ones(49), ii.ravel(), jj.ravel()]
    sl = (slice(i - 3, i + 4), slice(j - 3, j + 4))
    cf = [np.linalg.lstsq(A, q[c][sl].ravel(), rcond=None)[0] for c in range(3)]
    G = np.array([[cf[c][1], cf[c][2]] for c in range(3)])
    g0 = np.array([cf[c][0] for c in range(3)])
    d = np.linalg.lstsq(G, -g0, rcond=None)[0]
    return i + d[0], j + d[1]


def centroid_sigma(img, c, nw=25):
    """Poisson centroid error sqrt(sum d_i (x_i-c)^2)/N on background-
    subtracted counts in a (2nw)^2 window: the formula calib::peak uses."""
    c0 = [int(round(v)) for v in c]
    sl = (slice(c0[0] - nw, c0[0] + nw), slice(c0[1] - nw, c0[1] + nw))
    w = img[sl]
    bg = np.median(np.r_[w[:3].ravel(), w[-3:].ravel()])
    d = w - bg
    yy, xx = np.mgrid[sl]
    n = d.sum()
    return [float(np.sqrt((np.clip(w, 0, None) * (g - cc) ** 2).sum()) / n)
            for g, cc in ((yy, c[0]), (xx, c[1]))]


g = np.linspace(-15, 15, 5)
A1, A2 = [a.ravel() for a in np.meshgrid(g, g)]
yy, xx = np.mgrid[0:N, 0:N]
rng = np.random.default_rng(20260101)
frames, imgs = [], []
for a1, a2 in zip(A1, A2):
    t = beampix(a1, a2)
    img = rng.poisson(BG + AMP * np.exp(-((yy - t[0]) ** 2 + (xx - t[1]) ** 2)
                                        / (2 * SIGMA_PX ** 2))).astype(float)
    cen = sa._peak_position(img, NWINDOW)
    frames.append(dict(angle1=float(a1), angle2=float(a2), true_pixel=[float(t[0]), float(t[1])],
                       xu_pixel=[float(cen[0]), float(cen[1])],
                       sigma_pixel=centroid_sigma(img, cen)))
    imgs.append(img)

# the same beam, generated in the order the fit model applies the circles
# (R_x(detrot) R_y R_z, detrot as an explicit third circle; r13 'rev')
tc = xu.QConversion([], ["x+", "y-", "z+"], [1, 0, 0])
tc.init_area("z+", "y+", cch1=TRUTH["cch1"], cch2=TRUTH["cch2"], Nch1=N, Nch2=N,
             pwidth1=PW, pwidth2=PW, distance=D, tiltazimuth=0, tilt=0)
rev_pix = [beampix(a1, a2, lambda x, y, **k: tc.area(TRUTH["detrot"], y, x, **k)) for a1, a2 in zip(A1, A2)]
exact = {id(im): np.array(f["true_pixel"]) for im, f in zip(imgs, frames)}
rev_exact = {id(im): np.array(p) for im, p in zip(imgs, rev_pix)}
_peak = sa._peak_position

# capture every odr_fit result xu makes, without changing the call
calls = []
_odr = sa.odr_fit


def _record(*a, **k):
    k.pop("step_beta", None)
    r = _odr(*a, **k)
    calls.append(r)
    return r


sa.odr_fit = _record


def run(label, detrot_start, fix_detrot, pix=None):
    calls.clear()
    sa._peak_position = (lambda img, nw, plot=False: tuple(pix[id(img)])) if pix else _peak
    res, eps = sa.area_detector_calib(
        A1, A2, imgs, ["z+", "y-"], "x+", plot=False, wl=WL,
        start=(None, None, D, 0, 0, detrot_start, 0),
        fix=(False, False, True, True, True, fix_detrot, True), nwindow=NWINDOW)
    best = min(calls, key=lambda r: np.abs(np.asarray(r.beta) - np.asarray(
        [res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]])).max()
        if len(r.beta) == 9 else np.inf)
    return dict(
        label=label, start_detrot=detrot_start, fixed_detrot=fix_detrot,
        fixed=["distance", "tiltazimuth", "tilt", "offset"] + (["detrot"] if fix_detrot else []),
        fitted=dict(zip(NAMES, map(float, res))), eps=float(eps), n_odr_calls=len(calls),
        odr=dict(info=int(best.info), stopreason=str(best.stopreason),
                 niter=int(getattr(best, "niter", -1)), nfev=int(getattr(best, "nfev", -1)),
                 beta=list(map(float, best.beta)),
                 sd_beta=list(map(float, getattr(best, "sd_beta", []))),
                 sum_square=float(getattr(best, "sum_square", float("nan")))))


fits = [run("detrot free, start 0", 0.0, False), run("detrot fixed at truth", TRUTH["detrot"], True),
        run("detrot fixed at truth, exact beam pixels from Ang2Q.area (no peak finding)", TRUTH["detrot"], True, exact),
        run("detrot fixed at truth, exact pixels generated in the fit model's circle order", TRUTH["detrot"], True, rev_exact)]
json.dump(dict(
    program="xrayutilities 1.8.0 sample_align.area_detector_calib (step_beta removed), odrpack 0.7.1",
    units=dict(angle="deg", length="m", pixel="index, centres at integers, [slow, fast]"),
    note=("circles: outer 'z+' (angle1), arm 'y-' (angle2); detector directions z+ (slow), y+ (fast); "
          "r_i x+; detrot is xu's init_area detrot; true_pixel from QConversion.area, xu_pixel from "
          "_peak_position on the noisy image; sigma_pixel is our Poisson centroid error, xu has none; "
          "xu's fit does not recover the truth it generated the data from (pwidth 0.00205/0.00195 vs "
          "0.002, detrot free runs off to 3e6 with info 12). Cause: XU-1, not peak finding: the same bias "
          "remains with exact Ang2Q beam pixels (fit 3), and goes away when the pixels are generated with the "
          "circles in the fit model's order (fit 4)"),
    wavelength=WL, shape=[N, N], truth=TRUTH,
    image=dict(amplitude=AMP, background=BG, sigma_px=SIGMA_PX, noise="poisson, numpy default_rng(20260101)", nwindow=NWINDOW),
    frames=frames, fits=fits), open(OUT, "w"), indent=1)
print(json.dumps(fits, indent=1)[:1500])
