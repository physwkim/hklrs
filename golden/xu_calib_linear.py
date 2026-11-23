"""Golden for linear-detector calibration from xrayutilities 1.8.0
(sample_align.psd_chdeg and linear_detector_calib, pin deff08f). One detector
circle 'x+', beam y+, detector direction z+. Peak channels come from xu's own
Ang2Q (QConversion.linear: the channel whose |Q| is smallest, refined by a
local line fit); spectra are Gaussians (sigma 2.5 ch, amplitude 3000, seeded
Poisson noise). Cases: true tilt 0 and 3 deg, each fitted with usetilt True and
False, once from exact positions via psd_chdeg and once from spectra via
linear_detector_calib. Run: python -I xu_calib_linear.py"""
import json
import os

import numpy as np
import xrayutilities as xu
from xrayutilities.analysis import sample_align as sa

assert xu.__version__ == "1.8.0", xu.__version__
xu.config.VERBOSITY = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_calib_linear.json")
N, CCH, PW, D, WL = 1000, 300.3, 2e-4, 1.0, 1.54
AMP, SIGMA_CH = 3000.0, 2.5
ANGLES = np.arange(-5.0, 3.01, 0.5)  # channel falls with angle: -5..3 deg stays on 0..999
rng = np.random.default_rng(20260102)


def channel(conv, ang):
    q = np.array(conv.linear(ang, wl=WL, deg=True))  # (3, N)
    m = np.linalg.norm(q, axis=0)
    i = int(m.argmin())
    k = np.arange(i - 3, i + 4)
    A = np.c_[np.ones(7), k - i]
    cf = [np.linalg.lstsq(A, q[c][k], rcond=None)[0] for c in range(3)]
    g = np.array([c[1] for c in cf])
    g0 = np.array([c[0] for c in cf])
    return float(i + (g @ -g0) / (g @ g))


calls = []
_odr = sa.odr_fit


def _record(*a, **k):
    r = _odr(*a, **k)
    calls.append(r)
    return r


sa.odr_fit = _record


def odr_info(r):
    return dict(info=int(r.info), stopreason=str(r.stopreason), niter=int(getattr(r, "niter", -1)),
                beta=list(map(float, r.beta)), sd_beta=list(map(float, getattr(r, "sd_beta", []))))


cases = []
for tilt in (0.0, 3.0):
    conv = xu.QConversion([], ["x+"], [0, 1, 0])
    conv.init_linear("z+", CCH, N, distance=D, pixelwidth=PW, tilt=tilt)
    pos = np.array([channel(conv, a) for a in ANGLES])
    assert pos.min() > 10 and pos.max() < N - 10, (tilt, pos.min(), pos.max())
    ch = np.arange(N)
    spectra = np.array([rng.poisson(AMP * np.exp(-(ch - p) ** 2 / (2 * SIGMA_CH ** 2))).astype(float)
                        for p in pos])
    # Poisson centroid error of a Gaussian line: sigma / sqrt(counts)
    sigma = [float(SIGMA_CH / np.sqrt(s.sum())) for s in spectra]
    for usetilt in (True, False):
        calls.clear()
        r = sa.psd_chdeg(ANGLES.copy(), pos.copy(), stdev=np.array(sigma), plot=False, usetilt=usetilt)
        direct = dict(result=list(map(float, r)), fits=[odr_info(c) for c in calls])
        calls.clear()
        r = sa.linear_detector_calib(ANGLES.copy(), spectra, plot=False, usetilt=usetilt)
        spec = dict(result=list(map(float, r)), fits=[odr_info(c) for c in calls])
        cases.append(dict(true_tilt=tilt, usetilt=usetilt, psd_chdeg=direct, linear_detector_calib=spec,
                          angles=ANGLES.tolist(), xu_channel=pos.tolist(), sigma_channel=sigma))

json.dump(dict(
    program="xrayutilities 1.8.0 sample_align.psd_chdeg / linear_detector_calib, odrpack 0.7.1",
    units=dict(angle="deg", length="m", channel="index, centres at integers"),
    note=("psd_chdeg returns [pixelwidth at 1 m (negative if the channel falls with angle), center channel"
          "(, tilt if usetilt)]; fits lists the odr runs in xu's order: linear, tan, (tilt); "
          "xu_channel is xu's own channel for each angle; spectra are not stored, regenerate them from spectra_seed"),
    wavelength=WL, truth=dict(cch=CCH, pixelwidth=PW, distance=D, nchannel=N, sigma_channel_px=SIGMA_CH),
    spectra_seed=20260102, cases=cases), open(OUT, "w"), indent=1)
for c in cases:
    print(c["true_tilt"], c["usetilt"], c["psd_chdeg"]["result"], c["linear_detector_calib"]["result"],
          [f["info"] for f in c["psd_chdeg"]["fits"]])
