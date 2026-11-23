"""Golden intensity normalisation from xrayutilities 1.8.0 deff08f: IntensityNormalizer.__call__ on point,
PSD (one row per frame) and CCD data. Run: <xu python> -I xu_normalise.py

"compare" cases are those where xu is correct: "xu" is its corrected intensity. hklrs returns (counts,
weight) per pixel, so a case holds what the normaliser needs, per frame: raw counts, monitor, count time,
absfun value ("absfun" multiplies the intensity, hklrs takes its inverse as a transmission), and the
detector's dark rate (per second) and flat field. "divergence" cases are those where xu is wrong, with what
xu returned (or the exception it raised) beside the inputs.
"""
import json
import os

import numpy as np
import xrayutilities as xu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_normalise.json")
rng = np.random.default_rng(20260301)


def counts(*shape):
    return rng.integers(50, 5000, size=shape).astype(float)


def run(raw, mon=None, time=None, absf=None, dark=None, flat=None, av_mon=None, ccd=False):
    """xu's output for `raw` (frames first) and the inputs it was given."""
    n = raw.shape[0]
    fields = [("det", "f8", raw.shape[1:]), ("mon", "f8"), ("t", "f8"), ("absf", "f8")]
    data = np.zeros(n, dtype=fields)
    data["det"] = raw
    data["mon"] = mon if mon is not None else 1.0
    data["t"] = time if isinstance(time, np.ndarray) else 1.0
    data["absf"] = absf if absf is not None else 1.0
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
    case = dict(raw=raw.tolist(), mon=None if mon is None else mon.tolist(),
                time=time.tolist() if isinstance(time, np.ndarray) else time,
                absfun=None if absf is None else absf.tolist(),
                dark=None if dark is None else dark.tolist(),
                flat=None if flat is None else flat.tolist(),
                av_mon=av_mon, shape=list(raw.shape[1:]))
    try:
        out = corr(data, ccd=raw) if ccd else corr(data)
        case["xu"] = np.asarray(out).tolist()
    except Exception as e:  # recorded: it is the divergence
        case["xu_error"] = f"{type(e).__name__}: {e}"
    return case


def main():
    assert xu.__version__ == "1.8.0", xu.__version__
    n, nch, n1, n2 = 6, 9, 4, 5
    mon = rng.uniform(800, 1200, n)
    t = rng.uniform(0.5, 2.0, n)
    absf = np.array([1, 1, 3.2, 3.2, 10.5, 1.0])
    dark = rng.uniform(0.0, 0.3, nch)
    flat = rng.uniform(0.7, 1.3, nch)

    dark2 = rng.uniform(0.0, 0.3, (n1, n2))
    flat2 = rng.uniform(0.7, 1.3, (n1, n2))
    point = counts(n)
    psd = counts(n, nch)
    ccd = counts(n, n1, n2)
    compare = {
        "point_mon_time_abs": run(point, mon, t, absf),
        "point_mon": run(point, mon),
        "point_time_float": run(point, time=2.5),
        "point_av_mon": run(point, mon, av_mon=1000.0),
        "point_abs_only": run(point, absf=absf),
        "psd_mon_time_abs": run(psd, mon, t, absf),
        "psd_dark_flat_mon_time_abs": run(psd, mon, t, absf, dark, flat),
        "psd_dark_mon_time": run(psd, mon, t, None, dark),
        "psd_flat_mon": run(psd, mon, None, None, None, flat),
        "ccd_mon_time_abs": run(ccd, mon, t, absf, ccd=True),
    }
    # xu clamps a dark-subtracted pixel at 0 (hklrs keeps the sign): dark above the counts
    big = np.full(nch, 50.0 * 60)
    divergence = {
        # XU-9: no monitor/time array/absfun -> c is a float and the 2D branch skips flat and dark
        "xu9_psd_flat_no_monitor": run(psd[:1], flat=flat),
        "xu9_psd_dark_flat_time_float": run(psd[:1], time=2.5, dark=dark, flat=flat),
        # XU-10: a CCD stack with no monitor, time array or absfun
        "xu10_ccd_no_monitor": run(ccd, ccd=True),
        # 3D data: dark and flat are accepted and ignored ("still missing" in xu)
        "ccd_dark_flat_ignored": run(ccd, mon, t, absf, dark2, flat2, ccd=True),
        "dark_clamped": run(psd, mon, t, None, big),
        # a zero monitor: c is inf, xu sets the factor to 1 and returns the raw counts
        "zero_monitor": run(point[:3], np.array([1000.0, 0.0, 900.0]), np.ones(3)),
        # a flat pixel below 1e-5 is set to 1 and its counts scaled by the mean flat, not masked
        "flat_below_1e-5": run(psd, mon, t, None, None, np.r_[flat[:-1], 1e-6]),
    }
    doc = {"source": "xrayutilities 1.8.0 deff08fbe98d5995513757b7a6d57cffe21b9a3f",
           "flat_min": 1e-5, "compare": compare, "divergence": divergence}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT)


main()
