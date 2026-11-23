"""Golden line-cut coordinates from xrayutilities deff08f (1.8.0-11, line_cuts.py identical).
Run: <xu python> -I xu_cuts.py

xu's _get_cut(pos_along, pos_perp, intensity, dis, npoints) is wrapped to record its two coordinate arrays;
a pixel is selected when |pos_perp| < dis and dis = intrange / 2 (XU-17). Angles are xu's degrees; Q is
xu's experiment frame (x scattering-plane normal, y beam-ish, z surface normal) with 1/angstrom and 2 pi.
Per case: the call, the points, xu's "along" and "perp" per pixel (null = NaN), the returned used mask,
and "error" when xu raises on an empty window (XU-18). qx omega and 2theta raise for every array input
("xu_error"); their coordinates are the intended ones, from xu Q2Ang on broadcast cut arrays. intrange is chosen from the recorded |perp| so no pixel lies within
1e-6 of the selection bound.
"""
import json
import os

import numpy as np
import xrayutilities as xu
from xrayutilities.analysis import line_cuts as lc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_cuts.json")
WL = 1.5406
NPOINTS = 40
rng = np.random.default_rng(20260101)

# cloud A: a peak near qz = 5, one sign of qy; cloud B: qy of both signs and small qz, which puts the
# 'real' geometry's omega on both sides of 2theta/2 (|tt - om| > 90 in qy_2theta)
n = 400
CLOUDS = {
    "peak": [rng.uniform(-0.3, 0.3, n), rng.uniform(1.6, 2.4, n), rng.uniform(4.6, 5.4, n)],
    "wide": [rng.uniform(-0.4, 0.4, n), rng.uniform(-4.0, 4.0, n), rng.uniform(0.3, 3.0, n)],
}
INTENSITY = {k: np.round(rng.uniform(1.0, 100.0, n), 3) for k in CLOUDS}
CUTS = {"peak": [0.02, 2.0, 5.0], "wide": [0.05, 1.5, 1.8]}  # qx, qy, qz of the cut position

# (name, function, intdir values, how "at" feeds the call)
KINDS = [
    ("qz", lc.get_qz_scan, ["q", "omega", "2theta"], lambda a: [a[0], a[1]]),
    ("qy", lc.get_qy_scan, ["q", "omega", "2theta"], lambda a: [a[0], a[2]]),
    ("qx", lc.get_qx_scan, ["q", "omega", "2theta"], lambda a: [a[1], a[2]]),
    ("omega", lc.get_omega_scan, ["2theta", "radial"], lambda a: list(a)),
    ("radial", lc.get_radial_scan, ["omega", "2theta"], lambda a: list(a)),
    ("twotheta", lc.get_ttheta_scan, ["omega", "radial"], lambda a: list(a)),
]


def clean(a):
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(a)]


def capture(fn, q, at, intdir, intrange, inten=None):
    seen = {}
    real = lc._get_cut

    def spy(pa, pp, inten, dis, npts):
        seen["along"], seen["perp"] = np.array(pa, float).ravel(), np.array(pp, float).ravel()
        seen["inten"], seen["dis"] = np.array(inten, float).ravel(), dis
        return real(pa, pp, inten, dis, npts)

    lc._get_cut = spy
    try:
        res = fn(q, np.ones_like(q[0]) if inten is None else inten, at, NPOINTS, intrange, intdir=intdir, wl=WL)
        return seen, res, None
    except Exception as e:  # XU-18
        return seen, None, type(e).__name__
    finally:
        lc._get_cut = real


def sums(seen, res):
    """xu's two gridders on the pixels _get_cut selects: its own fuzzy result (res) and the point gridder."""
    ma = np.abs(seen["perp"]) < seen["dis"]
    a, v = seen["along"][ma], seen["inten"][ma]
    width = (a.max() - a.min()) / NPOINTS
    g = xu.Gridder1D(NPOINTS)
    g(a, v)
    out = dict(selected=int(ma.sum()), xmin=float(res[0][0]), xmax=float(res[0][-1]), width=float(width),
               fuzzy=[float(x) for x in res[1]], point=[float(x) for x in g.data],
               point_norm=[float(x) for x in g._gnorm])
    # no pixel (or footprint edge) may sit within 1e-6 bins of a bin edge, where floor and rint disagree
    dx = (out["xmax"] - out["xmin"]) / (NPOINTS - 1)
    for x in (a, a - width / 2, a + width / 2):
        fr = (x - out["xmin"]) / dx + 0.5
        assert np.all(np.abs(fr - np.round(fr)) > 1e-6), "pixel on a bin edge"
    return out


def pick_dis(perp, frac):
    s = np.sort(np.abs(perp[np.isfinite(perp)]))
    k = int(len(s) * frac)
    while k < len(s) - 1 and s[k] - s[k - 1] < 2e-6:
        k += 1
    return 0.5 * (s[k - 1] + s[k])


cases = []
for cname, q in CLOUDS.items():
    a = CUTS[cname]
    for kind, fn, intdirs, feed in KINDS:
        for intdir in intdirs:
            at = feed(a)
            if kind == "qx" and intdir != "q":
                # get_qx_scan builds ragged Q ([array, scalar, scalar]) at :334/:341 and raises for any array
                # input; the coordinates it intends are those of the same expressions on broadcast cut arrays
                _, _, err = capture(fn, q, at, intdir, 1e9)
                assert err == "ValueError"
                h = xu.HXRD([1, 0, 0], [0, 0, 1], wl=WL)
                om, _, _, tt = h.Q2Ang(*q, trans=False, geometry="realTilt")
                cut = [q[0], np.full_like(q[0], at[0]), np.full_like(q[0], at[1])]
                oc, _, _, tc = h.Q2Ang(*cut, trans=False, geometry="realTilt")
                perp = om - oc if intdir == "omega" else tt - tc
                dis = pick_dis(perp, 0.3)
                cases.append(dict(cloud=cname, kind=kind, intdir=intdir, at=at, intrange=2 * dis, xu_error=err,
                                  along=clean(q[0]), perp=clean(perp),
                                  mask=[int(abs(v) < dis) for v in perp]))
                continue
            seen, _, err = capture(fn, q, at, intdir, 1e9)  # everything selected: record coordinates
            if err:
                raise SystemExit(f"{kind}/{intdir}/{cname}: {err}")
            perp = seen["perp"]
            ang = intdir != "q"
            # intrange chosen so about 30% of the finite pixels are selected; xu halves it
            dis = pick_dis(perp, 0.3)
            seen2, res, err = capture(fn, q, at, intdir, 2 * dis, INTENSITY[cname])
            assert err is None
            cases.append(dict(cloud=cname, kind=kind, intdir=intdir, at=at, intrange=2 * dis,
                              along=clean(seen2["along"]), perp=clean(seen2["perp"]),
                              mask=[int(v) for v in np.ravel(res[2])], grid=sums(seen2, res)))
            if (cname, kind, intdir) == ("peak", "qz", "q"):
                # XU-19: NaN intensity on selected pixels; xu's mask keeps them as used
                nan = INTENSITY[cname].copy()
                hit = np.nonzero(np.ravel(res[2]))[0][::3]
                nan[hit] = np.nan
                seen3, res3, err = capture(fn, q, at, intdir, 2 * dis, nan)
                assert err is None
                ua = seen3["along"][np.ravel(res3[2]) == 1]
                cases.append(dict(cloud=cname, kind=kind, intdir=intdir, at=at, intrange=2 * dis, nan_at=[int(i) for i in hit],
                                  mask=[int(v) for v in np.ravel(res3[2])],
                                  grid=dict(xmin=float(res3[0][0]), xmax=float(res3[0][-1]),
                                            width=float((ua.max() - ua.min()) / NPOINTS), fuzzy=[float(x) for x in res3[1]])))
            # XU-18: the window excludes every pixel
            tiny = 0.5 * np.nanmin(np.abs(perp))
            if tiny > 1e-6:
                _, _, err = capture(fn, q, at, intdir, 2 * tiny)
                cases.append(dict(cloud=cname, kind=kind, intdir=intdir, at=at, intrange=2 * tiny,
                                  error=err))
    # arbitrary line
    for point, vec in (([0.0, 1.8, 4.6], [0.1, 1.0, 2.0]), ([0.1, 2.0, 5.0], [1.0, 1.0, 1.0])):
        seen, res, err = capture(lambda qq, I, at, npts, ir, **kw: lc.get_arbitrary_line(qq, I, point, vec, npts, ir),
                                 q, None, None, 1e9)
        dis = pick_dis(seen["perp"], 0.3)
        seen2, res, err = capture(lambda qq, I, at, npts, ir, **kw: lc.get_arbitrary_line(qq, I, point, vec, npts, ir),
                                  q, None, None, dis, INTENSITY[cname])
        cases.append(dict(cloud=cname, kind="line", intdir="q", point=point, dir=vec, intrange=dis,
                          along=clean(seen2["along"]), perp=clean(seen2["perp"]),
                          mask=[int(v) for v in np.ravel(res[2])], grid=sums(seen2, res)))

doc = dict(xu_version=xu.__version__, wavelength=WL, npoints=NPOINTS,
           clouds={k: [list(map(float, c)) for c in v] for k, v in CLOUDS.items()},
           intensity={k: [float(x) for x in v] for k, v in INTENSITY.items()}, cases=cases)
with open(OUT, "w") as f:
    json.dump(doc, f, allow_nan=False)
print(len(cases), "cases")
