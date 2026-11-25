"""Golden fuzzy-gridder sums from xrayutilities deff08f (1.8.0-11). Run with xu 1.8.0, whose
src/gridder{1,2,3}d.c, src/gridder_utils.c and FuzzyGridder __call__ paths are identical to deff08f.
Run: <xu python> -I xu_fuzzy.py

Each case grids n points (data = intensity, every point weight 1) with FuzzyGridder1D/2D/3D on a
fixed dataRange, one width per axis (or one scalar width), and records xu's raw sums _gdata and _gnorm
and its normalised .data, flattened in C order (x slowest). Points are drawn so every footprint lies
inside xu's grid, [min - d/2, max + d/2] per axis, and every centre inside [min, max], so xu neither
clamps nor drops; no footprint end lies within 1e-6 bins of a bin edge, where xu's rint and the floor
rule disagree. "defects" holds single-point cases of xu behaviour hklrs does not follow, each run
through xu: the edge bin receiving the part of a footprint beyond the grid, the range checks, a zero
width in 1D, and a NaN coordinate.
"""
import json
import os

import numpy as np
import xrayutilities as xu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_fuzzy.json")
rng = np.random.default_rng(20261125)
TIE = 1e-6

# (name, bins per axis, [min, max] per axis, width per axis in steps, n, scalar width)
CASES = [
    ("1d_narrow", [12], [[-1.0, 2.3]], [0.4], 300, True),
    ("1d_wide", [12], [[-1.0, 2.3]], [1.5], 300, True),
    ("1d_wider", [12], [[-1.0, 2.3]], [3.0], 300, True),
    ("2d_scalar", [8, 6], [[0.5, 3.0], [-2.0, -0.2]], [1.5, None], 400, True),
    ("2d_narrow_wide", [8, 6], [[0.5, 3.0], [-2.0, -0.2]], [0.4, 3.0], 400, False),
    ("2d_wide_mid", [8, 6], [[0.5, 3.0], [-2.0, -0.2]], [3.0, 1.5], 400, False),
    ("3d_scalar", [6, 5, 4], [[-0.5, 0.5], [1.0, 2.6], [10.0, 10.9]], [1.5, None, None], 500, True),
    ("3d_mixed", [6, 5, 4], [[-0.5, 0.5], [1.0, 2.6], [10.0, 10.9]], [1.5, 0.4, 3.0], 500, False),
    ("3d_wide", [6, 5, 4], [[-0.5, 0.5], [1.0, 2.6], [10.0, 10.9]], [3.0, 3.0, 1.5], 500, False),
]


def step(lo, hi, n):
    return (hi - lo) / (n - 1)


def widths(ns, ranges, rel, scalar):
    """Width per axis in coordinate units; a scalar width is one length on every axis."""
    if scalar:
        w = rel[0] * step(*ranges[0], ns[0])
        return [w] * len(ns)
    return [r * step(*rg, n) for r, rg, n in zip(rel, ranges, ns)]


def off_tie(v, lo, d):
    """True when v is farther than TIE bins from every edge lo - d/2 + k d."""
    t = (v - lo) / d + 0.5
    return np.abs(t - np.round(t)) > TIE


def draw(ns, ranges, ws, n):
    out = []
    for nb, (lo, hi), w in zip(ns, ranges, ws):
        d = step(lo, hi, nb)
        a, b = max(lo, lo - d / 2 + w / 2), min(hi, hi + d / 2 - w / 2)
        assert a < b, (lo, hi, w)
        v = rng.uniform(a, b, 4 * n)
        ok = off_tie(v - w / 2, lo, d) & off_tie(v + w / 2, lo, d)
        out.append(v[ok][:n])
        assert len(out[-1]) == n
    return out


def grid(ns, ranges, coords, data, width):
    """xu's fuzzy gridder of len(ns) dimensions on a fixed range; width as xu takes it."""
    cls = {1: xu.FuzzyGridder1D, 2: xu.FuzzyGridder2D, 3: xu.FuzzyGridder3D}[len(ns)]
    g = cls(*ns)
    g.dataRange(*[v for rg in ranges for v in rg])
    g(*[np.asarray(c, float) for c in coords], np.asarray(data, float), width=width)
    return g


def sums(g):
    return dict(gdata=[float(v) for v in g._gdata.ravel()], gnorm=[float(v) for v in g._gnorm.ravel()],
                normalised=[float(v) for v in g.data.ravel()])


cases = []
for name, ns, ranges, rel, n, scalar in CASES:
    ws = widths(ns, ranges, rel, scalar)
    coords = draw(ns, ranges, ws, n)
    data = np.round(rng.uniform(1.0, 100.0, n), 3)
    g = grid(ns, ranges, coords, data, ws[0] if scalar else tuple(ws))
    # every point inside: nothing dropped, nothing folded
    assert np.isclose(g._gnorm.sum(), n, rtol=1e-12), (name, g._gnorm.sum())
    cases.append(dict(name=name, bins=ns, ranges=ranges, width=ws, scalar=scalar,
                      coords=[[float(v) for v in c] for c in coords], data=[float(v) for v in data], **sums(g)))

# defects, on the grid 0..4 with step 1 per axis: bin k is [k - 0.5, k + 0.5)
R = [0.0, 4.0]
defects = {
    # x = 0.1, width 3: [-1.4, 1.6]; 0.3 of it is below bin 0, yet bin 0 gets 0.3 + 1/3
    "edge_overflow_1d": dict(x=0.1, width=3.0, **sums(grid([5], [R], [[0.1]], [1.0], 3.0))),
    "edge_overflow_2d": dict(x=0.1, y=2.1, width=[3.0, 0.4],
                             **sums(grid([5, 5], [R, R], [[0.1], [2.1]], [1.0], (3.0, 0.4)))),
    # x = -0.15, width 0.4: [-0.35, 0.05] lies in bin 0; 1D keeps it (x >= min - w/2), 2D drops it (x < min)
    "range_keep_1d": dict(x=-0.15, width=0.4, **sums(grid([5], [R], [[-0.15]], [1.0], 0.4))),
    "range_drop_2d": dict(x=-0.15, y=2.1, width=[0.4, 0.4],
                          **sums(grid([5, 5], [R, R], [[-0.15], [2.1]], [1.0], (0.4, 0.4)))),
    # x = -0.3, width 0.2: [-0.4, -0.2] lies in bin 0; 1D drops it too (x < min - w/2)
    "range_drop_1d": dict(x=-0.3, width=0.2, **sums(grid([5], [R], [[-0.3]], [1.0], 0.2))),
    # width 0: 1D takes half a step (gridder.py:354), 2D a point
    "zero_width_1d": dict(x=0.3, width=0.0, **sums(grid([5], [R], [[0.3]], [1.0], 0.0))),
    "zero_width_2d": dict(x=0.3, y=2.1, width=[0.0, 0.0],
                          **sums(grid([5, 5], [R, R], [[0.3], [2.1]], [1.0], 0.0))),
    # a NaN coordinate passes the range check and is cast by gindex (undefined behaviour; x86-64 here)
    "nan_coordinate_1d": dict(x=[None, 2.1], data=[7.0, 1.0], width=0.4,
                              **sums(grid([5], [R], [[np.nan, 2.1]], [7.0, 1.0], 0.4))),
}

with open(OUT, "w") as fh:
    json.dump(dict(xu=xu.__version__, cases=cases, defects=defects), fh, indent=1)
    fh.write("\n")
print(OUT, len(cases), "cases")
for k, v in defects.items():
    print(k, np.round(v["gnorm"], 6).tolist()[:5], np.round(v["gdata"], 6).tolist()[:5])
