"""hklrs.cut against xrayutilities line_cuts (get_*_scan, live).

Counts are xu's intensity and the weight is 1, so counts / weights per bin
is xu's FuzzyGridder1D output (width footprint) and counts / contributions
its Gridder1D output (point footprint). xu's node-centred grid of `n`
points from xmin to xmax is the lattice of step (xmax - xmin) / (n - 1)
anchored half a step below xmin; node j is bin j. Angles are degrees on
both sides. Run with the py2 interpreter: python -I -m pytest compare
"""
import re

import numpy as np
import pytest

import hklrs
from hklrs import cut as hc

xu = pytest.importorskip("xrayutilities")
lc = pytest.importorskip("xrayutilities.analysis.line_cuts")
# (name, kind, step, anchor) of each axis in a "grids differ" message
AXES = r'\("(\w+)", "(\w+)", ([^,]+), ([^)]+)\)'

WL = 1.5406
NP = 40
rng = np.random.default_rng(20260101)
N = 400
CLOUDS = {
    "peak": [rng.uniform(-0.3, 0.3, N), rng.uniform(1.6, 2.4, N), rng.uniform(4.6, 5.4, N)],
    "wide": [rng.uniform(-0.4, 0.4, N), rng.uniform(-4.0, 4.0, N), rng.uniform(0.3, 3.0, N)],
}
POS = {"peak": [0.02, 2.0, 5.0], "wide": [0.05, 1.5, 1.8]}
INTEN = {k: np.round(rng.uniform(1.0, 100.0, N), 3) for k in CLOUDS}

# (kind, intdir, xu function, xu cutpos from (qx, qy, qz), hklrs `at`)
KINDS = [
    ("qz", "q", lc.get_qz_scan, lambda p: [p[0], p[1]], lambda p: [p[0], p[1], 0.0]),
    ("qz", "omega", lc.get_qz_scan, lambda p: [p[0], p[1]], lambda p: [p[0], p[1], 0.0]),
    ("qy", "q", lc.get_qy_scan, lambda p: [p[0], p[2]], lambda p: [p[0], 0.0, p[2]]),
    ("qy", "2theta", lc.get_qy_scan, lambda p: [p[0], p[2]], lambda p: [p[0], 0.0, p[2]]),
    ("qx", "q", lc.get_qx_scan, lambda p: [p[1], p[2]], lambda p: [0.0, p[1], p[2]]),
    ("omega", "2theta", lc.get_omega_scan, list, list),
    ("omega", "radial", lc.get_omega_scan, list, list),
    ("radial", "omega", lc.get_radial_scan, list, list),
    ("twotheta", "omega", lc.get_ttheta_scan, list, list),
]
ANGULAR_ALONG = {"omega", "radial", "twotheta"}


def xu_run(monkeypatch, fn, q, inten, cutpos, intdir, intrange):
    """(seen, result): seen holds the along/perp arrays and `dis` _get_cut got."""
    seen = {}
    real = lc._get_cut

    def spy(pa, pp, i, dis, npts):
        seen["along"], seen["perp"] = np.asarray(pa, float).ravel(), np.asarray(pp, float).ravel()
        seen["dis"] = dis
        return real(pa, pp, i, dis, npts)

    monkeypatch.setattr(lc, "_get_cut", spy)
    return seen, fn(q, inten, cutpos, NP, intrange, intdir=intdir, wl=WL)


def pick_dis(perp, frac=0.3):
    """A window holding `frac` of the finite pixels, midway between two distances."""
    a = np.sort(np.abs(perp[np.isfinite(perp)]))
    k = int(frac * len(a))
    return float((a[k] + a[k + 1]) / 2)


def build(kind, intdir, at, dis, xu_x):
    dx = (xu_x[-1] - xu_x[0]) / (NP - 1)
    return hc.Cut(kind, intdir, dis, (dx, xu_x[0] - dx / 2), at=at)


def accumulate(cut, q, inten, fp, reasons=None, weight=None):
    q = np.ascontiguousarray(np.column_stack(q), dtype=float)
    return cut.accumulate(
        WL, fp, q, np.asarray(inten, float),
        np.ones(len(q)) if weight is None else weight,
        np.zeros(len(q), np.uint8) if reasons is None else reasons,
    )


def dense(r, what):
    out = np.zeros(NP)
    for b, v in zip(r.bins, what):
        assert 0 <= b < NP, f"bin {b} off xu's grid"
        out[b] = v
    return out


def edge_safe(along, xmin, dx, width):
    for x in (along, along - width / 2, along + width / 2):
        fr = (x - xmin) / dx + 0.5
        assert np.all(np.abs(fr - np.round(fr)) > 1e-6), "pixel on a bin edge: floor and rint differ"


@pytest.fixture(params=[(c, k) for c in CLOUDS for k in range(len(KINDS))],
                ids=lambda p: f"{p[0]}-{KINDS[p[1]][0]}-{KINDS[p[1]][1]}")
def case(request, monkeypatch):
    cname, k = request.param
    kind, intdir, fn, cutpos, at = KINDS[k]
    q, pos, inten = CLOUDS[cname], POS[cname], INTEN[cname]
    seen, _ = xu_run(monkeypatch, fn, q, inten, cutpos(pos), intdir, 1e9)
    dis = pick_dis(seen["perp"])
    seen, res = xu_run(monkeypatch, fn, q, inten, cutpos(pos), intdir, 2 * dis)
    sel = np.abs(seen["perp"]) < seen["dis"]
    return dict(kind=kind, intdir=intdir, q=q, inten=inten, at=at(pos), dis=dis, seen=seen, res=res, sel=sel,
                cutpos=cutpos(pos), fn=fn)


def test_width_footprint_is_xu_fuzzy_gridder(case):
    c = case
    x, data, mask = c["res"]
    along = c["seen"]["along"][c["sel"]]
    width = (along.max() - along.min()) / NP
    cut = build(c["kind"], c["intdir"], c["at"], c["dis"], x)
    dx = (x[-1] - x[0]) / (NP - 1)
    edge_safe(along, x[0], dx, width)
    r = accumulate(cut, c["q"], c["inten"], hc.Footprint.width(width))
    # the ratio of the sums is xu's value; an empty bin is NaN here and 0 in xu
    np.testing.assert_allclose(dense(r, r.intensity), data, rtol=1e-9, atol=1e-12)
    assert not np.isnan(r.intensity).any()
    assert np.all(data[np.setdiff1d(np.arange(NP), r.bins)] == 0)
    np.testing.assert_allclose(r.centres, x[r.bins], rtol=0, atol=1e-9)
    assert int(mask.sum()) == int(c["sel"].sum())
    # hklrs counts every selected pixel once somewhere; xu's `mask` is a bin mask
    assert sum(r.skipped.values()) == N - int(c["sel"].sum())


def test_point_footprint_is_xu_gridder1d(case):
    c = case
    x = c["res"][0]
    along, inten = c["seen"]["along"][c["sel"]], c["inten"][c["sel"]]
    g = xu.Gridder1D(NP)
    g(along, inten)
    cut = build(c["kind"], c["intdir"], c["at"], c["dis"], x)
    r = accumulate(cut, c["q"], c["inten"], hc.Footprint.point())
    n = dense(r, r.contributions)
    np.testing.assert_array_equal(n, g._gnorm)
    # weight 1: the ratio of the sums is Gridder1D's mean
    np.testing.assert_allclose(dense(r, r.intensity), g.data, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(r.weights, r.contributions)  # weight 1


def test_xu_17_intrange_is_a_full_width(case):
    """xu halves intrange; the hklrs half_width is the half-width. Fed xu's
    intrange it selects the wider window xu's docstring suggests."""
    c = case
    x = c["res"][0]
    inside = int(c["sel"].sum())
    cut = build(c["kind"], c["intdir"], c["at"], 2 * c["dis"], x)
    r = accumulate(cut, c["q"], c["inten"], hc.Footprint.point())
    assert int(r.contributions.sum()) > inside
    assert int(c["res"][2].sum()) <= inside


def test_xu_18_empty_window_is_an_empty_result(monkeypatch):
    q, pos = CLOUDS["peak"], POS["peak"]
    with pytest.raises(ValueError):  # xu: max() of an empty selection
        xu_run(monkeypatch, lc.get_qz_scan, q, INTEN["peak"], pos[:2], "q", 1e-12)
    cut = hc.Cut("qz", "q", 1e-12 / 2, (0.01, 0.0), at=[pos[0], pos[1], 0.0])
    r = accumulate(cut, q, INTEN["peak"], hc.Footprint.point())
    assert len(r) == 0 and r.skipped["outside"] == N


def test_xu_19_nan_intensity_is_not_a_contribution(monkeypatch):
    q, pos, inten = CLOUDS["peak"], POS["peak"], INTEN["peak"].copy()
    seen, res = xu_run(monkeypatch, lc.get_qz_scan, q, inten, pos[:2], "q", 2 * 0.4)
    sel = np.nonzero(np.abs(seen["perp"]) < seen["dis"])[0]
    hit = sel[::3]
    inten[hit] = np.nan
    seen, res = xu_run(monkeypatch, lc.get_qz_scan, q, inten, pos[:2], "q", 2 * 0.4)
    x, _, mask = res
    cut = build("qz", "q", [pos[0], pos[1], 0.0], 0.4, x)
    r = accumulate(cut, q, inten, hc.Footprint.point())
    assert r.skipped["BadCount"] == len(hit)
    # the counts rank ahead of the weight: weight 0 is Opaque only where the counts are finite
    w = np.ones(N)
    w[hit] = w[sel[1]] = 0.0
    r0 = accumulate(cut, q, inten, hc.Footprint.point(), weight=w)
    assert (r0.skipped["BadCount"], r0.skipped["Opaque"]) == (len(hit), 1)
    # xu's mask is per pixel and still marks the NaN pixels used, though its gridder drops them
    assert int(mask.sum()) == len(sel) and mask.ravel()[hit].all()
    assert int(r.contributions.sum()) == len(sel) - len(hit)


def test_xu_25_qx_omega_and_2theta_are_computed_here(monkeypatch):
    q, pos = CLOUDS["peak"], POS["peak"]
    for intdir in ("omega", "2theta"):
        with pytest.raises(ValueError):  # xu: ragged Q built for array input
            xu_run(monkeypatch, lc.get_qx_scan, q, INTEN["peak"], pos[1:], intdir, 1.0)
        h = xu.HXRD([1, 0, 0], [0, 0, 1], wl=WL)
        om, _, _, tt = h.Q2Ang(*q, trans=False, geometry="realTilt")
        cut_q = [q[0], np.full(N, pos[1]), np.full(N, pos[2])]
        oc, _, _, tc = h.Q2Ang(*cut_q, trans=False, geometry="realTilt")
        perp = om - oc if intdir == "omega" else tt - tc
        dis = pick_dis(perp)
        cut = hc.Cut("qx", intdir, dis, (0.01, 0.0), at=[0.0, pos[1], pos[2]])
        r = accumulate(cut, q, INTEN["peak"], hc.Footprint.point())
        assert int(r.contributions.sum()) == int((np.abs(perp) < dis).sum())


def test_merge_of_two_halves_is_the_whole():
    q, inten = CLOUDS["peak"], INTEN["peak"]
    cut = hc.Cut("qz", "q", 0.4, (0.05, 0.0), at=[0.02, 2.0, 0.0])
    reasons = np.zeros(N, np.uint8)
    reasons[::7] = 1
    for fp in (hc.Footprint.point(), hc.Footprint.width(0.03)):
        whole = accumulate(cut, q, inten, fp, reasons)
        h = N // 3
        lo = accumulate(cut, [a[:h] for a in q], inten[:h], fp, reasons[:h])
        hi = accumulate(cut, [a[h:] for a in q], inten[h:], fp, reasons[h:])
        for m in (lo.merge(hi), hi.merge(lo)):
            np.testing.assert_array_equal(m.bins, whole.bins)
            np.testing.assert_array_equal(m.contributions, whole.contributions)
            np.testing.assert_allclose(m.counts, whole.counts, rtol=1e-12)
            np.testing.assert_allclose(m.weights, whole.weights, rtol=1e-12)
            assert m.skipped == whole.skipped
        assert whole.skipped["Unlit"] == len(reasons[::7])
    a = accumulate(cut, q, inten, hc.Footprint.point())
    # another binning or kind is another axis; another position or window
    # alone is another cut on the same axis
    refused = {
        "grids differ": [
            hc.Cut("qz", "q", 0.4, (0.1, 0.0), at=[0.02, 2.0, 0.0]),
            hc.Cut("qx", "q", 0.4, (0.05, 0.0), at=[0.02, 2.0, 0.0]),
            hc.Cut("omega", "2theta", 0.4, (0.05, 0.0), at=[0.02, 2.0, 5.0]),
        ],
        "different cuts": [
            hc.Cut("qz", "q", 0.4, (0.05, 0.0), at=[0.02, 2.1, 0.0]),
            hc.Cut("qz", "q", 0.3, (0.05, 0.0), at=[0.02, 2.0, 0.0]),
        ],
    }
    for match, others in refused.items():
        for other in others:
            with pytest.raises(ValueError, match=match):
                a.merge(accumulate(other, q, inten, hc.Footprint.point()))
    with pytest.raises(ValueError, match="footprints"):
        a.merge(accumulate(cut, q, inten, hc.Footprint.width(0.03)))


def test_merge_refusals_of_an_omega_cut_are_in_degrees():
    """An ω cut's binning, 2θ window and footprint went in as degrees and
    come back so in every merge refusal."""
    q, inten = CLOUDS["peak"], INTEN["peak"]

    def refused(a, b, fa=hc.Footprint.point(), fb=hc.Footprint.point()):
        with pytest.raises(ValueError) as e:
            accumulate(a, q, inten, fa).merge(accumulate(b, q, inten, fb))
        return str(e.value)

    def omega(half_width, step):
        return hc.Cut("omega", "2theta", half_width, (step, 0.25), at=[0.02, 2.0, 5.0])

    m = refused(omega(0.4, 0.05), omega(0.4, 0.1))
    assert m.startswith("grids differ"), m
    axes = [(n, k, float(s), float(a)) for n, k, s, a in re.findall(AXES, m)]
    assert axes == [("Omega", "angle", pytest.approx(s), pytest.approx(0.25)) for s in (0.05, 0.1)], m
    m = refused(omega(0.4, 0.05), omega(0.3, 0.05))
    assert m.startswith("cut results are of different cuts"), m
    # an ω cut keeps |at| alone
    cuts = re.findall(r'Cut\("omega", "2theta", ([^,]+), \(([^,]+), ([^)]+)\), at=\[0\.0, 0\.0, ([^\]]+)\]\)', m)
    norm = np.linalg.norm([0.02, 2.0, 5.0])
    assert [tuple(map(float, c)) for c in cuts] == [pytest.approx((w, 0.05, 0.25, norm)) for w in (0.4, 0.3)], m
    m = refused(omega(0.4, 0.05), omega(0.4, 0.05), hc.Footprint.width(0.03))
    widths = re.fullmatch(r"cut results have different footprints: Footprint\.width\(([^)]+)\) vs Footprint\.point\(\)", m)
    assert widths and float(widths[1]) == pytest.approx(0.03), m


def test_reason_wins_and_a_bad_weight_is_opaque():
    q = CLOUDS["peak"]
    cut = hc.Cut("qz", "q", 10.0, (1.0, 0.0), at=[0.0, 2.0, 0.0])
    w = np.ones(N)
    w[0], w[1], w[2] = 0.0, np.nan, -1.0
    reasons = np.zeros(N, np.uint8)
    reasons[2] = int(hklrs.Masked.Extinct)  # a given reason wins over the bad weight
    r = accumulate(cut, q, INTEN["peak"], hc.Footprint.point(), reasons, w)
    assert r.skipped["Opaque"] == 2 and r.skipped["Extinct"] == 1
    assert r.skipped["Opaque"] + r.skipped["Extinct"] + int(r.contributions.sum()) + r.skipped["outside"] \
        + r.skipped["no_coordinate"] == N


def test_intensity_is_nan_where_the_weight_sums_to_zero():
    # every weight is positive, so Σweight is 0 only by underflow: a quarter of the smallest
    # subnormal rounds to 0 while the counts (8 * 0.25) do not
    cut = hc.Cut("qz", "q", 1.0, (1.0, 0.0))
    r = cut.accumulate(WL, hc.Footprint.width(1.0), np.array([[0.0, 0.0, 1.25]]),  # [0.75, 1.75]
                       np.array([8.0]), np.array([5e-324]), np.zeros(1, np.uint8))
    np.testing.assert_array_equal(r.bins, [0, 1])
    assert r.weights[0] == 0.0 and r.counts[0] == 2.0 and r.contributions[0] == 1
    assert np.isnan(r.intensity[0]) and not np.isnan(r.intensity[1])
    # nothing accumulated: no bins, an empty array
    r = cut.accumulate(WL, hc.Footprint.point(), np.zeros((2, 3)), np.ones(2), np.ones(2),
                       np.array([1, 2], np.uint8))
    assert r.intensity.shape == (0,) and len(r) == 0
    r = cut.accumulate(WL, hc.Footprint.point(), np.zeros((2, 3)), np.array([6.0, 2.0]),
                       np.array([4.0, 4.0]), np.zeros(2, np.uint8))
    np.testing.assert_array_equal(r.intensity, [1.0])


def test_argument_errors():
    cut = hc.Cut("qz", "q", 1.0, (1.0, 0.0))
    q = np.zeros((4, 3))
    ok = (WL, hc.Footprint.point(), q, np.ones(4), np.ones(4), np.zeros(4, np.uint8))
    with pytest.raises(ValueError):
        cut.accumulate(WL, hc.Footprint.width(0.0), *ok[2:])
    with pytest.raises(ValueError):
        cut.accumulate(WL, hc.Footprint.point(), q, np.ones(3), np.ones(4), np.zeros(4, np.uint8))
    with pytest.raises(ValueError):
        cut.accumulate(WL, hc.Footprint.point(), q, np.ones(4), np.ones(4), np.full(4, 99, np.uint8))
    for bad in (("nope", "q"), ("qx", "radial"), ("line", "q")):
        with pytest.raises(ValueError):
            hc.Cut(*bad, 1.0, (1.0, 0.0))
    assert not hasattr(hc.CutResult, "mean")
    c = hc.Cut("omega", "2theta", 0.5, (0.1, 0.05), at=[0.0, 2.0, 5.0])
    assert (c.kind, c.intdir, c.half_width) == ("omega", "2theta", 0.5)
    assert c.binning == pytest.approx((0.1, 0.05))
