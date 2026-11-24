"""hklrs against xrayutilities QConversion (point, area and linear detector).

xu frame: beam +y, z up, x completes the right-handed set. hkl/hklrs frame:
beam +x, z up. M maps xu coordinates to the hklrs frame: x_xu -> -y, y_xu -> x,
z_xu -> z, so a hklrs circle about -y is 'x+' in xu and one about +x is 'y+'.
"""
import math
import random

import numpy as np
import pytest

import _ref
import hklrs

xu = pytest.importorskip("xrayutilities")

M = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_I = (0, 1, 0)
# hklrs preset -> xu circles (same angle order: sample outer->inner, then detector)
XU_CIRCLES = {
    "E4CV": (("x+", "y+", "x+"), ("x+",)),
    "E6C": (("z+", "x+", "y+", "x+"), ("z+", "x+")),
    # XU-3: xu 'k+' turns left-handed, so the right-handed circle is 'k-'
    "K4CV": (("x+", "k-", "x+"), ("x+",)),
}
WL = 1.54
LAT = (4.1, 5.3, 6.7, 82.0, 97.0, 101.0)
U = _ref.rot((1.0, -2.0, 0.5), 0.9)


@pytest.fixture(autouse=True)
def kappa_axis_in_xu_frame():
    """xu takes the kappa direction from its config: hklrs's kappa axis
    (0, -cos 50, -sin 50) mapped to the xu frame is (cos 50, 0, -sin 50),
    i.e. plane 'xz' tilted by -50 degrees ('k+' base vector)."""
    old = xu.config.KAPPA_PLANE, xu.config.KAPPA_ANGLE
    xu.config.KAPPA_PLANE, xu.config.KAPPA_ANGLE = "xz", -50.0
    np.testing.assert_allclose(M @ xu.math.getVector("k+"), _ref.KAPPA, atol=1e-15)
    yield
    xu.config.KAPPA_PLANE, xu.config.KAPPA_ANGLE = old


def angles(name, n, seed):
    rng = random.Random(f"{name}-{seed}")
    na = len(hklrs.Geometry.preset(name).axis_names)
    return [[rng.uniform(-120, 120) for _ in range(na)] for _ in range(n)]


def sample_for(name):
    return hklrs.Sample("xu", hklrs.Lattice(*LAT), U.tolist())


@pytest.fixture(params=sorted(XU_CIRCLES))
def name(request):
    if request.param not in hklrs.presets():
        pytest.skip("preset not in hklrs")
    return request.param


def numpy_ref(name, ang, point=None, wl=WL):
    s, d = _ref.stacks_from_angles(name, ang)
    return _ref.forward(s, d, wl, point)


def test_point_q_sample(name):
    sc, dc = XU_CIRCLES[name]
    qc = xu.QConversion(sc, dc, R_I, wl=WL)
    geo = hklrs.Geometry.preset(name)
    smp = sample_for(name)
    for ang in angles(name, 30, "pt"):
        qx = np.array(qc.point(*ang))  # UB = identity: Q in the sample frame
        r = hklrs.forward(geo, smp, ang, WL)
        _, _, _, q_sample, _ = numpy_ref(name, ang)
        np.testing.assert_allclose(M @ qx, q_sample, atol=1e-12, err_msg="xu vs numpy")
        np.testing.assert_allclose(r["q_sample"], q_sample, atol=1e-12, err_msg="hklrs vs numpy")
        np.testing.assert_allclose(r["q_sample"], M @ qx, atol=1e-12)


def test_point_hkl_with_ub(name):
    """xu's hkl output with UB_xu = M^T U B against hklrs's hkl."""
    sc, dc = XU_CIRCLES[name]
    ub_h = U @ _ref.b_matrix(*LAT)
    qc = xu.QConversion(sc, dc, R_I, wl=WL, UB=M.T @ ub_h)
    geo = hklrs.Geometry.preset(name)
    smp = sample_for(name)
    for ang in angles(name, 30, "hkl"):
        hx = np.array(qc.point(*ang))
        r = hklrs.forward(geo, smp, ang, WL)
        np.testing.assert_allclose(r["hkl"], hx, atol=1e-9)


# hklrs's xu importer reads the direction strings in the frame of its beam, so
# it gets them mapped by M and the hklrs beam M r_i.
BEAM = (M @ R_I).tolist()
Q_TOL = 1e-11
POS_TOL = 1e-14


def hklrs_dir(d):
    """The xu direction string `d` in the hklrs frame."""
    v = M @ xu.math.getVector(d)
    i = int(np.argmax(np.abs(v)))
    return "xyz"[i] + ("+" if v[i] > 0 else "-")


def xu_vectors(xyz):
    """xu's (x, y, z) arrays as hklrs-frame vectors along a last axis."""
    return np.stack(xyz, axis=-1) @ M.T


# init_area / init_linear parameters, hklrs's names: cch, nch, pwidth and
# chpdeg are per detector direction; a None drops the key.
AREA_BASE = dict(dir1="z-", dir2="x+", cch=(4.0, 6.0), nch=(9, 13), distance=0.5,
                 pwidth=(50e-6, 50e-6))
AREA_CASES = {
    "tilt": dict(tilt=10.0),
    "tilt_tiltazimuth": dict(tilt=8.0, tiltazimuth=30.0),
    "detrot": dict(detrot=2.5),
    "roi": dict(roi=(1, 8, 3, 12)),
    "chpdeg": dict(distance=None, pwidth=None, chpdeg=(120.0, -95.5), tilt=5.0, tiltazimuth=40.0,
                   detrot=1.5, roi=(1, 8, 2, 12)),
    "combined": dict(dir1="x-", dir2="z+", cch=(4.3, 6.7), pwidth=(172e-6, 150e-6), tilt=-6.0,
                     tiltazimuth=120.0, detrot=-3.0, roi=(2, 8, 1, 11)),
}
LINEAR_BASE = dict(dir="z+", cch=4.0, nch=9, distance=1.0, pwidth=50e-6)
LINEAR_CASES = {
    "plain": {},
    "tilt": dict(tilt=10.0),
    "roi": dict(roi=(2, 8)),
    "chpdeg": dict(distance=None, pwidth=None, chpdeg=-80.0, tilt=4.0, roi=(2, 8)),
    "combined": dict(dir="x-", cch=3.6, pwidth=172e-6, tilt=-7.0, roi=(1, 7)),
}


def params(base, case):
    return {k: v for k, v in dict(base, **case).items() if v is not None}


def xu_area(name, p):
    qc = xu.QConversion(*XU_CIRCLES[name], R_I, wl=WL)
    kw = {k: p[k] for k in ("tilt", "tiltazimuth", "detrot") if k in p}
    if "roi" in p:
        kw["roi"] = list(p["roi"])
    if "nav" in p:
        kw["Nav"] = list(p["nav"])
    if "chpdeg" in p:
        kw.update(chpdeg1=p["chpdeg"][0], chpdeg2=p["chpdeg"][1])
    else:
        kw.update(distance=p["distance"], pwidth1=p["pwidth"][0], pwidth2=p["pwidth"][1])
    qc.init_area(p["dir1"], p["dir2"], *p["cch"], *p["nch"], **kw)
    return qc


def hklrs_area(p):
    kw = {k: p[k] for k in ("tilt", "tiltazimuth", "detrot", "nav", "roi") if k in p}
    d1, d2 = hklrs_dir(p["dir1"]), hklrs_dir(p["dir2"])
    if "chpdeg" in p:
        area = hklrs.xu.Area.from_chpdeg(d1, d2, p["cch"], p["nch"], p["chpdeg"], **kw)
    else:
        area = hklrs.xu.Area(d1, d2, p["cch"], p["nch"], p["distance"], p["pwidth"], **kw)
    return area.detector(BEAM)


def xu_linear(name, p):
    qc = xu.QConversion(*XU_CIRCLES[name], R_I, wl=WL)
    kw = {k: p[k] for k in ("tilt", "roi") if k in p}
    if "nav" in p:
        kw["Nav"] = p["nav"]
    if "chpdeg" in p:
        kw["chpdeg"] = p["chpdeg"]
    else:
        kw.update(distance=p["distance"], pixelwidth=p["pwidth"])
    qc.init_linear(p["dir"], p["cch"], p["nch"], **kw)
    return qc


def hklrs_linear(p):
    kw = {k: p[k] for k in ("tilt", "nav", "roi") if k in p}
    d = hklrs_dir(p["dir"])
    if "chpdeg" in p:
        linear = hklrs.xu.Linear.from_chpdeg(d, p["cch"], p["nch"], p["chpdeg"], **kw)
    else:
        linear = hklrs.xu.Linear(d, p["cch"], p["nch"], p["distance"], p["pwidth"], **kw)
    return linear.detector(BEAM)


def frames(name, qc, det, dim, tag):
    """Per goniometer angle set: the angles, xu's Q (Ang2Q.area / .linear)
    and detector positions (getDetectorPos), and hklrs's q_map and
    detector_positions, all hklrs-frame vectors; a linear detector is one row."""
    n_sample = len(XU_CIRCLES[name][0])
    geo = hklrs.Geometry.preset(name)
    xu_q = qc.area if dim == 2 else qc.linear
    out = []
    for ang in angles(name, 4, tag):
        q_h = np.stack(hklrs.q_map(geo, ang, WL, det), axis=-1)
        p_h = np.stack(hklrs.detector_positions(geo, ang, det), axis=-1)
        if dim == 1:
            q_h, p_h = q_h[0], p_h[0]
        q_x = xu_vectors(xu_q(*ang))
        p_x = xu_vectors(qc.getDetectorPos(*ang[n_sample:], dim=dim))
        out.append((ang, q_x, q_h, p_x, p_h))
    return out


def assert_matches_xu(name, qc, det, dim, tag):
    for ang, q_x, q_h, p_x, p_h in frames(name, qc, det, dim, tag):
        assert q_h.shape == q_x.shape == p_h.shape == p_x.shape
        np.testing.assert_allclose(q_h, q_x, rtol=0, atol=Q_TOL, err_msg=f"Q at {ang}")
        np.testing.assert_allclose(p_h, p_x, rtol=0, atol=POS_TOL, err_msg=f"position at {ang}")


def area_case(name, nch, pitch, dist, cch, tag):
    p = dict(dir1="z-", dir2="x+", cch=cch, nch=nch, distance=dist, pwidth=pitch)
    det = hklrs_area(p)
    assert det.shape == tuple(nch)
    d1 = np.array([0.0, 0.0, -1.0])  # xu 'z-' in the hklrs frame
    d2 = np.array([0.0, -1.0, 0.0])  # xu 'x+' in the hklrs frame
    worst = 0.0
    for ang, q_x, q_h, p_x, p_h in frames(name, xu_area(name, p), det, 2, tag):
        np.testing.assert_allclose(p_h, p_x, rtol=0, atol=POS_TOL, err_msg=f"position at {ang}")
        for i in range(nch[0]):
            for j in range(nch[1]):
                pt = dist * _ref.X + (i - cch[0]) * pitch[0] * d1 + (j - cch[1]) * pitch[1] * d2
                _, _, _, q_sample, _ = numpy_ref(name, ang, point=pt)
                np.testing.assert_allclose(q_x[i, j], q_sample, atol=1e-11,
                                           err_msg=f"xu vs numpy px({i},{j}) {ang}")
                np.testing.assert_allclose(q_h[i, j], q_sample, atol=1e-11,
                                           err_msg=f"hklrs vs numpy px({i},{j}) {ang}")
        worst = max(worst, np.abs(q_h - q_x).max())
    return worst


def test_area_small_detector(name):
    assert area_case(name, (20, 15), (75e-6, 75e-6), 0.7, (9.5, 7.0), "small") < 1e-11


def test_area_large_detector_square_and_rectangular_pixels(name):
    assert area_case(name, (40, 30), (172e-6, 100e-6), 0.25, (19.0, 14.5), "large") < 1e-11


@pytest.mark.parametrize("case", sorted(AREA_CASES))
def test_area_end_to_end(name, case):
    """xu's init_area parameters imported by hklrs, Ang2Q.area and
    getDetectorPos against q_map and detector_positions."""
    p = params(AREA_BASE, AREA_CASES[case])
    assert_matches_xu(name, xu_area(name, p), hklrs_area(p), 2, case)


def test_area_roi_of_the_whole_chip_is_the_imported_roi(name):
    """PixelDetector.roi on the imported chip gives xu's roi output."""
    p = params(AREA_BASE, AREA_CASES["roi"])
    r = p["roi"]
    det = hklrs_area(params(p, dict(roi=None))).roi(r[:2], r[2:])
    assert_matches_xu(name, xu_area(name, p), det, 2, "roi")


@pytest.mark.parametrize("case", sorted(LINEAR_CASES))
def test_linear_end_to_end(name, case):
    """xu's init_linear parameters imported by hklrs, Ang2Q.linear and
    getDetectorPos against q_map and detector_positions."""
    p = params(LINEAR_BASE, LINEAR_CASES[case])
    det = hklrs_linear(p)
    assert det.shape[0] == 1
    assert_matches_xu(name, xu_linear(name, p), det, 1, case)


def block_grid(cch, start, end, n):
    """Centre channel and count of the unbinned chip whose pixels are the
    centres of the Nav-blocks of [start, end)."""
    assert (end - start) % n == 0
    return (cch - start - (n - 1) / 2) / n, (end - start) // n


NAV_AREA = {"nav": dict(nch=(10, 12), nav=(2, 3)), "nav_roi": dict(nav=(2, 3), roi=(1, 9, 2, 11))}
NAV_LINEAR = {"nav": dict(nav=3), "nav_roi": dict(nav=2, roi=(1, 9))}


@pytest.mark.parametrize("case", sorted(NAV_AREA))
def test_xu13_area_nav_pixel_is_the_first_pixel_of_a_pixel_0_aligned_block(name, case):
    """XU-13: an hklrs Nav pixel is the centre of the raw block xu averages,
    as xu's unbinned chip on the block grid computes it; xu's own Nav pixel k
    sits on raw pixel Nav*(roi0//Nav + k). Fails when xu fixes it."""
    p = params(AREA_BASE, NAV_AREA[case])
    nav = p["nav"]
    r = p.get("roi", (0, p["nch"][0], 0, p["nch"][1]))
    grid = [block_grid(p["cch"][k], r[2 * k], r[2 * k + 1], nav[k]) for k in (0, 1)]
    reference = xu_area(name, params(p, dict(
        cch=tuple(g[0] for g in grid), nch=tuple(g[1] for g in grid),
        pwidth=tuple(w * n for w, n in zip(p["pwidth"], nav)), nav=None, roi=None)))
    det = hklrs_area(p)
    assert_matches_xu(name, reference, det, 2, case)
    whole = hklrs_area(params(p, dict(nav=None, roi=None)))
    assert_matches_xu(name, reference, whole.roi(r[:2], r[2:]).bin(*nav), 2, case)
    rows, cols = (nav[k] * (r[2 * k] // nav[k] + np.arange(grid[k][1])) for k in (0, 1))
    for ang, q_x, q_h, p_x, p_h in frames(name, xu_area(name, p), whole, 2, case):
        np.testing.assert_allclose(q_x, q_h[np.ix_(rows, cols)], rtol=0, atol=Q_TOL, err_msg=f"{ang}")
        np.testing.assert_allclose(p_x, p_h[np.ix_(rows, cols)], rtol=0, atol=POS_TOL)
    for _, q_x, q_h, _, _ in frames(name, xu_area(name, p), det, 2, case):
        assert np.abs(q_h - q_x).max() > 1e-6


@pytest.mark.parametrize("case", sorted(NAV_LINEAR))
def test_xu13_linear_nav_pixel_is_the_first_pixel_of_a_pixel_0_aligned_block(name, case):
    """XU-13 for a linear detector, as for the area above."""
    p = params(LINEAR_BASE, NAV_LINEAR[case])
    nav = p["nav"]
    r = p.get("roi", (0, p["nch"]))
    cch, n = block_grid(p["cch"], *r, nav)
    reference = xu_linear(name, params(p, dict(cch=cch, nch=n, pwidth=p["pwidth"] * nav,
                                               nav=None, roi=None)))
    det = hklrs_linear(p)
    assert_matches_xu(name, reference, det, 1, case)
    whole = hklrs_linear(params(p, dict(nav=None, roi=None)))
    assert_matches_xu(name, reference, whole.roi((0, 1), r).bin(1, nav), 1, case)
    pixels = nav * (r[0] // nav + np.arange(n))
    for ang, q_x, q_h, p_x, p_h in frames(name, xu_linear(name, p), whole, 1, case):
        np.testing.assert_allclose(q_x, q_h[pixels], rtol=0, atol=Q_TOL, err_msg=f"{ang}")
        np.testing.assert_allclose(p_x, p_h[pixels], rtol=0, atol=POS_TOL)
    for _, q_x, q_h, _, _ in frames(name, xu_linear(name, p), det, 1, case):
        assert np.abs(q_h - q_x).max() > 1e-6


def test_xu14_tilted_linear_along_r_i_is_rejected_where_xu_scales_its_pitch(name):
    """XU-14: a linear row along r_i has no tilt axis. hklrs rejects the tilt;
    xu zeroes the NaN axis, which scales the pitch by cos(tilt) and here moves
    pixels 0 and 1 from upstream of the sample to downstream. Fails when xu
    fixes it."""
    tilt = 60.0
    p = params(LINEAR_BASE, dict(dir="y+", pwidth=0.4, tilt=tilt))
    with pytest.raises(ValueError, match="parallel"):
        hklrs_linear(p)
    untilted = params(p, dict(tilt=None))
    scaled = params(untilted, dict(pwidth=p["pwidth"] * math.cos(math.radians(tilt))))
    assert_matches_xu(name, xu_linear(name, p), hklrs_linear(scaled), 1, "xu14")
    for _, q_x, q_h, _, _ in frames(name, xu_linear(name, p), hklrs_linear(untilted), 1, "xu14"):
        assert np.abs(q_h - q_x).max() > 1e-3


def test_xu16_roi_off_the_chip_is_rejected_where_xu_extrapolates(name):
    """XU-16: hklrs rejects a roi that leaves the chip; xu returns Q for the
    missing pixels, those of a larger chip. Fails when xu checks the roi."""
    area = params(AREA_BASE, dict(roi=(0, 12, 0, 15)))
    with pytest.raises(ValueError, match="outside the detector shape"):
        hklrs_area(area)
    larger = hklrs_area(params(area, dict(nch=(12, 15), roi=None)))
    assert_matches_xu(name, xu_area(name, area), larger, 2, "xu16")
    linear = params(LINEAR_BASE, dict(roi=(0, 12)))
    with pytest.raises(ValueError, match="outside the detector shape"):
        hklrs_linear(linear)
    longer = hklrs_linear(params(linear, dict(nch=12, roi=None)))
    assert_matches_xu(name, xu_linear(name, linear), longer, 1, "xu16")


def pixels(det):
    s, f = det.shape
    return np.array([[det.pixel(i, j) for j in range(f)] for i in range(s)])


def test_xu3_detector_tilt_turns_left_handed():
    """XU-3: rotation_arb turns left-handed, so xu's tilt about r_i x dir
    (linear, qconversion.c:431) and about -sin(az) dir1 + cos(az) dir2 (area,
    :598) is a right-handed turn by -tilt. hklrs imports the values unchanged
    and reproduces that turn. Fails when xu fixes rotation_arb."""
    t = 10.0
    r_i = np.array(R_I, float)

    p = params(LINEAR_BASE, dict(tilt=t))
    e = xu.math.getVector(p["dir"])
    k = np.arange(p["nch"]) - p["cch"]

    def row(turn):
        return p["distance"] * r_i + np.outer(k * p["pwidth"], turn @ e)

    axis = np.cross(r_i, e)
    got = np.stack(xu_linear("E4CV", p).getDetectorPos(0.0, dim=1), axis=-1)
    np.testing.assert_allclose(got, row(_ref.rot(axis, -math.radians(t))), rtol=0, atol=POS_TOL)
    assert np.abs(got - row(_ref.rot(axis, math.radians(t)))).max() > 1e-6
    np.testing.assert_allclose(pixels(hklrs_linear(p))[0], got @ M.T, rtol=0, atol=POS_TOL)

    p = params(AREA_BASE, dict(tilt=t, tiltazimuth=30.0))
    e1, e2 = xu.math.getVector(p["dir1"]), xu.math.getVector(p["dir2"])
    i, j = np.meshgrid(np.arange(p["nch"][0]) - p["cch"][0], np.arange(p["nch"][1]) - p["cch"][1],
                       indexing="ij")

    def chip(turn):
        return (p["distance"] * r_i + (i * p["pwidth"][0])[..., None] * (turn @ e1)
                + (j * p["pwidth"][1])[..., None] * (turn @ e2))

    az = math.radians(p["tiltazimuth"])
    axis = -math.sin(az) * e1 + math.cos(az) * e2
    got = np.stack(xu_area("E4CV", p).getDetectorPos(0.0, dim=2), axis=-1)
    np.testing.assert_allclose(got, chip(_ref.rot(axis, -math.radians(t))), rtol=0, atol=POS_TOL)
    assert np.abs(got - chip(_ref.rot(axis, math.radians(t)))).max() > 1e-6
    np.testing.assert_allclose(pixels(hklrs_area(p)), got @ M.T, rtol=0, atol=POS_TOL)


def test_kappa_plus_is_inverse_rotation():
    """XU-3: xu 'k+' is the inverse of the right-handed kappa rotation. Fails
    when xu fixes it; then drop the 'k-' workaround above."""
    qc = xu.QConversion(("x+", "k+", "x+"), ("x+",), R_I, wl=WL)
    for ang in angles("K4CV", 20, "k+"):
        s, d = _ref.stacks_from_angles("K4CV", ang)
        s[1] = (s[1][0], -s[1][1])
        _, _, _, q_sample, _ = _ref.forward(s, d, WL)
        np.testing.assert_allclose(M @ np.array(qc.point(*ang)), q_sample, atol=1e-12)
        s[1] = (s[1][0], -s[1][1])
        _, _, _, wrong, _ = _ref.forward(s, d, WL)
        if abs(ang[1]) > 1e-3 and abs(ang[1]) < 179.9:
            assert np.abs(M @ np.array(qc.point(*ang)) - wrong).max() > 1e-6


def test_surface_frame_is_xu_experiment_frame():
    """Sample.surface_frame against xu's HXRD(ipdir, ndir) transform, which
    keeps ndir and the part of ipdir normal to it (experiment.py:1581-1605).
    The default HXRD lab frame is the identity, so Transform is that
    surface frame in xu axes, which M relabels; in_plane is not normal to
    the surface in the sample frame."""
    s = sample_for("E4CV")
    ub = np.array(s.ub())
    normal, in_plane = (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)
    frame = np.array(s.surface_frame(normal, in_plane))
    with pytest.warns(UserWarning, match="not perpendicular"):
        hx = xu.HXRD(ub @ in_plane, ub @ normal, wl=WL)
    rng = np.random.default_rng(7)
    for v in rng.normal(size=(5, 3)):
        np.testing.assert_allclose(frame.T @ v, M @ hx.Transform(v), rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="parallel"):
        s.surface_frame(normal, (-2.0, -2.0, -2.0))
