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
        p_h = np.stack(hklrs.detector_positions(geo, ang, WL, det), axis=-1)
        if dim == 1:
            q_h, p_h = q_h[0], p_h[0]
        q_x = xu_vectors(xu_q(*ang))
        p_x = xu_vectors(qc.getDetectorPos(*ang[n_sample:], dim=dim))
        assert q_h.shape == q_x.shape == p_h.shape == p_x.shape
        out.append((ang, q_x, q_h, p_x, p_h))
    return out


def assert_matches_xu(name, qc, det, dim, tag):
    for ang, q_x, q_h, p_x, p_h in frames(name, qc, det, dim, tag):
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
