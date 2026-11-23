"""hklrs against xrayutilities QConversion (point and area detector).

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
}
WL = 1.54
LAT = (4.1, 5.3, 6.7, 82.0, 97.0, 101.0)
U = _ref.rot((1.0, -2.0, 0.5), 0.9)


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


def area_case(name, nch, pitch, dist, cch, tag):
    sc, dc = XU_CIRCLES[name]
    qc = xu.QConversion(sc, dc, R_I, wl=WL)
    qc.init_area("z-", "x+", cch[0], cch[1], nch[0], nch[1],
                 distance=dist, pwidth1=pitch[0], pwidth2=pitch[1])
    geo = hklrs.Geometry.preset(name)
    smp = sample_for(name)
    d1 = np.array([0.0, 0.0, -1.0])  # xu 'z-' in the hklrs frame
    d2 = np.array([0.0, -1.0, 0.0])  # xu 'x+' in the hklrs frame
    worst = 0.0
    for ang in angles(name, 4, tag):
        qx, qy, qz = qc.area(*ang)
        assert qx.shape == tuple(nch)
        for i in range(nch[0]):
            for j in range(nch[1]):
                pt = dist * _ref.X + (i - cch[0]) * pitch[0] * d1 + (j - cch[1]) * pitch[1] * d2
                r = hklrs.forward(geo, smp, ang, WL, point=pt.tolist())
                _, _, _, q_sample, _ = numpy_ref(name, ang, point=pt)
                q_xu = M @ np.array([qx[i, j], qy[i, j], qz[i, j]])
                np.testing.assert_allclose(q_xu, q_sample, atol=1e-11,
                                           err_msg=f"xu vs numpy px({i},{j}) {ang}")
                np.testing.assert_allclose(r["q_sample"], q_sample, atol=1e-11,
                                           err_msg=f"hklrs vs numpy px({i},{j}) {ang}")
                worst = max(worst, np.abs(r["q_sample"] - q_xu).max())
    return worst


def test_area_small_detector(name):
    assert area_case(name, (20, 15), (75e-6, 75e-6), 0.7, (9.5, 7.0), "small") < 1e-11


def test_area_large_detector_square_and_rectangular_pixels(name):
    assert area_case(name, (40, 30), (172e-6, 100e-6), 0.25, (19.0, 14.5), "large") < 1e-11
