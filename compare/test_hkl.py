"""hklrs against libhkl 5.1.9 (gi, live) and the numpy reference.

libhkl's B is wrong unless beta = gamma = 90 (upstream-defects.md HKL-12), so
the geometry is compared through libhkl's own UB (UB_h . hkl_h = q_sample) and
hklrs's hkl is compared with the correct B.
"""
import math
import random

import numpy as np
import pytest

import _ref
import hklrs

gi = pytest.importorskip("gi")
gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

USER = Hkl.UnitEnum.USER
# user unit -> (kind, factor from canonical deg|m to the user unit)
UNIT = {"Degree": ("rot", 1.0), "Milli Radian": ("rot", math.radians(1) * 1000.0),
        "Milli Meter": ("trans", 1e3), "Meter": ("trans", 1.0)}
SAMPLES = {
    "cubic": (1.54, 1.54, 1.54, 90.0, 90.0, 90.0),
    "triclinic": (4.1, 5.3, 6.7, 82.0, 97.0, 101.0),
}
EULER_U = (23.0, -35.0, 58.0)
WAVELENGTHS = (1.54, 0.71)
N_RANDOM = 12


def mat(m):
    return np.array([[m.get(i, j) for j in range(3)] for i in range(3)])


def quat_matrix(q):
    return mat(q.to_matrix())


def hkl_case(name, lat, wl, user_vals):
    """libhkl's view of one configuration."""
    factories = Hkl.factories()
    if name not in factories:
        pytest.skip(f"preset {name} is not in the installed libhkl {Hkl.VERSION}; "
                    "its parity is the Rust golden hklrs/tests/hkl_golden.rs "
                    "(hkl 7be4f79 driver)")
    fac = factories[name]
    geo = fac.create_new_geometry()
    geo.wavelength_set(wl, USER)
    geo.axis_values_set(user_vals, USER)
    det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
    sample = Hkl.Sample.new("cmp")
    sample.lattice_set(Hkl.Lattice.new(lat[0], lat[1], lat[2], *(math.radians(x) for x in lat[3:])))
    sample.U_set(Hkl.Matrix.new_euler(*(math.radians(x) for x in EULER_U)))
    el = fac.create_new_engine_list()
    el.init(geo, det, sample)
    engines = {e.name_get(): e for e in el.engines_get()}
    el.get()
    out = {
        "U": mat(sample.U_get()), "UB": mat(sample.UB_get()),
        "rs": quat_matrix(geo.sample_rotation_get(sample)),
        "rd": quat_matrix(geo.detector_rotation_get(det)),
        "axes": list(geo.axis_names_get()),
        "units": [geo.axis_get(a).user_unit_get() for a in geo.axis_names_get()],
    }
    if "hkl" in engines:
        out["hkl"] = np.array(engines["hkl"].pseudo_axis_values_get(USER))
    if "q" in engines:
        out["q"] = engines["q"].pseudo_axis_values_get(USER)[0]
    return out


def configs(name):
    """Deterministic random axis sets for preset `name`: (lattice name,
    wavelength, canonical axes [deg|m], user-unit axes)."""
    rng = random.Random(f"hkl-{name}")
    probe = hkl_case(name, SAMPLES["cubic"], 1.54, [0.0] * len(hklrs.Geometry.preset(name).axis_names))
    units = probe["units"]
    for sname in SAMPLES:
        for wl in WAVELENGTHS:
            for _ in range(N_RANDOM):
                canon = [rng.uniform(-0.05, 0.05) if UNIT[u][0] == "trans"
                         else rng.uniform(-180.0, 180.0) for u in units]
                user = [c * UNIT[u][1] for c, u in zip(canon, units)]
                yield sname, wl, canon, user


def run(preset, sname, wl, canon, user):
    lat = SAMPLES[sname]
    h = hkl_case(preset, lat, wl, user)
    geo = hklrs.Geometry.preset(preset)
    assert h["axes"] == geo.axis_names, "axis order differs from libhkl"
    sample = hklrs.Sample("cmp", hklrs.Lattice(*lat), h["U"].tolist())
    r = hklrs.forward(geo, sample, canon, wl)
    return h, r, lat


def test_presets_listed():
    assert {"E4CV", "E4CH", "E6C", "K4CV"} <= set(hklrs.presets())


def test_preset_axes_match_libhkl(preset):
    h = hkl_case(preset, SAMPLES["cubic"], 1.54, [0.0] * len(hklrs.Geometry.preset(preset).axis_names))
    assert hklrs.Geometry.preset(preset).axis_names == h["axes"]


def test_forward_against_libhkl(preset):
    n = 0
    for sname, wl, canon, user in configs(preset):
        h, r, lat = run(preset, sname, wl, canon, user)
        k = 2 * math.pi / wl
        # libhkl's own rotations, numpy kinematics (beam +x, kf = Rd x)
        q_lab = k * (h["rd"] @ _ref.X - _ref.X)
        q_sample = h["rs"].T @ q_lab
        np.testing.assert_allclose(r["q_lab"], q_lab, atol=1e-12, err_msg=f"{preset} {sname} q_lab")
        np.testing.assert_allclose(r["q_sample"], q_sample, atol=1e-12, err_msg=f"{preset} {sname} q_sample")
        np.testing.assert_allclose(r["kf"], h["rd"] @ _ref.X, atol=1e-14)
        # libhkl's own hkl through libhkl's own UB (independent of B)
        if "hkl" in h:
            np.testing.assert_allclose(h["UB"] @ h["hkl"], q_sample, atol=1e-9,
                                       err_msg=f"{preset} {sname} UB_h hkl_h")
        if "q" in h:
            # libhkl's q engine is signed by the side of kf (kf_y < 0 or kf_z < 0)
            assert r["q"] == pytest.approx(abs(h["q"]), rel=1e-9, abs=1e-12)
        assert np.linalg.norm(r["q_lab"]) == pytest.approx(r["q"], rel=1e-9, abs=1e-12)
        n += 1
    assert n == len(SAMPLES) * len(WAVELENGTHS) * N_RANDOM


def test_hkl_against_correct_B(preset):
    for sname, wl, canon, user in configs(preset):
        h, r, lat = run(preset, sname, wl, canon, user)
        b = _ref.b_matrix(*lat)
        k = 2 * math.pi / wl
        q_sample = h["rs"].T @ (k * (h["rd"] @ _ref.X - _ref.X))
        expect = np.linalg.solve(h["U"] @ b, q_sample)
        np.testing.assert_allclose(r["hkl"], expect, atol=1e-9, err_msg=f"{preset} {sname}")
        if lat[4] == 90.0 and lat[5] == 90.0 and "hkl" in h:
            # libhkl's B is right here, so the two hkl must agree
            np.testing.assert_allclose(r["hkl"], h["hkl"], atol=1e-9)


def test_preset_kinematics_first_principles(preset):
    """hklrs against the hand-written stacks, no libhkl involved."""
    if preset not in _ref.PRESET_STACKS:
        pytest.skip(f"no hand-written stack for {preset}")
    rng = random.Random(f"fp-{preset}")
    geo = hklrs.Geometry.preset(preset)
    lat = SAMPLES["triclinic"]
    u = _ref.rot((1, 2, 3), 0.7)
    sample = hklrs.Sample("fp", hklrs.Lattice(*lat), u.tolist())
    for _ in range(20):
        ang = [rng.uniform(-180, 180) for _ in geo.axis_names]
        s, d = _ref.stacks_from_angles(preset, ang)
        rs, rd, q_lab, q_sample, kf = _ref.forward(s, d, 1.2)
        r = hklrs.forward(geo, sample, ang, 1.2)
        np.testing.assert_allclose(r["q_lab"], q_lab, atol=1e-12)
        np.testing.assert_allclose(r["q_sample"], q_sample, atol=1e-12)
        np.testing.assert_allclose(r["hkl"], np.linalg.solve(u @ _ref.b_matrix(*lat), q_sample), atol=1e-9)


@pytest.mark.parametrize("lat", list(SAMPLES.values()) + [(5, 5, 5, 70, 70, 70), (3, 4, 5, 90, 100, 90)])
def test_lattice_b_matrix(lat):
    b = np.array(hklrs.Lattice(*lat).b_matrix())
    np.testing.assert_allclose(b, _ref.b_matrix(*lat), atol=1e-12)
    np.testing.assert_allclose(b, _ref.b_matrix_busing_levy(*lat), atol=1e-12)
    g = _ref.metric(*lat)
    np.testing.assert_allclose(b.T @ b, (2 * math.pi) ** 2 * np.linalg.inv(g), atol=1e-12)
    for hkl in ([1, 0, 0], [0, 0, 1], [1, 2, 3]):
        d = 2 * math.pi / np.linalg.norm(b @ hkl)
        assert hklrs.Lattice(*lat).d_spacing(hkl) == pytest.approx(d, rel=1e-12)
