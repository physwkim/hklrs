"""hklrs.absorption against the closed forms the Rust absorption tests use
(hklrs/src/absorption.rs), evaluated here in numpy at the pixel positions of
hklrs.detector_positions: the sample's F ∫₀ᵗ exp(−μz/s_i − μ L_f(z)) dz,
with L_f = z/s_f in reflection and (t − z)/|s_f| in transmission, times the
detector path's exp(−μ_air |p|) exp(−μ_w t_w/|cos|) (1 − exp(−μ_s t_s/|cos|)).
E4CV turns the sample by omega about −y; chi and phi stay 0."""
import math

import numpy as np
import pytest

import hklrs

GEOMETRY = hklrs.Geometry.preset("E4CV")
WAVELENGTH = 1.54
X, Z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
KI = X
TOL = 1e-13
MU = 1e4
FACE = 5e-3
# footprint inside the face at every incidence used here: F = 1/s_i
PENCIL = hklrs.Beam.top_hat(Z.tolist(), [1e-7, 1e-7])
AIR, WINDOW, SENSOR = 1.2, (1e4, 25e-6), (3e3, 450e-6)
# the Merlin centred on the arm 0.5 m out, normal to it; at 1 cm its pixels
# are seen up to 45 degrees off the normal
FAR = hklrs.sixs(hklrs.detector("Merlin"), (128, 128), 0.5, 0.0)
NEAR = hklrs.sixs(hklrs.detector("Merlin"), (128, 128), 0.01, 0.0)
CENTRE = (128, 128)
MASKED = {m.name: int(m) for m in hklrs.Masked}


def omega(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def slab(normal, thickness, length=FACE, width=FACE):
    axis = (Z if normal[2] == 0 else X).tolist()
    if thickness is None:
        return hklrs.Slab.bulk(list(normal), axis, length, width)
    return hklrs.Slab(list(normal), axis, length, width, thickness)


def phi(x):
    """expm1(x)/x, 1 at 0."""
    with np.errstate(invalid="ignore"):
        return np.where(x == 0, 1.0, np.expm1(x) / x)


def reference(axes, det, normal, thickness, footprint=None, air=None, window=None,
              sensor=None):
    """(values, reasons) by the closed forms; F defaults to the pencil's."""
    p = np.stack(hklrs.detector_positions(GEOMETRY, axes, det), -1)
    dist = np.linalg.norm(p, axis=-1)
    kf = p / dist[..., None]
    n = omega(axes[0]) @ (np.array(normal) / np.linalg.norm(normal))
    s_i, s_f = -KI @ n, kf @ n
    assert s_i > 0 and np.abs(s_f).min() > 1e-9, "a case the closed forms decide by sign"
    f = 1 / s_i if footprint is None else footprint
    b, c = MU / s_i, MU / np.abs(s_f)
    with np.errstate(over="ignore", under="ignore"):
        if thickness is None:
            sample = np.where(s_f > 0, f / (b + c), np.nan)
        else:
            t = thickness
            sample = np.where(s_f > 0, f * t * phi(-(b + c) * t),
                              f * t * np.exp(-np.minimum(b, c) * t) * phi(-np.abs(b - c) * t))
        path = np.ones_like(dist)
        if air is not None:
            path = path * np.exp(-air * dist)
        if window or sensor:
            normal_det = np.cross(p[1, 0] - p[0, 0], p[0, 1] - p[0, 0])
            cos = np.abs(kf @ (normal_det / np.linalg.norm(normal_det)))
            if window:
                path = path * np.exp(-window[0] * window[1] / cos)
            if sensor:
                path = path * -np.expm1(-sensor[0] * sensor[1] / cos)
        values = sample * path
    reasons = np.where(np.isnan(sample), MASKED["ThroughBulk"],
                       np.where(values > 0, 0, MASKED["Opaque"])).astype(np.uint8)
    return np.where(reasons == 0, values, np.nan), reasons


def absorption(axes, det, normal, thickness, beam=PENCIL, **path):
    absorber = hklrs.Absorber(slab(normal, thickness), beam, MU, **path)
    return hklrs.absorption(GEOMETRY, axes, WAVELENGTH, det, absorber)


def check(got, want):
    (gv, gr), (wv, wr) = got, want
    assert (gv.dtype, gr.dtype, gv.shape) == (np.float64, np.uint8, wv.shape)
    assert np.array_equal(gr, wr)
    ok = wr == 0
    np.testing.assert_allclose(gv[ok], wv[ok], rtol=TOL, atol=0)
    assert np.isnan(gv[~ok]).all()


PATH = {"air": AIR, "window": WINDOW, "sensor": SENSOR}
CASES = {
    # omega 15, 2theta 30: the centre is the symmetric reflection
    "bulk-reflection": ([15, 0, 0, 30], FAR, Z, None, {}),
    "slab-reflection": ([15, 0, 0, 30], FAR, Z, 50e-6, {}),
    "slab-transmission": ([0, 0, 0, 0], FAR, -X, 100e-6, {}),
    "slab-transmission-oblique": ([0, 0, 0, 20], NEAR, -X, 100e-6, PATH),
    # pixels below the face pass through the bulk; through the slab they
    # transmit
    "bulk-near": ([15, 0, 0, 30], NEAR, Z, None, PATH),
    "slab-near": ([15, 0, 0, 30], NEAR, Z, 50e-6, PATH),
    "slab-far-path": ([15, 0, 0, 30], FAR, Z, 50e-6, PATH),
    "tilted-sample": ([40, 0, 0, 60], NEAR, [1.0, 0.0, 2.0], 30e-6, {"sensor": SENSOR}),
    # e^-1000 in the slab, e^-5000 in the air
    "opaque-slab": ([0, 0, 0, 0], FAR, -X, 0.1, {}),
    "opaque-air": ([15, 0, 0, 30], FAR, Z, None, {"air": 1e4}),
}


@pytest.mark.parametrize("name", CASES)
def test_absorption_matches_the_closed_forms(name):
    axes, det, normal, thickness, path = CASES[name]
    check(absorption(axes, det, normal, thickness, **path),
          reference(axes, det, normal, thickness, **path))


def at_centre(arrays):
    values, reasons = arrays
    assert reasons[CENTRE] == 0
    return values[CENTRE]


def test_bulk_symmetric_reflection_is_one_over_two_mu():
    got = at_centre(absorption([15, 0, 0, 30], FAR, Z, None))
    assert abs(got * 2 * MU - 1) < TOL


def test_normal_transmission_is_t_exp_minus_mu_t():
    t = 100e-6
    got = at_centre(absorption([0, 0, 0, 0], FAR, -X, t))
    assert abs(got / (t * math.exp(-MU * t)) - 1) < TOL


def test_detector_path_at_normal_incidence():
    """0.5 m of air at 1.2/m, a window of optical depth 0.25 and a sensor of
    1.35, on the centre pixel, which faces the sample."""
    bare = at_centre(absorption([15, 0, 0, 30], FAR, Z, None))
    full = at_centre(absorption([15, 0, 0, 30], FAR, Z, None, **PATH))
    want = math.exp(-0.6) * math.exp(-0.25) * -math.expm1(-1.35)
    assert abs(full / bare / want - 1) < TOL


@pytest.mark.parametrize("om", [5, 15])
def test_top_hat_footprint_is_min_of_l_over_w_and_one_over_s_i(om):
    """A 1 mm face under a beam 0.1 mm high: at 5 degrees the footprint
    spills (F = L/w = 10), at 15 it does not (F = 1/s_i)."""
    length, w = 1e-3, 1e-4
    beam = hklrs.Beam.top_hat(Z.tolist(), [w, 1e-7])
    absorber = hklrs.Absorber(slab(Z, None, length=length), beam, MU)
    axes = [om, 0, 0, 2 * om]
    f = min(length / w, 1 / math.sin(math.radians(om)))
    check(hklrs.absorption(GEOMETRY, axes, WAVELENGTH, FAR, absorber),
          reference(axes, FAR, Z, None, footprint=f))


def test_gaussian_footprint_is_an_erf():
    """The face's projected length L s_i under a Gaussian of σ along z
    intercepts erf(L s_i / (2√2 σ)) of the flux."""
    length, fwhm, om = 1e-3, 1e-4, 5
    beam = hklrs.Beam.gaussian_fwhm(Z.tolist(), [fwhm, 1e-7])
    absorber = hklrs.Absorber(slab(Z, None, length=length), beam, MU)
    axes = [om, 0, 0, 2 * om]
    s_i, sigma = math.sin(math.radians(om)), fwhm / (2 * math.sqrt(2 * math.log(2)))
    f = math.erf(length * s_i / (2 * math.sqrt(2) * sigma)) / s_i
    check(hklrs.absorption(GEOMETRY, axes, WAVELENGTH, FAR, absorber),
          reference(axes, FAR, Z, None, footprint=f))


MASK_CASES = {
    # the face looks away from the beam; the air alone would be opaque
    "Unlit": ([0, 0, 0, 0], FAR, X, None, {"air": 1e4}),
    # the raw table Merlin lies in the plane x = 0, the face's, and is
    # seen edge-on through its window
    "GrazingExit": ([0, 0, 0, 0], hklrs.detector("Merlin"), -X, None, {"window": WINDOW}),
    "ThroughBulk": ([15, 0, 0, 0], FAR, Z, None, {}),
    "Opaque": ([15, 0, 0, 30], FAR, Z, None, {"air": 1e4}),
}


@pytest.mark.parametrize("reason", MASK_CASES)
def test_masked_pixels_carry_the_sample_reason_first(reason):
    axes, det, normal, thickness, path = MASK_CASES[reason]
    values, reasons = absorption(axes, det, normal, thickness, **path)
    assert (reasons == MASKED[reason]).all() and np.isnan(values).all()


def test_mu_is_the_same_at_every_wavelength():
    absorber = hklrs.Absorber(slab(Z, 50e-6), PENCIL, MU, **PATH)
    a, b = (hklrs.absorption(GEOMETRY, [15, 0, 0, 30], wl, NEAR, absorber)[0]
            for wl in (WAVELENGTH, 0.7))
    assert np.array_equal(a, b, equal_nan=True)


HC = 12.398419843320026  # keV Å, hklrs::units::HC_KEV_ANGSTROM
KEV = [5.0, 8.04778, 17.47934, 30.0, 80.0]
# formula, density (g/cm³); air is dry air at 20 °C by atoms
MATERIALS = {
    "Si": ([("Si", 1.0)], 2.33),
    "air": ([("N", 1.56168), ("O", 0.41892), ("Ar", 0.00934)], 1.205e-3),
    "Be": ([("Be", 1.0)], 1.848),
    "Kapton": ([("C", 22.0), ("H", 10.0), ("N", 2.0), ("O", 5.0)], 1.43),
}


@pytest.mark.parametrize("name", MATERIALS)
def test_material_mu_matches_xrt(name):
    """Against Python xrt on Chantler's total cross-sections, which takes eV
    and gives 1/cm."""
    xm = pytest.importorskip("xrt.backends.raycing.materials")
    formula, density = MATERIALS[name]
    elements, quantities = zip(*formula)
    want = xm.Material(list(elements), list(quantities), rho=density,
                       table="Chantler total")
    got = hklrs.Material(formula, density)
    for kev in KEV:
        wl = HC / kev
        per_cm = want.get_absorption_coefficient(np.array([HC / wl * 1e3]))[0]
        assert abs(got.mu(wl) / (per_cm * 100) - 1) < 1e-14, kev


def test_material_mu_follows_the_wavelength():
    """Each Material weighs as its mu at the beam energy given as a number."""
    si, air, be, kapton = (hklrs.Material(*MATERIALS[n]) for n in ("Si", "air", "Be", "Kapton"))
    by_material = hklrs.Absorber(slab(Z, 50e-6), PENCIL, si, air=air, window=(be, 25e-6),
                                 sensor=(kapton, 450e-6))
    got = {}
    for wl in (WAVELENGTH, 0.7, HC / 30.0):
        by_number = hklrs.Absorber(slab(Z, 50e-6), PENCIL, si.mu(wl), air=air.mu(wl),
                                   window=(be.mu(wl), 25e-6), sensor=(kapton.mu(wl), 450e-6))
        got[wl], reasons = hklrs.absorption(GEOMETRY, [15, 0, 0, 30], wl, NEAR, by_material)
        want = hklrs.absorption(GEOMETRY, [15, 0, 0, 30], wl, NEAR, by_number)
        assert not reasons.any()
        assert np.array_equal(got[wl], want[0]) and np.array_equal(reasons, want[1])
    a, b, c = got.values()
    assert not (np.array_equal(a, b) or np.array_equal(b, c))


def test_layers_need_a_pixel_area():
    """An xu linear detector's pixels have no area: air is fine, a window or
    sensor is an error."""
    linear = hklrs.xu.Linear("z+", 4.0, 9, 1.0, 50e-6).detector(X.tolist())
    axes = [0, 0, 0, 0]
    values, reasons = absorption(axes, linear, -X, 100e-6, air=AIR)
    assert not reasons.any() and np.isfinite(values).all()
    for layer in ({"window": WINDOW}, {"sensor": SENSOR}):
        with pytest.raises(ValueError, match="area"):
            absorption(axes, linear, -X, 100e-6, **layer)


Z_, X_ = Z.tolist(), X.tolist()
INVALID = {
    "zero-normal": lambda: hklrs.Slab([0, 0, 0], X_, FACE, FACE, 1e-4),
    "axis-along-normal": lambda: hklrs.Slab(Z_, Z_, FACE, FACE, 1e-4),
    "zero-length": lambda: hklrs.Slab(Z_, X_, 0.0, FACE, 1e-4),
    "negative-thickness": lambda: hklrs.Slab(Z_, X_, FACE, FACE, -1e-4),
    "nan-width": lambda: hklrs.Slab.bulk(Z_, X_, FACE, math.nan),
    "zero-beam-width": lambda: hklrs.Beam.top_hat(Z_, [0.0, 1e-4]),
    "infinite-fwhm": lambda: hklrs.Beam.gaussian_fwhm(Z_, [1e-4, math.inf]),
    "zero-mu": lambda: hklrs.Absorber(slab(Z, None), PENCIL, 0.0),
    "negative-air": lambda: hklrs.Absorber(slab(Z, None), PENCIL, MU, air=-1.0),
    "zero-window": lambda: hklrs.Absorber(slab(Z, None), PENCIL, MU, window=(1e4, 0.0)),
    "nan-sensor-mu": lambda: hklrs.Absorber(slab(Z, None), PENCIL, MU, sensor=(math.nan, 1e-4)),
    # a profile axis along the beam has no part across it
    "beam-axis-along-the-beam": lambda: absorption(
        [15, 0, 0, 30], FAR, Z, None, beam=hklrs.Beam.top_hat(X_, [1e-4, 1e-4])),
    "zero-wavelength": lambda: hklrs.absorption(
        GEOMETRY, [15, 0, 0, 30], 0.0, FAR, hklrs.Absorber(slab(Z, None), PENCIL, MU)),
    "unknown-element": lambda: hklrs.Material([("Xx", 1.0)], 1.0),
    "empty-formula": lambda: hklrs.Material([], 1.0),
    "zero-density": lambda: hklrs.Material([("Si", 1.0)], 0.0),
    "negative-quantity": lambda: hklrs.Material([("Si", 1.0), ("O", -2.0)], 2.2),
    # 620 keV, past the end of every table at 432.9 keV
    "material-past-its-table": lambda: hklrs.Material(*MATERIALS["Si"]).mu(0.02),
    "air-past-its-table": lambda: hklrs.absorption(
        GEOMETRY, [15, 0, 0, 30], 0.02, FAR,
        hklrs.Absorber(slab(Z, None), PENCIL, MU, air=hklrs.Material(*MATERIALS["air"]))),
}


@pytest.mark.parametrize("make", INVALID.values(), ids=list(INVALID))
def test_invalid_input_is_an_error(make):
    with pytest.raises(ValueError):
        make()
