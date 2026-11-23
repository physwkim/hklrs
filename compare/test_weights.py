"""hklrs.solid_angle and hklrs.polarisation against pyFAI 2026.9.0
Geometry.solidAngleArray and Geometry.polarization (golden/pyfai_weights.json),
on detectors imported from the PONI. pyFAI lives in another env, so only its
golden is read here."""
import json
import math
import os

import numpy as np
import pytest

import hklrs

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden",
                       "pyfai_weights.json")) as f:
    G = json.load(f)

# the hkl presets' lab: x along the beam, z up; pyFAI's axis 2 is up x beam
BEAM, UP = [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
A1, A2 = np.array(UP), np.cross(UP, BEAM)
GEOMETRY = "E4CV"
AXES = [0.0] * 4
# neither weight depends on it
WAVELENGTH = 1.0
SOLID_ANGLE_TOL = 1e-14


def detector(case):
    return hklrs.poni.Poni(
        G["dist"], (G["poni1"], G["poni2"]),
        tuple(math.degrees(case[k]) for k in ("rot1", "rot2", "rot3")),
        (G["pixel1"], G["pixel2"]), tuple(G["shape"]), case["orientation"]).detector(BEAM, UP)


def weighed(arrays):
    """The values of a (values, reasons) pair whose every pixel has one."""
    values, reasons = arrays
    assert (values.dtype, reasons.dtype) == (np.float64, np.uint8)
    assert values.shape == reasons.shape == tuple(G["shape"])
    assert not reasons.any() and np.isfinite(values).all()
    return values


def test_the_golden_covers_every_rotation_and_orientation():
    assert sorted((c["rots"], c["orientation"]) for c in G["cases"]) == sorted(
        [(r, 3) for r in ("none", "rot1", "rot2", "rot3", "all", "all_large")]
        + [("all", o) for o in (1, 2, 4)])


@pytest.mark.parametrize("case", G["cases"], ids=lambda c: f"{c['rots']}-{c['orientation']}")
def test_solid_angle_matches_pyfai_solid_angle_array(case):
    """solidAngleArray is cos^3 of the incidence, dist^3/|p|^3; hklrs's is
    pixel1 pixel2 dist/|p|^3, the same number once scaled by
    dist^2/(pixel1 pixel2)."""
    geo = hklrs.Geometry.preset(GEOMETRY)
    values = weighed(hklrs.solid_angle(geo, AXES, WAVELENGTH, detector(case)))
    scale = G["dist"] ** 2 / (G["pixel1"] * G["pixel2"])
    want = np.array(case["solid_angle"])
    np.testing.assert_allclose(values * scale, want, rtol=SOLID_ANGLE_TOL, atol=0)


@pytest.mark.parametrize("case", G["cases"], ids=lambda c: f"{c['rots']}-{c['orientation']}")
def test_polarisation_matches_pyfai_polarization(case):
    """pyFAI's axis_offset d turns its field to e = cos d a2 - sin d a1;
    pyFAI casts its result to float32."""
    geo, det = hklrs.Geometry.preset(GEOMETRY), detector(case)
    for pol in case["polarization"]:
        d = pol["axis_offset"]
        e = (math.cos(d) * A2 - math.sin(d) * A1).tolist()
        values = weighed(hklrs.polarisation(geo, AXES, WAVELENGTH, det, e, pol["factor"]))
        want = np.array(pol["values"])
        tol = want * np.finfo(np.float32).eps + 4 * np.finfo(np.float64).eps
        assert (np.abs(values - want) <= tol).all(), \
            f"factor {pol['factor']} axis_offset {d}: {np.abs(values - want).max()}"


def test_a_pixel_along_a_fully_polarised_field_is_extinct():
    # rot1 = 90 deg turns the beam axis onto a2, and the PONI is pixel (0, 0)
    det = hklrs.poni.Poni(0.1, (0.5e-4, 0.5e-4), (90, 0, 0), (1e-4, 1e-4), (3, 4), 3) \
        .detector(BEAM, UP)
    assert np.allclose(det.pixel(0, 0), [0.0, 0.1, 0.0], rtol=0, atol=1e-17)
    geo = hklrs.Geometry.preset(GEOMETRY)
    values, reasons = hklrs.polarisation(geo, AXES, WAVELENGTH, det, A2.tolist(), 1.0)
    assert reasons[0, 0] == hklrs.Masked.Extinct and math.isnan(values[0, 0])
    rest = np.ones(reasons.shape, bool)
    rest[0, 0] = False
    assert not reasons[rest].any() and (values[rest] > 0).all()


def test_masked_codes_are_the_reasons():
    assert [(m.name, int(m)) for m in hklrs.Masked] == [
        ("Unlit", 1), ("GrazingExit", 2), ("ThroughBulk", 3), ("Opaque", 4),
        ("EdgeOn", 5), ("Extinct", 6), ("Detector", 7), ("BadFlat", 8),
        ("BadCount", 9)]


def test_weights_refuse_what_has_none():
    geo = hklrs.Geometry.preset(GEOMETRY)
    det = detector(G["cases"][0])
    with pytest.raises(ValueError, match="polarisation factor"):
        hklrs.polarisation(geo, AXES, WAVELENGTH, det, A2.tolist(), 1.5)
    with pytest.raises(ValueError, match="normal to the beam"):
        hklrs.polarisation(geo, AXES, WAVELENGTH, det, BEAM, 0.5)
    # an xu linear detector's rows have no height, so its pixels no area
    linear = hklrs.xu.Linear("z+", 4.0, 9, 1.0, 50e-6).detector(BEAM)
    with pytest.raises(ValueError, match="area"):
        hklrs.solid_angle(geo, AXES, WAVELENGTH, linear)
