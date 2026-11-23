"""hklrs.poni against pyFAI 2026.9.0: Geometry.calc_pos_zyx
(golden/pyfai_poni.json) and the .poni files pyFAI wrote and read back
(golden/pyfai_poni_files.json, golden/poni/). pyFAI lives in another env, so
only its goldens are read here."""
import json
import math
import os

import numpy as np
import pytest

import hklrs

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden")
# metres; the pixels are about 0.3 m from the sample
TOL = 1e-14


def load(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


ZYX = load("pyfai_poni.json")
FILES = load("pyfai_poni_files.json")

_beam = np.array([0.3, -0.5, 0.81]) / np.linalg.norm([0.3, -0.5, 0.81])
_up = np.cross(_beam, [0.2, 0.7, -0.4])
# beam and up, up normal to the beam: the hkl presets' lab and an oblique one
LABS = {
    "hkl": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    "oblique": (_beam, _up / np.linalg.norm(_up)),
}


@pytest.fixture(params=sorted(LABS))
def lab(request):
    return LABS[request.param]


def pyfai_positions(lab, zyx):
    """pyFAI's (t3, t1, t2) of each pixel as lab vectors: axis 3 along the
    beam, axis 1 up and axis 2 = axis 1 x axis 3."""
    beam, up = lab
    t3, t1, t2 = np.array(zyx).T
    return np.outer(t1, up) + np.outer(t2, np.cross(up, beam)) + np.outer(t3, beam)


def positions(det, pixels):
    return np.array([det.pixel(i, j) for i, j in pixels])


def golden_poni(case):
    return hklrs.poni.Poni(
        ZYX["dist"], (ZYX["poni1"], ZYX["poni2"]),
        tuple(math.degrees(case[k]) for k in ("rot1", "rot2", "rot3")),
        (ZYX["pixel1"], ZYX["pixel2"]), tuple(ZYX["shape"]), case["orientation"])


def test_import_matches_pyfai_calc_pos_zyx(lab):
    seen = set()
    for case in ZYX["cases"]:
        det = golden_poni(case).detector(*lab)
        assert det.shape == tuple(ZYX["shape"])
        np.testing.assert_allclose(positions(det, ZYX["pixels"]), pyfai_positions(lab, case["zyx"]),
                                   rtol=0, atol=TOL, err_msg=f"{case['orientation']} {case['rots']}")
        seen.add((case["orientation"], case["rots"]))
    assert seen == {(o, r) for o in range(1, 5)
                    for r in ("none", "rot1", "rot2", "rot3", "all", "all_large")}


def test_export_gives_back_the_golden_fields(lab):
    """Every golden rotation lies in the export's ranges, so the export of
    the imported detector is the golden PONI itself."""
    for case in ZYX["cases"]:
        p = golden_poni(case)
        back = hklrs.poni.Poni.from_detector(p.detector(*lab), p.orientation, *lab)
        assert (back.shape, back.orientation) == (p.shape, p.orientation)
        for k in ("dist", "poni1", "poni2", "pixel1", "pixel2"):
            assert getattr(back, k) == pytest.approx(getattr(p, k), rel=0, abs=1e-15), k
        for k in ("rot1", "rot2", "rot3"):
            assert getattr(back, k) == pytest.approx(getattr(p, k), rel=0, abs=1e-12), k


def test_export_in_any_orientation_places_the_same_pixels(lab):
    for case in ZYX["cases"]:
        det = golden_poni(case).detector(*lab)
        for o in range(1, 5):
            back = hklrs.poni.Poni.from_detector(det, o, *lab)
            assert back.orientation == o
            np.testing.assert_allclose(positions(back.detector(*lab), ZYX["pixels"]),
                                       pyfai_positions(lab, case["zyx"]), rtol=0, atol=TOL)


def test_import_and_export_refuse_what_has_no_pyfai_axes():
    p = golden_poni(ZYX["cases"][0])
    with pytest.raises(ValueError, match="orientation 5"):
        hklrs.poni.Poni(p.dist, (p.poni1, p.poni2), (0, 0, 0), (p.pixel1, p.pixel2), p.shape, 5)
    beam = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="along the beam"):
        p.detector(beam, [-2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="non-zero"):
        p.detector(beam, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="orientation 0"):
        hklrs.poni.Poni.from_detector(p.detector(beam, [0, 0, 1]), 0, beam, [0, 0, 1])


def read(name):
    with open(os.path.join(GOLDEN, "poni", name)) as f:
        return hklrs.poni.parse(f.read())


@pytest.mark.parametrize("entry", FILES["files"], ids=lambda e: e["file"])
def test_parse_reads_pyfai_s_own_files(entry, lab):
    """Each field as pyFAI read the file back, and every pixel where
    pyFAI's Geometry.load of it puts it."""
    f, wavelength = read(entry["file"])
    assert (f.version, f.detector_name, f.parallax) == (entry["version"], entry["detector"],
                                                   entry["parallax"])
    assert (f.pixel1, f.pixel2) == (entry["pixel1"], entry["pixel2"])
    assert (f.orientation or 3) == entry["orientation"]
    # no binning key: the class's default instance keeps its own
    assert (f.binning, f.splinefile) == (None, None)
    for k in ("dist", "poni1", "poni2"):
        assert getattr(f, k) == entry[k], k
    for k in ("rot1", "rot2", "rot3"):
        assert math.radians(getattr(f, k)) == pytest.approx(entry[k], rel=2e-16, abs=0), k
    if entry["wavelength"] is None:
        assert wavelength is None
    else:
        assert wavelength * 1e-10 == pytest.approx(entry["wavelength"], rel=2.5e-16, abs=0)

    if f.parallax:
        with pytest.raises(ValueError, match="parallax"):
            f.poni()
        return
    if f.detector_name == "Detector":
        assert f.max_shape == (tuple(entry["max_shape"]) if entry["max_shape"] else None)
    else:
        # pyFAI's class gives a named detector its shape, not the file
        assert f.max_shape is None
        with pytest.raises(ValueError, match=f.detector_name):
            f.poni()
    if f.max_shape is None:
        if f.detector_name == "Detector":
            # version 1 has no shape, which orientation 3 does not need
            with pytest.raises(ValueError, match="max_shape"):
                f.poni()
        shape = tuple(entry["max_shape"] or (213, 347))
        p = hklrs.poni.Poni(f.dist, (f.poni1, f.poni2), (f.rot1, f.rot2, f.rot3),
                            (entry["pixel1"], entry["pixel2"]), shape, entry["orientation"])
    else:
        p = f.poni()
    np.testing.assert_allclose(positions(p.detector(*lab), FILES["pixels"]),
                               pyfai_positions(lab, entry["zyx"]), rtol=0, atol=TOL)

    back = hklrs.poni.Poni.from_detector(p.detector(*lab), p.orientation, *lab)
    for k in ("dist", "poni1", "poni2"):
        assert getattr(back, k) == pytest.approx(entry[k], rel=0, abs=1e-15), f"export {k}"
    for k in ("rot1", "rot2", "rot3"):
        assert math.radians(getattr(back, k)) == pytest.approx(entry[k], rel=0, abs=1e-14), \
            f"export {k}"


def test_the_files_cover_every_version():
    seen = {e["file"]: e["version"] for e in FILES["files"]}
    assert {1.0, 2.0, 2.1, 3.0} <= set(seen.values())
    assert {"v1_unversioned.poni", "pilatus1m.poni", "v3_parallax.poni"} <= set(seen)
    assert set(seen) == set(os.listdir(os.path.join(GOLDEN, "poni")))
