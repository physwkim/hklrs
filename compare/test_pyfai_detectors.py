"""hklrs.poni.PoniFile.detector against pyFAI 2026.9.0's named detector
classes (golden/pyfai_detectors.json): each file pyFAI wrote for a modelled
class, default or binned, gives pyFAI's shape, Detector.mask and
calc_pos_zyx; another class, a binning pyFAI's setter does not give the
class, a binned mask pyFAI cannot make and a 0 pitch pyFAI's constructor
does not take as the default's are ValueErrors."""
import re

import numpy as np
import pytest

import hklrs
from test_poni import LABS, TOL, load, positions, pyfai_positions

G = load("pyfai_detectors.json")
CLASSES = {c["class"]: c for c in G["classes"]}
MODELLED = [c for c in G["classes"] if c["layout"] in ("uniform", "edges")]


@pytest.fixture(params=sorted(LABS))
def lab(request):
    return LABS[request.param]


def fingerprint(det):
    """count, sum and sum of squares (mod 2**64) of the masked row-major indices"""
    if det.mask is None:
        return [0, 0, 0]
    k = np.flatnonzero(det.mask).astype(np.uint64)
    return [len(k), int(k.sum(dtype=np.uint64)), int((k * k).sum(dtype=np.uint64))]


def file(name, config):
    return hklrs.poni.parse(
        f"poni_version: 2.1\nDetector: {name}\nDetector_config: {config}\n"
        "Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\nRot3: 0.3\n"
        "Wavelength: 1e-10\n")[0]


def binned(c, b):
    return next(e for e in c["binned"] if e["binning"] == b)


@pytest.mark.parametrize("case", G["cases"],
                         ids=lambda c: f"{c['class']}-o{c['orientation']}-b{c['binning']}")
def test_pyfai_s_files_give_its_pixels(case, lab):
    c = CLASSES[case["class"]]
    f = hklrs.poni.parse(case["poni"])[0]
    with pytest.raises(ValueError, match=case["class"]):
        f.poni()
    if case["binning"] is None:
        shape, mask = c["shape"], c["fingerprint"]
    else:
        e = binned(c, case["binning"])
        shape, mask = e["shape"], e["mask"] or [0, 0, 0]
    det = f.detector(*lab)
    assert det.shape == tuple(shape)
    assert fingerprint(det) == mask
    np.testing.assert_allclose(positions(det, case["pixels"]),
                               pyfai_positions(lab, case["zyx"]), rtol=0, atol=TOL)


def test_every_modelled_class_has_a_file():
    assert {c["class"] for c in G["cases"]} == {c["class"] for c in MODELLED}


@pytest.mark.parametrize("c", [c for c in G["classes"]
                               if c["layout"] not in ("uniform", "edges", "generic")],
                         ids=lambda c: c["class"])
def test_other_classes_are_value_errors(c):
    reason = re.escape(f"{c['class']} is not supported: {c['layout']}")
    with pytest.raises(ValueError, match=reason):
        file(c["class"], "{}").detector(*LABS["hkl"])


@pytest.mark.parametrize("c", MODELLED, ids=lambda c: c["class"])
def test_binnings_are_pyfai_s(c):
    """each binning of the golden: pyFAI's binned shape and mask where its
    setter bins the class, else a ValueError naming the cause"""
    for e in c.get("binned", []):
        f = file(c["class"], f'{{"binning": {e["binning"]}}}')
        if not c["binnable"]:
            if e["binning"] == c["binning"]:
                assert f.detector(*LABS["hkl"]).shape == tuple(c["shape"])
            else:
                with pytest.raises(ValueError, match="binning is not the class's own"):
                    f.detector(*LABS["hkl"])
        elif 0 in e["shape"]:
            with pytest.raises(ValueError, match="non-zero"):
                f.detector(*LABS["hkl"])
        elif e["mask_error"]:
            with pytest.raises(ValueError, match="does not bin"):
                f.detector(*LABS["hkl"])
        else:
            det = f.detector(*LABS["hkl"])
            assert det.shape == tuple(e["shape"]), e["binning"]
            assert fingerprint(det) == (e["mask"] or [0, 0, 0]), e["binning"]


ZERO = {"pixel1": ('{"pixel1": 0}', "pixel1"), "pixel2": ('{"pixel2": 0}', "pixel2"),
        "both": ('{"pixel1": 0, "pixel2": 0}', "pixel1"),
        "pixel1 null": ('{"pixel1": null}', "pixel1"),
        "pixel2 null": ('{"pixel2": null}', "pixel2"), "version 1": (None, "pixel1")}


@pytest.mark.parametrize("c", MODELLED, ids=lambda c: c["class"])
def test_zero_pitches_are_pyfai_s(c):
    """the default instance where pyFAI's constructor gives it for a 0 or null
    pitch, else a ValueError on the key; version 1 gives a class that does
    not force its pitch the generic detector, which has no shape"""
    for form, outcome in c["zero_pitch"].items():
        config, key = ZERO[form]
        if config is None:
            f = hklrs.poni.parse(
                f"Detector: {c['class']}\nPixelSize1: 0\nPixelSize2: 0\nDistance: 0.3\n"
                "Poni1: 0.01\nPoni2: 0.02\nRot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n")[0]
        else:
            f = file(c["class"], config)
        if outcome == "default":
            assert f.detector(*LABS["hkl"]).shape == tuple(c["shape"]), form
        elif outcome == "generic":
            with pytest.raises(ValueError, match="no max_shape"):
                f.detector(*LABS["hkl"])
        else:
            with pytest.raises(ValueError, match=f"Detector_config {key} is not"):
                f.detector(*LABS["hkl"])
