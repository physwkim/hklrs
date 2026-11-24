"""hklrs.poni.PoniFile.detector against pyFAI 2026.9.0's named detector
classes (golden/pyfai_detectors.json): each file pyFAI wrote for a modelled
class, default or binned, and each file of other pitches and binnings gives
pyFAI's shape, Detector.mask and calc_pos_zyx where Geometry.load builds an
instance whose pitch, binning and shape agree; another class, a file pyFAI
raises on or builds without a pitch or with disagreeing values, and a binned
mask pyFAI cannot make are ValueErrors."""
import re

import numpy as np
import pytest

import hklrs
from test_poni import LABS, TOL, load, positions, pyfai_positions

G = load("pyfai_detectors.json")
CLASSES = {c["class"]: c for c in G["classes"]}
MODELLED = [c for c in G["classes"] if c["layout"] in ("uniform", "edges")]
GEOMETRY = ("Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\nRot3: 0.3\n"
            "Wavelength: 1e-10\n")


@pytest.fixture(params=sorted(LABS))
def lab(request):
    return LABS[request.param]


def fingerprint(det):
    """count, sum and sum of squares (mod 2**64) of the masked row-major indices"""
    k = np.flatnonzero(det.mask).astype(np.uint64)
    return [len(k), int(k.sum(dtype=np.uint64)), int((k * k).sum(dtype=np.uint64))]


def file(name, config):
    return hklrs.poni.parse(
        f"poni_version: 2.1\nDetector: {name}\nDetector_config: {config}\n{GEOMETRY}")[0]


def agrees(c, e):
    """pyFAI's instance e of class c has the class's pitch at its binning
    and, for ImXPad, the default binning: a Rayonix's table
    pitch, a Mar345 plate within the class's max_shape, else the default
    instance's pitch scaled by the binning, to a relative 1e-9"""
    rule = c["pitch"]
    for k in range(2):
        if e["pixel"][k] is None:
            return False
        p, b, own = float(e["pixel"][k]), e["binning"][k], c["binning"][k]
        if rule["rule"] == "table":
            want = dict((t[0], t[1]) for t in rule["table"]).get(b)
        elif rule["rule"] == "plate":
            want = p if 1 <= e["max_shape"][k] <= rule["max_shape"][k] else None
        else:
            want = float(c["pixel"][k]) * b / own
        if not (b >= 1 and np.isfinite(p) and p > 0 and want is not None
                and abs(p - want) <= 1e-9 * want and (c["layout"] != "edges" or b == own)):
            return False
    return True


def rust_f64(text):
    """Rust's Debug of the float `text`: Python's repr, its exponent unpadded"""
    m = re.fullmatch(r"(.*)e([+-])0*(\d+)", repr(float(text)))
    return f"{m[1]}e{'-' if m[2] == '-' else ''}{m[3]}" if m else repr(float(text))


def refusal(c, e, outcome):
    """the ValueError text of a file of class c that pyFAI reads as e, its
    instance's facts, with outcome the exception the read or the mask
    raises; None for a detector"""
    name = c["class"]
    if "raises" in outcome:
        return f"pyFAI raises building detector {name} from the file"
    if e["class"] == "Detector":
        return "gives no max_shape"
    for k in range(2):
        if e["pixel"][k] is None:
            return f"pyFAI builds detector {name} from the file with no pixel{k + 1}"
    if not agrees(c, e):
        own = ", ".join(rust_f64(p) for p in c["pixel"])
        return f"pyFAI builds detector {name} of pitch .* which disagree; " + re.escape(
            f"{name} defines its own pixel, ({own}) at binning {tuple(c['binning'])}, "
            "and another pitch belongs on the generic Detector")
    if 0 in e["shape"]:
        return "detector shape must be non-zero"
    if "mask_error" in outcome:
        b = tuple(e["binning"])
        return re.escape(f"pyFAI detector {name} binned {b}") + ".* does not bin"
    return None


@pytest.mark.parametrize("case", G["cases"],
                         ids=lambda c: f"{c['class']}-o{c['orientation']}-b{c['binning']}")
def test_pyfai_s_files_give_its_pixels(case, lab):
    f = hklrs.poni.parse(case["poni"])[0]
    with pytest.raises(ValueError, match=case["class"]):
        f.poni()
    det = f.detector(*lab)
    assert det.shape == tuple(case["shape"])
    assert fingerprint(det) == case["fingerprint"]
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
def test_reads_are_pyfai_s_instance_or_value_errors(c):
    """each file the golden reads for the class (Detector_config pitches and
    binnings, version 1 pixel sizes): pyFAI's shape and mask, or the
    ValueError of what pyFAI does"""
    for e in c["reads"]:
        f = hklrs.poni.parse(e["head"] + GEOMETRY)[0]
        want = refusal(c, e, e)
        if want is None:
            det = f.detector(*LABS["hkl"])
            assert det.shape == tuple(e["shape"]), e["head"]
            assert fingerprint(det) == (e.get("mask") or [0, 0, 0]), e["head"]
        else:
            with pytest.raises(ValueError, match=want):
                f.detector(*LABS["hkl"])


@pytest.mark.parametrize("c", MODELLED, ids=lambda c: c["class"])
def test_round_trips_are_pyfai_s_read(c, lab):
    """PoniFile.write of the class binned by the factory, read back: pyFAI's
    calc_pos_zyx and mask of what Geometry.load builds, or its ValueError"""
    for e in c["round_trip"]:
        f = hklrs.poni.parse(e["poni"])[0]
        read = e.get("read", e)
        want = refusal(c, read, e)
        if want is not None:
            with pytest.raises(ValueError, match=want):
                f.detector(*lab)
            continue
        det = f.detector(*lab)
        assert det.shape == tuple(read["shape"]), e["binning"]
        assert fingerprint(det) == (e.get("mask") or [0, 0, 0]), e["binning"]
        np.testing.assert_allclose(positions(det, e["pixels"]),
                                   pyfai_positions(lab, e["zyx"]), rtol=0, atol=TOL)


GENERIC = next(c for c in G["classes"] if c["layout"] == "generic")


@pytest.mark.parametrize("e", GENERIC["reads"] + GENERIC["round_trip"],
                         ids=lambda e: e.get("head", "").split("config: ")[-1].strip()
                         or f"round trip {e['binning']}")
def test_generic_binnings_are_pyfai_s(e, lab):
    """the generic Detector with a binning key: pyFAI's binned pitch and shape
    at its calc_pos_zyx, a ValueError where its setter raises or the core
    refuses the instance; a round trip reads back as pyFAI reads it"""
    f = hklrs.poni.parse(e["poni"] if "poni" in e else e["head"] + GEOMETRY)[0]
    facts = e.get("read", e)
    if "raises" in e:
        want = "pyFAI raises building detector Detector from the file"
    elif 0 in facts["shape"]:
        want = "detector shape must be non-zero"
    elif float(facts["pixel"][0]) <= 0:
        want = "pixel1"
    else:
        want = None
    if want is not None:
        with pytest.raises(ValueError, match=want):
            f.detector(*lab)
        return
    det = f.detector(*lab)
    assert det.shape == tuple(facts["shape"])
    assert det.mask.shape == det.shape and not det.mask.any()
    np.testing.assert_allclose(positions(det, e["pixels"]),
                               pyfai_positions(lab, e["zyx"]), rtol=0, atol=TOL)
