"""hklrs.sixs against binoculars-ng 7be4f79 (golden/bin_7be4f79_sixs.json):
corner and central pixel positions after calibration, on every table row.
Where binoculars' pixels differ before calibration (BIN-11, BIN-12) or its
central pixel keeps an x (BIN-9), the golden is moved by exactly those
differences. The golden's detrot is radians; hklrs.sixs takes degrees."""
import json
import math
import os

import numpy as np
import pytest

import hklrs

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden",
                       "bin_7be4f79_sixs.json")) as f:
    CASES = json.load(f)["cases"]

TOL = 1e-14
NAMES = hklrs.detectors()


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def calibrated(case):
    det = hklrs.detector(NAMES[case["id"]])
    centre = (case["iy0"], case["ix0"])
    return det, centre, hklrs.sixs(det, centre, case["sdd"], math.degrees(case["detrot"]))


def test_every_row_has_a_case():
    assert {c["id"] for c in CASES} == set(range(len(NAMES)))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{NAMES[c['id']]}-{c['detrot']}")
def test_sixs_matches_binoculars(case):
    """binoculars' calibrated pixel is rot (p_b + (sdd, -c_b.y, -c_b.z)),
    hklrs's rot (p_h + (sdd, 0, 0) - c_h); they differ by
    rot ((p_h - p_b) - (c_h - c_b) - c_b.x x)."""
    det, centre, out = calibrated(case)
    if det.mask is None:
        assert out.mask is None
    else:
        assert np.array_equal(out.mask, det.mask)
    rot = rx(case["detrot"])
    before = [((s, f), np.array(p)) for s, f, *p in case["before"]]
    assert before[4][0] == centre
    c_b, c_h = before[4][1], np.array(det.pixel(*centre))
    for (s, f, *want), (at, p_b) in zip(case["pixels"], before):
        assert (s, f) == at
        p_h = np.array(det.pixel(s, f))
        moved = rot @ ((p_h - p_b) - (c_h - c_b) - np.array([c_b[0], 0.0, 0.0]))
        err = np.abs(np.array(out.pixel(s, f)) - (np.array(want) + moved)).max()
        assert err < TOL, f"pixel ({s}, {f}) off by {err}"


@pytest.mark.parametrize("case", [c for c in CASES if NAMES[c["id"]] == "Cirpad"],
                         ids=lambda c: str(c["detrot"]))
def test_bin_9_cirpad_central_pixel_at_sdd(case):
    """binoculars leaves the central pixel at sdd plus its x before
    calibration."""
    _, centre, out = calibrated(case)
    sdd = case["sdd"]
    got = np.array(out.pixel(*centre))
    assert np.abs(got - rx(case["detrot"]) @ [sdd, 0.0, 0.0]).max() < TOL
    before, binoculars = case["before"][4][2:], case["pixels"][4][2:]
    assert abs(before[0]) > 1e-3
    assert abs(binoculars[0] - (sdd + before[0])) < TOL


def test_centre_outside_the_detector_is_an_error():
    with pytest.raises(ValueError):
        hklrs.sixs(hklrs.detector("Merlin"), (256, 0), 1.0, 0.0)


def test_non_finite_input_is_an_error():
    det = hklrs.detector("Merlin")
    for sdd, detrot in [(math.nan, 0.0), (1.0, math.inf)]:
        with pytest.raises(ValueError):
            hklrs.sixs(det, (0, 0), sdd, detrot)
