"""hklrs.detector against binoculars-ng 7be4f79 (golden/bin_7be4f79_detectors.json):
shape, pixel positions and mask of every table row. Equal except where
binoculars is wrong: the Pilatus CdTe 2M centre strips (BIN-1), the imXpad
chip-end pixels (BIN-12) and the single-precision Cirpad placement (BIN-11)."""
import json
import os

import numpy as np
import pytest

import hklrs

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "golden",
                       "bin_7be4f79_detectors.json")) as f:
    ROWS = json.load(f)["detectors"]

POS_TOL = 1e-14
# BIN-11: binoculars places the Cirpad modules in float; 1.106e-7 m at most
# over the golden samples
CIRPAD_TOL = 1.11e-7
# the imXpad rows: chips of (slow, fast) pixels at one pitch, rows along +z
IMXPAD = {"ImXpadS140", "ImXpadS70"}
CHIP, PITCH = (120, 80), 130e-6
# a Cirpad module is an S70: one chip high
CIRPAD_MODULE = (120, 560)


def bin_12_offset(n, chip):
    """BIN-12: binoculars' minus pyFAI's centre of each of `n` pixels in chips
    of `chip`, in pitches: binoculars puts the last pixel of a chip at its far
    edge, and makes the detector's last pixel wide too."""
    off = np.where(np.arange(n) % chip == chip - 1, 1.25, 0.0)
    off[-1] = 2.0
    return off


def checked_indices(n, chip):
    """Both ends, the middle and both sides of every chip boundary."""
    edges = {k * chip + d for k in range(1, n // chip + 1) for d in (-1, 0)} if chip else set()
    return sorted({0, 1, n // 2, n - 2, n - 1} | {e for e in edges if e < n})


def golden_mask(row):
    mask = np.zeros(row["height"] * row["width"], bool)
    for start, n in row["mask_runs"]:
        mask[start:start + n] = True
    return mask.reshape(row["height"], row["width"])


def test_the_table_is_binoculars_in_id_order():
    assert [r["id"] for r in ROWS] == list(range(len(ROWS)))
    assert hklrs.detectors() == [r["name"] for r in ROWS]


def test_unknown_name_is_an_error():
    with pytest.raises(ValueError, match="unknown detector"):
        hklrs.detector("Pilatus")


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_shape(row):
    assert hklrs.detector(row["name"]).shape == (row["height"], row["width"])


@pytest.mark.parametrize("row", [r for r in ROWS if r["name"] != "Cirpad"],
                         ids=lambda r: r["name"])
def test_pixel_positions(row):
    """Pixel (j, i) is (0, y[i], z[j]); every row is checked over all columns
    at the sampled rows and all rows at the sampled columns."""
    (h, w), name = (row["height"], row["width"]), row["name"]
    y, z = np.array(row["y"]), np.array(row["z"])
    chip = CHIP if name in IMXPAD else None
    if chip:
        y = y + bin_12_offset(w, chip[1]) * PITCH
        z = z - bin_12_offset(h, chip[0]) * PITCH
    det = hklrs.detector(name)
    js, is_ = checked_indices(h, chip and chip[0]), checked_indices(w, chip and chip[1])
    pairs = [(j, i) for j in js for i in range(w)] + [(j, i) for j in range(h) for i in is_]
    got = np.array([det.pixel(j, i) for j, i in pairs])
    want = np.array([(0.0, y[i], z[j]) for j, i in pairs])
    err = np.abs(got - want).max()
    assert err < POS_TOL, f"{name}: off by {err}"


def test_bin_11_cirpad_placed_in_single_precision():
    """Each module's chip-end pixels are also BIN-12's, along its own slow and
    fast directions."""
    row = next(r for r in ROWS if r["name"] == "Cirpad")
    det = hklrs.detector("Cirpad")
    mh, mw = CIRPAD_MODULE
    slow_off, fast_off = bin_12_offset(mh, CHIP[0]), bin_12_offset(mw, CHIP[1])

    def direction(a, b):
        d = np.array(det.pixel(*b)) - np.array(det.pixel(*a))
        return d / np.linalg.norm(d)

    worst = 0.0
    assert row["samples"]
    for j, i, *want in row["samples"]:
        top = j - j % mh
        slow, fast = direction((top, 0), (top + 1, 0)), direction((top, 0), (top, 1))
        bin_12 = (slow * slow_off[j - top] + fast * fast_off[i]) * PITCH
        err = np.abs(np.array(want) - np.array(det.pixel(j, i)) - bin_12).max()
        worst = max(worst, err)
    assert 1e-9 < worst <= CIRPAD_TOL, worst


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["name"])
def test_mask(row):
    """None when no pixel is masked. Pilatus: binoculars steps the centre
    strip by the module width (487), not the module pitch (487 + 7 gap), so
    modules 1 and 2 lose columns 729..732 and 1216..1219 where 736..739 and
    1230..1233 are right (BIN-1)."""
    want = golden_mask(row)
    got = hklrs.detector(row["name"]).mask
    if not want.any():
        assert got is None
        return
    assert got.dtype == np.bool_ and got.shape == want.shape
    differ = np.argwhere(got != want).tolist()
    expected = []
    if row["name"] == "PilatusCdTe2M":
        strips = [729, 730, 731, 736, 737, 738, 1216, 1217, 1218, 1230, 1231, 1232]
        expected = [[j, c] for j in range(row["height"]) if j % 212 < 195 for c in strips]
    assert differ == expected
