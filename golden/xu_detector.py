"""Golden area and linear detectors from xrayutilities deff08f (1.8.0-11). Run: <xu python> -I xu_detector.py

Modelled on xrayutilities tests/test_qconversion_area.py and _linear.py, on a bare QConversion
(UB = I, so Q is in the sample frame): sample x+, detector z- then x+ (outermost first), r_i along +y.
Per case: the init_area / init_linear parameters (distance and pixel widths, or chpdeg), the pixel
positions at zero detector angles
(getDetectorPos, metres) and Q (1/angstrom, with 2 pi) at each angle set.

xu places a Nav-binned pixel on the first raw pixel of a block aligned to pixel 0, not on the centre of
the block it averages, and keeps a trailing partial block. For a Nav case "pos"/"q" therefore come from the
equivalent unbinned xu detector "reference" (pitch * Nav, the centre channel moved to the block grid, whole
blocks only, Nav 1, no roi); "xu_nav_pos" is xu's own Nav output and "full_pos" the unbinned chip.
"""
import json
import os

import numpy as np
import xrayutilities as xu

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xu_detector.json")
SAMPLE = ["x+"]
DETECTOR = ["z-", "x+"]
R_I = [0.0, 1.0, 0.0]
WL = 1.5406
ANGLES = [[10.0, 25.0, 0.0], [-3.5, 40.0, 7.5]]

AREA_BASE = dict(dir1="z+", dir2="x+", cch1=4.0, cch2=6.0, nch1=9, nch2=13, distance=0.5,
                 pwidth1=50e-6, pwidth2=50e-6, tilt=0.0, tiltazimuth=0.0, detrot=0.0,
                 chpdeg1=None, chpdeg2=None, nav=[1, 1], roi=None)
AREA = [
    ("plain", {}),
    ("tilt", dict(tilt=10.0)),
    ("tilt_tiltazimuth", dict(tilt=8.0, tiltazimuth=30.0)),
    ("detrot", dict(detrot=2.5)),
    ("roi", dict(roi=[1, 8, 3, 12])),
    ("nav", dict(nch1=10, nch2=12, nav=[2, 3])),
    ("nav_roi", dict(nav=[2, 3], roi=[1, 9, 2, 11])),
    ("combined", dict(dir1="x-", dir2="z+", cch1=4.3, cch2=6.7, pwidth1=172e-6, pwidth2=150e-6,
                      tilt=-6.0, tiltazimuth=120.0, detrot=-3.0, nav=[3, 2], roi=[2, 8, 1, 11])),
    ("chpdeg", dict(distance=None, pwidth1=None, pwidth2=None, chpdeg1=120.0, chpdeg2=-95.5, tilt=5.0,
                    tiltazimuth=40.0, detrot=1.5, roi=[1, 8, 2, 12])),
]
LINEAR_BASE = dict(dir="z+", cch=4.0, nch=9, distance=1.0, pwidth=50e-6, chpdeg=None, tilt=0.0, nav=1,
                   roi=None)
LINEAR = [
    ("plain", {}),
    ("tilt", dict(tilt=10.0)),
    ("roi", dict(roi=[2, 8])),
    ("nav", dict(nav=3)),
    ("nav_roi", dict(nav=2, roi=[1, 9])),
    ("combined", dict(dir="x-", cch=3.6, pwidth=172e-6, tilt=-7.0, nav=2, roi=[1, 7])),
    ("chpdeg", dict(distance=None, pwidth=None, chpdeg=-80.0, tilt=4.0, roi=[2, 8])),
]
# XU-14: a tilted row along r_i has no tilt axis; xu scales its pitch by cos(tilt), hklrs rejects it.
LINEAR_ALONG_R_I = dict(LINEAR_BASE, dir="y+", tilt=60.0)


def conversion():
    return xu.QConversion(SAMPLE, DETECTOR, R_I)


def stack(xyz):
    return np.stack(xyz, axis=-1)


def area_detector(p):
    qc = conversion()
    kw = {"Nav": p["nav"]}
    if p["roi"] is not None:
        kw["roi"] = p["roi"]
    qc.init_area(p["dir1"], p["dir2"], p["cch1"], p["cch2"], p["nch1"], p["nch2"], p["distance"],
                 p["pwidth1"], p["pwidth2"], chpdeg1=p["chpdeg1"], chpdeg2=p["chpdeg2"],
                 detrot=p["detrot"], tiltazimuth=p["tiltazimuth"],
                 tilt=p["tilt"], **kw)
    pos = stack(qc.getDetectorPos(0.0, 0.0, dim=2))
    q = [stack(qc.area(*a, wl=WL)) for a in ANGLES]
    return pos, q


def linear_detector(p):
    qc = conversion()
    kw = {"Nav": p["nav"]}
    if p["roi"] is not None:
        kw["roi"] = p["roi"]
    qc.init_linear(p["dir"], p["cch"], p["nch"], p["distance"], p["pwidth"], chpdeg=p["chpdeg"],
                   tilt=p["tilt"], **kw)
    pos = stack(qc.getDetectorPos(0.0, 0.0, dim=1))
    q = [stack(qc.linear(*a, wl=WL)) for a in ANGLES]
    return pos, q


def block_grid(cch, start, end, n):
    """Centre channel and channel count of the unbinned detector whose pixels are the block centres."""
    assert (end - start) % n == 0, "the importer requires whole Nav blocks"
    return (cch - start - (n - 1) / 2) / n, (end - start) // n


def area_reference(p):
    r = p["roi"] or [0, p["nch1"], 0, p["nch2"]]
    n1, n2 = p["nav"]
    cch1, nch1 = block_grid(p["cch1"], r[0], r[1], n1)
    cch2, nch2 = block_grid(p["cch2"], r[2], r[3], n2)
    return dict(p, cch1=cch1, cch2=cch2, nch1=nch1, nch2=nch2, pwidth1=p["pwidth1"] * n1,
                pwidth2=p["pwidth2"] * n2, nav=[1, 1], roi=None)


def linear_reference(p):
    r = p["roi"] or [0, p["nch"]]
    cch, nch = block_grid(p["cch"], r[0], r[1], p["nav"])
    return dict(p, cch=cch, nch=nch, pwidth=p["pwidth"] * p["nav"], nav=1, roi=None)


def case(name, p, detector, reference, unbinned):
    c = {"name": name, "params": p}
    if p["nav"] in (1, [1, 1]):
        pos, q = detector(p)
    else:
        c["reference"] = reference(p)
        pos, q = detector(c["reference"])
        c["xu_nav_pos"] = detector(p)[0].tolist()
        c["full_pos"] = detector(unbinned(p))[0].tolist()
    assert np.isfinite(pos).all() and all(np.isfinite(x).all() for x in q)
    c["pos"] = pos.tolist()
    c["q"] = [x.tolist() for x in q]
    return c


def main():
    assert xu.__version__ == "1.8.0", xu.__version__
    doc = {
        "source": "xrayutilities 1.8.0 deff08fbe98d5995513757b7a6d57cffe21b9a3f",
        "sample_axes": SAMPLE, "detector_axes": DETECTOR, "r_i": R_I, "wavelength": WL,
        "angles": ANGLES,
        "units": {"angle": "deg", "length": "m", "q": "1/angstrom"},
        "area": [case(n, dict(AREA_BASE, **d), area_detector, area_reference,
                      lambda p: dict(p, nav=[1, 1], roi=None)) for n, d in AREA],
        "linear": [case(n, dict(LINEAR_BASE, **d), linear_detector, linear_reference,
                        lambda p: dict(p, nav=1, roi=None)) for n, d in LINEAR],
        "linear_along_r_i": {"params": LINEAR_ALONG_R_I,
                             "pos": linear_detector(LINEAR_ALONG_R_I)[0].tolist()},
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(doc["area"]), len(doc["linear"]))


main()
