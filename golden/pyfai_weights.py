"""Golden pixel weights from pyFAI 2026.9.0: Geometry.solidAngleArray (the
solid angle relative to the PONI direction's, cos^3 of the incidence) and
Geometry.polarization. Run: python -I pyfai_weights.py"""
import json
import os

import numpy
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry

assert pyFAI.version == "2026.9.0", pyFAI.version

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyfai_weights.json")
# solidAngleArray indexes pixels with float32 (numpy.fromfunction) and keeps
# positions in float32; dyadic pitches and PONI offsets make those exact
PIXEL1, PIXEL2 = 3 * 2.0**-11, 2.0**-9
SHAPE = (9, 13)
DIST, PONI1, PONI2 = 0.041, 11 * 2.0**-10, 9 * 2.0**-10
# 2theta spans 0-24 deg (none), 40-74 (all) and 91-113 (all_large), across
# the 90 deg where polarisation is smallest
ROTS = {
    "none": (0.0, 0.0, 0.0),
    "rot1": (0.83, 0.0, 0.0),
    "rot2": (0.0, -0.61, 0.0),
    "rot3": (0.0, 0.0, 0.47),
    "all": (0.83, -0.61, 0.47),
    "all_large": (2.5, -1.2, -2.9),
}
CASES = [(name, 3) for name in ROTS] + [("all", o) for o in (1, 2, 4)]
FACTORS = [-1.0, -0.6, 0.0, 0.35, 0.95, 1.0]
AXIS_OFFSETS = [0.0, 0.4, -1.3]


def geometry(rots, orientation):
    det = Detector(pixel1=PIXEL1, pixel2=PIXEL2, max_shape=SHAPE,
                   orientation=orientation)
    assert int(det.orientation) == orientation
    r1, r2, r3 = rots
    return Geometry(dist=DIST, poni1=PONI1, poni2=PONI2,
                    rot1=r1, rot2=r2, rot3=r3, detector=det)


def main():
    d1, d2 = numpy.indices(SHAPE, dtype=numpy.float64)
    cases = []
    for name, orientation in CASES:
        rots = ROTS[name]
        sa = geometry(rots, orientation).solidAngleArray(shape=SHAPE)
        assert sa.dtype == numpy.float64 and (sa > 0).all(), (name, orientation)
        # the same cos^3 from float64 indices, by the cython kernel and by
        # numpy, or the float32 indexing has rounded a position
        for path in ("cython", "numpy"):
            cosa = geometry(rots, orientation).cos_incidence(d1, d2, path=path)
            spread = numpy.abs(sa / cosa**3 - 1).max()
            assert spread < 1e-15, (name, orientation, path, spread)
        pols = []
        for factor in FACTORS:
            for offset in AXIS_OFFSETS:
                pol = geometry(rots, orientation).polarization(
                    shape=SHAPE, factor=factor, axis_offset=offset)
                assert pol.dtype == numpy.float32 and (pol > 0).all()
                # numexpr and numpy evaluate in float64 before the float32
                # cast, so they may differ by that one rounding
                alt = geometry(rots, orientation).polarization(
                    shape=SHAPE, factor=factor, axis_offset=offset,
                    path="numpy")
                assert (numpy.abs(pol - alt) <= numpy.spacing(pol)).all()
                pols.append({"factor": factor, "axis_offset": offset,
                             "values": pol.astype(numpy.float64).tolist()})
        cases.append({
            "orientation": orientation, "rots": name,
            "rot1": rots[0], "rot2": rots[1], "rot3": rots[2],
            "solid_angle": sa.tolist(),
            "polarization": pols,
        })
    doc = {
        "program": f"pyFAI {pyFAI.version} Geometry.solidAngleArray, "
                   "Geometry.polarization",
        "units": {"length": "m", "angle": "rad"},
        "dist": DIST, "poni1": PONI1, "poni2": PONI2,
        "pixel1": PIXEL1, "pixel2": PIXEL2, "shape": list(SHAPE),
        "note": "values[slow][fast]; solid_angle is relative to the PONI "
                "direction's (order 3); polarization is float32",
        "cases": cases,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(cases))


main()
