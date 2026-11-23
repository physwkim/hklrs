"""Golden pixel positions from pyFAI 2026.9.0 Geometry.calc_pos_zyx for the
PONI importer. Run: python -I pyfai_poni.py"""
import json
import os

import numpy
import pyFAI
from pyFAI.detectors import Detector
from pyFAI.geometry import Geometry

assert pyFAI.version == "2026.9.0", pyFAI.version

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyfai_poni.json")
# unequal pitches and an odd, non-square shape, so a swap of the two axes
# or an off-by-one in an orientation flip changes the positions
PIXEL1, PIXEL2 = 7.5e-5, 1.72e-4
SHAPE = (213, 347)
DIST, PONI1, PONI2 = 0.31, 0.0071, 0.0253
ROTS = {
    "none": (0.0, 0.0, 0.0),
    "rot1": (0.21, 0.0, 0.0),
    "rot2": (0.0, -0.13, 0.0),
    "rot3": (0.0, 0.0, 0.37),
    "all": (0.21, -0.13, 0.37),
    "all_large": (2.5, -1.2, -2.9),
}
# (slow, fast): the four corners and interior pixels
PIXELS = [(0, 0), (0, 346), (212, 0), (212, 346), (17, 291), (101, 5), (200, 173)]


def main():
    d1 = numpy.array([p[0] for p in PIXELS], dtype=numpy.float64)
    d2 = numpy.array([p[1] for p in PIXELS], dtype=numpy.float64)
    cases = []
    for orientation in (1, 2, 3, 4):
        det = Detector(pixel1=PIXEL1, pixel2=PIXEL2, max_shape=SHAPE,
                       orientation=orientation)
        assert int(det.orientation) == orientation
        for name, (r1, r2, r3) in ROTS.items():
            geo = Geometry(dist=DIST, poni1=PONI1, poni2=PONI2,
                           rot1=r1, rot2=r2, rot3=r3, detector=det)
            z, y, x = geo.calc_pos_zyx(d1=d1, d2=d2)
            # pyFAI's numpy path must agree with the cython kernel it
            # returns by default, or the vectors are not pyFAI's geometry
            zn, yn, xn = geo.calc_pos_zyx(d1=d1, d2=d2, use_cython=False)
            spread = numpy.abs(numpy.array([z - zn, y - yn, x - xn])).max()
            assert spread < 1e-15, (orientation, name, spread)
            cases.append({
                "orientation": orientation, "rots": name,
                "rot1": r1, "rot2": r2, "rot3": r3,
                "zyx": [[float(z[i]), float(y[i]), float(x[i])]
                        for i in range(len(PIXELS))],
            })
    doc = {
        "program": f"pyFAI {pyFAI.version} Geometry.calc_pos_zyx",
        "units": {"length": "m", "angle": "rad"},
        "dist": DIST, "poni1": PONI1, "poni2": PONI2,
        "pixel1": PIXEL1, "pixel2": PIXEL2, "shape": list(SHAPE),
        "pixels": [list(p) for p in PIXELS],
        "note": "zyx = (t3, t1, t2): along the beam, axis 1, axis 2",
        "cases": cases,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(cases))


main()
