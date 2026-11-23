"""Golden pixel edges and centres of the ImXPad S70 and S140 from pyFAI
2026.9.0 (calc_pixels_edges, calc_cartesian_positions), default orientation 3,
so neither axis is reversed.

Produced with pyFAI 2026.9.0, numpy 2.5.5, CPython 3.12.16 (the binpy env of
the hkl repro scripts). Run: python -I pyfai_imxpad.py"""
import json
import os

import numpy
import pyFAI
from pyFAI.detectors import ImXPadS70, ImXPadS140

assert pyFAI.version == "2026.9.0", pyFAI.version

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyfai_imxpad.json")


def main():
    detectors = []
    for name, cls in (("ImXpadS70", ImXPadS70), ("ImXpadS140", ImXPadS140)):
        det = cls()
        assert int(det.orientation) == 3, det.orientation
        edges1, edges2 = det.calc_pixels_edges()
        p1, p2, p3 = det.calc_cartesian_positions()
        assert p3 is None
        # a flat detector: p1 depends only on the row, p2 only on the column
        assert (p1 == p1[:, :1]).all() and (p2 == p2[:1, :]).all()
        # decimal strings: serde_json without float_roundtrip reads some
        # doubles one ulp off, and the test compares to the bit
        exact = lambda a: [repr(float(x)) for x in a]
        detectors.append({
            "name": name,
            "shape": list(det.shape),
            "module": list(det.module_size),
            "pixel": [det.pixel1, det.pixel2],
            "edges1": exact(edges1),
            "edges2": exact(edges2),
            "centre1": exact(p1[:, 0]),
            "centre2": exact(p2[0, :]),
        })
    doc = {"pyfai": pyFAI.version, "numpy": numpy.__version__, "detectors": detectors}
    with open(OUT, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
