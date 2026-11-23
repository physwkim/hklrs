"""Golden .poni files written by pyFAI 2026.9.0 PoniFile.write, and what
pyFAI reads back from each, for the PONI text parser.
Run: python -I pyfai_poni_files.py"""
import json
import os

import numpy
import pyFAI
from pyFAI.detectors import Detector, Pilatus1M
from pyFAI.geometry import Geometry
from pyFAI.io.ponifile import PoniFile

assert pyFAI.version == "2026.9.0", pyFAI.version

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "poni")
OUT = os.path.join(HERE, "pyfai_poni_files.json")
PIXEL1, PIXEL2 = 7.5e-5, 1.72e-4
SHAPE = (213, 347)
PIXELS = [(0, 0), (0, 346), (212, 0), (212, 346), (17, 291), (101, 5)]
GEOMETRY = {"dist": 0.31, "poni1": 0.0071, "poni2": 0.0253}


def generic(orientation):
    return Detector(pixel1=PIXEL1, pixel2=PIXEL2, max_shape=SHAPE,
                    orientation=orientation)


# name, poni_version written, detector, rotations, wavelength (m), parallax
FILES = [
    ("v1", 1, generic(3), (0.21, -0.13, 0.37), 1.54e-10, None),
    ("v2", 2, generic(3), (2.5, -1.2, -2.9), 7.1e-11, None),
    ("v2_1", 2.1, generic(1), (0.21, -0.13, 0.37), 1.54e-10, None),
    ("v3", 3, generic(4), (-0.4, 0.3, 1.1), None, None),
    ("v3_parallax", 3, generic(2), (0.21, -0.13, 0.37), 1e-10, True),
    ("pilatus1m", 2.1, Pilatus1M(orientation=2), (0.1, 0.2, -0.3), 1e-10,
     None),
]


def write(name, version, det, rots, wavelength, parallax):
    geo = Geometry(rot1=rots[0], rot2=rots[1], rot3=rots[2], detector=det,
                   wavelength=wavelength, **GEOMETRY)
    pf = PoniFile(geo)
    if parallax:
        # Geometry needs a sensor to turn parallax on; the dict form is
        # PoniFile's own route to a Parallax: True file
        pf = PoniFile(data={**pf.as_dict(), "parallax": True})
    pf.API_VERSION = version
    pf.extra["Calibrant"] = "LaB6"
    path = os.path.join(DIR, name + ".poni")
    with open(path, "w") as f:
        pf.write(f)
    return path


def unversioned(path):
    """The v1 format before poni_version existed: the same file without its
    version line (ponifile.py PoniFile docstring, version 1)."""
    with open(path) as f:
        lines = [line for line in f if not line.startswith("poni_version:")]
    out = path.replace("v1.poni", "v1_unversioned.poni")
    with open(out, "w") as f:
        f.writelines(lines)
    return out


def read_back(path):
    pf = PoniFile(path)
    det = pf.detector
    shape = det.max_shape
    doc = {
        "file": os.path.basename(path),
        "version": pf.API_VERSION,
        "detector": det.__class__.__name__,
        "pixel1": det.pixel1, "pixel2": det.pixel2,
        "max_shape": None if shape is None else [int(n) for n in shape],
        "orientation": int(det.orientation),
        "dist": pf.dist, "poni1": pf.poni1, "poni2": pf.poni2,
        "rot1": pf.rot1, "rot2": pf.rot2, "rot3": pf.rot3,
        "wavelength": pf.wavelength,
        "parallax": bool(pf.parallax),
    }
    if not pf.parallax:
        geo = Geometry()
        geo.load(path)
        d1 = numpy.array([p[0] for p in PIXELS], dtype=numpy.float64)
        d2 = numpy.array([p[1] for p in PIXELS], dtype=numpy.float64)
        z, y, x = geo.calc_pos_zyx(d1=d1, d2=d2)
        doc["zyx"] = [[float(z[i]), float(y[i]), float(x[i])]
                      for i in range(len(PIXELS))]
    return doc


def main():
    os.makedirs(DIR, exist_ok=True)
    paths = []
    for spec in FILES:
        paths.append(write(*spec))
        if spec[0] == "v1":
            paths.append(unversioned(paths[-1]))
    files = [read_back(p) for p in paths]
    doc = {
        "program": f"pyFAI {pyFAI.version} PoniFile.write and PoniFile(path)",
        "units": {"length": "m", "angle": "rad", "wavelength": "m"},
        "pixels": [list(p) for p in PIXELS],
        "note": "zyx = Geometry.load(file).calc_pos_zyx = (t3, t1, t2); "
                "absent for parallax files",
        "files": files,
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(files))


main()
