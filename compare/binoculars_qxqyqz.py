"""Python binoculars 93956b2's sixs:qxqyqzprojection run as its
FlyScanUHV2.process_job runs it (sixs.py:690-710): dataframes() on a NeXus
file, get_pixels, process_image and QxQyQzProjection.project. The file is
written in memory with the layout of the SIXS scans; the frames' images are
zeros, the projection does not read them. Runs under the binhkl env (Python
3.8, pyFAI 0.19, libhkl 7be4f79's typelib); test_binoculars.py launches it.

usage: python -I -B binoculars_qxqyqz.py <binoculars tree> <cases.json> <out.npz>

Writes, per case i, "q<i>": (frames, 3, slow, fast) Qx, Qy, Qz.
"""
import json
import sys

import h5py
import numpy

tree, cases, out = sys.argv[1:4]
sys.path.insert(0, tree)
from binoculars.backends import sixs  # noqa: E402

# imxpads140, the detector get_detector names for any other image shape
# (sixs.py:170-178)
SHAPE = (240, 560)


def nexus(case):
    f = h5py.File("frames.nxs", "w", driver="core", backing_store=False)
    diff = f.create_group("com/SIXS/i14-c-cx2-ex-cm-uhv")
    diff.attrs["NX_class"] = numpy.bytes_(b"NXdiffractometer")
    diff["type"] = numpy.bytes_(b"ZAXIS")
    diff["UB"] = numpy.eye(3)
    for name, value in case["sample"].items():
        diff[name] = numpy.array([value])
    mono = f.create_group("com/SIXS/i14-c-c02-op-mono")
    mono.attrs["NX_class"] = numpy.bytes_(b"NXmonochromator")
    mono["lambda"] = numpy.array([case["wavelength"]])
    data = f.create_group("com/scan_data")
    data.attrs["NX_class"] = numpy.bytes_(b"NXdata")
    frames = numpy.array(case["frames"], dtype=float)
    for i, axis in enumerate(("mu", "omega", "delta", "gamma")):
        data[axis] = frames[:, i]
    data["xpad_image"] = numpy.zeros((len(frames),) + SHAPE, numpy.uint16)
    return f


def run(case):
    source = sixs.FlyScanUHV2(dict(case["input"]))
    projection = sixs.QxQyQzProjection(dict(case["projection"]))
    with nexus(case) as f:
        (dataframe,) = sixs.dataframes(f, source.HPATH, source.config)
        assert dataframe.detector.name == "imxpads140"
        pixels = source.get_pixels(dataframe.detector)
        mask = numpy.zeros(SHAPE, bool)
        q = []
        for i in range(len(case["frames"])):
            _, _, (index, pdataframe) = source.process_image(i, dataframe, pixels, mask)
            q.append(numpy.array(projection.project(index, pdataframe)))
    return numpy.array(q)


with open(cases) as f:
    numpy.savez(out, **{f"q{i}": run(c) for i, c in enumerate(json.load(f))})
