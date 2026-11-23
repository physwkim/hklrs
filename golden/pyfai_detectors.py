"""pyFAI 2026.9.0's detector classes as Detector.factory resolves a PONI
file's detector name, for hklrs's PONI import.

Every Detector subclass the factory reaches, by registry key or by attribute
of pyFAI.detectors, with its constructor arguments and the default instance's
pitch, shape, max_shape, binning and orientation, and the layout hklrs gives
it: "generic" (Detector itself), "uniform" (calc_cartesian_positions equals
the generic Detector of that pitch, shape and orientation, both index paths,
every orientation), "edges" (ImXPad: the chip rule gives the pixel edges of
calc_cartesian_positions over the whole image, every orientation), or the
reason it has no model. A modelled class's mask rule reproduces its calc_mask
and Detector.mask element by element in every orientation; the golden keeps a
fingerprint of that mask. A uniform class is "binnable" when the factory's
binning key, at each of BINNINGS, gives the default instance binned by
Detector.binning's rule: the pitch times the binning ratio, max_shape floor
divided, positions those of the generic Detector of that pitch and shape, and
Detector.mask the mask rule's, drawn over max_shape and summed over binning
blocks for a gap rule or Lambda9M's empty array (pyFAI raises where the
binning does not divide max_shape) or over the binned shape for the disc; "binned" keeps pyFAI's
instances either way, "unbinned" its 1x1 one. "zero_pitch" is what the
factory makes of a 0 or null pixel1, pixel2 or both, and Geometry.load of a
version 1 file with PixelSize1 and PixelSize2 0: the default instance, an
exception, an instance with no pitch, or, for version 1, the generic Detector;
"zero_pitch_default" holds when it is
the default instance for every factory form. "probes" are names put to
detector_factory and the class it returns; "cases" are a PONI file pyFAI writes for a modelled class
and, after Geometry.load, calc_pos_zyx of some of its pixels where the image
is flipped once per orientation: by index for a uniform class, whose index
path is its whole-image path, over the whole image for ImXPad, whose index
path flips twice ("zyx_by_index", upstream defect FAI-2).

Writes pyfai_detectors.json next to this script and the class table
hklrs/src/poni/detectors/table.rs. Produced with pyFAI 2026.9.0, numpy 2.5.5,
CPython 3.12.16 (the binpy env of the hkl repro scripts).
Run: python -I pyfai_detectors.py"""
import inspect
import json
import logging
import os
import tempfile

import numpy
import pyFAI
import pyFAI.detectors
from pyFAI.detectors import ALL_DETECTORS, Detector, detector_factory
from pyFAI.geometry import Geometry
from pyFAI.io.ponifile import PoniFile

assert pyFAI.version == "2026.9.0", pyFAI.version
# the factory logs every unknown probe name as an error
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pyfai_detectors.json")
TABLE = os.path.join(HERE, "..", "hklrs", "src", "poni", "detectors", "table.rs")

# calc_mask's owner -> the rule that reproduces it; strip is the [lo, hi)
# window around each module's middle column (Eiger2CdTe, PilatusCdTe,
# Pilatus4_CdTe add it to the module gaps)
GAPS = {"_Dectris": None, "_Lambda": None, "Maxipix": None,
        "Eiger2CdTe": (-1, 1), "PilatusCdTe": (-1, 2), "Pilatus4_CdTe": (0, 1)}
DISC = {"Mar345", "Rayonix133", "RayonixSx165"}
CHIPS = {"ImXPadS10"}
BINNINGS = [(1, 1), (2, 2), (3, 3), (2, 1), (1, 4)]


def gaps_mask(shape, module, gap, strip):
    (n0, n1), (m0, m1), (g0, g1) = shape, module, gap
    mask = numpy.zeros(shape, dtype=bool)
    for i in range(m0, n0, m0 + g0):
        mask[i:i + g0, :] = True
    for i in range(m1, n1, m1 + g1):
        mask[:, i:i + g1] = True
    if strip is not None:
        assert m1 // 2 + strip[0] >= 0
        for i in range(m1 // 2, n1, m1 + g1):
            mask[:, i + strip[0]:i + strip[1]] = True
    return mask


def disc_mask(shape):
    c0, c1 = shape[0] // 2, shape[1] // 2
    x, y = numpy.ogrid[:shape[0], :shape[1]]
    return (x + 0.5 - c0) ** 2 + (y + 0.5 - c1) ** 2 > c0 ** 2


def chips_mask(shape, module):
    edges = []
    for n, m in zip(shape, module):
        pos = numpy.zeros(n, dtype=bool)
        for i in range(1, n // m):
            pos[i * m - 1] = pos[i * m] = True
        pos[0] = pos[-1] = True
        edges.append(pos)
    return edges[0][:, None] | edges[1][None, :]


def rebin_any(mask, b):
    """utils.mathutil.binning(mask, b) != 0, None where it raises"""
    (n0, n1), (b0, b1) = mask.shape, b
    if n0 % b0 or n1 % b1:
        return None
    return mask.reshape(n0 // b0, b0, n1 // b1, b1).any(axis=(1, 3))


def binned_mask(rule, max_shape, shape, b):
    """the rule's Detector.mask of the instance binned b, None where pyFAI's
    rebin raises"""
    if rule is None:
        return numpy.zeros(shape, dtype=bool)
    if rule["rule"] == "disc":
        return disc_mask(shape)
    if rule["rule"] == "empty":
        m = numpy.zeros(max_shape, dtype=bool)
    else:
        assert rule["rule"] == "gaps", rule
        m = gaps_mask(max_shape, rule["module"], rule["gap"], rule["strip"])
    return m if tuple(max_shape) == tuple(shape) else rebin_any(m, b)


def fingerprint(mask):
    """count, sum and sum of squares (mod 2**64) of the row-major indices of
    the masked pixels"""
    k = numpy.flatnonzero(mask).astype(numpy.uint64)
    return [int(k.size), int(k.sum(dtype=numpy.uint64)),
            int((k * k).sum(dtype=numpy.uint64))]


def owner(cls, method):
    return getattr(cls, method).__qualname__.split(".")[0]


def mask_rule(det):
    """(rule, mask) reproducing det.calc_mask and det.mask, or (reason, None)"""
    m = det.calc_mask()
    for o in (1, 2, 3, 4):
        other = type(det)(orientation=o).calc_mask()
        assert (m is None and other is None) or (numpy.asarray(m) == numpy.asarray(other)).all()
    family = owner(type(det), "calc_mask")
    if family in GAPS:
        gap = [int(g) for g in type(det).MODULE_GAP]
        rule = {"rule": "gaps", "module": [int(n) for n in det.module_size],
                "gap": gap, "strip": None if GAPS[family] is None else list(GAPS[family])}
        assert tuple(det.shape) == tuple(det.max_shape)
        mine = gaps_mask(det.shape, det.module_size, gap, GAPS[family])
    elif family in DISC:
        rule, mine = {"rule": "disc"}, disc_mask(det.shape)
    elif family in CHIPS:
        rule = {"rule": "chips", "module": [int(n) for n in det.module_size]}
        assert tuple(det.shape) == tuple(det.max_shape)
        mine = chips_mask(det.shape, det.module_size)
    elif m is None:
        rule, mine = None, numpy.zeros(det.shape, dtype=bool)
    elif not numpy.asarray(m).any():
        # Lambda9M: an array over max_shape, which Detector.mask bins
        assert numpy.asarray(m).shape == tuple(det.max_shape) == tuple(det.shape)
        rule, mine = {"rule": "empty"}, numpy.zeros(det.shape, dtype=bool)
    else:
        return f"mask ({family}.calc_mask)", None
    if m is None:
        assert not mine.any()
    else:
        assert (numpy.asarray(m) != 0).shape == mine.shape
        assert ((numpy.asarray(m) != 0) == mine).all(), det
    full = det.mask
    assert (not mine.any()) if full is None else ((numpy.asarray(full) != 0) == mine).all()
    return rule, mine


def same_as_generic(make, rng):
    """calc_cartesian_positions of make(orientation) equals the generic
    Detector's of the same pitch, shape and orientation: explicit indices
    (calc_pos_zyx's path) and no indices (position_array's), every
    orientation"""
    for o in (1, 2, 3, 4):
        det = make(o)
        ref = Detector(pixel1=det.pixel1, pixel2=det.pixel2, max_shape=det.shape,
                       orientation=o)
        n0, n1 = det.shape
        d1 = numpy.concatenate([numpy.arange(n0), numpy.arange(n0), numpy.zeros(n1),
                                numpy.full(n1, n0 - 1), rng.integers(0, n0, 4000)])
        d2 = numpy.concatenate([numpy.zeros(n0), numpy.full(n0, n1 - 1), numpy.arange(n1),
                                numpy.arange(n1), rng.integers(0, n1, 4000)])
        for args in ((d1.astype(float), d2.astype(float)), ()):
            a, b = det.calc_cartesian_positions(*args), ref.calc_cartesian_positions(*args)
            if a[2] is not None or not (numpy.array_equal(a[0], b[0])
                                        and numpy.array_equal(a[1], b[1])):
                return False
    return True


def chip_edges(n, module, pitch, border):
    """ImXPadS10._calc_pixels_size and calc_pixels_edges along one axis: the
    pixels either side of an interior chip boundary are border pitches wide"""
    size = numpy.ones(n)
    for i in range(1, n // module):
        size[i * module - 1] = size[i * module] = border
    return numpy.concatenate([[0.0], numpy.cumsum(pitch * size)])


def chip_layout(cls):
    """the chip rule over max_shape and the class's pitch and module gives
    calc_cartesian_positions of the whole image, the edges reversed along
    each axis the orientation flips; the index path agrees in orientation 3"""
    for o in (1, 2, 3, 4):
        det = cls(orientation=o)
        if (det.pixel1, det.pixel2) != tuple(cls.PIXEL_SIZE) or det.shape != det.max_shape:
            return False
        centres = []
        for k, flip in ((0, o in (1, 2)), (1, o in (1, 4))):
            e = chip_edges(det.max_shape[k], det.module_size[k], cls.PIXEL_SIZE[k],
                           cls.BORDER_SIZE_RELATIVE)
            e = e[::-1] if flip else e
            centres.append(0.5 * (e[:-1] + e[1:]))
        p1, p2, p3 = det.calc_cartesian_positions()
        if p3 is not None or not (numpy.array_equal(p1, numpy.outer(centres[0], numpy.ones(det.shape[1])))
                                  and numpy.array_equal(p2, numpy.outer(numpy.ones(det.shape[0]), centres[1]))):
            return False
        if o == 3:
            d1, d2 = numpy.indices(det.shape).reshape(2, -1).astype(float)
            q1, q2, _ = det.calc_cartesian_positions(d1, d2)
            if not (numpy.array_equal(q1, p1.ravel()) and numpy.array_equal(q2, p2.ravel())):
                return False
    return True


def binned(cls, doc, rule, rng):
    """pyFAI's instances of cls for the factory's binning key at each of
    BINNINGS, and whether each is the default instance binned by
    Detector.binning's rule with the rule's binned mask"""
    pd = [float(p) for p in doc["pixel"]]
    bd, mx = doc["binning"], doc["max_shape"]
    out, binnable = [], True
    for b in BINNINGS:
        def make(o, b=b):
            det = detector_factory(doc["class"], {"binning": list(b), "orientation": o})
            assert type(det) is cls
            return det
        det = make(3)
        try:
            m, err = det.mask, None
            fp = None if m is None else fingerprint(numpy.asarray(m) != 0)
        except RuntimeError as e:
            m, fp, err = None, None, str(e)
        out.append({"binning": list(b), "pixel": [repr(det.pixel1), repr(det.pixel2)],
                    "shape": [int(n) for n in det.shape], "mask": fp, "mask_error": err})
        shape = (mx[0] // b[0], mx[1] // b[1])
        ok = ((det.pixel1, det.pixel2) == (pd[0] * (b[0] / bd[0]), pd[1] * (b[1] / bd[1]))
              and tuple(det.shape) == shape and tuple(det.binning) == b)
        mine = binned_mask(rule, mx, shape, b) if ok else None
        if err is not None:
            ok = ok and mine is None
        elif ok:
            ok = mine is not None and ((not mine.any()) if m is None
                                       else bool(((numpy.asarray(m) != 0) == mine).all()))
        # an empty image has no positions
        binnable = binnable and ok and (b == tuple(bd) or 0 in shape or same_as_generic(make, rng))
    unbinned = out[BINNINGS.index((1, 1))]
    return binnable, {"pixel": unbinned["pixel"], "shape": unbinned["shape"]}, out


ZERO_FORMS = {"pixel1": {"pixel1": 0}, "pixel2": {"pixel2": 0},
              "both": {"pixel1": 0, "pixel2": 0}, "pixel1 null": {"pixel1": None},
              "pixel2 null": {"pixel2": None}}


def instance_key(det):
    return (type(det).__name__, det.pixel1, det.pixel2, det.shape, tuple(det.binning),
            int(det.orientation))


def zero_pitch(name, cls, tmp):
    """the factory's instance for each of ZERO_FORMS and Geometry.load's for
    a version 1 file of 0 pixel sizes, against the default instance"""
    default = instance_key(cls())

    def outcome(make):
        try:
            det = make()
        except Exception as err:
            return f"raises {type(err).__name__}"
        if instance_key(det) == default:
            return "default"
        assert det.pixel1 is None or det.pixel2 is None, (name, instance_key(det))
        return "no pitch"

    out = {k: outcome(lambda c=c: detector_factory(name, c)) for k, c in ZERO_FORMS.items()}
    path = os.path.join(tmp, "zero.poni")
    with open(path, "w") as f:
        f.write(f"Detector: {name}\nPixelSize1: 0\nPixelSize2: 0\nDistance: 0.3\n"
                "Poni1: 0.01\nPoni2: 0.02\nRot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n")

    def load():
        geo = Geometry()
        geo.load(path)
        return geo.detector
    # PoniFile.read_from_dict: a NexusDetector kept, the class's constructor
    # where it forces the pitch, else the generic Detector with the pitches set
    if issubclass(cls, pyFAI.detectors.NexusDetector):
        out["version 1"] = outcome(load)
    elif cls.force_pixel:
        out["version 1"] = outcome(load)
        assert out["version 1"] == out["both"], (name, out)
    else:
        det = load()
        assert type(det) is Detector and (det.pixel1, det.pixel2, det.shape) == (0.0, 0.0, None)
        out["version 1"] = "generic"
    # a null is a 0 to every constructor
    for k in ("pixel1", "pixel2"):
        assert (out[k] == "default") == (out[f"{k} null"] == "default"), (name, out)
    return out


def describe(name, cls, rng):
    args = [a for a in inspect.getfullargspec(cls).args if a != "self"]
    doc = {"class": name, "args": args, "force_pixel": bool(cls.force_pixel)}
    try:
        det = cls()
    except Exception as err:  # base classes that need a file or a layout
        return {**doc, "layout": f"no default instance ({type(err).__name__})"}
    shape = det.shape
    # ModuleDetector's is (None, None)
    module = getattr(det, "module_size", None)
    module = [int(n) for n in module] if module and None not in module else None
    doc.update({
        "pixel": [repr(det.pixel1), repr(det.pixel2)] if det.pixel1 else None,
        "shape": None if shape is None else [int(n) for n in shape],
        "max_shape": None if det.max_shape is None else [int(n) for n in det.max_shape],
        "binning": [int(b) for b in det.binning],
        "orientation": int(det.orientation),
        "module_size": module,
    })
    with tempfile.TemporaryDirectory() as tmp:
        zero = zero_pitch(name, cls, tmp)
    doc["zero_pitch"] = zero
    doc["zero_pitch_default"] = all(zero[k] == "default" for k in ZERO_FORMS)
    if cls is Detector:
        return {**doc, "layout": "generic", "mask": None}
    if shape is None or det.pixel1 is None:
        return {**doc, "layout": "no fixed shape or pitch"}
    if not det.IS_FLAT:
        return {**doc, "layout": "curved"}
    if det.splinefile is not None:
        return {**doc, "layout": "spline"}
    if det._pixel_corners is not None or not det.IS_CONTIGUOUS:
        return {**doc, "layout": "pixel corners"}
    if same_as_generic(lambda o: cls(orientation=o), rng):
        layout = "uniform"
    elif owner(cls, "calc_cartesian_positions") == "ImXPadS10" and chip_layout(cls):
        layout = "edges"
        doc["border"] = cls.BORDER_SIZE_RELATIVE
    else:
        return {**doc, "layout": "non-uniform pixels"}
    rule, mask = mask_rule(det)
    if mask is None:
        return {**doc, "layout": rule}
    if rule is not None and rule.get("module") is not None:
        assert rule["module"] == doc["module_size"]
    assert tuple(det.shape) == tuple(m // b for m, b in zip(det.max_shape, det.binning))
    doc = {**doc, "layout": layout, "mask": rule, "fingerprint": fingerprint(mask)}
    if layout == "edges":
        # calc_pixels_edges spans max_shape whatever the binning
        two = detector_factory(name, {"binning": [2, 2]})
        assert two.shape != two.max_shape and two.calc_cartesian_positions()[0].shape[0] == two.max_shape[0]
        return {**doc, "binnable": False}
    binnable, unbinned, instances = binned(cls, doc, rule, rng)
    return {**doc, "binnable": binnable, "unbinned": unbinned, "binned": instances}


def probes(attributes, registry):
    names = set(attributes) | {n.lower() for n in attributes} | {n.upper() for n in attributes}
    for key in registry:
        names |= {key, key.upper(), key.replace("_", " "), key.replace("_", " ").upper(),
                  key.replace(" ", "_")}
    # pyFAI.detectors attributes that are not Detector classes fail the lookup
    names |= {"no such detector", "Pilatus 1 M", "detector", "DETECTOR", "Orientation",
              "SensorConfig", "logger", "ALL_DETECTORS"}
    out = {}
    for name in sorted(names):
        try:
            out[name] = type(detector_factory(name)).__name__
        except Exception as err:
            # the class the factory had picked when its constructor raised
            # (HexDetector needs arguments); none when the name is unknown
            found, tb = None, err.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is Detector.factory.__func__.__code__:
                    found = tb.tb_frame.f_locals.get("detectorClass")
                tb = tb.tb_next
            out[name] = found.__name__ if found else None
    return out


def case(doc, tmp, o, b=None):
    """a PONI file pyFAI writes for doc's class in orientation o, binned b
    by the binning key, and calc_pos_zyx after load"""
    cls = getattr(pyFAI.detectors, doc["class"], None) or next(
        c for c in ALL_DETECTORS.values() if c.__name__ == doc["class"])
    det = cls(orientation=o) if b is None else detector_factory(
        doc["class"], {"binning": list(b), "orientation": o})
    n0, n1 = det.shape
    geo = Geometry(dist=0.31, poni1=0.37 * n0 * det.pixel1, poni2=0.61 * n1 * det.pixel2,
                   rot1=0.1, rot2=-0.2, rot3=0.3, detector=det, wavelength=1e-10)
    path = os.path.join(tmp, f"{doc['class']}.poni")
    with open(path, "w") as f:
        PoniFile(geo).write(f)
    with open(path) as f:
        text = f.read()
    # the date line changes on every run
    text = "".join(line for line in text.splitlines(True) if not line.startswith("# Calibration done"))
    if b is not None:
        # PoniFile.write gives the binned pitch, not the binning
        config = json.dumps({"binning": list(b), "orientation": o})
        lines = text.splitlines(True)
        k = next(i for i, line in enumerate(lines) if line.startswith("Detector_config:"))
        lines[k] = f"Detector_config: {config}\n"
        text = "".join(lines)
        with open(path, "w") as f:
            f.write(text)
    back = Geometry()
    back.load(path)
    assert type(back.detector) is cls and int(back.detector.orientation) == o
    assert (back.detector.pixel1, back.detector.pixel2, back.detector.shape) == (
        det.pixel1, det.pixel2, det.shape)
    pixels = [(0, 0), (n0 - 1, 0), (0, n1 - 1), (n0 - 1, n1 - 1), (n0 // 3, n1 // 2)]
    if doc["layout"] == "edges":
        # either side of a chip boundary
        m0, m1 = doc["module_size"]
        pixels += [(n0 // 3, m1 - 1), (min(m0, n0 - 1), min(m1, n1 - 1))]
    d1 = numpy.array([p[0] for p in pixels], dtype=float)
    d2 = numpy.array([p[1] for p in pixels], dtype=float)
    out = {"class": doc["class"], "orientation": o, "binning": None if b is None else list(b),
           "poni": text, "pixels": [list(p) for p in pixels]}
    by_index = back.calc_pos_zyx(d1=d1, d2=d2)
    if doc["layout"] == "edges":
        whole = back.calc_pos_zyx()
        zyx = [w[d1.astype(int), d2.astype(int)] for w in whole]
        out["zyx_by_index"] = [[float(c[k]) for c in by_index] for k in range(len(pixels))]
    else:
        zyx = by_index
    out["zyx"] = [[float(c[k]) for c in zyx] for k in range(len(pixels))]
    return out


def rust_pair(v):
    return "None" if v is None else f"Some(({v[0]}, {v[1]}))"


def rust_rule(rule):
    if rule["rule"] == "gaps":
        strip = "None" if rule["strip"] is None else f"Some(({rule['strip'][0]}, {rule['strip'][1]}))"
        return (f"Rule::Gaps {{ module: ({rule['module'][0]}, {rule['module'][1]}), "
                f"gap: ({rule['gap'][0]}, {rule['gap'][1]}), strip: {strip} }}")
    if rule["rule"] == "empty":
        return "Rule::Empty"
    assert rule["rule"] == "disc", rule
    return "Rule::Disc"


def rust_grid(c):
    return (f"Grid {{ pixel: ({c['pixel'][0]}, {c['pixel'][1]}), "
            f"shape: ({c['shape'][0]}, {c['shape'][1]}), "
            f"max_shape: ({c['max_shape'][0]}, {c['max_shape'][1]}), "
            f"binning: ({c['binning'][0]}, {c['binning'][1]}), "
            f"orientation: {c['orientation']}, "
            f"module_size: {rust_pair(c['module_size'])}, "
            f"binnable: {'true' if c['binnable'] else 'false'}, "
            f"zero_pitch_default: {'true' if c['zero_pitch_default'] else 'false'} }}")


def rust_class(c):
    args = ", ".join(f'"{a}"' for a in c["args"])
    layout = c["layout"]
    if layout == "generic":
        lay = "Layout::Generic"
    elif layout == "uniform":
        rule = "None" if c["mask"] is None else f"Some({rust_rule(c['mask'])})"
        lay = f"Layout::Uniform({rust_grid(c)}, {rule})"
    elif layout == "edges":
        m = c["mask"]
        assert m["rule"] == "chips" and m["module"] == c["module_size"]
        lay = (f"Layout::Edges({rust_grid(c)}, Chips {{ module: ({m['module'][0]}, "
               f"{m['module'][1]}), border: {c['border']!r} }})")
    else:
        lay = f'Layout::Unsupported("{layout}")'
    force = "true" if c.get("force_pixel") else "false"
    return (f'    Class {{ name: "{c["class"]}", args: &[{args}], force_pixel: {force}, '
            f"layout: {lay} }},\n")


def write_table(doc):
    lines = ["// @generated by golden/pyfai_detectors.py from pyFAI "
             f"{doc['pyfai']}; do not edit.\n",
             "use super::{Chips, Class, Grid, Layout, Rule};\n\n",
             "/// Every class Detector.factory reaches, sorted by name.\n",
             "pub(super) static CLASSES: &[Class] = &[\n"]
    lines += [rust_class(c) for c in doc["classes"]]
    lines += ["];\n\n",
              "/// Detector.factory's first lookup: the Detector subclasses among the\n",
              "/// attributes of pyFAI.detectors, sorted by attribute name.\n",
              "pub(super) static ATTRIBUTES: &[(&str, &str)] = &[\n"]
    lines += [f'    ("{k}", "{v}"),\n' for k, v in sorted(doc["attributes"].items())]
    lines += ["];\n\n",
              "/// Detector.registry: lower-case names and aliases, sorted.\n",
              "pub(super) static REGISTRY: &[(&str, &str)] = &[\n"]
    lines += [f'    ("{k}", "{v}"),\n' for k, v in sorted(doc["registry"].items())]
    lines += ["];\n"]
    for line in lines:
        assert "\\" not in line
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    with open(TABLE, "w") as f:
        f.writelines(lines)


def main():
    rng = numpy.random.default_rng(20261123)
    attributes = {n: c.__name__ for n in dir(pyFAI.detectors)
                  if isinstance(c := getattr(pyFAI.detectors, n), type) and issubclass(c, Detector)}
    registry = {k: c.__name__ for k, c in ALL_DETECTORS.items()}
    classes = {c.__name__: c for c in ALL_DETECTORS.values()}
    for n in attributes:
        c = getattr(pyFAI.detectors, n)
        assert classes.setdefault(c.__name__, c) is c
    described = [describe(n, classes[n], rng) for n in sorted(classes)]
    doc = {
        "program": f"pyFAI {pyFAI.version} Detector.factory, calc_mask, "
                   "calc_cartesian_positions, PoniFile.write and Geometry.calc_pos_zyx",
        "pyfai": pyFAI.version,
        "numpy": numpy.__version__,
        "note": "pixel: repr of the default instance's pixel1, pixel2; fingerprint: "
                "count, sum and sum of squares mod 2**64 of the row-major indices of "
                "the masked pixels; zyx = calc_pos_zyx = (t3, t1, t2), by index for a "
                "uniform class and from the whole image for ImXPad; zyx_by_index: "
                "ImXPad's calc_pos_zyx(d1, d2); a case's binning is its Detector_config's "
                "binning key, null for the default instance",
        "attributes": attributes,
        "registry": registry,
        "classes": described,
    }
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)  # the factory takes an existing file name for a NeXus file
        try:
            doc["probes"] = probes(attributes, registry)
        finally:
            os.chdir(cwd)
        uniform = [c for c in described if c["layout"] == "uniform"]
        doc["cases"] = [case(c, tmp, 1 + i % 4) for i, c in enumerate(uniform)]
        doc["cases"] += [case(c, tmp, o) for c in described if c["layout"] == "edges"
                         for o in (1, 2, 3, 4)]
        # one binned instance of each binnable class whose mask pyFAI bins
        binnable = [c for c in uniform if c["binnable"]]
        for i, c in enumerate(binnable):
            usable = [tuple(e["binning"]) for e in c["binned"] if e["mask_error"] is None
                      and e["binning"] != c["binning"] and 0 not in e["shape"]]
            if usable:
                doc["cases"].append(case(c, tmp, 1 + (i + 1) % 4, usable[i % len(usable)]))
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    write_table(doc)
    counts = {}
    for c in described:
        counts[c["layout"]] = counts.get(c["layout"], 0) + 1
    print(counts)
    print("not binnable:", [c["class"] for c in described if c.get("binnable") is False])
    print("zero pitch not default:", [(c["class"], c["zero_pitch"]["both"]) for c in described
                                      if c.get("zero_pitch_default") is False])
    print("mask raises:", [(c["class"], e["binning"]) for c in described
                           for e in c.get("binned", []) if e["mask_error"]])


if __name__ == "__main__":
    main()
