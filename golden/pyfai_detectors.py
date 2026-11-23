"""pyFAI 2026.9.0's detector classes as Geometry.load builds a PONI file's
detector (Detector.factory of its name and Detector_config), for hklrs's
PONI import.

Every Detector subclass the factory reaches, by registry key or by attribute
of pyFAI.detectors, with its constructor arguments and the default instance's
pitch, shape, max_shape, binning and orientation, and the layout hklrs gives
it: "generic" (Detector itself), "uniform" (calc_cartesian_positions equals
the generic Detector of that pitch, shape and orientation, both index paths,
every orientation), "edges" (ImXPad: the chip rule gives the pixel edges of
calc_cartesian_positions over the whole image, every orientation), or the
reason it has no model. A modelled class's mask rule reproduces its calc_mask
and Detector.mask element by element in every orientation; the golden keeps a
fingerprint of that mask. "pitch" is how the class's constructor takes a
given pitch: "fixed" (it takes none), "as given", "rounded" (force_pixel:
binned by the rounded ratio to PIXEL_SIZE, the pitch kept), "rebinned"
(Apex2, Pixium), "perkin", "table" (a Rayonix's BINNED_PIXEL_SIZE) or
"plate" (Mar345 scales max_shape by 100e-6 over the pitch).

"reads" are the instances Geometry.load builds from files of the class with
other pitches and binnings (heads): each file's Detector lines and the
instance's class, pitch, shape, max_shape and binning, or the exception it
raises; Detector.mask's fingerprint or exception where the instance has a
positive pitch, a binning and a shape within the default's max_shape, whose
positions, for a uniform class, are asserted to be the generic Detector's of
that pitch, shape and orientation. "round_trip" is, at each of BINNINGS, the
factory's binning key on the default instance, PoniFile.write of a Geometry
of it and Geometry.load of that, with what the load builds, its mask and
calc_pos_zyx at some pixels. "probes" are names put to detector_factory and
the class it returns; "cases" are a PONI file pyFAI writes for a modelled
class, default or binned by the binning key where that binning gives the
class's own pitch and the mask, and, after Geometry.load, calc_pos_zyx of
some of its pixels where the image is flipped once per orientation: by index
for a uniform class, whose index path is its whole-image path, over the
whole image for ImXPad, whose index path flips twice ("zyx_by_index",
upstream defect FAI-2).

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
from pyFAI.detectors._rayonix import _Rayonix
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
# the classes whose __init__ hands pixel1 and pixel2 to Detector.__init__ as
# given
PASS = {"Detector", "ModuleDetector", "Eiger", "Pilatus", "Pilatus4", "Maxipix", "_Lambda",
        "Jungfrau4M", "ImXPadS10", "ImXPadS70", "Mar555"}
RATIOS = [1.5, 0.5, 2.4, 2.5, 0.9999]
GEOMETRY = ("Distance: 0.3\nPoni1: 0.01\nPoni2: 0.02\nRot1: 0.1\nRot2: -0.2\nRot3: 0.3\n"
            "Wavelength: 1e-10\n")


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


def pitch_rule(cls, det):
    """how cls's constructor takes a given pitch, by the class that defines
    __init__; the reads test it against pyFAI"""
    params = inspect.signature(cls).parameters
    if "pixel1" not in params:
        assert "pixel2" not in params and not cls.force_pixel, cls
        return {"rule": "fixed"}
    size = [float(s) for s in cls.PIXEL_SIZE] if "PIXEL_SIZE" in dir(cls) else None
    rounded = bool(cls.force_pixel) and size is not None
    # a pitch the file omits is the constructor's default, which gives the
    # default instance's: as given, or None, which force_pixel takes as size
    defaults = (params["pixel1"].default, params["pixel2"].default)
    assert defaults == (det.pixel1, det.pixel2) or (
        defaults == (None, None) and rounded and size == [det.pixel1, det.pixel2]), cls
    own = owner(cls, "__init__")
    if issubclass(cls, _Rayonix):
        assert not rounded, cls
        return {"rule": "table",
                "table": [[int(b), float(p)] for b, p in cls.BINNED_PIXEL_SIZE.items()]}
    if own in PASS:
        return {"rule": "rounded", "size": size} if rounded else {"rule": "as given"}
    assert rounded == (own != "Mar345"), cls
    if own in ("Apex2", "Pixium"):
        unit = size if own == "Apex2" else [cls.DEFAULT_PIXEL1, cls.DEFAULT_PIXEL2]
        return {"rule": "rebinned", "size": size, "unit": unit}
    if own == "Perkin":
        return {"rule": "perkin", "size": size, "unit": [cls.DEFAULT_PIXEL1, cls.DEFAULT_PIXEL2]}
    if own == "Mar345":
        # Mar345.__init__ scales MAX_SHAPE by 100e-6 / pixel
        return {"rule": "plate", "max_shape": [int(n) for n in cls.MAX_SHAPE], "unit": 100e-6}
    raise AssertionError(f"{cls.__name__}: no pitch rule for {own}.__init__")


def facts(det):
    shape = None if det.shape is None else [int(n) for n in det.shape]
    return {"class": type(det).__name__,
            "pixel": [None if p is None else repr(p) for p in (det.pixel1, det.pixel2)],
            "shape": shape,
            "max_shape": None if det.max_shape is None else [int(n) for n in det.max_shape],
            "binning": [int(b) for b in det.binning]}


def in_range(det, doc):
    """a class's instance with a positive pitch, a binning and a shape within
    the default instance's max_shape"""
    return (type(det).__name__ == doc["class"]
            and all(p is not None and 0 < p < numpy.inf for p in (det.pixel1, det.pixel2))
            and all(b >= 1 for b in det.binning)
            and all(1 <= n <= m for n, m in zip(det.shape, doc["max_shape"])))


def mask_outcome(det, masks):
    """Detector.mask's fingerprint, or the exception it raises"""
    key = (type(det).__name__, tuple(det.shape), tuple(det.max_shape), tuple(det.binning))
    if key not in masks:
        try:
            m = det.mask
        except Exception as err:
            masks[key] = {"mask_error": type(err).__name__}
        else:
            m = numpy.zeros(det.shape, dtype=bool) if m is None else numpy.asarray(m) != 0
            assert m.shape == tuple(det.shape), key
            masks[key] = {"mask": fingerprint(m)}
    return masks[key]


def assert_generic(det, rng, seen):
    """calc_cartesian_positions(d1, d2) of det is the generic Detector's of
    its pitch, shape and orientation"""
    key = (type(det).__name__, det.pixel1, det.pixel2, tuple(det.shape), int(det.orientation))
    if key in seen:
        return
    seen.add(key)
    n0, n1 = det.shape
    d1 = numpy.concatenate([[0, n0 - 1, 0, n0 - 1], rng.integers(0, n0, 200)]).astype(float)
    d2 = numpy.concatenate([[0, 0, n1 - 1, n1 - 1], rng.integers(0, n1, 200)]).astype(float)
    ref = Detector(pixel1=det.pixel1, pixel2=det.pixel2, max_shape=det.shape,
                   orientation=int(det.orientation))
    a, b = det.calc_cartesian_positions(d1, d2), ref.calc_cartesian_positions(d1, d2)
    assert a[2] is None and numpy.array_equal(a[0], b[0]) and numpy.array_equal(a[1], b[1]), key


def heads(doc):
    """the detector lines of the files read: a Detector_config of pitches
    (the native pitch at each of BINNINGS, a Rayonix table's, RATIOS of the
    default's, one axis, 0, null, negative), of binnings, of both, and
    version 1's pixel sizes"""
    name, rule = doc["class"], doc["pitch"]
    (p0, p1), (b0, b1) = [float(p) for p in doc["pixel"]], doc["binning"]

    def native(b):
        return {"pixel1": p0 * b[0] / b0, "pixel2": p1 * b[1] / b1}

    two = native((2, 2))
    configs = [native(b) for b in BINNINGS]
    if rule["rule"] == "table":
        configs += [{"pixel1": p, "pixel2": p} for _, p in rule["table"]]
    configs += [{"pixel1": p0 * r, "pixel2": p1 * r} for r in RATIOS]
    configs += [{"pixel1": two["pixel1"]}, {"pixel2": two["pixel2"]},
                {"pixel1": 0, "pixel2": 0}, {"pixel1": None, "pixel2": None},
                {"pixel1": -p0, "pixel2": -p1}, {"pixel1": 0}]
    configs += [{"binning": list(b)} for b in BINNINGS]
    configs += [{"binning": [0, 0]}, {"binning": 2}, {"binning": 0}]
    configs += [{**two, "binning": [1, 1]}, {**two, "binning": [2, 2]},
                {"pixel1": 0, "pixel2": 0, "binning": [2, 2]},
                {"pixel1": p0 * 1.5, "pixel2": p1 * 1.5, "binning": [2, 2]}]
    out = [f"poni_version: 2.1\nDetector: {name}\nDetector_config: {json.dumps(c)}\n"
           for c in configs]
    for p in (native((1, 1)), two, {"pixel1": 0.0, "pixel2": 0.0},
              {"pixel1": p0 * 1.5, "pixel2": p1 * 1.5}):
        out.append(f"Detector: {name}\nPixelSize1: {p['pixel1']!r}\nPixelSize2: {p['pixel2']!r}\n")
    return out


def load(text, tmp):
    """Geometry.load of a PONI file of text, or the exception's name"""
    path = os.path.join(tmp, "read.poni")
    with open(path, "w") as f:
        f.write(text)
    geo = Geometry()
    try:
        geo.load(path)
    except Exception as err:
        return None, type(err).__name__
    return geo, None


def reads(doc, tmp, masks, rng, seen):
    out = []
    for head in heads(doc):
        geo, err = load(head + GEOMETRY, tmp)
        if geo is None:
            out.append({"head": head, "raises": err})
            continue
        det = geo.detector
        entry = {"head": head, **facts(det)}
        if in_range(det, doc):
            entry.update(mask_outcome(det, masks))
            if doc["layout"] == "uniform":
                assert_generic(det, rng, seen)
        out.append(entry)
    return out


def written(geo, tmp):
    """PoniFile.write of geo, without the date line, which changes on every run"""
    path = os.path.join(tmp, "written.poni")
    with open(path, "w") as f:
        PoniFile(geo).write(f)
    with open(path) as f:
        text = f.read()
    return "".join(line for line in text.splitlines(True) if not line.startswith("# Calibration done"))


def sample(doc, shape):
    n0, n1 = shape
    pixels = [(0, 0), (n0 - 1, 0), (0, n1 - 1), (n0 - 1, n1 - 1), (n0 // 3, n1 // 2)]
    if doc["layout"] == "edges":
        # either side of a chip boundary
        m0, m1 = doc["module_size"]
        pixels += [(n0 // 3, m1 - 1), (min(m0, n0 - 1), min(m1, n1 - 1))]
    return pixels


def zyx_at(geo, doc, pixels):
    """calc_pos_zyx at pixels: by index for a uniform class, from the whole
    image for ImXPad (FAI-2)"""
    d1 = numpy.array([p[0] for p in pixels], dtype=float)
    d2 = numpy.array([p[1] for p in pixels], dtype=float)
    by_index = geo.calc_pos_zyx(d1=d1, d2=d2)
    if doc["layout"] == "edges":
        whole = geo.calc_pos_zyx()
        zyx = [w[d1.astype(int), d2.astype(int)] for w in whole]
    else:
        zyx = by_index
    return ([[float(c[k]) for c in zyx] for k in range(len(pixels))],
            [[float(c[k]) for c in by_index] for k in range(len(pixels))])


def round_trip(doc, tmp, masks):
    """at each of BINNINGS: the factory's binning key on the default
    instance, PoniFile.write of a Geometry of it, and Geometry.load of that"""
    out = []
    for b in BINNINGS:
        det = detector_factory(doc["class"], {"binning": list(b)})
        n0, n1 = det.shape
        geo = Geometry(dist=0.31, poni1=0.37 * n0 * det.pixel1, poni2=0.61 * n1 * det.pixel2,
                       rot1=0.1, rot2=-0.2, rot3=0.3, detector=det, wavelength=1e-10)
        text = written(geo, tmp)
        entry = {"binning": list(b), "written": facts(det), "poni": text}
        back, err = load(text, tmp)
        if back is None:
            entry["raises"] = err
        else:
            entry["read"] = facts(back.detector)
            if in_range(back.detector, doc):
                entry.update(mask_outcome(back.detector, masks))
            # ImXPad's chip edges span max_shape whatever the binning
            if in_range(back.detector, doc) and (
                    doc["layout"] == "uniform" or entry["read"]["binning"] == doc["binning"]):
                pixels = sample(doc, back.detector.shape)
                entry["pixels"] = [list(p) for p in pixels]
                entry["zyx"] = zyx_at(back, doc, pixels)[0]
        out.append(entry)
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
    if layout == "edges":
        # calc_pixels_edges spans max_shape and PIXEL_SIZE whatever the binning
        two = detector_factory(name, {"binning": [2, 2]})
        assert two.shape != two.max_shape and two.calc_cartesian_positions()[0].shape[0] == two.max_shape[0]
    return {**doc, "layout": layout, "mask": rule, "fingerprint": fingerprint(mask),
            "pitch": pitch_rule(cls, det)}


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


def native(doc, entry):
    """entry's pitch is its class's at entry's binning: a Rayonix's table's,
    else the default instance's scaled by the binning"""
    if entry.get("pixel") is None or None in entry["pixel"]:
        return False
    rule, out = doc["pitch"], []
    for k in (0, 1):
        b = entry["binning"][k]
        if rule["rule"] == "table":
            want = dict(rule["table"]).get(b)
        elif rule["rule"] == "plate":
            want = float(entry["pixel"][k])
        else:
            want = float(doc["pixel"][k]) * b / doc["binning"][k]
        out.append(want is not None and abs(float(entry["pixel"][k]) - want) <= 1e-9 * want)
    return all(out)


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
    text = written(geo, tmp)
    if b is not None:
        # PoniFile.write gives the binned pitch, not the binning
        config = json.dumps({"binning": list(b), "orientation": o})
        lines = text.splitlines(True)
        k = next(i for i, line in enumerate(lines) if line.startswith("Detector_config:"))
        lines[k] = f"Detector_config: {config}\n"
        text = "".join(lines)
    back, err = load(text, tmp)
    assert err is None, err
    assert type(back.detector) is cls and int(back.detector.orientation) == o
    assert (back.detector.pixel1, back.detector.pixel2, back.detector.shape) == (
        det.pixel1, det.pixel2, det.shape)
    pixels = sample(doc, det.shape)
    mask = back.detector.mask
    mask = numpy.zeros(det.shape, dtype=bool) if mask is None else numpy.asarray(mask) != 0
    out = {"class": doc["class"], "orientation": o, "binning": None if b is None else list(b),
           "poni": text, "shape": [int(n) for n in det.shape], "fingerprint": fingerprint(mask),
           "pixels": [list(p) for p in pixels]}
    out["zyx"], by_index = zyx_at(back, doc, pixels)
    if doc["layout"] == "edges":
        out["zyx_by_index"] = by_index
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
            f"module_size: {rust_pair(c['module_size'])} }}")


def rust_floats(v):
    return f"({v[0]!r}, {v[1]!r})"


def rust_pitch(p):
    rule = p["rule"]
    if rule == "fixed":
        return "Pitch::Fixed"
    if rule == "as given":
        return "Pitch::AsGiven"
    if rule == "rounded":
        return f"Pitch::Rounded {{ size: {rust_floats(p['size'])} }}"
    if rule in ("rebinned", "perkin"):
        kind = "Rebinned" if rule == "rebinned" else "Perkin"
        return f"Pitch::{kind} {{ size: {rust_floats(p['size'])}, unit: {rust_floats(p['unit'])} }}"
    if rule == "table":
        return "Pitch::Table(&[" + ", ".join(f"({b}, {v!r})" for b, v in p["table"]) + "])"
    assert rule == "plate", p
    m = p["max_shape"]
    return f"Pitch::Plate {{ max_shape: ({m[0]}, {m[1]}), unit: {p['unit']!r} }}"


def rust_class(c):
    args = ", ".join(f'"{a}"' for a in c["args"])
    layout = c["layout"]
    if layout == "generic":
        lay = "Layout::Generic"
    elif layout == "uniform":
        rule = "None" if c["mask"] is None else f"Some({rust_rule(c['mask'])})"
        lay = f"Layout::Uniform({rust_grid(c)}, {rust_pitch(c['pitch'])}, {rule})"
    elif layout == "edges":
        m = c["mask"]
        assert m["rule"] == "chips" and m["module"] == c["module_size"]
        lay = (f"Layout::Edges({rust_grid(c)}, {rust_pitch(c['pitch'])}, Chips {{ module: "
               f"({m['module'][0]}, {m['module'][1]}), border: {c['border']!r} }})")
    else:
        lay = f'Layout::Unsupported("{layout}")'
    force = "true" if c.get("force_pixel") else "false"
    return (f'    Class {{ name: "{c["class"]}", args: &[{args}], force_pixel: {force}, '
            f"layout: {lay} }},\n")


def write_table(doc):
    lines = ["// @generated by golden/pyfai_detectors.py from pyFAI "
             f"{doc['pyfai']}; do not edit.\n",
             "use super::{Chips, Class, Grid, Layout, Pitch, Rule};\n\n",
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
        modelled = [c for c in described if c["layout"] in ("uniform", "edges")]
        masks, seen = {}, set()
        for c in modelled:
            c["reads"] = reads(c, tmp, masks, rng, seen)
            c["round_trip"] = round_trip(c, tmp, masks)
        uniform = [c for c in modelled if c["layout"] == "uniform"]
        doc["cases"] = [case(c, tmp, 1 + i % 4) for i, c in enumerate(uniform)]
        doc["cases"] += [case(c, tmp, o) for c in modelled if c["layout"] == "edges"
                         for o in (1, 2, 3, 4)]
        # one instance of each uniform class binned by the binning key alone
        # to the class's pitch at that binning, whose mask pyFAI bins
        for i, c in enumerate(uniform):
            usable = []
            for e in c["reads"]:
                config = json.loads(e["head"].split("Detector_config: ")[1]) if (
                    "Detector_config: " in e["head"]) else {}
                if (set(config) == {"binning"} and isinstance(config["binning"], list)
                        and "mask" in e and native(c, e) and e["binning"] != c["binning"]):
                    usable.append(tuple(config["binning"]))
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
    every = [(c["class"], e) for c in modelled for e in c["reads"]]
    print("reads:", len(every), "raise:", sorted({(n, e["raises"]) for n, e in every if "raises" in e}))
    print("no pitch:", sorted({n for n, e in every if "pixel" in e and None in e["pixel"]}))
    print("mask raises:", sorted({(n, tuple(e["binning"]), e["mask_error"]) for n, e in every
                                  if "mask_error" in e}))
    print("round trip loses the binning:", sorted({(c["class"], tuple(e["binning"])) for c in modelled
                                                   for e in c["round_trip"] if "read" in e
                                                   and e["read"]["binning"] != e["binning"]}))
    print("round trip raises:", sorted({(c["class"], tuple(e["binning"]), e["raises"])
                                        for c in modelled for e in c["round_trip"] if "raises" in e}))


if __name__ == "__main__":
    main()
