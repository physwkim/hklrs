"""Independent first-principles reference (numpy only; no hklrs, hkl or xu).

Frame: x along the beam, z up, right-handed rotations, Q = k_f - k_i with
|k| = 2*pi/lambda, hkl = (U B)^-1 R_sample^-1 Q.
"""
import math
from collections import namedtuple

import numpy as np

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def rot(axis, angle_rad):
    """Rodrigues rotation matrix, right-handed about `axis`."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * k + (1 - math.cos(angle_rad)) * (k @ k)


def metric(a, b, c, al, be, ga):
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    return np.array([
        [a * a, a * b * math.cos(ga), a * c * math.cos(be)],
        [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
        [a * c * math.cos(be), b * c * math.cos(al), c * c],
    ])


def b_matrix(a, b, c, al, be, ga):
    """Upper-triangular B with positive diagonal and B^T B = (2 pi)^2 G^-1
    (the Busing-Levy B), from a Cholesky factor of the reciprocal metric."""
    gstar = np.linalg.inv(metric(a, b, c, al, be, ga))
    return 2 * math.pi * np.linalg.cholesky(gstar).T


def b_matrix_busing_levy(a, b, c, al, be, ga):
    """The closed form of Busing & Levy (1967) eq. 3, with 2 pi."""
    al, be, ga = (math.radians(x) for x in (al, be, ga))
    v = a * b * c * math.sqrt(1 - math.cos(al) ** 2 - math.cos(be) ** 2
                              - math.cos(ga) ** 2 + 2 * math.cos(al) * math.cos(be) * math.cos(ga))
    a_, b_, c_ = (b * c * math.sin(al) / v, a * c * math.sin(be) / v, a * b * math.sin(ga) / v)
    cal = (math.cos(be) * math.cos(ga) - math.cos(al)) / (math.sin(be) * math.sin(ga))
    cbe = (math.cos(al) * math.cos(ga) - math.cos(be)) / (math.sin(al) * math.sin(ga))
    cga = (math.cos(al) * math.cos(be) - math.cos(ga)) / (math.sin(al) * math.sin(be))
    sga = math.sqrt(1 - cga * cga)
    sbe = math.sqrt(1 - cbe * cbe)
    return 2 * math.pi * np.array([
        [a_, b_ * cga, c_ * cbe],
        [0.0, b_ * sga, -c_ * sbe * math.cos(al)],
        [0.0, 0.0, 1.0 / c],
    ])


# Stack entry kinds besides a rotation axis (a 3-vector, value in rad):
Translation = namedtuple("Translation", "direction")  # value in m
RotationAbout = namedtuple("RotationAbout", "axis origin")  # pivot off the sample
PARAMETER = None  # a value that moves nothing


def transform(kind, value):
    """4x4 homogeneous transform of one stack entry at `value`."""
    t = np.eye(4)
    if kind is PARAMETER:
        return t
    if isinstance(kind, Translation):
        t[:3, 3] = value * np.asarray(kind.direction, float)
        return t
    axis, origin = kind if isinstance(kind, RotationAbout) else (kind, np.zeros(3))
    r = rot(axis, value)
    t[:3, :3] = r
    t[:3, 3] = origin - r @ origin
    return t


def pose(stack):
    p = np.eye(4)
    for kind, value in stack:
        p = p @ transform(kind, value)
    return p


def forward(sample_stack, det_stack, wl, point=None, source_stack=()):
    """sample_stack/det_stack/source_stack: [(kind, value)] outermost first.
    k_f points from the sample position to the detector pose applied to
    `point` (default +x), k_i is the source rotation applied to +x.
    Returns R_sample, R_detector, q_lab (2 pi, no wavelength-dependent
    approximation), q_sample, kf."""
    ps, pd = pose(sample_stack), pose(det_stack)
    p = X if point is None else np.asarray(point, float)
    kf = pd[:3, :3] @ p + pd[:3, 3] - ps[:3, 3]
    kf = kf / np.linalg.norm(kf)
    q_lab = 2 * math.pi / wl * (kf - pose(source_stack)[:3, :3] @ X)
    rs = ps[:3, :3]
    return rs, pd[:3, :3], q_lab, rs.T @ q_lab, kf


# Hand-written stacks (axis names, kinds, outermost first) of the hkl presets,
# from the hkl_holder_add_* calls in hkl 7be4f79 hkl/hkl-engine-*.c; PYFAI
# from pyFAI.
KAPPA_ALPHA = math.radians(50.0)
KAPPA = np.array([0.0, -math.cos(KAPPA_ALPHA), -math.sin(KAPPA_ALPHA)])
MX, MY, MZ = -X, -Y, -Z
LISA_D_SD = 0.5  # m, D_SD in hkl-engine-petra3-p08-lisa.c
PRESET_STACKS = {
    "E4CV": ([("omega", MY), ("chi", X), ("phi", MY)], [("tth", MY)]),
    "E4CH": ([("omega", Z), ("chi", X), ("phi", Z)], [("tth", Z)]),
    "E6C": ([("mu", Z), ("omega", MY), ("chi", X), ("phi", MY)],
            [("gamma", Z), ("delta", MY)]),
    "K4CV": ([("komega", MY), ("kappa", KAPPA), ("kphi", MY)], [("tth", MY)]),
    "K6C": ([("mu", Z), ("komega", MY), ("kappa", KAPPA), ("kphi", MY)],
            [("gamma", Z), ("delta", MY)]),
    "TwoC": ([("omega", MY)], [("tth", MY)]),
    "ZAXIS": ([("mu", Z), ("omega", MY)], [("mu", Z), ("delta", MY), ("gamma", Z)]),
    "SOLEIL MARS": ([("omega", MY), ("chi", MX), ("phi", Z)], [("tth", MY)]),
    "ESRF BM28 PSIC": ([("mu", Z), ("eta", MY), ("chi", X), ("phi", MY)],
                       [("nu", Z), ("delta", MY)]),
    "ESRF ID01 PSIC": ([("mu", MZ), ("eta", MY), ("phi", MZ)], [("nu", MZ), ("delta", MY)]),
    "APS POLAR": ([("tau", MY), ("mu", Z), ("chi", X), ("phi", Z)],
                  [("tau", MY), ("gamma", Z), ("delta", MY)]),
    "PETRA3 P09 EH2": ([("mu", MY), ("omega", Z), ("chi", X), ("phi", Z)],
                       [("mu", MY), ("delta", Z), ("gamma", MY)]),
    "PETRA3 P23 4C": ([("omega_t", MY), ("mu", MZ)], [("gamma", MZ), ("delta", MY)]),
    "PETRA3 P23 6C": ([("omega_t", MY), ("mu", MZ), ("omega", MY), ("chi", MX), ("phi", Y)],
                      [("gamma", MZ), ("delta", MY)]),
    "SOLEIL SIRIUS KAPPA": ([("mu", MZ), ("komega", MY), ("kappa", KAPPA), ("kphi", MY)],
                            [("delta", MZ), ("gamma", MY)]),
    "SOLEIL SIRIUS TURRET": ([("basepitch", Y), ("thetah", MZ), ("alphay", Y), ("alphax", X)],
                             [("basepitch", Y), ("delta", MZ), ("gamma", MY)]),
    "SOLEIL SIXS MED1+2": ([("pitch", MY), ("mu", Z)],
                           [("pitch", MY), ("gamma", Z), ("delta", MY)]),
    "SOLEIL SIXS MED2+2": ([("beta", MY), ("mu", Z), ("omega", MY)],
                           [("beta", MY), ("gamma", Z), ("delta", MY)]),
    "SOLEIL SIXS MED2+3": ([("beta", MY), ("mu", Z), ("omega", MY)],
                           [("beta", MY), ("gamma", Z), ("delta", MY), ("eta_a", MX)]),
    "SOLEIL SIXS MED2+3 v2": ([("mu", Z), ("omega", MY)],
                              [("gamma", Z), ("delta", MY), ("eta_a", MX)]),
    # dtx moves the detector point; hkl's default kf skips holder translations (HKL-15)
    "SOLEIL CRISTAL K4C": ([("komega", MY), ("kappa", KAPPA), ("kphi", MY)],
                           [("tth", MY), ("dtx", Translation(X))]),
    # r is a length no code reads (HKL-35)
    "SOLEIL NANOSCOPIUM ROBOT": ([("rz", MZ), ("rs", Y), ("rx", MX)],
                                [("r", PARAMETER), ("delta", MY), ("gamma", MZ)]),
    "PETRA3 P08 LISA": ([("sphi", Z)],
                        [("dtth", MZ), ("dh", Translation(Z)),
                         ("drot", RotationAbout(MY, np.array([LISA_D_SD, 0.0, 0.0])))]),
    # pyFAI geometry/core.py: pixel = rot3 rot2 rot1 (p1 - poni1, p2 - poni2, dist),
    # rot1/rot2 right-handed about -t1/-t2, rot3 about +t3; with x=t3, y=-t2,
    # z=-t1 rot3 turns about -x, not hkl's +x (HKL-37), and poni1, poni2, sdd
    # move the pixel, which hkl's kf skips (HKL-15)
    "PYFAI": ([], [("rot3", MX), ("rot2", MY), ("rot1", MZ), ("poni1", Translation(Z)),
                   ("poni2", Translation(Y)), ("sdd", Translation(X))]),
}
# Source stacks: k_i = R_source x. LISA's k_i0 = (cos a, 0, -sin a) is R(+y, a) x;
# hklrs takes a = _alpha_max(lambda) as the axis `tilt`.
PRESET_SOURCE = {"PETRA3 P08 LISA": [("mchi", X), ("tilt", Y)]}


def _stacks(name, angles_deg):
    """Sample, detector and source [(kind, value)] stacks of preset `name`;
    `angles_deg` in preset axis order (source, sample, then detector axes,
    each at its first holder, as hkl lists them), rotations in degrees,
    translations in metres."""
    src = PRESET_SOURCE.get(name, [])
    s, d = PRESET_STACKS[name]
    names = list(dict.fromkeys(n for n, _ in src + s + d))
    val = dict(zip(names, angles_deg, strict=True))

    def value(n, kind):
        moves_by_angle = kind is not PARAMETER and not isinstance(kind, Translation)
        return math.radians(val[n]) if moves_by_angle else val[n]

    return tuple([(kind, value(n, kind)) for n, kind in stack] for stack in (s, d, src))


def stacks_from_angles(name, angles_deg):
    """[(kind, value)] sample and detector stacks for preset `name`; see _stacks."""
    s, d, _ = _stacks(name, angles_deg)
    return s, d


def preset_forward(name, angles_deg, wl, point=None):
    """`forward` of preset `name`, source stack included."""
    s, d, src = _stacks(name, angles_deg)
    return forward(s, d, wl, point, src)


def triad(a, b):
    """Right-handed orthonormal triad from two non-parallel vectors."""
    t1 = a / np.linalg.norm(a)
    t3 = np.cross(a, b)
    t3 = t3 / np.linalg.norm(t3)
    return np.column_stack([t1, np.cross(t3, t1), t3])


def busing_levy_ub(b, hkls, qs):
    """Busing-Levy U and UB: hkl[0] matched exactly, hkl[1] in plane.
    `qs` are measured Q in the sample frame, `b` the B with 2 pi."""
    u = triad(np.asarray(qs[0], float), np.asarray(qs[1], float)) @ triad(
        b @ np.asarray(hkls[0], float), b @ np.asarray(hkls[1], float)).T
    return u, u @ b
