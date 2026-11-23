"""Independent first-principles reference (numpy only; no hklrs, hkl or xu).

Frame: x along the beam, z up, right-handed rotations, Q = k_f - k_i with
|k| = 2*pi/lambda, hkl = (U B)^-1 R_sample^-1 Q.
"""
import math

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


def forward(sample_stack, det_stack, wl, point=None):
    """sample_stack/det_stack: [(axis, angle_rad)] outermost first.
    Returns R_sample, R_detector, q_lab (2 pi, no wavelength-dependent
    approximation), q_sample, kf."""
    rs = np.eye(3)
    for ax, ang in sample_stack:
        rs = rs @ rot(ax, ang)
    rd = np.eye(3)
    for ax, ang in det_stack:
        rd = rd @ rot(ax, ang)
    p = X if point is None else np.asarray(point, float)
    kf = rd @ p
    kf = kf / np.linalg.norm(kf)
    q_lab = 2 * math.pi / wl * (kf - X)
    return rs, rd, q_lab, rs.T @ q_lab, kf


# Hand-written stacks (axis names, rotation axes, outermost first) of the
# hkl presets, from the instrument drawings in hkl's documentation.
KAPPA_ALPHA = math.radians(50.0)
KAPPA = np.array([0.0, -math.cos(KAPPA_ALPHA), -math.sin(KAPPA_ALPHA)])
MY = -Y
PRESET_STACKS = {
    "E4CV": ([("omega", MY), ("chi", X), ("phi", MY)], [("tth", MY)]),
    "E4CH": ([("omega", Z), ("chi", X), ("phi", Z)], [("tth", Z)]),
    "E6C": ([("mu", Z), ("omega", MY), ("chi", X), ("phi", MY)],
            [("gamma", Z), ("delta", MY)]),
    "K4CV": ([("komega", MY), ("kappa", KAPPA), ("kphi", MY)], [("tth", MY)]),
}


def stacks_from_angles(name, angles_deg):
    """[(axis, rad)] stacks for preset `name`; angles in preset axis order
    (sample axes then detector axes, as hkl lists them)."""
    s, d = PRESET_STACKS[name]
    names = [n for n, _ in s] + [n for n, _ in d]
    val = dict(zip(names, (math.radians(a) for a in angles_deg)))
    return ([(ax, val[n]) for n, ax in s], [(ax, val[n]) for n, ax in d])
