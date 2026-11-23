"""Golden read-only pseudos (reals -> pseudo) from libhkl 5.1.9 (gi Hkl 5.0).
Run: python -I hkl_pseudo.py

Engines of hkl/hkl-pseudoaxis-common-q.c: "q" (E4CV, E4CH, K4CV, MARS),
"q2" (q, alpha), "tth2" (tth, alpha; common-tth.c), "qper_qpar", and "incidence"/"emergence"
(common-readonly.c) (parameters x, y, z = surface normal in the
sample frame) on every other preset that carries them. Per case the axes are
set, engine_list.get() is called and the engine's pseudo values are recorded
in hkl's user units (alpha in degrees), with an independent numpy forward
built from hkl's own holder rotations (kf = Rd.x, ki = x, q = k (kf - ki)).
Cases: seeded random axes within hkl's ranges, the zero position, and
small detector angles (hkl_vector_angle acos precision, HKL-11).
"""
import json
import math
import os
import random

import gi
import numpy as np

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

SEED = 20260302
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_pseudo.json")
WL = 1.54
N_RANDOM = 40
UNIT = {"Degree": ("rot", 1.0), "Milli Radian": ("rot", math.pi / 180.0 * 1000.0),
        "Milli Meter": ("trans", 1.0), "Meter": ("trans", 1e-3)}
ENGINES = ("q", "q2", "qper_qpar", "tth2", "incidence", "emergence")
NORMALS = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (-1.0, 2.0, 1.0), (0.0, 3.0, 4.0)]
AZ_MIN = 1e-9  # |projection of the unit normal| below which azimuth is None (undefined)


def mat(m):
    return np.array([[m.get(i, j) for j in range(3)] for i in range(3)])


def numpy_pseudo(geo, smp, det, n):
    """q, q_signed (negative when kf.y<0 or kf.z<0), alpha, qper, qpar from first principles."""
    rs = mat(geo.sample_rotation_get(smp).to_matrix())
    rd = mat(geo.detector_rotation_get(det).to_matrix())
    k = 2 * math.pi / WL
    ki = k * np.array([1.0, 0, 0])
    kf = k * rd @ np.array([1.0, 0, 0])
    tth = math.atan2(np.linalg.norm(np.cross(ki, kf)), ki @ kf)
    q = 2 * k * math.sin(tth / 2)
    # engine "q" signs it by kf (common-q.c:96); "q2" does not (common-q.c:141)
    q_signed = -q if (kf[1] < 0 or kf[2] < 0) else q
    out = {"q": q, "q_signed": q_signed, "alpha": math.degrees(math.atan2(kf[2], kf[1])), "tth": math.degrees(tth)}
    if n is not None:
        qv = kf - ki
        nn = rs @ np.array(n, float)
        nn /= np.linalg.norm(nn)
        npar = np.cross(ki, nn)
        qper_v = nn * (qv @ nn)
        qpar_v = qv - qper_v
        out["qper"] = float(np.linalg.norm(qper_v)) * (-1 if qv @ nn < 0 else 1)
        out["qpar"] = float(np.linalg.norm(qpar_v)) * (-1 if qv @ npar < 0 else 1)
        out["qpar_sign_dot"] = float(qv @ npar)

        def ang(a, b):
            return math.atan2(np.linalg.norm(np.cross(a, b)), a @ b)

        def az(nv, normal):
            pr = nv - (nv @ normal) * normal
            return math.degrees(ang(pr, np.array([0.0, 1, 0]))) if np.linalg.norm(pr) > AZ_MIN else None
        x = np.array([1.0, 0, 0])
        ki_u = ki / k
        kf_u = kf / k
        out["incidence"] = math.degrees(ang(nn, ki_u) - math.pi / 2)
        out["emergence"] = math.degrees(math.pi / 2 - ang(nn, kf_u))
        out["azimuth_incidence"] = az(nn, ki_u)
        out["azimuth_emergence"] = az(nn, x)
    return out


def main():
    rng = random.Random(SEED)
    cases = []
    for gname, fac in Hkl.factories().items():
        geo = fac.create_new_geometry()
        geo.wavelength_set(WL, USER)
        det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        smp = Hkl.Sample.new("pseudo")
        el = fac.create_new_engine_list()
        el.init(geo, det, smp)
        found = {e.name_get(): e for e in el.engines_get() if e.name_get() in ENGINES}
        if not found:
            continue
        names = list(geo.axis_names_get())
        units = [geo.axis_get(n).user_unit_get() for n in names]
        ranges = {n: [v / UNIT[u][1] for v in geo.axis_get(n).min_max_get(USER)] for n, u in zip(names, units)}
        rot = [UNIT[u][0] == "rot" for u in units]

        def set_axes(canon):
            geo.axis_values_set([v * UNIT[u][1] for v, u in zip(canon, units)], USER)

        starts = [[0.0] * len(names)]
        for _ in range(N_RANDOM):
            starts.append([rng.uniform(*ranges[n]) if r else rng.uniform(-5, 5) for n, r in zip(names, rot)])
        # small detector angles: every rotation named tth/gamma/delta tiny, the rest random
        for scale in (1e-2, 1e-4, 1e-6):
            s = [rng.uniform(*ranges[n]) if r else 0.0 for n, r in zip(names, rot)]
            for i, n in enumerate(names):
                if n in ("tth", "gamma", "delta"):
                    s[i] = scale
            starts.append(s)
        for canon in starts:
            for ename, eng in found.items():
                params_list = [None]
                if ename in ("qper_qpar", "incidence", "emergence"):
                    params_list = [list(n) for n in NORMALS]
                for params in params_list:
                    set_axes(canon)
                    if params is not None:
                        eng.parameters_values_set(params, USER)
                    el.get()
                    vals = [float(x) for x in eng.pseudo_axis_values_get(USER)]
                    case = {"geometry": gname, "engine": ename, "wavelength": WL,
                            "axes": dict(zip(names, canon)),
                            "pseudo_names": list(eng.pseudo_axis_names_get()),
                            "parameters": params, "pseudo": vals,
                            "numpy": numpy_pseudo(geo, smp, det, params)}
                    cases.append(case)
    with open(OUT, "w") as f:
        json.dump({"hkl_version": "5.1.9", "seed": SEED, "cases": cases}, f, separators=(",", ":"))
    print(len(cases), "cases")


main()
