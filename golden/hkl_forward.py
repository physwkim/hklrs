"""Golden forward vectors from libhkl (gi Hkl 5.0). Run: python -I hkl_forward.py"""
import json
import math
import os
import random

import gi

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

SEED = 20260101
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_forward.json")
SAMPLES = {
    "cubic": dict(a=1.54, b=1.54, c=1.54, alpha=90.0, beta=90.0, gamma=90.0,
                  ux=0.0, uy=0.0, uz=0.0),
    "triclinic": dict(a=4.1, b=5.3, c=6.7, alpha=82.0, beta=97.0, gamma=101.0,
                      ux=12.0, uy=-31.0, uz=47.0),
}
WAVELENGTHS = [1.54, 0.71]
N_RANDOM = 10


def mat(m):
    return [[m.get(i, j) for j in range(3)] for i in range(3)]


def vec(v):
    if v is None:  # e.g. NANOSCOPIUM ROBOT 'r' (Meter) has no axis vector
        return None
    return [float(x) for x in v.data]


# user unit -> (kind, factor from canonical deg|mm to user unit)
UNIT = {"Degree": ("rot", 1.0), "Milli Radian": ("rot", math.pi / 180.0 * 1000.0),
        "Milli Meter": ("trans", 1.0), "Meter": ("trans", 1e-3)}


def main():
    factories = Hkl.factories()
    names = list(factories.keys())
    rng = random.Random(SEED)
    geometries = {}
    cases = []
    for gname in names:
        fac = factories[gname]
        geo = fac.create_new_geometry()
        axes = []
        for an in geo.axis_names_get():
            ax = geo.axis_get(an)
            axes.append({"name": an, "axis_v": vec(ax.axis_v_get()),
                         "unit": ax.user_unit_get()})
        geometries[gname] = {"axes": axes}
        units = [a["unit"] for a in axes]
        for sname, s in SAMPLES.items():
            for wl in WAVELENGTHS:
                for k in range(N_RANDOM):
                    canon = [rng.uniform(-50.0, 50.0) if UNIT[u][0] == "trans"
                             else rng.uniform(-180.0, 180.0) for u in units]
                    vals = [c * UNIT[u][1] for c, u in zip(canon, units)]
                    case = {"geometry": gname, "sample": sname, "lattice": dict(s),
                            "wavelength": wl, "axes": dict(zip([a["name"] for a in axes], vals)),
                            "axes_canonical_deg_mm": dict(zip([a["name"] for a in axes], canon)),
                            "index": k}
                    try:
                        geo = fac.create_new_geometry()
                        geo.wavelength_set(wl, USER)
                        geo.axis_values_set(vals, USER)
                        det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
                        sample = Hkl.Sample.new("golden")
                        sample.lattice_set(Hkl.Lattice.new(
                            s["a"], s["b"], s["c"], math.radians(s["alpha"]),
                            math.radians(s["beta"]), math.radians(s["gamma"])))
                        sample.U_set(Hkl.Matrix.new_euler(
                            math.radians(s["ux"]), math.radians(s["uy"]),
                            math.radians(s["uz"])))
                        case["U"] = mat(sample.U_get())
                        case["UB"] = mat(sample.UB_get())
                        el = fac.create_new_engine_list()
                        el.init(geo, det, sample)
                        eng = el.engine_get_by_name("hkl")
                        if eng is None:
                            raise RuntimeError("no 'hkl' engine in this geometry")
                        el.get()
                        case["pseudo_names"] = list(eng.pseudo_axis_names_get())
                        case["hkl"] = [float(x) for x in eng.pseudo_axis_values_get(USER)]
                    except Exception as e:  # recorded, never dropped
                        case["error"] = f"{type(e).__name__}: {e}"
                    cases.append(case)
    doc = {"hkl_version": Hkl.VERSION, "seed": SEED,
           "units": {"angle": "deg", "length": "mm", "wavelength": "angstrom"},
           "geometries": geometries, "cases": cases}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(cases))


main()
