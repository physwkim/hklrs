"""Golden kappa <-> eulerian conversion from libhkl 5.1.9 (gi Hkl 5.0).
Run: python -I hkl_eulerians.py

The "eulerians" engine of K4CV and K6C (hkl-pseudoaxis-common-eulerians.c,
alpha = 50 deg hard-coded there and in the kappa axis), for both values of
its "solutions" parameter:
- forward: axes set, engine list get, the engine's (omega, chi, phi);
- inverse: (omega, chi, phi) set from a start geometry; every solution hkl
  returns (its kappa twin and axis-range images included, sorted by hkl), or
  the error text. Out-of-reach chi (|chi| > 2*alpha, HKL-9), chi given
  outside (-180, 180] (HKL-19) and chi within 1e-11 deg of 2*alpha, where
  kappa is ill-conditioned in chi, are included.
Angles are degrees, as hkl's user unit for these axes.
"""
import json
import os
import random

import gi

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

SEED = 20261123
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_eulerians.json")
GEOMETRIES = ["K4CV", "K6C"]
SOLUTIONS = [0.0, 1.0]
N_RANDOM = 30
# kappa at and past the axis range: the get path restricts kappa to (-180, 180]
KAPPA_SPECIAL = [0.0, 180.0, -180.0, 90.0, -90.0, 1e-9, 200.0, -250.0, 359.0]
# chi at the reach 2*alpha = 100 deg, past it (HKL-9), and outside (-180, 180]
CHI_SPECIAL = [0.0, 100.0, -100.0, 99.999999, 100.0001, -100.0001, 150.0, -180.0,
               350.0, -300.0, 450.0]
# chi within 1e-11 deg of 2*alpha: asin near 1, kappa ill-conditioned in chi
CHI_NEAR_REACH = [99.99999999999999, -99.99999999999997, 99.9999999999999,
                  -99.9999999999997, 99.999999999999, -99.99999999999]


class Ctx:
    def __init__(self, gname):
        self.fac = Hkl.factories()[gname]
        self.geo = self.fac.create_new_geometry()
        self.names = list(self.geo.axis_names_get())
        self.det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.smp = Hkl.Sample.new("eul")
        self.el = self.fac.create_new_engine_list()
        self.el.init(self.geo, self.det, self.smp)
        self.eng = self.el.engine_get_by_name("eulerians")

    def set_axes(self, values):
        self.geo.axis_values_set([float(v) for v in values], USER)

    def axes(self, geo=None):
        return [float(v) for v in (geo or self.geo).axis_values_get(USER)]

    def solutions(self, s):
        self.eng.parameters_values_set([s], USER)


def named(ctx, values):
    return dict(zip(ctx.names, values))


def forward_case(ctx, axes, s):
    case = {"direction": "forward", "geometry": ctx.gname, "solutions": s,
            "axes": named(ctx, axes)}
    try:
        ctx.solutions(s)
        ctx.set_axes(axes)
        case["axes_set"] = named(ctx, ctx.axes())
        ctx.el.get()
        case["eulerians"] = [float(v) for v in ctx.eng.pseudo_axis_values_get(USER)]
    except Exception as e:  # recorded, never dropped
        case["error"] = f"{type(e).__name__}: {e}"
    return case


def inverse_case(ctx, eulerians, start, s):
    case = {"direction": "inverse", "geometry": ctx.gname, "solutions": s,
            "eulerians": list(eulerians), "start_axes": named(ctx, start)}
    try:
        ctx.solutions(s)
        ctx.set_axes(start)
        sols = ctx.eng.pseudo_axis_values_set([float(v) for v in eulerians], USER)
        case["solution_axes"] = [named(ctx, ctx.axes(it.geometry_get()))
                                 for it in sols.items()]
    except Exception as e:  # recorded, never dropped
        case["error"] = f"{type(e).__name__}: {e}"
    return case


def main():
    cases = []
    for gname in GEOMETRIES:
        rng = random.Random(f"{SEED}-{gname}")
        ctx = Ctx(gname)
        ctx.gname = gname

        def start():
            return [rng.uniform(-180, 180) for _ in ctx.names]

        kappas = [rng.uniform(-180, 180) for _ in range(N_RANDOM)] + KAPPA_SPECIAL
        for kappa in kappas:
            axes = start()
            axes[ctx.names.index("kappa")] = kappa
            for s in SOLUTIONS:
                cases.append(forward_case(ctx, axes, s))
        chis = [rng.uniform(-100, 100) for _ in range(N_RANDOM)] + CHI_SPECIAL + CHI_NEAR_REACH
        for chi in chis:
            eul = [rng.uniform(-180, 180), chi, rng.uniform(-180, 180)]
            st = start()
            for s in SOLUTIONS:
                cases.append(inverse_case(ctx, eul, st, s))
    doc = {"hkl_version": Hkl.VERSION, "seed": SEED, "alpha_deg": 50.0,
           "units": {"angle": "deg"}, "cases": cases}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    fwd = [c for c in cases if c["direction"] == "forward"]
    inv = [c for c in cases if c["direction"] == "inverse"]
    print("wrote", OUT, "forward", len(fwd), "inverse", len(inv))
    print("forward errors", sum("error" in c for c in fwd),
          "axes changed by set", sum(c.get("axes_set") != c["axes"] for c in fwd))
    print("inverse errors", sorted({c["error"] for c in inv if "error" in c}),
          sum("error" in c for c in inv))
    print("inverse solution counts", sorted({len(c.get("solution_axes", [])) for c in inv}))


main()
