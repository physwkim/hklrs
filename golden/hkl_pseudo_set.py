"""Golden pseudo-axis engine solves (pseudo -> angles) from libhkl 5.1.9 (gi Hkl 5.0).
Run: python -I hkl_pseudo_set.py

Engines: "psi" (hkl-pseudoaxis-common-psi.c) on its 9 presets, "q", "q2",
"qper_qpar" (common-q.c) and "tth2" (common-tth.c) on theirs, "eulerians"
(common-eulerians.c) on the 4 kappa presets with solutions = 0 and 1. Per
case the engine's mode and parameters are set at the start axes, psi is
initialised there (hkl0 = the hkl engine's get at the start, recorded), and
pseudo_axis_values_set(target) is called from the start axes; every solution
hkl returns is recorded, or hkl's error. The solver restarts with libc
rand(); srand() is seeded per call so the output is reproducible.
Seam targets, each with its equivalent in (-180, 180] at the same start and
other target values: psi and alpha at -180, 180, 190, -190, 350 (HKL-38);
chi at 350, -300, 450 (HKL-19); chi = 0, where kappa = 0 leaves komega + kphi
alone fixed. Per psi preset one case sets psi before initialised_set on a
fresh engine list (HKL-39) and records how many of hkl's solutions are at
the hkl0 of its start axes.
Axes are canonical degrees and millimetres, by axis name.
"""
import ctypes
import json
import math
import os
import random

import gi

gi.require_version("Hkl", "5.0")
from gi.repository import GLib, Hkl  # noqa: E402

SEED = 20261123
USER = Hkl.UnitEnum.USER
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_pseudo_set.json")
LIBC = ctypes.CDLL(None)
LATTICE = (4.1, 5.3, 6.7, 90.0, 90.0, 90.0)
WL = 1.54
UNIT = {"Degree": 1.0, "Milli Radian": math.pi / 180.0 * 1000.0,
        "Milli Meter": 1.0, "Meter": 1e-3}
PSI = ["APS POLAR", "E4CH", "E4CV", "E6C", "K4CV", "K6C", "PETRA3 P23 6C",
       "SOLEIL MARS", "SOLEIL SIRIUS KAPPA"]
Q = ["E4CV", "E4CH", "K4CV", "SOLEIL MARS"]
DETECTOR = ["E6C", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL SIRIUS TURRET", "PETRA3 P23 4C",
            "PETRA3 P23 6C", "ZAXIS", "SOLEIL SIXS MED1+2", "SOLEIL SIXS MED2+2",
            "SOLEIL SIXS MED2+3", "SOLEIL SIXS MED2+3 v2"]
EULERIANS = ["K4CV", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL CRISTAL K4C"]
# seam -> its equivalent in (-180, 180]; +-180 are their own
ANGLE_SEAMS = [-180.0, 180.0, 190.0, -170.0, -190.0, 170.0, 350.0, -10.0]
CHI_SEAMS = [350.0, -10.0, -300.0, 60.0, 450.0, 90.0, 0.0]
DETECTOR_AXES = ("tth", "delta", "gamma")
REFERENCES = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
SURFACES = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 2.0, -1.0)]


class Ctx:
    def __init__(self, gname, rng):
        self.gname = gname
        self.fac = Hkl.factories()[gname]
        self.geo = self.fac.create_new_geometry()
        self.geo.wavelength_set(WL, USER)
        self.names = list(self.geo.axis_names_get())
        self.units = [self.geo.axis_get(n).user_unit_get() for n in self.names]
        self.det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.smp = Hkl.Sample.new("set")
        a, b, c, al, be, ga = LATTICE
        self.smp.lattice_set(Hkl.Lattice.new(a, b, c, *(math.radians(x) for x in (al, be, ga))))
        self.u_euler = [rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)]
        self.smp.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in self.u_euler)))
        self.ub = [[self.smp.UB_get().get(i, j) for j in range(3)] for i in range(3)]
        self.el = self.fac.create_new_engine_list()
        self.el.init(self.geo, self.det, self.smp)

    def set_axes(self, canon):
        self.geo.axis_values_set([v * UNIT[u] for v, u in zip(canon, self.units)], USER)

    def axes(self, geo):
        return dict(zip(self.names, (v / UNIT[u] for v, u in
                                     zip(geo.axis_values_get(USER), self.units))))

    def hkl(self, canon):
        self.set_axes(canon)
        self.el.get()
        return [float(x) for x in self.el.engine_get_by_name("hkl").pseudo_axis_values_get(USER)]


def start_axes(ctx, rng):
    """Detector axes in [20, 120], the others within +-60: away from Q = 0 and from the stops."""
    return [rng.uniform(20, 120) if n in DETECTOR_AXES else rng.uniform(-60, 60)
            for n in ctx.names]


def solve(ctx, engine, mode, params, target, start, c_seed, init=True, hkl0=None):
    case = {"geometry": ctx.gname, "engine": engine, "mode": mode, "parameters": params,
            "target": target, "start_axes": dict(zip(ctx.names, start)), "c_srand": c_seed,
            "initialized": init}
    if hkl0 is not None:
        case["hkl0"] = hkl0
    eng = ctx.el.engine_get_by_name(engine)
    try:
        ctx.set_axes(start)
        eng.current_mode_set(mode)
        if params:
            eng.parameters_values_set(params, USER)
        if init and engine == "psi":
            eng.initialized_set(True)
        case["written"] = list(eng.axis_names_get(Hkl.EngineAxisNamesGet.WRITE))
        ctx.set_axes(start)
        LIBC.srand(c_seed)
        sols = eng.pseudo_axis_values_set([float(x) for x in target], USER)
        case["solutions"] = [ctx.axes(it.geometry_get()) for it in sols.items()]
    except GLib.Error as e:  # recorded, never dropped
        case["error"] = f"GError: {e}"
        case["solutions"] = []
    if not init:
        # hkl solved against unwritten memory (HKL-39): its error or solutions
        # change from run to run, only that none is at hkl0 does not.
        case.pop("error", None)
        at = [ctx.hkl([s[n] for n in ctx.names]) for s in case.pop("solutions")]
        case["solutions_at_hkl0"] = sum(max(abs(a - b) for a, b in zip(h, hkl0)) < 1e-4
                                        for h in at)
    return case


def main():
    cases, geometries = [], {}
    runs = ([("psi", g) for g in PSI] + [("q", g) for g in Q]
            + [(e, g) for e in ("q2", "tth2", "qper_qpar") for g in DETECTOR]
            + [("eulerians", g) for g in EULERIANS])
    for engine, gname in runs:
        rng = random.Random(f"{SEED}-{engine}-{gname}")
        ctx = Ctx(gname, random.Random(f"{SEED}-{gname}-U"))
        geometries[gname] = {"axes": ctx.names, "units": ctx.units, "UB": ctx.ub,
                             "U_euler_deg": ctx.u_euler, "wavelength": WL}
        mode = ctx.el.engine_get_by_name(engine).modes_names_get()[0]

        def run(params, target, start, **kw):
            cases.append(solve(ctx, engine, mode, params, target, start,
                               rng.randint(1, 2**30), **kw))

        if engine == "psi":
            for i in range(2):
                start = start_axes(ctx, rng)
                hkl0 = ctx.hkl(start)
                ref = list(REFERENCES[i])
                for psi in [rng.uniform(-170, 170) for _ in range(3)] + ANGLE_SEAMS:
                    run(ref, [psi], start, hkl0=hkl0)
            # HKL-39: a fresh engine list, psi set before initialised_set
            ctx = Ctx(gname, random.Random(f"{SEED}-{gname}-U"))
            start = start_axes(ctx, rng)
            hkl0 = ctx.hkl(start)
            ctx = Ctx(gname, random.Random(f"{SEED}-{gname}-U"))
            run(list(REFERENCES[0]), [10.0], start, init=False, hkl0=hkl0)
        elif engine == "q":
            for _ in range(2):
                start = start_axes(ctx, rng)
                for q in [rng.uniform(-7.5, 7.5) for _ in range(4)] + [0.0]:
                    run([], [q], start)
        elif engine in ("q2", "tth2"):
            def first():
                return rng.uniform(1.0, 7.5) if engine == "q2" else rng.uniform(10, 150)
            for _ in range(2):
                start = start_axes(ctx, rng)
                for _ in range(3):
                    run([], [first(), rng.uniform(-170, 170)], start)
                seam = first()
                for alpha in ANGLE_SEAMS:
                    run([], [seam, alpha], start)
        elif engine == "qper_qpar":
            for surface in SURFACES:
                start = start_axes(ctx, rng)
                for _ in range(3):
                    run(list(surface), [rng.uniform(-3, 3), rng.uniform(-4, 4)], start)
        else:
            for solutions in (0.0, 1.0):
                start = start_axes(ctx, rng)
                for _ in range(3):
                    run([solutions], [rng.uniform(-180, 180), rng.uniform(-100, 100),
                                      rng.uniform(-180, 180)], start)
                omega, phi = rng.uniform(-180, 180), rng.uniform(-180, 180)
                for chi in CHI_SEAMS:
                    run([solutions], [omega, chi, phi], start)
    doc = {"hkl_version": Hkl.VERSION, "seed": SEED, "wavelength": WL,
           "units": {"angle": "deg", "length": "mm", "q": "1/angstrom with 2pi"},
           "geometries": geometries, "cases": cases}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, allow_nan=False)
        f.write("\n")
    print("wrote", OUT, "cases", len(cases))
    for engine in ("psi", "q", "q2", "tth2", "qper_qpar", "eulerians"):
        cs = [c for c in cases if c["engine"] == engine]
        print(engine, len(cs), "solved", sum(bool(c.get("solutions")) for c in cs),
              "errors", sorted({c["error"] for c in cs if "error" in c}))


main()
