"""Golden hklpy2 configuration files that hklrs-bsrs imports.
Run: python -I hklpy2_config.py (hklpy2 352700b6, hkl_soleil solver)

hklpy2 builds an E4CV with a hexagonal sample oriented by Busing-Levy from
two reflections, limits and cut points that drop some solutions, and
exports it (hklpy2_e4cv.yml), then the same in psi_constant with its extras
(hklpy2_e4cv_psi.yml). hklpy2_config.json holds, per file, the UB and every
forward (hkl -> angles) solution hklpy2 keeps after its constraints, with
digits raised so they are not rounded to 4 places. hklpy2_e4cv_orient.yml
is hklpy2's own test file, copied unchanged. libhkl's solver restarts with
libc rand(); srand(C_SRAND) is set through ctypes before every forward, so a
target's solutions do not depend on the targets solved before it.
"""
import ctypes
import json
import math
import os
import shutil
import warnings

import numpy as np

import hklpy2

HERE = os.path.dirname(os.path.abspath(__file__))
HKLPY2 = os.path.dirname(os.path.abspath(hklpy2.__file__))
LIBC = ctypes.CDLL(None)
C_SRAND = 20260201
WAVELENGTH = 1.54
LATTICE = dict(a=3.0, b=3.0, c=5.0, alpha=90.0, beta=90.0, gamma=120.0)
# omega, chi, phi, tth; tth from Bragg's law, the others arbitrary so U is
# far from the identity
REFLECTIONS = [("r100", (1, 0, 0), (-12.0, 31.0, 47.0)),
               ("r001", (0, 0, 1), (8.0, -54.0, 112.0))]
CONSTRAINTS = {"omega": dict(limits=(-120.0, 120.0), cut_point=-180.0),
               "chi": dict(limits=(-100.0, 100.0), cut_point=-180.0),
               "phi": dict(limits=(0.0, 360.0), cut_point=0.0),
               "tth": dict(limits=(-5.0, 170.0), cut_point=-180.0)}
TARGETS = [(1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 2), (-1, 2, 1), (0.5, 0.25, 1.5)]
PSI = dict(h2=0.0, k2=0.0, l2=1.0, psi=30.0)
PSI_TARGETS = [(1, 0, 0), (1, 1, 0), (2, -1, 0)]


def d_spacing(hkl):
    a, c = LATTICE["a"], LATTICE["c"]
    h, k, l = hkl
    return 1.0 / math.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / a**2 + l * l / c**2)


def build():
    f = hklpy2.creator(name="e4cv")
    f.digits = 12
    f.beam.wavelength.put(WAVELENGTH)
    f.add_sample("hexagonal", **LATTICE)
    for name, hkl, (omega, chi, phi) in REFLECTIONS:
        tth = 2.0 * math.degrees(math.asin(WAVELENGTH / (2.0 * d_spacing(hkl))))
        reals = dict(omega=omega + tth / 2.0, chi=chi, phi=phi, tth=tth)
        f.add_reflection(hkl, reals, wavelength=WAVELENGTH, name=name)
    f.core.calc_UB(*[name for name, _, _ in REFLECTIONS])
    for axis, c in CONSTRAINTS.items():
        f.core.constraints[axis].limits = c["limits"]
        f.core.constraints[axis].cut_point = c["cut_point"]
    return f


def solutions(f, targets):
    out = []
    for hkl in targets:
        LIBC.srand(C_SRAND)
        found = f.core.forward(dict(zip("hkl", hkl)))
        out.append({"hkl": list(hkl), "solutions": [list(s) for s in found]})
    return out


def export(f, name):
    path = os.path.join(HERE, name)
    f.export(path, comment="golden/hklpy2_config.py")
    # the header records the absolute path and the time; keep the file stable
    with open(path) as y:
        text = y.read()
    lines = [ln for ln in text.splitlines()
             if not ln.startswith(("  datetime:", "  file:"))]
    with open(path, "w") as y:
        y.write("\n".join(lines) + "\n")
    return name


def case(f, name, targets):
    return {"file": export(f, name), "mode": f.core.mode,
            "axes": list(f.real_axis_names),
            "UB": np.array(f.sample.UB).tolist(), "U": np.array(f.sample.U).tolist(),
            "forward": solutions(f, targets)}


def main():
    warnings.simplefilter("ignore")
    f = build()
    f.core.mode = "bissector"
    bissector = case(f, "hklpy2_e4cv.yml", TARGETS)
    f.core.mode = "psi_constant"
    f.core.extras = PSI
    psi = case(f, "hklpy2_e4cv_psi.yml", PSI_TARGETS)
    shutil.copyfile(os.path.join(HKLPY2, "tests", "e4cv_orient.yml"),
                    os.path.join(HERE, "hklpy2_e4cv_orient.yml"))
    doc = {"hklpy2_version": hklpy2.__version__, "c_srand": C_SRAND, "wavelength": WAVELENGTH,
           "lattice": LATTICE, "constraints": CONSTRAINTS,
           "units": {"angle": "deg", "UB": "1/angstrom, with 2 pi"},
           "cases": [bissector, psi]}
    with open(os.path.join(HERE, "hklpy2_config.json"), "w") as out:
        json.dump(doc, out, indent=1)
    for c in doc["cases"]:
        print(c["file"], c["mode"], [len(t["solutions"]) for t in c["forward"]])


if __name__ == "__main__":
    main()
