"""Golden th_tth vectors from hklpy2's ThTthSolver. Run: python -I th_tth.py"""
import json
import math
import os

from hklpy2.backends.th_tth_q import TH_TTH_Q_GEOMETRY, ThTthSolver

WL = 1.54
A = 5.431020511
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "th_tth.json")
THETAS = [14.9131, 0.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 75.0, 89.0]
# (theta, tth): bisecting pairs plus non-bisecting ones (inverse uses tth only)
PAIRS = [(t, 2 * t) for t in THETAS] + [(10.0, 30.0), (14.9131, 40.0), (0.0, 0.0)]


def main():
    s = ThTthSolver(TH_TTH_Q_GEOMETRY)
    s.wavelength = WL
    cases = []
    for th, tth in PAIRS:
        c = {"theta": th, "tth": tth}
        q = s.inverse({"th": th, "tth": tth})["q"]
        c["q"] = q
        # th_tth has no UB/hkl; for cubic powder |hkl| = q*a/(2*pi)
        c["hkl_norm"] = q * A / (2 * math.pi)
        fw = s.forward({"q": q})
        c["forward"] = fw[0] if fw else None
        cases.append(c)
    doc = {"solver": "th_tth", "geometry": TH_TTH_Q_GEOMETRY, "wavelength": WL,
           "lattice_a": A, "units": {"angle": "deg", "q": "1/angstrom"},
           "cases": cases}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote", OUT, len(cases))


main()
