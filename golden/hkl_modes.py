"""Golden mode lists from libhkl 5.1.9 (gi Hkl 5.0).
Run: python -I hkl_modes.py

Per factory: the "hkl" engine's mode names in hkl's order, the mode it is
in on a fresh engine list (hkl's default_mode), and per mode its parameter
names, each flagged angle when hkl converts it to degrees for the user; the
"psi" engine's modes with theirs, where the list has that engine; and the
engine list's own parameters, flagged so, with their value on a new list in
user units. TwoC's list has no engine; PYFAI and PETRA3 P08 LISA have no gi
factory.
"""
import json
import os

import gi

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hkl_modes.json")


def parameters(engine):
    names = list(engine.parameters_names_get())
    engine.parameters_values_set([1.0] * len(names), Hkl.UnitEnum.DEFAULT)
    user = engine.parameters_values_get(Hkl.UnitEnum.USER)
    return [{"name": n, "angle": v != 1.0} for n, v in zip(names, user)]


def engine_list(engines):
    out = []
    for name in engines.parameters_names_get():
        p = engines.parameter_get(name)
        value = p.value_get(Hkl.UnitEnum.USER)
        p.value_set(1.0, Hkl.UnitEnum.DEFAULT)
        angle = p.value_get(Hkl.UnitEnum.USER) != 1.0
        out.append({"name": name, "angle": angle, "value": value})
    return out


def modes(engine):
    out = []
    for name in engine.modes_names_get():
        engine.current_mode_set(name)
        out.append({"name": name, "parameters": parameters(engine)})
    return out


def main():
    presets = {}
    for name, factory in sorted(Hkl.factories().items()):
        engines = factory.create_new_engine_list()
        engines.init(factory.create_new_geometry(),
                     Hkl.Detector.factory_new(Hkl.DetectorType(0)),
                     Hkl.Sample.new("s"))
        by_name = {e.name_get(): e for e in engines.engines_get()}
        hkl, psi = by_name.get("hkl"), by_name.get("psi")
        presets[name] = {
            "default": None if hkl is None else hkl.current_mode_get(),
            "modes": [] if hkl is None else modes(hkl),
            "psi": None if psi is None else modes(psi),
            "engine_list": engine_list(engines),
        }
    with open(OUT, "w") as f:
        json.dump({"hkl_version": "5.1.9", "presets": presets}, f, indent=1)
        f.write("\n")


main()
