"""hklrs.pseudo_solve against libhkl 5.1.9 (gi, live): the psi, q, q2, tth2,
qper_qpar and eulerians engines on every preset the installed libhkl has
them for. Cases are drawn as golden/hkl_pseudo_set.py draws them, seeds
included, and held to hklrs/tests/hkl_pseudo_set.rs's rules.

Every libhkl solution is one of hklrs's within MATCH. Every hklrs solution
holds the axes the mode neither writes nor turns for the slits, and reads the
target back within READ through hklrs's forward (test_pseudo.py holds that
to libhkl's get), eulerians through libhkl's get. libhkl's psi get is no
read-back: hkl_vector_angle's acos (HKL-11) leaves it 1.5e-8 rad off at
psi = 0 and 180. MED2+3 fits its slits after
every engine; libhkl's coarser fit is matched after one Newton step.
HKL-16: libhkl appends the kappa twin of every solution whatever the mode
writes; a twin that moves a held axis is counted against the hklrs solution
it shares the sample rotation with instead of matched.
HKL-19, HKL-38: libhkl fails at chi, psi or alpha outside (-180, 180]; hklrs
gives the set of the equivalent target inside.
HKL-39: psi set before initialized_set leaves libhkl's hkl0 unwritten, so
none of its solutions is at the hkl0 of the start; hklrs solves there.
One Rust error raises one Python type from solve and from pseudo_solve.
"""
import ctypes
import functools
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pytest

import hklrs
from test_hkl import UNIT, USER, Hkl, mat, quat_matrix as quat
from gi.repository import GLib  # test_hkl pins the gi version

SEED = 20261123
LIBC = ctypes.CDLL(None)
LATTICE = (4.1, 5.3, 6.7, 90.0, 90.0, 90.0)
WL = 1.54
PSI = ["APS POLAR", "E4CH", "E4CV", "E6C", "K4CV", "K6C", "PETRA3 P23 6C",
       "SOLEIL MARS", "SOLEIL SIRIUS KAPPA"]
Q = ["E4CV", "E4CH", "K4CV", "SOLEIL MARS"]
DETECTOR = ["E6C", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL SIRIUS TURRET", "PETRA3 P23 4C",
            "PETRA3 P23 6C", "ZAXIS", "SOLEIL SIXS MED1+2", "SOLEIL SIXS MED2+2",
            "SOLEIL SIXS MED2+3", "SOLEIL SIXS MED2+3 v2"]
EULERIANS = ["K4CV", "K6C", "SOLEIL SIRIUS KAPPA", "SOLEIL CRISTAL K4C"]
RUNS = ([("psi", g) for g in PSI] + [("q", g) for g in Q]
        + [(e, g) for e in ("q2", "tth2", "qper_qpar") for g in DETECTOR]
        + [("eulerians", g) for g in EULERIANS])
ANGLE_SEAMS = [-180.0, 180.0, 190.0, -170.0, -190.0, 170.0, 350.0, -10.0]
CHI_SEAMS = [350.0, -10.0, -300.0, 60.0, 450.0, 90.0, 0.0]
DETECTOR_AXES = ("tth", "delta", "gamma")
REFERENCES = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
SURFACES = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 2.0, -1.0)]
KAPPA = ("komega", "kappa", "kphi")
# The slits axis and the axis whose vector is the surface (SIXS MED2+3).
SLITS = {"SOLEIL SIXS MED2+3": ("eta_a", "omega")}
# Which target values are angles, and where the one a seam moves sits.
ANGLES = {"psi": (True,), "q": (False,), "q2": (False, True), "tth2": (True, True),
          "qper_qpar": (False, False), "eulerians": (True, True, True)}
SEAM = {"psi": 0, "q2": 1, "tth2": 1, "eulerians": 1}

MATCH = math.degrees(1e-6)  # deg: libhkl's residual leaves its angles within ~1e-7 rad
READ = 1e-9                 # rad, 1/Angstrom or Miller units
SAME = math.degrees(1e-9)   # deg: one solution
HELD = 1e-12                # deg: a held axis through the deg -> rad -> deg round trip


def wrap(d):
    return 180.0 - ((180.0 - d) % 360.0)


def close(a, b, tol):
    return all(abs(wrap(x - y)) <= tol for x, y in zip(a, b))


def same_sets(a, b):
    def within(x, y):
        return all(any(close(s, t, SAME) for t in y) for s in x)
    return len(a) == len(b) and within(a, b) and within(b, a)


def miss(a, b, angles):
    """Largest difference of pseudo values, angles in rad modulo 2 pi; NaN when one is."""
    d = [abs(math.radians(wrap(x - y))) if ang else abs(x - y)
         for x, y, ang in zip(a, b, angles)]
    return math.nan if any(math.isnan(x) for x in d) else max(d)


class Rig:
    """One libhkl engine list on `preset` with golden's sample, and hklrs's twin of it."""

    def __init__(self, preset, rng):
        fac = Hkl.factories()[preset]
        self.preset = preset
        self.geo = fac.create_new_geometry()
        self.geo.wavelength_set(WL, USER)
        self.names = list(self.geo.axis_names_get())
        self.units = [self.geo.axis_get(n).user_unit_get() for n in self.names]
        self.det = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.smp = Hkl.Sample.new("set")
        a, b, c, al, be, ga = LATTICE
        self.smp.lattice_set(Hkl.Lattice.new(a, b, c, *(math.radians(x) for x in (al, be, ga))))
        self.smp.U_set(Hkl.Matrix.new_euler(*(math.radians(v) for v in (
            rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)))))
        self.el = fac.create_new_engine_list()  # owns the engines: keep it alive
        self.el.init(self.geo, self.det, self.smp)
        self.g = hklrs.Geometry.preset(preset)
        assert self.g.axis_names == self.names
        self.sample = hklrs.Sample("set", hklrs.Lattice(*LATTICE), mat(self.smp.U_get()).tolist())
        np.testing.assert_allclose(self.sample.ub(), mat(self.smp.UB_get()), atol=1e-12)

    def set_axes(self, canon):
        """`canon` in degrees / metres."""
        self.geo.axis_values_set([v * UNIT[u][1] for v, u in zip(canon, self.units)], USER)

    def axes(self, geo):
        return [v / UNIT[u][1] for v, u in zip(geo.axis_values_get(USER), self.units)]

    def start(self, rng):
        """Detector axes in [20, 120] deg, the others within +-60 deg or mm."""
        return [rng.uniform(20, 120) if n in DETECTOR_AXES
                else rng.uniform(-60, 60) * (1e-3 if UNIT[u][0] == "trans" else 1.0)
                for n, u in zip(self.names, self.units)]

    def get(self, engine, canon):
        """libhkl's hkl or eulerians values at `canon`. Neither get can fail
        here, and the list's flag is no help: it is every engine's at once,
        psi's included, which fails at Q = 0."""
        self.set_axes(canon)
        self.el.get()
        return list(self.el.engine_get_by_name(engine).pseudo_axis_values_get(USER))

    def rotation(self, canon):
        self.set_axes(canon)
        return quat(self.geo.sample_rotation_get(self.smp))


@dataclass
class Case:
    preset: str
    engine: str
    params: list
    target: list
    start: list
    init: bool
    theirs: list = field(default_factory=list)
    ours: object = None
    problems: list = field(default_factory=list)
    extras: int = 0
    newton: int = 0
    held_twins: int = 0
    at_hkl0: int = 0

    @property
    def solved(self):
        return isinstance(self.ours, list) and bool(self.ours)


def solve_hkl(rig, case, mode, c_seed):
    """libhkl's WRITE axes and solutions (degrees / metres), as golden records them."""
    eng = rig.el.engine_get_by_name(case.engine)
    rig.set_axes(case.start)
    eng.current_mode_set(mode)
    if case.params:
        eng.parameters_values_set(case.params, USER)
    if case.init and case.engine == "psi":
        eng.initialized_set(True)
    written = list(eng.axis_names_get(Hkl.EngineAxisNamesGet.WRITE))
    rig.set_axes(case.start)
    LIBC.srand(c_seed)
    try:
        sols = eng.pseudo_axis_values_set([float(x) for x in case.target], USER)
    except GLib.Error:
        return written, []
    return written, [rig.axes(it.geometry_get()) for it in sols.items()]


def read_back(rig, case, hkl0, values):
    """The engine's values at `values` through hklrs's forward, as the
    target; psi is NaN unless hkl there is hkl0. eulerians, which hklrs-py
    has no forward for, through libhkl's get on the image of the
    `solutions` parameter nearer the target. None when the forward raises."""
    e = case.engine
    if e == "eulerians":
        eng = rig.el.engine_get_by_name(e)
        images = []
        for solutions in (0.0, 1.0):
            eng.parameters_values_set([solutions], USER)
            images.append(rig.get(e, values))
        return min(images, key=lambda x: miss(x, case.target, ANGLES[e]))
    try:
        if e != "psi":
            surface = case.params if e == "qper_qpar" else None
            return hklrs.pseudo(rig.g, e, values, WL, surface=surface)
        hkl = hklrs.forward(rig.g, rig.sample, values, WL)["hkl"]
        if max(abs(a - b) for a, b in zip(hkl, hkl0)) > READ:
            return [math.nan]
        return [hklrs.psi(rig.g, rig.sample, case.params, values)]
    except (ValueError, hklrs.InverseError):
        return None


def slits_newton(rig, t):
    """`t` after one Newton step on the slits equation over the slits axis:
    libhkl stops its slits fit at |surface . normal| < 1e-6, coarser than
    MATCH where the equation turns slowly. None without slits."""
    if rig.preset not in SLITS:
        return None
    axis, surface_axis = SLITS[rig.preset]
    i = rig.names.index(axis)
    surface = np.array(rig.geo.axis_get(surface_axis).axis_v_get().data)

    def at(v):
        x = list(t)
        x[i] = v
        return x

    def f(v):
        rig.set_axes(at(v))
        normal = quat(rig.geo.detector_rotation_get(rig.det)) @ np.array([0.0, 0.0, 1.0])
        return (quat(rig.geo.sample_rotation_get(rig.smp)) @ surface) @ normal

    v, h = t[i], math.degrees(1e-6)
    return at(wrap(v - f(v) * 2.0 * h / (f(v + h) - f(v - h))))


def twin_of(rig, s, t):
    """`t` is the kappa twin of `s`: the other axes agree and so does the
    sample rotation, the one thing a twin keeps."""
    if not set(KAPPA) <= set(rig.names):
        return False
    others = [i for i, n in enumerate(rig.names) if n not in KAPPA]
    return (close([s[i] for i in others], [t[i] for i in others], MATCH)
            and np.abs(rig.rotation(s) - rig.rotation(t)).max() <= 1e-6)


def check(rig, case, mode, hkl0):
    """hklrs's solve of `case` against libhkl's, into `case.problems`."""
    fail = case.problems.append
    moved = set(mode.written()) | ({SLITS[rig.preset][0]} if rig.preset in SLITS else set())
    held = [i for i, n in enumerate(rig.names) if n not in moved]
    theirs = case.theirs if case.init else []
    angles = ANGLES[case.engine]
    ours = case.ours
    if isinstance(ours, list):
        for s in ours:
            kept = all(abs(s[i] - case.start[i]) <= HELD for i in held)
            got = read_back(rig, case, hkl0, s)
            if not kept or got is None or not miss(got, case.target, angles) <= READ:
                fail(f"{s} reads back {got} for {case.target}, held {kept}")
        case.extras = sum(not any(close(s, t, MATCH) for t in theirs) for s in ours)
        for t in theirs:
            shifted = any(abs(wrap(t[i] - case.start[i])) > MATCH for i in held)
            if shifted and any(twin_of(rig, s, t) for s in ours):
                case.held_twins += 1
            elif any(close(s, t, MATCH) for s in ours):
                pass
            elif (p := slits_newton(rig, t)) is not None and any(close(s, p, MATCH) for s in ours):
                case.newton += 1
            else:
                fail(f"libhkl solution {t} not in {ours}")
    elif (isinstance(ours, hklrs.Degenerate) and case.engine == "eulerians"
          and wrap(case.target[1]) == 0.0):
        # kappa = 0 and chi = 0: komega + kphi alone is fixed, as omega + phi.
        komega, kphi = rig.names.index("komega"), rig.names.index("kphi")
        for t in theirs:
            turned = list(t)
            turned[komega] += math.degrees(0.3)
            turned[kphi] -= math.degrees(0.3)
            for v in (t, turned):
                got = read_back(rig, case, hkl0, v)
                total = [[e[1], e[0] + e[2]] for e in (got, case.target)]
                if not miss(*total, (True, True)) <= math.radians(MATCH):
                    fail(f"Degenerate, but {v} reads back {got}")
    elif isinstance(ours, hklrs.NoSolution) and not theirs:
        pass
    else:
        fail(f"{type(ours).__name__}: {ours}, libhkl {theirs}")


def run(rig, case, c_seed, hkl0=None):
    eng = rig.el.engine_get_by_name(case.engine)
    written, case.theirs = solve_hkl(rig, case, eng.modes_names_get()[0], c_seed)
    kw = {"surface": case.params} if case.engine == "qper_qpar" else {}
    if case.engine == "psi":
        kw["reference"] = case.params
    mode = hklrs.PseudoMode(rig.g, case.engine, case.target, slits=rig.preset in SLITS, **kw)
    if case.init and mode.written() != written:
        case.problems.append(f"writes {mode.written()}, libhkl {written}")
    try:
        case.ours = hklrs.pseudo_solve(mode, WL, case.start, **(
            {"sample": rig.sample, "hkl0": hkl0} if case.engine == "psi" else {}))
    except hklrs.InverseError as e:
        case.ours = e
    if not case.init:
        # libhkl solved against unwritten memory (HKL-39): its error or
        # solutions change from run to run, only that none is at hkl0 does not.
        case.at_hkl0 = sum(max(abs(a - b) for a, b in zip(rig.get("hkl", t), hkl0)) < 1e-4
                           for t in case.theirs)
    check(rig, case, mode, hkl0)
    return case


@functools.cache
def cases(engine, preset):
    """golden's cases for `engine` on `preset`, solved live by both."""
    if preset not in Hkl.factories():
        pytest.skip(f"preset {preset} is not in the installed libhkl {Hkl.VERSION}")
    rng = random.Random(f"{SEED}-{engine}-{preset}")
    rig = Rig(preset, random.Random(f"{SEED}-{preset}-U"))
    out = []

    def add(params, target, start, hkl0=None, init=True, r=None):
        case = Case(preset, engine, list(params), list(target), start, init)
        out.append(run(r or rig, case, rng.randint(1, 2**30), hkl0))

    if engine == "psi":
        for i in range(2):
            start = rig.start(rng)
            hkl0 = rig.get("hkl", start)
            for psi in [rng.uniform(-170, 170) for _ in range(3)] + ANGLE_SEAMS:
                add(REFERENCES[i], [psi], start, hkl0)
        # HKL-39: a fresh engine list, psi set before initialized_set
        fresh = Rig(preset, random.Random(f"{SEED}-{preset}-U"))
        start = fresh.start(rng)
        hkl0 = fresh.get("hkl", start)
        add(REFERENCES[0], [10.0], start, hkl0, init=False,
            r=Rig(preset, random.Random(f"{SEED}-{preset}-U")))
    elif engine == "q":
        for _ in range(2):
            start = rig.start(rng)
            for q in [rng.uniform(-7.5, 7.5) for _ in range(4)] + [0.0]:
                add([], [q], start)
    elif engine in ("q2", "tth2"):
        def first():
            return rng.uniform(1.0, 7.5) if engine == "q2" else rng.uniform(10, 150)
        for _ in range(2):
            start = rig.start(rng)
            for _ in range(3):
                add([], [first(), rng.uniform(-170, 170)], start)
            seam = first()
            for alpha in ANGLE_SEAMS:
                add([], [seam, alpha], start)
    elif engine == "qper_qpar":
        for surface in SURFACES:
            start = rig.start(rng)
            for _ in range(3):
                add(surface, [rng.uniform(-3, 3), rng.uniform(-4, 4)], start)
    else:
        for solutions in (0.0, 1.0):
            start = rig.start(rng)
            for _ in range(3):
                add([solutions], [rng.uniform(-180, 180), rng.uniform(-100, 100),
                                  rng.uniform(-180, 180)], start)
            omega, phi = rng.uniform(-180, 180), rng.uniform(-180, 180)
            for chi in CHI_SEAMS:
                add([solutions], [omega, chi, phi], start)
    return out


@pytest.mark.parametrize("engine,preset", RUNS)
def test_pseudo_set_against_libhkl(engine, preset):
    cs = cases(engine, preset)
    problems = [f"{c.target} from {c.start}: {p}" for c in cs for p in c.problems]
    print(f"{engine} {preset}: {len(cs)} cases, libhkl "
          f"{sum(len(c.theirs) for c in cs if c.init)} solutions in hklrs's "
          f"{sum(len(c.ours) for c in cs if isinstance(c.ours, list))} "
          f"({sum(c.extras for c in cs)} beyond, {sum(c.newton for c in cs)} after the "
          f"slits Newton step, {sum(c.held_twins for c in cs)} held-axis twins)")
    assert not problems, "\n".join(problems[:20])


@pytest.mark.parametrize("engine", ["q", "q2", "tth2", "qper_qpar"])
def test_hkl16_kappa_twins_move_held_axes(engine):
    """HKL-16: on a kappa preset libhkl appends the kappa twin of each
    solution although the mode holds komega, kappa and kphi; each such twin
    is that of an hklrs solution, which holds them."""
    kappa = [g for e, g in RUNS if e == engine and g in EULERIANS]
    cs = [c for g in kappa for c in cases(engine, g)]
    problems = [p for c in cs for p in c.problems]
    assert not problems, "\n".join(problems[:20])
    assert sum(c.held_twins for c in cs) > 0


def seams(engine):
    """[targets at |angle| >= 180 (outside (-180, 180] for eulerians), of
    them libhkl failed and hklrs solved, outside with the equivalent's set],
    and the problems found."""
    k = SEAM[engine]
    n, problems = [0, 0, 0], []
    for g in [g for e, g in RUNS if e == engine]:
        cs = [c for c in cases(engine, g) if c.init]
        for c in cs:
            a = c.target[k]
            if engine != "eulerians" and abs(a) >= 180.0:
                n[0] += 1
                n[1] += not c.theirs and c.solved
            if -180.0 < a <= 180.0:
                continue
            inside = wrap(a)
            eq = [e for e in cs if e.params == c.params and e.start == c.start
                  and e.target == c.target[:k] + [inside] + c.target[k + 1:]]
            if not eq:
                problems.append(f"{g} {c.target}: no equivalent at {inside}")
                continue
            if abs(a) > 180.0 and c.theirs:
                problems.append(f"{g} {c.target}: libhkl solved at {a}")
            o, e = c.ours, eq[0].ours
            agree = (same_sets(o, e) if isinstance(o, list) and isinstance(e, list)
                     else type(o) is type(e) and str(o) == str(e))
            if not agree:
                problems.append(f"{g} {c.target}: {o!r}, at {inside} {e!r}")
            if engine == "eulerians":
                n[0] += 1
                n[1] += c.solved
            n[2] += 1
    print(f"{engine}: {n[0]} seam targets, libhkl fails and hklrs solves {n[1]}; "
          f"{n[2]} outside (-180, 180] give their equivalent's set")
    return n, problems


@pytest.mark.parametrize("engine", ["psi", "q2", "tth2"])
def test_hkl38_psi_alpha_seam(engine):
    """HKL-38: psi and alpha at |angle| >= 180 fail in libhkl; hklrs solves
    them, outside (-180, 180] as the equivalent target inside."""
    n, problems = seams(engine)
    assert not problems, "\n".join(problems[:20])
    assert n[1] > 0 and n[2] > 0, n


def test_hkl19_eulerians_chi_seam():
    """HKL-19: chi outside (-180, 180] fails in libhkl's eulerians; hklrs
    solves it as the equivalent chi inside."""
    n, problems = seams("eulerians")
    assert not problems, "\n".join(problems[:20])
    assert n[1] > 0 and n[2] > 0, n


@pytest.mark.parametrize("preset", PSI)
def test_hkl39_psi_before_initialisation(preset):
    """HKL-39: psi set before initialized_set solves against an unwritten
    hkl0 in libhkl, none of whose solutions is at the start's hkl0; hklrs,
    given hkl0, solves there."""
    (c,) = [c for c in cases("psi", preset) if not c.init]
    assert c.at_hkl0 == 0, f"libhkl solved at hkl0: {c.theirs}"
    assert not c.problems, "\n".join(c.problems)
    assert c.solved, c.ours


@pytest.mark.parametrize("error", ["Unreachable", "Degenerate", "NonFinite",
                                   "NoSolutionInLimits", "GeometryError"])
def test_solve_and_pseudo_solve_raise_one_type_per_error(error):
    """One Rust error, one Python type: solve and pick on an hkl mode and
    pseudo_solve and pseudo_pick on a pseudo engine raise the InverseError
    subclass named after the variant."""
    g = hklrs.Geometry.preset("E4CV")
    sample = hklrs.Sample("errors", hklrs.Lattice(*LATTICE))
    current = [30.0, 10.0, 20.0, 60.0]
    hkl, engine, values, hkl0 = [1.0, 1.0, 1.0], "q", [2.0], None
    if error == "Unreachable":
        hkl, values = [10.0, 10.0, 10.0], [100.0]
    elif error == "Degenerate":
        hkl, engine, values, hkl0 = [0.0, 0.0, 0.0], "psi", [10.0], [0.0, 0.0, 0.0]
    elif error == "NonFinite":
        hkl, engine, values, hkl0 = [math.nan, 0.0, 1.0], "psi", [10.0], [math.nan, 0.0, 1.0]
    elif error == "NoSolutionInLimits":
        for a in g.axis_names:
            g.set_limits(a, 170.0, 171.0)
    elif error == "GeometryError":
        current = current[:3]
    mode = hklrs.Mode(g, "bissector")
    pseudo = hklrs.PseudoMode(g, engine, values,
                              reference=[1.0, 0.0, 0.0] if engine == "psi" else None)
    raised = []
    for f in (hklrs.solve, hklrs.pick):
        with pytest.raises(hklrs.InverseError) as e:
            f(mode, sample, hkl, WL, current)
        raised.append(type(e.value))
    for f in (hklrs.pseudo_solve, hklrs.pseudo_pick):
        with pytest.raises(hklrs.InverseError) as e:
            f(pseudo, WL, current, sample, hkl0)
        raised.append(type(e.value))
    assert raised == [getattr(hklrs, error)] * 4


def test_mode_and_pseudo_mode_raise_one_type_per_error():
    """As above for the error a mode raises when it is made."""
    g = hklrs.Geometry.preset("E4CV")
    made = (lambda: hklrs.Mode(g, "psi_constant", h2=0.0, k2=0.0, l2=0.0, psi=10.0),
            lambda: hklrs.PseudoMode(g, "psi", [10.0], reference=[0.0] * 3))
    raised = []
    for f in made:
        with pytest.raises(hklrs.InverseError) as e:
            f()
        raised.append(type(e.value))
    assert raised == [hklrs.ZeroReference] * 2


def test_another_presets_mode_is_unknown_and_its_engine_a_geometry_error():
    """Mode takes its name from the geometry's preset, so E6C's mode on E4CV
    is unknown; PseudoMode checks the engine's axes instead."""
    g = hklrs.Geometry.preset("E4CV")
    with pytest.raises(hklrs.UnknownMode):
        hklrs.Mode(g, "bissector_vertical")
    with pytest.raises(hklrs.GeometryError):
        hklrs.PseudoMode(g, "eulerians", [0.0, 0.0, 0.0])


def test_pick_and_pseudo_pick_read_picker_alike():
    """"closest" is the first of solve's list, "first" one of them, anything
    else one ValueError from both."""
    g = hklrs.Geometry.preset("E4CV")
    sample = hklrs.Sample("picker", hklrs.Lattice(*LATTICE))
    current = [30.0, 10.0, 20.0, 60.0]
    mode, pseudo = hklrs.Mode(g, "bissector"), hklrs.PseudoMode(g, "q", [2.0])
    calls = [(hklrs.solve, hklrs.pick, (mode, sample, [1.0, 1.0, 1.0], WL, current)),
             (hklrs.pseudo_solve, hklrs.pseudo_pick, (pseudo, WL, current))]
    messages = []
    for solve, pick, args in calls:
        every = solve(*args)
        assert pick(*args) == pick(*args, picker="closest") == every[0]
        assert pick(*args, picker="first") in every
        with pytest.raises(ValueError) as e:
            pick(*args, picker="nearest")
        assert not isinstance(e.value, hklrs.InverseError)
        messages.append(str(e.value))
    assert messages[0] == messages[1]
