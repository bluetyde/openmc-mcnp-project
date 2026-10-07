"""mcnpy_speed may only change how many round trips MCNPy makes, never what it builds. These tests run MCNPy's
and metapy's own source (read from the installed packages; nothing is imported from them, so no Java gateway
starts and port 25333 is never touched) against a fake gateway that counts round trips the way py4j makes them:

  - fast_add: the real Deck.universes / Deck.get_universe and UniverseList, cells added the way openmc_to_mcnp
    adds them: the universe lists come out the same, and the universe reads drop from N^2 to N;
  - cached_reflection: the real metapy.wrap property getters and setters (e_class_body) on fake features of every
    kind MCNPy has (attribute, enum, enum list, reference, containment, 'name'): the same values come back and
    the same eSet calls reach the fake Java side, with far fewer trips, and the module is restored afterwards;
  - the source guard: the installed MCNPy is the one the speedups were written for;
  - mcnpy.deck_formatter.line_wrap never ends on a blank-free token longer than 114 characters (why a
    dummy cell holding a long union hung deck.write), and mcnpy_speed.line_wrap gives the same text wherever
    MCNPy's ends and still ends where it loops.

Needs openmc (Python only) and the installed mcnpy/metapy sources, not the MCNPy gateway.
Run: python tests/test_mcnpy_speed.py
"""
import ast
import collections
import importlib.util
import itertools
import os
import random
import subprocess
import sys
import textwrap
import types
import unittest
import warnings
from pathlib import Path

import openmc
from py4j.java_collections import JavaList
from py4j.java_gateway import JavaClass, JavaPackage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mcnpy_speed  # noqa: E402


def package_dir(name):
    spec = importlib.util.find_spec(name)
    return spec.submodule_search_locations[0] if spec else None


MCNPY, METAPY = package_dir("mcnpy"), package_dir("metapy")
needs_sources = unittest.skipUnless(MCNPY and METAPY, "mcnpy/metapy are not installed")


def segment(path, qualname):
    text = Path(path).read_text(encoding="utf-8")
    body, node = ast.parse(text).body, None
    for part in qualname.split("."):
        node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        body = node.body
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
    return "\n".join(text.splitlines()[start - 1:node.end_lineno])


# ---------------------------------------------------------------- fake py4j gateway

TRIPS = collections.Counter()
_ids = itertools.count(1)


class FakeClass(JavaClass):
    def __init__(self, fqn):
        self._fqn = fqn

    def __getattr__(self, name):
        raise AttributeError(name)


class FakePackage(JavaPackage):
    """Each step of gateway.jvm.a.b.C is one reflection round trip in py4j (JVMView/JavaPackage.__getattr__)."""

    def __init__(self, fqn):
        self._fqn = fqn

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        TRIPS["reflect"] += 1
        fqn = f"{self._fqn}.{name}" if self._fqn else name
        return FakeClass(fqn) if name[0].isupper() else FakePackage(fqn)


class FakeObject:
    def __init__(self, kinds=()):
        self._target_id = f"o{next(_ids)}"
        self.kinds = set(kinds)


EATTR, EREF, EENUM = ("org.eclipse.emf.ecore." + n for n in ("EAttribute", "EReference", "EEnum"))


class EnumValue(FakeObject):
    def __init__(self, literal):
        super().__init__()
        self.literal = literal

    def toString(self):
        TRIPS["call"] += 1
        return self.literal

    def __eq__(self, other):
        return isinstance(other, EnumValue) and other.literal == self.literal

    def __repr__(self):
        return f"Enum({self.literal})"


class Literal(FakeObject):
    def __init__(self, literal):
        super().__init__()
        self.literal = literal

    def getInstance(self):
        TRIPS["call"] += 1
        return EnumValue(self.literal)


class DataType(FakeObject):
    def __init__(self, literals):
        super().__init__([EENUM] if literals is not None else ["org.eclipse.emf.ecore.EDataType"])
        self.literals = literals or {}

    def getEEnumLiteral(self, name):
        TRIPS["call"] += 1
        return Literal(self.literals[name]) if name in self.literals else None

    def getEEnumLiteralByLiteral(self, literal):
        TRIPS["call"] += 1
        return Literal(literal) if literal in self.literals.values() else None


class Feature(FakeObject):
    def __init__(self, name, attribute, literals=None, containment=False):
        super().__init__([EATTR] if attribute else [EREF])
        self.name, self.literals, self.containment = name, literals, containment

    def getName(self):
        TRIPS["call"] += 1
        return self.name

    def isContainment(self):
        TRIPS["call"] += 1
        return self.containment

    def getEAttributeType(self):
        TRIPS["call"] += 4  # the call, toString probe + call in wrap_e_object, and the object's later release
        return DataType(self.literals)


class EnumList(JavaList):
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        return iter(self._values)


SETS = []


class EObject(FakeObject):
    def __init__(self):
        super().__init__()
        self.store = {}

    def eGet(self, feature, resolve):
        TRIPS["call"] += 1
        return self.store.get(feature.name)

    def eSet(self, feature, value):
        TRIPS["call"] += 1
        self.store[feature.name] = value
        SETS.append((feature.name, value if isinstance(value, (str, int, float, type(None), EnumValue)) else "obj"))


class FakeGateway:
    def __init__(self):
        self.jvm = FakePackage("")

    def __getattr__(self, name):  # entry point fields: one round trip each read
        TRIPS["field"] += 1
        return FakeObject()


def fake_is_instance_of(java_object, java_class, gateway=None):
    TRIPS["reflect"] += 5  # jvm.py4j.reflection.TypeUtil, .isInstanceOf, the call
    return java_class._fqn in java_object.kinds


def load_wrap():
    """metapy/wrap.py run as module 'metapy.wrap' against the fake gateway (metapy/__init__ would start Java)."""
    saved = {k: sys.modules.get(k) for k in ("metapy", "metapy.gateway", "metapy.util", "metapy.wrap")}
    pkg = types.ModuleType("metapy")
    pkg.__path__ = []
    gw = types.ModuleType("metapy.gateway")
    gw.gateway, gw.is_instance_of, gw.get_documentation = FakeGateway(), fake_is_instance_of, lambda e: None
    util = types.ModuleType("metapy.util")
    exec(Path(METAPY, "util.py").read_text(), util.__dict__)
    sys.modules.update({"metapy": pkg, "metapy.gateway": gw, "metapy.util": util})
    try:
        spec = importlib.util.spec_from_file_location("metapy.wrap", os.path.join(METAPY, "wrap.py"))
        wrap = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # its regexes are plain strings
            spec.loader.exec_module(wrap)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return wrap


FEATURES = [Feature("name", True), Feature("count", True), Feature("density", True),
            Feature("side", True, literals={"PLUS": "+", "MINUS": "-"}),
            Feature("unit", True, literals={"G_CM3": "-", "NONE": "¥×¥"}),
            Feature("particles", True, literals={"N": "N", "P": "P"}),
            Feature("surface", False), Feature("nodes", False, containment=True)]


class EClass(FakeObject):
    def getESuperTypes(self):
        return []

    def getEStructuralFeatures(self):
        return FEATURES

    def getName(self):
        return "Thing"

    def toString(self):
        return "Thing"


class Factory:
    def create(self, e_class):
        TRIPS["call"] += 1
        return EObject()


def exercise(Thing):
    """Reads and writes of every feature kind, as MCNPy's translate does them. Returns what the reads gave."""
    t = Thing()
    t._e_object.store.update(name="12", count="3", density=1.5, side=EnumValue("-"), unit=EnumValue("¥×¥"),
                             particles=EnumList([EnumValue("N"), EnumValue("¥×¥")]), surface=None, nodes=None)
    seen = [t.name, t.count, t.density, t.side, t.unit, t.particles, t.surface, t.nodes]
    t.side = "minus"
    t.side = "+"
    t.unit = "g_cm3"
    t.count = 5
    t.name = 7
    t.surface = None
    t.density = 2.0
    seen += [t.side, t.count, t.name, t.density, t.unit]
    return seen


@needs_sources
class CachedReflectionTest(unittest.TestCase):
    def setUp(self):
        self.wrap = load_wrap()
        body = self.wrap.e_class_body(EClass(), Factory(), {}, numeric_ids=True, package="mcnpy")
        self.Thing = type("Thing", (self.wrap.InternalEObject,), body)

    def run_once(self):
        TRIPS.clear()
        SETS.clear()
        seen = exercise(self.Thing)
        return seen, list(SETS), sum(TRIPS.values())

    def test_same_values_and_writes_fewer_trips(self):
        plain_seen, plain_sets, plain_trips = self.run_once()
        with mcnpy_speed.cached_reflection(self.wrap):
            cold_seen, cold_sets, cold_trips = self.run_once()
            warm_seen, warm_sets, warm_trips = self.run_once()
        self.assertEqual(plain_seen, cold_seen)
        self.assertEqual(plain_seen, warm_seen)
        self.assertEqual(plain_sets, cold_sets)
        self.assertEqual(plain_sets, warm_sets)
        print(f"\n  round trips for the same reads and writes: plain {plain_trips}, cached (first time) {cold_trips}, "
              f"cached (warm) {warm_trips}")
        self.assertLess(warm_trips * 3, plain_trips)

    def test_known_answers(self):
        seen, sets, _ = self.run_once()
        self.assertEqual(seen[:8], ["12", 3, 1.5, "-", None, ["N", None], None, None])
        self.assertEqual(seen[8:], ["+", 5, 7, 2.0, "-"])  # the fake keeps the int that Java would store as "7"
        self.assertIn(("side", EnumValue("-")), sets)  # 'minus' upper-cased and found by name

    def test_getter_is_one_trip_when_warm(self):
        t = self.Thing()
        with mcnpy_speed.cached_reflection(self.wrap):
            t.surface
            TRIPS.clear()
            self.assertIsNone(t.surface)
            self.assertEqual(sum(TRIPS.values()), 1)  # eGet only
        TRIPS.clear()
        t.surface
        self.assertEqual(sum(TRIPS.values()), 11)  # eGet + EAttribute lookup (5) + is_instance_of (5)

    def test_module_restored(self):
        before = {n: getattr(self.wrap, n) for n in ("gateway", "is_instance_of", "return_value_converter", "is_enum")}
        with mcnpy_speed.cached_reflection(self.wrap):
            self.assertIsNot(self.wrap.gateway, before["gateway"])
        for n, v in before.items():
            self.assertIs(getattr(self.wrap, n), v)

    def test_cache_is_per_object(self):
        """An answer about one Java object is never given for another: a data type fetched fresh still gets
        its own is_instance_of answer."""
        r = mcnpy_speed._Reflection(self.wrap)
        attr, ref = Feature("a", True), Feature("r", False)
        cls = self.wrap.gateway.jvm.org.eclipse.emf.ecore.EAttribute
        self.assertTrue(r.is_instance_of(attr, cls))
        self.assertFalse(r.is_instance_of(ref, cls))
        self.assertTrue(r.is_instance_of(attr, cls))


def load_deck_class():
    """The real Deck.universes property and Deck.get_universe (mcnpy/deck.py) and UniverseList
    (mcnpy/geometry.py) in a Deck holding only what they use, with add() doing what Deck.add does for cells."""
    deck_py = os.path.join(MCNPY, "deck.py")
    parts = [segment(deck_py, f"Deck.{n}") for n in ("universes", "get_universe")]
    setter = next(p for p in Path(deck_py).read_text().split("\n\n") if "@universes.setter" in p)
    src = segment(os.path.join(MCNPY, "geometry.py"), "UniverseList") + "\n\n\nclass Deck:\n" + textwrap.indent(
        textwrap.dedent("""\
            def __init__(self):
                self._is_reading = False
                self._universes = {}
                self.cells = {}

            def add(self, cell):
                self.cells[cell.name] = cell
                self.get_universe(cell)
        """), "    ") + "\n" + "\n\n".join(parts + [setter]) + "\n"
    ns = {}
    exec(compile(src, "mcnpy-deck-extract", "exec"), ns)
    return ns["Deck"]


class Universe:
    def __init__(self, name, sign=None):
        self.name, self.sign, self._e_object = name, sign, object()


class Cell:
    def __init__(self, name, universe=None):
        self.name, self._universe = name, universe

    @property
    def universe(self):  # a Java read in MCNPy: 11 round trips
        TRIPS["universe"] += 1
        return self._universe


def state(deck):
    return [(k, u.name, u.sign, u._e_object, list(u.cells)) for k, u in deck._universes.items()]


@needs_sources
class FastAddTest(unittest.TestCase):
    def setUp(self):
        self.Deck = load_deck_class()

    def build(self, cells, fast):
        deck = self.Deck()
        TRIPS.clear()
        if fast:
            with mcnpy_speed.fast_add(self.Deck):
                for c in cells:
                    deck.add(c)
        else:
            for c in cells:
                deck.add(c)
        return deck, TRIPS["universe"]

    def test_root_cells(self):
        n = 120
        plain, plain_reads = self.build([Cell(i) for i in range(1, n + 1)], fast=False)
        fast, fast_reads = self.build([Cell(i) for i in range(1, n + 1)], fast=True)
        self.assertEqual(state(plain)[0][4], state(fast)[0][4])
        self.assertEqual([s[:3] for s in state(plain)], [s[:3] for s in state(fast)])
        self.assertEqual(plain_reads, n * n)  # each add walks every earlier cell twice
        self.assertEqual(fast_reads, n)
        print(f"\n  universe reads adding {n} root cells: walking {plain_reads}, stored {fast_reads} "
              f"(x11 round trips each; 267 cells: {11 * 267 * 267:,} vs {11 * 267:,})")

    def test_cells_with_universes(self):
        us = {5: Universe(5, "-"), 9: Universe(9)}

        def cells():
            return [Cell(i, us[5] if i % 3 == 0 else us[9] if i % 3 == 1 else None) for i in range(1, 40)]
        plain, _ = self.build(cells(), fast=False)
        fast, _ = self.build(cells(), fast=True)
        self.assertEqual(state(plain), state(fast))

    def test_restored(self):
        original = self.Deck.__dict__["get_universe"]
        with mcnpy_speed.fast_add(self.Deck):
            self.assertIsNot(self.Deck.__dict__["get_universe"], original)
        self.assertIs(self.Deck.__dict__["get_universe"], original)

    def test_walk_still_matters_after_changes(self):
        """Why fast_add is limited to the adds inside openmc_to_mcnp: if a cell's universe changes after it was
        added, the walking property moves it on the next add; the stored dict doesn't. MCNPy sets universes
        only after the last add ("Making Universes"), which the source guard pins."""
        a, b = Cell(1), Cell(2)
        deck = self.Deck()
        with mcnpy_speed.fast_add(self.Deck):
            deck.add(a)
            a._universe = Universe(4)
            deck.add(b)
            self.assertEqual(list(deck._universes), [0])
        self.assertEqual(sorted(deck.universes), [0, 4])  # the property (left as is) still moves it when read


@needs_sources
class SourceGuardTest(unittest.TestCase):
    def test_installed_sources_are_the_analyzed_ones(self):
        self.assertTrue(mcnpy_speed.sources_match("fast_add"))
        self.assertTrue(mcnpy_speed.sources_match("cached_reflection"))
        self.assertTrue(mcnpy_speed.sources_match("terminating_line_wrap"))

    def test_changed_source_is_refused_and_says_why(self):
        saved = mcnpy_speed.SOURCES["fast_add"]
        mcnpy_speed.SOURCES["fast_add"] = [saved[0][:3] + ("0000000000000000",)]
        try:
            self.assertFalse(mcnpy_speed.sources_match("fast_add"))
            self.assertRegex(mcnpy_speed.source_status("fast_add"), r"Deck\.get_universe differs from the version analyzed")
        finally:
            mcnpy_speed.SOURCES["fast_add"] = saved

    def test_an_unreadable_or_missing_source_says_why_too(self):
        saved = mcnpy_speed.SOURCES["fast_add"]
        try:
            mcnpy_speed.SOURCES["fast_add"] = [("mcnpy", "no_such_file.py", "Deck.get_universe", "0" * 16)]
            self.assertRegex(mcnpy_speed.source_status("fast_add"), r"no_such_file.py .* could not be read \(FileNotFoundError\)")
            mcnpy_speed.SOURCES["fast_add"] = [("no_such_package_xyz", "x.py", "f", "0" * 16)]
            self.assertEqual(mcnpy_speed.source_status("fast_add"), "no_such_package_xyz is not installed")
        finally:
            mcnpy_speed.SOURCES["fast_add"] = saved


@needs_sources
class LineWrapTest(unittest.TestCase):
    """mcnpy.deck_formatter.line_wrap, loaded by path in a child process (it only imports re)."""

    def wraps(self, line, timeout=10):
        code = textwrap.dedent(f"""
            import importlib.util
            spec = importlib.util.spec_from_file_location("deck_formatter", {os.path.join(MCNPY, "deck_formatter.py")!r})
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            out = m.line_wrap({line!r}, "", 120)
            print(max(len(x) for x in out.split(chr(10))))
        """)
        try:
            subprocess.run([sys.executable, "-c", code], check=True, timeout=timeout, capture_output=True)
            return True
        except subprocess.TimeoutExpired:
            return False

    @staticmethod
    def card(terms):
        return "268 0 (" + ":".join(f"-{1000 + i}" for i in range(terms)) + ")"

    def test_world_cell_union_wraps(self):
        self.assertTrue(self.wraps("267 0 1 -2 3 -4 5 -6 " + " ".join(["(-7:8:-9:10:-11:12:-13:14)"] * 40)))

    def test_long_union_never_ends(self):
        self.assertTrue(self.wraps(self.card(18)))   # the union is one 109-character token: wrapped
        self.assertFalse(self.wraps(self.card(19), timeout=5))  # 115 characters: line_wrap loops for ever


@needs_sources
class TerminatingLineWrapTest(unittest.TestCase):
    """mcnpy_speed.line_wrap against MCNPy's own: the same text wherever MCNPy's ends, an end where it loops."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mcnpy_deck_formatter", os.path.join(MCNPY, "deck_formatter.py"))
        cls.fmt = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(cls.fmt)

    def test_same_text_where_mcnpy_ends(self):
        """Tokens of at most 40 characters: a blank in every 114 columns, so MCNPy's line_wrap always ends."""
        rng = random.Random(7)
        for _ in range(4000):
            tokens = ["".join(rng.choice("0123456789-:()#") for _ in range(rng.randint(1, 40)))
                      for _ in range(rng.randint(0, 40))]
            line = rng.choice(["", "     ", "268 0 "]) + (" " * rng.randint(1, 3)).join(tokens)
            comment, limit = rng.choice(["", "$ cell"]), rng.choice([120, 115])
            self.assertEqual(mcnpy_speed.line_wrap(line, comment, limit), self.fmt.line_wrap(line, comment, limit))

    def test_ends_where_mcnpy_loops(self):
        for terms in (19, 60, 300):
            card = LineWrapTest.card(terms)
            out = mcnpy_speed.line_wrap(card, "", 120)
            self.assertTrue(all(len(x) <= 120 for x in out.split("\n")), terms)
            self.assertEqual("".join(out.split()), "".join(card.split()))
        out = mcnpy_speed.line_wrap("1 0 " + "7" * 200, "", 120)  # no colon to cut after: one long line
        self.assertEqual("".join(out.split()), "10" + "7" * 200)

    def test_formatter_uses_it(self):
        deck = "TITLE\n268 0 " + LineWrapTest.card(300)[6:] + "\n     IMP:N=1.0\n"
        with mcnpy_speed.terminating_line_wrap(self.fmt):
            out = self.fmt.formatter(deck)
        self.assertEqual(self.fmt.line_wrap.__module__, "mcnpy_deck_formatter")  # restored afterwards
        self.assertIn("IMP:N=1.0", out)
        self.assertTrue(all(len(x) <= 120 for x in out.split("\n")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
