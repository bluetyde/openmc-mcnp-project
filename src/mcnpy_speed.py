"""Make MCNPy's translate cheaper without changing what it writes.

MCNPy (0.0.7, Java behind py4j) spends almost all of translate in py4j round trips, and most of those trips
ask the same questions again and again:

1. Deck.add -> get_universe reads the Deck.universes property twice per cell (mcnpy/deck.py:544 and 545, or
   557 and 558 for root cells). The property walks every cell added so far and reads `cell.universe` from
   Java each time (deck.py:228), 11 round trips per read (metapy/wrap.py:448, 216-217: eGet, the five-step
   lookup of org.eclipse.emf.ecore.EAttribute, and the five-step is_instance_of). Adding N cells costs
   11 N^2 trips: 784,000 for the 267-cell graphite pile. The walk only moves cells whose universe was
   changed after they were added; inside openmc_to_mcnp no cell has a universe until "Making Universes",
   after the last add, so every walk done by an add changes nothing. fast_add() gives get_universe the
   stored dict instead of the property; the later reads of deck.universes (fills, lattices) still walk.

2. Every generated property read or write in metapy.wrap looks up the EMF classes EAttribute, EReference,
   EEnum by name (five round trips each, nothing is cached by py4j), asks Java is_instance_of about the
   feature (five more), and for attributes fetches the feature's data type (a new Java object each time).
   These are questions about the metamodel, which never changes while the gateway runs. cached_reflection()
   answers each one once per feature: same answers, so the same eGet/eSet calls reach Java.

3. terminating_line_wrap() replaces that line_wrap with one that gives the same text whenever the original
   ends, and cuts a too-long union after a colon where the original would loop.

mcnpy_speedups() applies 1, 2 and 3 around openmc_to_mcnp and deck.write and yields what is on and, for
each one left off, why. MCNPY_SPEEDUPS=0 turns 1 and 2 off
(for comparing decks); 3 stays on. Each applies only to the MCNPy source it was worked out from (SOURCES).
"""
import contextlib
import os
import warnings

from py4j.java_collections import JavaList
from py4j.java_gateway import JavaClass, JavaObject, JavaPackage


# 1. Deck.add without re-reading every cell's universe

def _get_universe_stored(self, cell):
    """mcnpy.deck.Deck.get_universe (0.0.7, deck.py:541-563) with self._universes in place of the property."""
    UniverseList = _get_universe_stored.UniverseList
    if cell.universe is not None:
        u_id = cell.universe.name
        if u_id in self._universes:
            _universe = self._universes[u_id]
            _universe.add_only(cell)
        else:
            _universe = UniverseList(name=u_id, cells=None)
            if cell.universe.sign is not None:
                _universe.sign = cell.universe.sign
            _universe.add_only(cell)
            self._universes[u_id] = _universe
            _universe._e_object = cell.universe
    else:
        u_id = 0
        if u_id in self._universes:
            _universe = self._universes[u_id]
            _universe.add_only(cell)
        else:
            _universe = UniverseList(name=u_id, cells=None)
            _universe.add_only(cell)
            self._universes[u_id] = _universe


@contextlib.contextmanager
def fast_add(deck_cls):
    """While inside: deck_cls.get_universe uses the stored universe dict, not the walking property."""
    original = deck_cls.__dict__["get_universe"]
    _get_universe_stored.UniverseList = original.__globals__["UniverseList"]
    deck_cls.get_universe = _get_universe_stored
    try:
        yield
    finally:
        deck_cls.get_universe = original


# 2. Metamodel questions answered once

class _JVMNode:
    """gateway.jvm.<package>...<Class> with every step resolved once. Classes are returned as py4j JavaClass
    objects (is_instance_of needs one); packages as nodes."""

    def __init__(self, real, cache, path=""):
        self._real, self._cache, self._path = real, cache, path

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        path = f"{self._path}.{name}" if self._path else name
        hit = self._cache.get(path)
        if hit is None:
            hit = getattr(self._real, name)
            self._cache[path] = hit
        if isinstance(hit, JavaClass):
            return hit
        return _JVMNode(hit, self._cache, path)


class _GatewayView:
    """metapy's gateway as metapy.wrap sees it: the JVM view and the entry point's fixed fields (copier,
    equalityHelper: fields of metapy's EntryPoint that are never reassigned) are fetched once; anything
    else goes to the real gateway."""

    FIXED_FIELDS = ("copier", "equalityHelper")

    def __init__(self, real):
        self._real = real
        self._fields = {}
        self.jvm = _JVMNode(real.jvm, {})

    def __getattr__(self, name):
        if name in self.FIXED_FIELDS:
            if name not in self._fields:
                self._fields[name] = getattr(self._real, name)
            return self._fields[name]
        return getattr(self._real, name)


def _class_name(java_class):
    if isinstance(java_class, str):
        return java_class
    if isinstance(java_class, JavaClass):
        return java_class._fqn
    return None


class _Reflection:
    """Answers about metamodel objects (EStructuralFeatures and their data types), cached by py4j object id.
    py4j ids are never reused while the gateway runs (the JVM numbers objects o1, o2, ... upward), and a Java
    object's class never changes, so an answer cached under an id stays right."""

    def __init__(self, wrap):
        self.wrap = wrap
        self.gateway = wrap.gateway
        self.is_instance_of_real = wrap.is_instance_of
        self.instance = {}
        self.features = {}
        self.names = {}

    def is_instance_of(self, java_object, java_class, *args, **kwargs):
        oid, fqn = getattr(java_object, "_target_id", None), _class_name(java_class)
        if oid is None or fqn is None or args or kwargs:
            return self.is_instance_of_real(java_object, java_class, *args, **kwargs)
        key = (oid, fqn)
        if key not in self.instance:
            self.instance[key] = self.is_instance_of_real(java_object, java_class)
        return self.instance[key]

    def feature(self, feature):
        """(is an EAttribute, its data type or None, the data type is an EEnum) for a structural feature."""
        oid = feature._target_id
        info = self.features.get(oid)
        if info is None:
            ecore = self.gateway.jvm.org.eclipse.emf.ecore
            attribute = self.is_instance_of(feature, ecore.EAttribute)
            data_type = feature.getEAttributeType() if attribute else None
            enum = attribute and self.is_instance_of_real(data_type, ecore.EEnum)
            info = self.features[oid] = (attribute, data_type, enum)
        return info

    def name(self, feature):
        oid = feature._target_id
        if oid not in self.names:
            self.names[oid] = feature.getName()
        return self.names[oid]

    # metapy.wrap.return_value_converter (wrap.py:195-243), with the metamodel answers cached
    def return_value_converter(self, feature, value):
        if isinstance(value, str) and self.name(feature) != 'name':
            try:
                if float(value) % 1 == 0:
                    if int(value) != 0 and value.startswith('0') is False:
                        return int(value)
                    elif int(value) == 0 and len(value) == 1:
                        return int(value)
                    else:
                        return value
                else:
                    return float(value)
            except ValueError:
                return value
        else:
            attribute, data_type, enum = self.feature(feature)
            if attribute:
                if isinstance(value, JavaList):
                    if enum:
                        e_list = []
                        for v in value:
                            val_str = v.toString()
                            if val_str == '¥×¥':
                                val_str = None
                            e_list.append(val_str)
                        return e_list
                if enum:
                    val_str = value.toString()
                    if val_str == '¥×¥':
                        return None
                    return val_str
            return value

    # metapy.wrap.is_enum (wrap.py:167-184), with the metamodel answers cached
    def is_enum(self, value, feature):
        attribute, data_type, enum = self.feature(feature)
        if attribute:
            str_or_int = isinstance(value, str) or isinstance(value, int)
            if enum and str_or_int:
                if isinstance(value, str):
                    value = value.upper()
                literal = data_type.getEEnumLiteral(value)
                if literal is None:
                    literal = data_type.getEEnumLiteralByLiteral(value)
                if literal is not None:
                    value = literal.getInstance()
        return value


@contextlib.contextmanager
def cached_reflection(wrap):
    """While inside: metapy.wrap (the module `wrap`) answers metamodel questions from a cache."""
    names = ("gateway", "is_instance_of", "return_value_converter", "is_enum")
    saved = {n: getattr(wrap, n) for n in names}
    r = _Reflection(wrap)
    wrap.gateway = _GatewayView(saved["gateway"])
    r.gateway = wrap.gateway
    wrap.is_instance_of = r.is_instance_of
    wrap.return_value_converter = r.return_value_converter
    wrap.is_enum = r.is_enum
    try:
        yield r
    finally:
        for n, v in saved.items():
            setattr(wrap, n, v)


def enabled():
    return os.environ.get("MCNPY_SPEEDUPS", "1") != "0"


# The MCNPy 0.0.7 / metapy 0.0.1 source each speedup was worked out from (sha256 of the function's text, first
# 16 hex digits). If any differs, that speedup is left off: its equivalence argument no longer applies.
SOURCES = {
    "fast_add": [("mcnpy", "deck.py", "Deck.get_universe", "9d63678ef7df44e0"),
                 ("mcnpy", "deck.py", "Deck.add", "b5d599cd859dfdcc"),
                 ("mcnpy", "translate_mcnp_openmc.py", "openmc_to_mcnp", "2e7f82d91c6752b9")],
    "cached_reflection": [("metapy", "wrap.py", "return_value_converter", "955b6ce11d580223"),
                          ("metapy", "wrap.py", "is_enum", "a06db6d739d233c8"),
                          ("metapy", "wrap.py", "e_class_body", "0c934a834e471f77")],
    "terminating_line_wrap": [("mcnpy", "deck_formatter.py", "line_wrap", "e9715d55c0924e98"),
                              ("mcnpy", "deck_formatter.py", "formatter", "beb00d1ca2e7844e")],
}


def source_digest(path, qualname):
    import ast
    import hashlib
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MCNPy's regexes are plain strings
        body, node = ast.parse(text).body, None
    for part in qualname.split("."):
        node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        body = node.body
    return hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()[:16]


def source_status(name):
    """None if the installed MCNPy has the source speedup `name` was written for, else the reason it was not
    (reads files; imports nothing)."""
    import importlib.util
    for package, filename, qualname, digest in SOURCES[name]:
        spec = importlib.util.find_spec(package)
        if spec is None or not spec.submodule_search_locations:
            return f"{package} is not installed"
        path = os.path.join(list(spec.submodule_search_locations)[0], filename)
        try:
            got = source_digest(path, qualname)
        except (OSError, SyntaxError, StopIteration) as e:
            return f"{package}/{filename} {qualname} could not be read ({type(e).__name__})"
        if got != digest:
            return f"{package}/{filename} {qualname} differs from the version analyzed"
    return None


def sources_match(name):
    return source_status(name) is None


@contextlib.contextmanager
def mcnpy_speedups():
    """fast_add + cached_reflection + terminating_line_wrap around MCNPy's translate (import mcnpy first: that
    starts its gateway). Yields {name: "on" | "off (why)"} so the caller can say what was left off."""
    status = {}
    with contextlib.ExitStack() as stack:
        for name, switchable in (("fast_add", True), ("cached_reflection", True), ("terminating_line_wrap", False)):
            if switchable and not enabled():  # terminating_line_wrap is a fix, not a speedup: on even then
                status[name] = "off (MCNPY_SPEEDUPS=0)"
                continue
            why = source_status(name)
            if why:
                status[name] = f"off ({why})"
                continue
            if name == "fast_add":
                import mcnpy.deck
                stack.enter_context(fast_add(mcnpy.deck.Deck))
            elif name == "cached_reflection":
                import metapy.wrap
                stack.enter_context(cached_reflection(metapy.wrap))
            else:
                import mcnpy.deck_formatter
                stack.enter_context(terminating_line_wrap(mcnpy.deck_formatter))
            status[name] = "on"
        yield status


# 3. deck.write without the endless line_wrap

def line_wrap(before_comment, comment, line_limit):
    """mcnpy.deck_formatter.line_wrap (0.0.7, deck_formatter.py:3-24) with a way out of its one endless case.

    After a wrap the rest of the line is a newline + 5 blanks + text. If that text has no blank before column
    line_limit, the original finds ws_index == 5, keeps before_comment[:5] and rebuilds the very same string,
    for ever (MCNPy prints a union as one blank-free token, so any union longer than 114 characters does it).
    Only in that state this version cuts after the last colon instead (MCNP allows a line break after ':'),
    or, with no colon, leaves the token whole on one long line. Every other step is the original's, so for
    every input on which the original ends, the output is the same."""
    if (len(before_comment) > line_limit):
        line = ''
        ws_index = 0
        while (len(before_comment) > line_limit):
            for i in range(len(before_comment)):
                if (before_comment[i] == ' '):
                    ws_index = i
                if (i >= line_limit-1):
                    if ws_index == 5 and before_comment.startswith('\n     '):
                        cut = before_comment.rfind(':', 6, line_limit)
                        if cut < 0:
                            line = line + before_comment
                            before_comment = ''
                        else:
                            line = line + before_comment[:cut + 1]
                            before_comment = '\n     ' + before_comment[cut + 1:]
                        break
                    if (before_comment[ws_index:].lstrip() != ''):
                        line = line + before_comment[:ws_index]
                        before_comment = '\n     ' + before_comment[ws_index:].lstrip()
                        ws_index = 5
                    else:
                        line = line + before_comment[:ws_index]
                        before_comment = ''
                    break
        line = line + before_comment + comment
    else:
        line = before_comment + comment

    return line


@contextlib.contextmanager
def terminating_line_wrap(deck_formatter):
    """While inside: MCNPy's deck formatter (formatter, print_lattice, print_material) uses line_wrap above."""
    original = deck_formatter.line_wrap
    deck_formatter.line_wrap = line_wrap
    try:
        yield
    finally:
        deck_formatter.line_wrap = original
