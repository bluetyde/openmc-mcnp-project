"""Write "everything outside the other cells" as #cell complements, not as expanded clauses.

OpenMC stores the rest-of-the-world cell of a Studio model as  box & ~(part | part | ...). When OpenMC writes
or reads it, the complement is expanded by De Morgan into one parenthesised union per cell, so for a model
of N cells with k surfaces each, that one cell carries N x k terms. MCNPy translates a cell term by term, one
round trip to its Java process each (stack samples on a 267-cell import: all the time is in py4j socket reads),
so that cell costs about as much as all the real cells together. MCNP says the same thing in N tokens:

    267 0  1 -2 3 -4 5 -6  #1 #2 #3 ... #266       ($ outside the box's contents; manual: complement operator #)

A clause that is the complement of another root cell's region (leaving out the world-box planes every Studio cell repeats) is that cell's `#n` inside the box: the intersection
of the complements of cells is the complement of their union, in every case (cells may overlap, fill with a
universe or hold a material; `#n` uses the region of cell n only). So the replacement never changes the
geometry; validate_deck checks the written deck against the OpenMC model point by point either way.

plan() finds such a cell; mcnpy_view() hands MCNPy the cell without those clauses; patch_deck() adds the #n
tokens to the card MCNPy wrote. Used only above MIN_TERMS replaced terms, so small models keep their
familiar expanded form.
"""
import contextlib
import re

import openmc

MIN_TERMS = 200   # replaced literals below which MCNPy's cost doesn't matter
LINE = 78         # card lines stay within the 80 columns validate_deck allows


def _canon(node, negate=False):
    """A region as nested tuples ('s', surface id, side) / ('and'|'or', sorted children); negate=True gives its
    complement (De Morgan), so equal regions compare equal whatever order their terms are written in."""
    if isinstance(node, openmc.Halfspace):
        side = node.side
        if negate:
            side = "+" if side == "-" else "-"
        return ("s", node.surface.id, side)
    if isinstance(node, openmc.Complement):
        return _canon(node.node, not negate)
    if isinstance(node, (openmc.Intersection, openmc.Union)):
        kind = "and" if isinstance(node, openmc.Intersection) != negate else "or"
        kids = []
        for child in node:
            k = _canon(child, negate)
            if k[0] == kind:  # a nested group of the same kind is the same group
                kids.extend(k[1])
            else:
                kids.append(k)
        return (kind, tuple(sorted(kids)))
    raise TypeError(f"unknown region node {type(node).__name__}")


def _terms(node):
    if isinstance(node, openmc.Halfspace):
        return 1
    if isinstance(node, openmc.Complement):
        return _terms(node.node)
    return sum(_terms(c) for c in node)


def plan(geometry, min_terms=None):
    """None, or {'cell': the world cell, 'kept': the region MCNPy is given, 'ids': cell ids to complement,
    'terms': literals replaced}. Needs a root cell shaped  K & C1 & C2 ... where K are plain half-spaces (the
    world box) and each clause C is the complement of the region of another root cell, ignoring the half-spaces
    of K that cell repeats (Studio gives every cell the world box; the world cell's clauses leave it out).

    Why that is exact: with K true, K & ~(P & K') = K & (~P | ~K') = K & ~P when K' is made of K's literals,
    so K & ~P is K & #cell for the cell P & K'."""
    min_terms = MIN_TERMS if min_terms is None else min_terms
    cells = {c.id: c for c in geometry.root_universe.cells.values()}
    best = None
    for w in cells.values():
        r = w.region
        if not isinstance(r, openmc.Intersection) or sum(isinstance(n, openmc.Union) for n in r) < 2:
            continue
        box = [n for n in r if isinstance(n, openmc.Halfspace)]
        box_keys = {_canon(n) for n in box}
        neg = {}
        for cid, c in cells.items():
            if cid == w.id or c.region is None:
                continue
            rest = [n for n in c.region if not (isinstance(n, openmc.Halfspace) and _canon(n) in box_keys)] \
                if isinstance(c.region, openmc.Intersection) else [c.region]
            if rest and not (len(rest) == 1 and isinstance(rest[0], openmc.Halfspace) and _canon(rest[0]) in box_keys):
                part = rest[0] if len(rest) == 1 else openmc.Intersection(rest)
                neg.setdefault(_canon(part, negate=True), cid)
        ids, kept, terms = [], list(box), 0
        for node in r:
            if isinstance(node, openmc.Halfspace):
                continue
            cid = neg.get(_canon(node))
            if cid is None:
                kept.append(node)
            else:
                ids.append(cid)
                terms += _terms(node)
        if ids and box and terms >= min_terms and (best is None or terms > best["terms"]):
            best = {"cell": w, "kept": kept[0] if len(kept) == 1 else openmc.Intersection(kept),
                    "ids": sorted(set(ids)), "terms": terms}
    return best


@contextlib.contextmanager
def mcnpy_view(wc):
    """While MCNPy translates: the world cell holds only the clauses that are not complements of cells."""
    if wc is None:
        yield
        return
    cell, saved = wc["cell"], wc["cell"].region
    cell.region = wc["kept"]
    try:
        yield
    finally:
        cell.region = saved


def patch_deck(text, wc):
    """The deck text with `#n` for every complemented cell added to the world cell's card (before its keywords,
    such as IMP:N=1, which end the geometry)."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{wc['cell'].id}\s+\d+(\s|$)", ln)), None)
    if start is None:
        raise ValueError(f"cell {wc['cell'].id} is not in the translated deck")
    end = start + 1
    while end < len(lines) and re.match(r"^\s{5,}\S", lines[end]):
        end += 1
    tokens = " ".join(lines[start:end]).split()
    at = next((k for k, t in enumerate(tokens) if k > 1 and re.match(r"^[A-Za-z*]", t)), len(tokens))
    tokens[at:at] = [f"#{i}" for i in wc["ids"]]
    out, cur = [], ""
    for t in tokens:
        if cur and len(cur) + 1 + len(t) > LINE:
            out.append(cur)
            cur = "     " + t
        else:
            cur = f"{cur} {t}" if cur else t
    out.append(cur)
    lines[start:end] = out
    return "\n".join(lines)


def patch_file(path, wc):
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(patch_deck(text, wc))
