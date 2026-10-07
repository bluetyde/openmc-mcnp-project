"""Write the regions of plain cells directly, so MCNPy only has to translate what it alone can.

MCNPy takes a round trip to its Java process for every term of every region (about 300 per term; the 267-cell
graphite-pile import costs 2.0 million of them after world_complement; its "Making Universes" phase is 292 of the
311 s). A cell that is only a material (or void) and a region of half-spaces needs none of that: its region is
-7 8 (-9:10) in MCNP, one token per half-space with the surface's own number (MCNPy keeps OpenMC's ids). So while
MCNPy translates, each such cell gets a short placeholder region, and the card text is patched afterwards: each
placeholder is replaced by the real region. MCNPy still writes the surface cards (boundary flags, every surface
type), the materials, the densities and the importances; nothing here knows a surface formula. validate_deck then
checks the written deck against the OpenMC model point by point, as for any deck.

A surface only appears in MCNPy's deck if some cell it translates uses it, so the placeholders between them name
every surface the plain cells use: each cell's placeholder holds the half-spaces of its region on surfaces no
earlier cell has claimed (at least one half-space), which makes the placeholders about one term per cell. (A
single dummy cell naming all the leftover surfaces was tried first: MCNPy translated 2.5 times faster, but writing
the deck with one 300-term cell then took more than eight minutes.)

plan() decides which cells; mcnpy_view() swaps the regions; patch_deck() writes the cards. Used only above
MIN_TERMS terms in total, so small models keep MCNPy's own text. If a card isn't what plan() expected,
patch_deck() raises DirectCardsMismatch and export_mcnp.translate() translates the whole model the normal way and
says so.
"""
import contextlib
import re

import openmc

MIN_TERMS = 400   # region terms (all plain cells together) below which MCNPy's cost doesn't matter
LINE = 78         # card lines stay within 80 columns (the exporter's tests and MCNP's 128 are looser)
LONG_UNION = 100  # a parenthesised union longer than this is split after its colons


class DirectCardsMismatch(Exception):
    """A translated card isn't what plan() expected; the caller translates the whole model the normal way."""


def _terms(node):
    if isinstance(node, openmc.Halfspace):
        return 1
    return sum(_terms(c) for c in node)


def _literals(node):
    if isinstance(node, openmc.Halfspace):
        yield node
    else:
        for c in node:
            yield from _literals(c)


def _is_plain(node):
    """Only half-spaces joined by intersection and union (no complement operator)."""
    if isinstance(node, openmc.Halfspace):
        return True
    if isinstance(node, (openmc.Intersection, openmc.Union)):
        return all(_is_plain(c) for c in node)
    return False


def region_text(node, top=True):
    """The MCNP geometry text of a region: -7 (the negative side), 8, blank = intersection, colon = union.
    Blank binds tighter than colon, so a union inside an intersection and an intersection inside a union get
    parentheses."""
    if isinstance(node, openmc.Halfspace):
        return f"{'-' if node.side == '-' else ''}{node.surface.id}"
    if isinstance(node, openmc.Intersection):
        return " ".join(region_text(c, False) for c in node)
    parts = [region_text(c, False) for c in node]
    text = ":".join(f"({p})" if " " in p and not p.startswith("(") else p for p in parts)
    return text if top else f"({text})"


def _tokens(text):
    """Whitespace tokens of a region text; a very long parenthesised union is cut after its colons (MCNP lets
    blanks follow a colon) so no token passes the line width."""
    out = []
    for tok in text.split():
        if len(tok) <= LONG_UNION:
            out.append(tok)
            continue
        pieces = tok.split(":")
        out.extend(p + ":" for p in pieces[:-1])
        out.append(pieces[-1])
    return out


def plan(geometry, skip=(), extra_seen=(), min_terms=None):
    """None, or {'texts': {cell id: region text}, 'place': {cell id: placeholder text}, 'placeholders': {cell id:
    placeholder region}, 'cells', 'regions'}. Plain cells are those filled with a material or void, not moved,
    whose region is half-spaces joined by intersection and union. `skip` are cells handled elsewhere (the world
    cell of world_complement) and `extra_seen` the surfaces they still give MCNPy."""
    min_terms = MIN_TERMS if min_terms is None else min_terms
    cells = geometry.get_all_cells()
    plain, other = {}, []
    for cid, c in cells.items():
        if cid in skip:
            continue
        ok = c.region is not None and (c.fill is None or isinstance(c.fill, openmc.Material)) \
            and c.translation is None and c.rotation is None and _is_plain(c.region) and _terms(c.region) > 1
        if ok:
            plain[cid] = c
        else:
            other.append(c)
    if sum(_terms(c.region) for c in plain.values()) < min_terms:
        return None
    covered = set(extra_seen)
    for c in other:
        if c.region is None:
            continue
        if isinstance(c.region, openmc.Complement):
            return None  # a complement elsewhere: leave the model to MCNPy alone
        covered.update(h.surface.id for h in _literals(c.region))
    lits = {}
    for cid, c in plain.items():
        mine = []
        for h in _literals(c.region):
            if h.surface.id not in covered:
                covered.add(h.surface.id)
                mine.append(h)
        lits[cid] = mine or [next(_literals(c.region))]  # every cell keeps at least one term
    placeholders = {cid: h[0] if len(h) == 1 else openmc.Intersection(h) for cid, h in lits.items()}
    return {"texts": {cid: region_text(c.region) for cid, c in plain.items()},
            "place": {cid: region_text(p) for cid, p in placeholders.items()},
            "regions": {cid: c.region for cid, c in plain.items()}, "cells": plain, "placeholders": placeholders,
            "surfaces": {h.surface.id for c in plain.values() for h in _literals(c.region)}}


@contextlib.contextmanager
def mcnpy_view(cp):
    """While MCNPy translates: each plain cell holds its placeholder (a few half-spaces)."""
    if cp is None:
        yield
        return
    for cid, c in cp["cells"].items():
        c.region = cp["placeholders"][cid]
    try:
        yield
    finally:
        for cid, c in cp["cells"].items():
            c.region = cp["regions"][cid]


def _wrap(tokens):
    out, cur = [], ""
    for t in tokens:
        if cur and len(cur) + 1 + len(t) > LINE:
            out.append(cur)
            cur = "     " + t
        else:
            cur = f"{cur} {t}" if cur else t
    out.append(cur)
    return out


def patch_deck(text, cp):
    """The deck text with each plain cell's placeholder replaced by its region.
    Raises DirectCardsMismatch if a card isn't what plan() expected."""
    lines = text.split("\n")
    out = [lines[0]]
    i, texts, done = 1, cp["texts"], set()
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break  # the end of the cell block
        m = re.match(r"^(\d+)\s+(\d+)(\s|$)", line)
        if not m:
            out.append(line)
            i += 1
            continue
        j = i + 1
        while j < len(lines) and re.match(r"^\s{5,}\S", lines[j]):
            j += 1
        cid = int(m.group(1))
        if cid in texts:
            tokens = " ".join(lines[i:j]).split()
            start = 2 if int(tokens[1]) == 0 else 3
            at = next((k for k in range(start, len(tokens)) if re.match(r"^[A-Za-z*]", tokens[k])), len(tokens))
            if tokens[start:at] != cp["place"][cid].split():
                raise DirectCardsMismatch(f"cell {cid}: expected the placeholder {cp['place'][cid]!r} in the translated "
                                          f"card, found {' '.join(tokens[start:at])!r}")
            out.extend(_wrap(tokens[:start] + _tokens(texts[cid]) + tokens[at:]))
            done.add(cid)
        else:
            out.extend(lines[i:j])
        i = j
    if done != set(texts):
        raise DirectCardsMismatch(f"cells {sorted(set(texts) - done)[:5]} are not in the translated deck")
    rest = lines[i:]  # a blank line, then the surface cards
    cards, k = set(), 1
    while k < len(rest) and rest[k].strip():
        m = re.match(r"^[*+]?(\d+)\s+[A-Za-z]", rest[k])
        if m:
            cards.add(int(m.group(1)))
        k += 1
    absent = cp["surfaces"] - cards
    if absent:  # the written regions would name surfaces the deck never defines
        raise DirectCardsMismatch(f"surfaces {sorted(absent)[:5]} have no card in the translated deck")
    out.extend(rest)
    return "\n".join(out)


def patch_file(path, cp):
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(patch_deck(text, cp))
