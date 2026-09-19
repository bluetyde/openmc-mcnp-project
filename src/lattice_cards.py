"""
Rectangular lattices (openmc.RectLattice) as MCNP LAT=1 / FILL cards.

MCNPy translates the rest of a model faithfully but gets lattices wrong: it moves the unit universe's cells
with TRCL, writes an element box that isn't centred on them, and picks index ranges that don't cover the
lattice. This module rewrites those cards from the OpenMC lattice after MCNPy has run. MCNPy keeps OpenMC's
cell and universe numbers (a lattice with id L becomes universe L), which is how the cards are found.
Page numbers are PDF pages of the MCNP 6.3.0 manual (LA-UR-22-30006 Rev. 1).

  lattice cell  N 0 -a b -c d -e f U=L LAT=1 FILL=...
                The [0,0,0] element is a box centred on its own origin. Beyond the 1st listed surface is
                element [1,0,0], beyond the 2nd [-1,0,0], then [0,1,0], [0,-1,0], [0,0,1], [0,0,-1] (p. 290),
                so a = PX +px/2 and b = PX -px/2 make the first index increase along +x.
  FILL          one universe when every element holds it and the lattice covers the filled cell, else
                FILL=i1:i2 j1:j2 k1:k2 u... with i varying fastest, then j, then k (p. 291-292).
                OpenMC lists RectLattice rows top (+y) first, so y is flipped.
  filled cell   FILL=L (x0 y0 z0): the lattice universe's origin, i.e. the centre of element [0,0,0], in the
                filled cell's coordinates (p. 291, "FILL = n (o1 o2 o3 ...)").
  unit cells    in the unit universes' own coordinates, so MCNPy's TRCL on them is removed.
Surfaces of repeated structures must be numbered <= 999 (p. 271).
"""
import math
import re

import openmc

from mcnp_cards import UnsupportedFeature

MAX_RS_SURFACE = 999  # p. 271: surface numbers used in repeated structures
LINE = 78
COMMENT = re.compile(r"^\s{0,4}[cC](\s|$)")


def _fmt(v):
    v = float(v)
    return "0" if v == 0 else repr(round(v, 10))


def prepare(model):
    """Before MCNPy: give region-less cells an explicit everywhere region (MCNPy writes None as text and fails).
    Returns notes."""
    cells = [c for c in model.geometry.get_all_cells().values() if c.region is None]
    if not cells:
        return []
    sid = max(model.geometry.get_all_surfaces(), default=0) + 1
    big = openmc.Sphere(surface_id=sid, r=1.0e9)
    for c in cells:
        c.region = -big | +big
    return [f"Cells {sorted(c.id for c in cells)} had no region (they fill everything); "
            f"written as '-{sid}:{sid}' around a far sphere {sid}."]


def _split_cards(block):
    """Cell block -> (header lines, [card text]). A card is its first line plus continuation lines (5+ leading
    spaces); comment lines stay with the card after them."""
    header, cards, pending = [], [], []
    for ln in block.split("\n"):
        if COMMENT.match(ln):
            pending.append(ln)
        elif ln.startswith("     ") and cards:
            cards[-1] += "\n" + ln
        elif re.match(r"^\s*\d+\s", ln):
            cards.append("\n".join(pending + [ln]))
            pending = []
        elif not cards:
            header.append(ln)
        else:
            pending.append(ln)
    if pending:
        cards.append("\n".join(pending))
    return header, cards


def _body(card):
    return " ".join(ln.strip() for ln in card.split("\n") if not COMMENT.match(ln)).strip()


def _number(card):
    b = _body(card)
    return int(b.split()[0]) if b and b.split()[0].isdigit() else None


def _geom_surfs(body):
    """Surface numbers in a cell card's geometry: after the cell number and material (and density), before
    the first keyword."""
    toks = body.replace("(", " ( ").replace(")", " ) ").replace(":", " : ").split()
    if len(toks) < 2:
        return set()
    rest = toks[2:] if toks[1] == "0" else toks[3:]
    out = set()
    for t in rest:
        if re.match(r"^[A-Za-z*]", t):
            break
        m = re.match(r"^[-+]?(\d+)(\.\d+)?$", t)
        if m:
            out.add(int(m.group(1)))
    return out


def _universe_of(body):
    m = re.search(r"\bU\s*=?\s*(-?\d+)\b", body, re.I)
    return abs(int(m.group(1))) if m else 0


def _trcl_numbers(body):
    return {int(m.group(1)) for m in re.finditer(r"\bTRCL\s*=?\s*(\d+)\b", body, re.I)}


def _wrap(text):
    out, line = [], ""
    for tok in text.split():
        if line and len(line) + 1 + len(tok) > LINE:
            out.append(line)
            line = "     " + tok
        else:
            line = f"{line} {tok}" if line else tok
    out.append(line)
    return "\n".join(out)


def rewrite(blocks, model):
    """Rewrite MCNPy's lattice cards in a deck split into blocks [cells, surfaces, data...], in place.
    Returns notes."""
    geometry = model.geometry
    universes = geometry.get_all_lattices().values()  # get_all_universes() leaves lattices out
    hexes = [u.id for u in universes if isinstance(u, openmc.HexLattice)]
    if hexes:
        raise UnsupportedFeature(f"Hexagonal lattices {hexes} aren't exported as MCNP LAT=2 yet; write them cell by cell.")
    lattices = [u for u in universes if isinstance(u, openmc.RectLattice)]
    if not lattices:
        return []
    header, cards = _split_cards(blocks[0])
    by_num = {_number(c): i for i, c in enumerate(cards) if _number(c) is not None}
    next_surf = max((int(m.group(1)) for m in re.finditer(r"^\s*\*?(\d+)\s", blocks[1], re.M)), default=0) + 1
    all_cells = geometry.get_all_cells()
    new_surfaces, dropped_tr, old_elem_surfs, notes = [], set(), set(), []

    for lat in lattices:
        L = lat.id
        elem = [i for i, c in enumerate(cards) if _universe_of(_body(c)) == L and re.search(r"\bLAT\b", _body(c), re.I)]
        if len(elem) != 1:
            raise UnsupportedFeature(f"Lattice {L}: expected one MCNP LAT cell in universe {L}, found {len(elem)}.")
        body = _body(cards[elem[0]])
        num = int(body.split()[0])
        old_elem_surfs |= _geom_surfs(body)
        dropped_tr |= _trcl_numbers(body)
        imp = " ".join(re.findall(r"\bIMP:[A-Za-z,]+\s*=\s*\S+", body, re.I)) or "IMP:N=1"

        pitch = [float(p) for p in lat.pitch]
        u = lat.universes if len(pitch) == 3 else [lat.universes]  # [z][y, top row first][x]
        nz, ny, nx = len(u), len(u[0]), len(u[0][0])
        dims = [nx, ny, nz]

        # the [0,0,0] element around the origin: +x, -x, +y, -y (, +z, -z) planes in that order (p. 290)
        planes = []
        for axis, p in zip("XYZ", pitch):
            for sign in (1, -1):
                if next_surf > MAX_RS_SURFACE:
                    raise UnsupportedFeature(f"Lattice {L}: MCNP needs repeated-structure surfaces numbered <= 999 "
                                             f"(manual p. 271); this deck already reaches {next_surf - 1}.")
                new_surfaces.append(f"{next_surf} P{axis} {_fmt(sign * p / 2)}")
                planes.append(next_surf)
                next_surf += 1
        region = " ".join(f"-{s}" if i % 2 == 0 else f"{s}" for i, s in enumerate(planes))

        # index ranges: 0..n-1, widened (with lat.outer) where a filled cell reaches past the lattice
        filled = [c for c in all_cells.values() if c.fill is lat]
        if not filled:
            raise UnsupportedFeature(f"Lattice {L} isn't used by any cell.")
        ll = [float(v) for v in lat.lower_left]
        lo_idx, hi_idx = [0, 0, 0], [nx - 1, ny - 1, nz - 1]
        for c in filled:
            if c.translation is not None or c.rotation is not None:
                raise UnsupportedFeature(f"Cell {c.id} fills lattice {L} with a translation or rotation; not supported yet.")
            blo, bhi = c.region.bounding_box
            for k in range(len(pitch)):
                if not (math.isfinite(blo[k]) and math.isfinite(bhi[k])):
                    if lat.outer is None:
                        continue  # unbounded along k with no outer: OpenMC loses particles there too
                    raise UnsupportedFeature(f"Cell {c.id} (filled by lattice {L}) is unbounded along {'xyz'[k]}.")
                a = math.floor((blo[k] - ll[k]) / pitch[k] + 1e-9)
                b = math.ceil((bhi[k] - ll[k]) / pitch[k] - 1e-9) - 1
                if (a < 0 or b > dims[k] - 1) and lat.outer is None:
                    raise UnsupportedFeature(f"Lattice {L} doesn't cover cell {c.id} and has no outer universe.")
                lo_idx[k], hi_idx[k] = min(lo_idx[k], a), max(hi_idx[k], b)

        def uni(i, j, k):
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                return u[k][ny - 1 - j][i].id
            return lat.outer.id
        ids = [uni(i, j, k) for k in range(lo_idx[2], hi_idx[2] + 1)
               for j in range(lo_idx[1], hi_idx[1] + 1) for i in range(lo_idx[0], hi_idx[0] + 1)]
        if lo_idx == [0, 0, 0] and hi_idx == [nx - 1, ny - 1, nz - 1] and len(set(ids)) == 1:
            fill = f"FILL={ids[0]}"
        else:
            fill = "FILL=" + " ".join(f"{a}:{b}" for a, b in zip(lo_idx, hi_idx)) + " " + " ".join(map(str, ids))
        cards[elem[0]] = _wrap(f"{num} 0 {region} U={L} LAT=1 {fill} {imp}")

        # the filled cells look at the lattice universe with its origin at the centre of element [0,0,0]
        origin = [ll[k] + pitch[k] / 2 for k in range(len(pitch))] + ([0.0] if len(pitch) == 2 else [])
        for c in filled:
            i = by_num.get(c.id)
            if i is None:
                raise UnsupportedFeature(f"Lattice {L}: MCNP cell {c.id}, which fills it, isn't in the deck.")
            new, n = re.subn(rf"\bFILL\s*=?\s*{L}\b(?!\s*:)(\s*\([^)]*\))?",
                             f"FILL={L} ({' '.join(_fmt(v) for v in origin)})", _body(cards[i]), flags=re.I)
            if n != 1:
                raise UnsupportedFeature(f"Lattice {L}: couldn't find 'FILL {L}' on cell {c.id}.")
            cards[i] = _wrap(new)

        # unit universes are in their own coordinates: drop MCNPy's TRCL on their cells
        used = set(ids)
        for idx, card in enumerate(cards):
            b = _body(card)
            if _universe_of(b) in used and re.search(r"\bTRCL\b", b, re.I):
                oc = all_cells.get(int(b.split()[0]))
                if oc is not None and (oc.translation is not None or oc.rotation is not None):
                    raise UnsupportedFeature(f"Cell {oc.id} in a lattice universe has its own translation or rotation; "
                                             f"not supported yet.")
                dropped_tr |= _trcl_numbers(b)
                cards[idx] = _wrap(re.sub(r"\s\*?TRCL\s*=?\s*(\([^)]*\)|\d+)", "", b, flags=re.I))
        notes.append(f"Lattice {L}: MCNP LAT=1 cell {num}, pitch {pitch}, element [0,0,0] centred at "
                     f"{[round(v, 6) for v in origin]}, indices {lo_idx} to {hi_idx}.")

    # surfaces: drop MCNPy's element boxes if nothing uses them now, then add the new planes
    used_surfs = set().union(*(_geom_surfs(_body(c)) for c in cards))
    kept = []
    for ln in blocks[1].split("\n"):
        m = re.match(r"^\s*\*?(\d+)\s", ln)
        if m and int(m.group(1)) in old_elem_surfs and int(m.group(1)) not in used_surfs:
            continue
        kept.append(ln)
    blocks[1] = "\n".join(kept + new_surfaces)
    blocks[0] = "\n".join(header + cards)
    # TR cards that only MCNPy's lattice TRCLs used
    still = set().union(*(_trcl_numbers(_body(c)) for c in cards))
    gone = dropped_tr - still
    if gone:
        for bi in range(2, len(blocks)):
            blocks[bi] = "\n".join(ln for ln in blocks[bi].split("\n")
                                   if not ((m := re.match(r"^\*?TR(\d+)\s", ln.strip(), re.I)) and int(m.group(1)) in gone))
    return notes
