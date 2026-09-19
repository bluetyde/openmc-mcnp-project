"""
OpenMC lattices as MCNP lattice cards: RectLattice -> LAT=1, HexLattice -> LAT=2 (with FILL).

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

HexLattice (LAT=2): an element is a hexagonal prism; its 8 surfaces are listed [1,0,0], [-1,0,0], [0,1,0],
[0,-1,0], [-1,1,0], [1,-1,0], then the two base planes (p. 290, example p. 766). [0,1,0] must be next to
[1,0,0], so the index steps T1 and T2 are neighbour directions 60 degrees apart and [-1,1,0] is at T2 - T1.
MCNPy can't translate hex lattices at all, so mcnpy_view() shows it a placeholder while it runs.
"""
import contextlib
import math
import re
import warnings

import openmc

from mcnp_cards import UnsupportedFeature

MAX_RS_SURFACE = 999  # p. 271: surface numbers used in repeated structures
LINE = 78
COMMENT = re.compile(r"^\s{0,4}[cC](\s|$)")


def _fmt(v):
    v = float(v)
    return "0" if v == 0 else repr(round(v, 10))


FAR = 1.0e9  # cm: "infinite" for the two workarounds below


def fix_loaded_hex_lattices(model):
    """OpenMC 0.15.3's Python XML reader drops the axial level of a HexLattice with n_axial="1": the lattice
    comes back with a z pitch but universes [ring][position] and num_axial None, so openmc.Geometry.find()
    treats it as 2D (no z shift), while OpenMC's transport code (checked with openmc.lib.find_cell) keeps it
    3D. Re-nest such lattices so Python agrees with transport. Returns the lattice IDs fixed."""
    fixed = []
    for lat in model.geometry.get_all_lattices().values():
        if isinstance(lat, openmc.HexLattice) and len(lat.pitch) == 2 and not lat.num_axial:
            u = lat.universes
            if len(u) and isinstance(u[0], (list, tuple)) and len(u[0]) and not isinstance(u[0][0], (list, tuple)):
                lat.universes = [[list(ring) for ring in u]]
                fixed.append(lat.id)
    return fixed


def prepare(model):
    """Before MCNPy, change the model (without changing its geometry) where MCNPy 0.0.7 would fail:
    - region-less cells get an explicit everywhere region (MCNPy writes None as text and then can't parse it);
    - 2D RectLattices (infinite along z) become one z layer 2*FAR tall (MCNPy only handles 3D lattices).
    Returns notes."""
    notes = []
    cells = [c for c in model.geometry.get_all_cells().values() if c.region is None]
    if cells:
        sid = max(model.geometry.get_all_surfaces(), default=0) + 1
        big = openmc.Sphere(surface_id=sid, r=FAR)
        for c in cells:
            c.region = -big | +big
        notes.append(f"Cells {sorted(c.id for c in cells)} had no region (they fill everything); "
                     f"written as '-{sid}:{sid}' around a far sphere {sid}.")
    for lat in model.geometry.get_all_lattices().values():
        if isinstance(lat, openmc.RectLattice) and len(lat.pitch) == 2:
            universes = [list(row) for row in lat.universes]
            lat.pitch = (*lat.pitch, 2 * FAR)
            lat.lower_left = (*lat.lower_left, -FAR)
            lat.universes = [universes]
            notes.append(f"Lattice {lat.id} is 2D (infinite along z); written as one z layer {2 * FAR:g} cm tall.")
    return notes


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


@contextlib.contextmanager
def mcnpy_view(model):
    """While MCNPy translates: swap each HexLattice for a placeholder RectLattice with the same ID holding the
    same universes. MCNPy 0.0.7 can't translate hex lattices at all ('NoneType' object is not subscriptable),
    but it still has to translate the universes inside them; rewrite() then writes the real LAT=2 card."""
    swaps = []
    cells = model.geometry.get_all_cells().values()
    for lat in list(model.geometry.get_all_lattices().values()):
        if not isinstance(lat, openmc.HexLattice):
            continue
        us = list({u.id: u for u in lat.get_unique_universes().values()}.values())
        if lat.outer is not None and all(u.id != lat.outer.id for u in us):
            us.append(lat.outer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the placeholder reuses the hex lattice's ID on purpose
            ph = openmc.RectLattice(lattice_id=lat.id, name="placeholder for a hex lattice")
        ph.pitch, ph.lower_left, ph.universes = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), [[us]]
        filled = [c for c in cells if c.fill is lat]
        for c in filled:
            c.fill = ph
        swaps.append((filled, lat))
    try:
        yield
    finally:
        for filled, lat in swaps:
            for c in filled:
                c.fill = lat


def _pad(idx):
    """An OpenMC lattice index as a 3-tuple (2D lattices have no z index)."""
    return tuple(int(v) for v in idx) + (0,) * (3 - len(idx))


def _filled_bounds(lat, filled):
    for c in filled:
        if c.translation is not None or c.rotation is not None:
            raise UnsupportedFeature(f"Cell {c.id} fills lattice {lat.id} with a translation or rotation; not supported yet.")
        yield c, c.region.bounding_box


def _rect_layout(lat, filled):
    """LAT=1 from an openmc.RectLattice. Element [0,0,0] is the lattice's first element (lower_left corner);
    OpenMC's rows are listed top (+y) first, so j = ny - 1 - row."""
    pitch = [float(p) for p in lat.pitch]
    u = lat.universes if len(pitch) == 3 else [lat.universes]  # [z][y, top row first][x]
    nz, ny, nx = len(u), len(u[0]), len(u[0][0])
    dims = [nx, ny, nz]
    pz = pitch[2] if len(pitch) == 3 else 2 * FAR
    # Base element is an RPP macrobody centered on the origin (manual §5.3.4, Listing 5.15, p. 289).
    # Its facets are in MCNP Table 5.2 order: xmax (+x), xmin (-x), ymax (+y), ymin (-y), zmax (+z), zmin (-z),
    # matching the element index directions (manual p. 278, 290).
    planes = [("RPP", (-pitch[0] / 2, pitch[0] / 2, -pitch[1] / 2, pitch[1] / 2, -pz / 2, pz / 2), "-")]
    ll = [float(v) for v in lat.lower_left]
    lo, hi = [0, 0, 0], [nx - 1, ny - 1, nz - 1]
    for c, (blo, bhi) in _filled_bounds(lat, filled):
        for k in range(len(pitch)):
            if dims[k] == 1 and pitch[k] >= 2 * FAR:
                continue  # the one z layer prepare() made from a 2D lattice covers everything
            if not (math.isfinite(blo[k]) and math.isfinite(bhi[k])):
                if lat.outer is None:
                    continue  # unbounded along k with no outer: OpenMC loses particles there too
                raise UnsupportedFeature(f"Cell {c.id} (filled by lattice {lat.id}) is unbounded along {'xyz'[k]}.")
            a = math.floor((blo[k] - ll[k]) / pitch[k] + 1e-9)
            b = math.ceil((bhi[k] - ll[k]) / pitch[k] - 1e-9) - 1
            if (a < 0 or b > dims[k] - 1) and lat.outer is None:
                raise UnsupportedFeature(f"Lattice {lat.id} doesn't cover cell {c.id} and has no outer universe.")
            lo[k], hi[k] = min(lo[k], a), max(hi[k], b)

    def uni(i, j, k):
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            return u[k][ny - 1 - j][i].id
        return lat.outer.id
    ids = [uni(i, j, k) for k in range(lo[2], hi[2] + 1) for j in range(lo[1], hi[1] + 1) for i in range(lo[0], hi[0] + 1)]
    single = lo == [0, 0, 0] and hi == [nx - 1, ny - 1, nz - 1] and len(set(ids)) == 1
    origin = [ll[k] + pitch[k] / 2 for k in range(len(pitch))] + ([0.0] if len(pitch) == 2 else [])
    index = {(i, j, k): (i, j, k) for k in range(nz) for j in range(ny) for i in range(nx)}  # OpenMC's (x, y, z) = MCNP's
    return 1, planes, lo, hi, ids, single, origin, f"pitch {pitch}", index


def _hex_layout(lat, filled):
    """LAT=2 from an openmc.HexLattice. Element [0,0,0] is the lattice's centre element (bottom axial level).
    The index steps are two neighbour directions 60 degrees apart, T1 and T2, so [-1,1,0] (5th surface) sits at
    T2 - T1 as the manual requires (p. 290, 766). Which universe fills element [i,j,k] is asked from OpenMC
    (find_element at the element's centre), so OpenMC's own ring and index conventions don't matter here."""
    p = float(lat.pitch[0])
    three_d = len(lat.pitch) == 2
    pz = float(lat.pitch[1]) if three_d else None
    s3 = math.sqrt(3.0) / 2
    T1, T2 = ((s3 * p, p / 2), (0.0, p)) if lat.orientation == "y" else ((p, 0.0), (p / 2, s3 * p))
    dirs = []
    for v in (T1, T2, (T2[0] - T1[0], T2[1] - T1[1])):
        n = (v[0] / p, v[1] / p)
        dirs += [n, (-n[0], -n[1])]
    # faces n . x = p/2, listed [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [-1,1,0], [1,-1,0], then the bases
    planes = [("P", (n[0], n[1], 0.0, p / 2), "-") for n in dirs]
    if three_d:
        planes += [("PZ", (pz / 2,), "-"), ("PZ", (-pz / 2,), "+")]
    cx, cy = float(lat.center[0]), float(lat.center[1])
    # After a model.xml round trip num_axial is None and a one-level lattice comes back as [ring][position], so
    # count axial levels only when the universes really are nested three deep ([axial][ring][position]).
    u = lat.universes
    nested = len(u) > 0 and isinstance(u[0], (list, tuple)) and len(u[0]) > 0 and isinstance(u[0][0], (list, tuple))
    nz = (lat.num_axial or (len(u) if nested else 1)) if three_d else 1
    z0 = float(lat.center[2]) - (nz - 1) / 2 * pz if three_d else 0.0
    M, klo, khi = lat.num_rings, 0, nz - 1
    for c, (blo, bhi) in _filled_bounds(lat, filled):
        if not all(math.isfinite(v) for v in (blo[0], blo[1], bhi[0], bhi[1])):
            raise UnsupportedFeature(f"Cell {c.id} (filled by hex lattice {lat.id}) is unbounded across the lattice.")
        rmax = max(math.hypot(x - cx, y - cy) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]))
        M = max(M, math.ceil(rmax / (s3 * p)) + 1)
        if three_d and math.isfinite(blo[2]) and math.isfinite(bhi[2]):
            klo = min(klo, math.floor((blo[2] - (z0 - pz / 2)) / pz + 1e-9))
            khi = max(khi, math.ceil((bhi[2] - (z0 - pz / 2)) / pz - 1e-9) - 1)
    lo, hi = [-M, -M, klo], [M, M, khi]
    ids, index = [], {}
    for k in range(klo, khi + 1):
        for j in range(-M, M + 1):
            for i in range(-M, M + 1):
                pt = (cx + i * T1[0] + j * T2[0], cy + i * T1[1] + j * T2[1], z0 + k * pz if three_d else 0.0)
                idx, _ = lat.find_element(pt)
                if lat.is_valid_index(idx):
                    ids.append(lat.get_universe(idx).id)
                    index[_pad(idx)] = (i, j, k)
                elif lat.outer is not None:
                    ids.append(lat.outer.id)
                else:
                    raise UnsupportedFeature(f"Hex lattice {lat.id} doesn't cover its cell and has no outer universe.")
    return (2, planes, lo, hi, ids, len(set(ids)) == 1, [cx, cy, z0],
            f"pitch {list(lat.pitch)}, {lat.orientation} orientation", index)


def rewrite(blocks, model, index_maps=None):
    """Rewrite MCNPy's lattice cards in a deck split into blocks [cells, surfaces, data...], in place.
    Returns notes. If `index_maps` is a dict, it gets {lattice id: {"cell": MCNP LAT cell number,
    "index": {OpenMC element index (x, y, z): MCNP [i, j, k]}}} for tally chains (mcnp_cards.tally_cards)."""
    geometry = model.geometry
    lattices = list(geometry.get_all_lattices().values())  # get_all_universes() leaves lattices out
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
        filled = [c for c in all_cells.values() if c.fill is lat]
        if not filled:
            raise UnsupportedFeature(f"Lattice {L} isn't used by any cell.")
        if isinstance(lat, openmc.RectLattice):
            lat_type, planes, lo, hi, ids, single, origin, desc, index = _rect_layout(lat, filled)
        else:
            lat_type, planes, lo, hi, ids, single, origin, desc, index = _hex_layout(lat, filled)
        if index_maps is not None:
            index_maps[L] = {"cell": num, "index": index}

        region = []
        for kind, params, sense in planes:
            if next_surf > MAX_RS_SURFACE:
                raise UnsupportedFeature(f"Lattice {L}: MCNP needs repeated-structure surfaces numbered <= 999 "
                                         f"(manual p. 271); this deck already reaches {next_surf - 1}.")
            new_surfaces.append(f"{next_surf} {kind} " + " ".join(_fmt(v) for v in params))
            region.append(f"{sense if sense == '-' else ''}{next_surf}")
            next_surf += 1
        fill = f"FILL={ids[0]}" if single else ("FILL=" + " ".join(f"{a}:{b}" for a, b in zip(lo, hi)) + " "
                                                   + " ".join(map(str, ids)))
        cards[elem[0]] = _wrap(f"{num} 0 {' '.join(region)} U={L} LAT={lat_type} {fill} {imp}")

        # the filled cells look at the lattice universe with its origin at the centre of element [0,0,0]
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
        notes.append(f"Lattice {L}: MCNP LAT={lat_type} cell {num}, {desc}, element [0,0,0] centred at "
                     f"{[round(v, 6) for v in origin]}, indices {lo} to {hi}.")

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
