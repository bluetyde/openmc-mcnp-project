"""
Automated MCNP Deck Validator

Checks MCNP decks for runnability and physics completeness beyond MontePy syntax parsing:
1. Mode card: MODE N is present (and P when the model transports photons, with IMP:P on every cell).
2. Run control, matched to the OpenMC run mode:
   - eigenvalue: KCODE plus an initial source (KSRC or SDEF)
   - fixed source: SDEF and NPS, and no KCODE
   Without OpenMC settings to compare against, KCODE is required (the original pin-cell check).
3. Thermal scattering: an MT card for every material with S(a,b) in the OpenMC model.
4. Universe 0: at least one cell in universe 0, and no orphaned non-zero universes.
5. Vacuum boundaries: when the OpenMC geometry has vacuum surfaces, a cell with IMP:N=0 exists.
6. Tallies: one F or FMESH tally per OpenMC tally score, referring to cells that exist.
7. Geometry equivalence (src/geometry_check.py): sampled points land in the same cell,
   material and density in the deck as in the OpenMC model, with no gaps or overlaps.

Usage:
    python src/validate_deck.py pin_cell_runnable.mcnp [materials.xml]
    python src/validate_deck.py deck.mcnp --model path/to/model.xml [--samples 20000]

With no --model, the model is read from geometry.xml/materials.xml/settings.xml(/tallies.xml)
in the current folder when they exist.
"""
import argparse
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

import montepy


def _load_model(model_path):
    from remediate_deck import load_model
    return load_model(model_path)


def _data_lines(problem):
    lines = []
    for card in problem.data_inputs:
        try:
            lines.extend(l.strip().upper() for l in card.format_for_mcnp_input(problem.mcnp_version))
        except Exception:
            lines.append(str(card).strip().upper())
    return lines


def _read_deck(deck_path):
    """montepy.read_input, with each tally chain (1 < 7[0 0 0] < 3) replaced by its bottom cell first:
    MontePy 1.1.3 can't parse chains. The tally checks read the chains from the deck's text."""
    text = open(deck_path).read()
    plain = re.sub(r"\(\s*(\d+)[^()]*<[^()]*\)", r"\1", text)
    if plain == text:
        return montepy.read_input(deck_path)
    fd, tmp = tempfile.mkstemp(suffix=".mcnp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(plain)
        return montepy.read_input(tmp)
    finally:
        os.remove(tmp)


def _cards_by_name(raw_text):
    """{card name: words} for every data card, continuation lines (leading spaces) joined."""
    cards, cur = {}, None
    for l in raw_text.splitlines():
        if l.startswith(" ") and l.strip():
            if cur is not None:
                cards[cur] += l.split()
            continue
        w = l.split()
        cur = w[0] if w else None
        if cur is not None:
            cards[cur] = w[1:]
    return cards


def _close_all(a, b, rel=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= rel * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def _check_sources(raw_text, sources):
    """Read a multi-source SDEF back (ERG=Dn with SI S; other keywords =FERG=Dm with DS L/S, manual p. 379-408)
    and compare each source with the OpenMC source in the same position: strength, particle, shape and
    position, energy distribution and direction. The reading here is written independently of the writer in
    mcnp_cards, so entries listed in the wrong order are caught. Returns a list of errors."""
    import openmc
    cards = _cards_by_name(raw_text)
    sdef = " ".join(cards.get("SDEF", []))
    kw = {m.group(1): m.group(2).strip() for m in re.finditer(r"(\w+)=((?:(?!\s\w+=).)*)", sdef)}
    k = len(sources)
    m = re.fullmatch(r"D(\d+)", kw.get("ERG", ""))
    if not m:
        return [f"{k} sources, but SDEF ERG isn't a distribution (ERG=Dn) that picks the source."]
    sel = m.group(1)
    si, sp = cards.get(f"SI{sel}", []), cards.get(f"SP{sel}", [])
    if not si or si[0] != "S" or len(si) - 1 != k:
        return [f"SDEF ERG=D{sel}: SI{sel} must be 'S' with one energy distribution per source ({k}), found {si}."]
    errors = []
    strengths = [float(s.strength) for s in sources]
    got = [float(v) for v in sp]
    if len(got) != k or not _close_all([g / sum(got) for g in got], [s / sum(strengths) for s in strengths]):
        errors.append(f"SP{sel} {sp} doesn't match the OpenMC source strengths {strengths}.")

    def dist(n):
        n = str(n).lstrip("D")
        si, sp = cards.get(f"SI{n}", []), cards.get(f"SP{n}", [])
        opt = si[0] if si and si[0] in ("H", "L", "A", "S") else ("H" if si else None)
        spopt = sp[0] if sp and sp[0] in ("D", "C") else None
        return (opt, [float(v) for v in (si[1:] if si and si[0] in ("H", "L", "A", "S") else si)],
                [float(v) for v in (sp[1:] if spopt else sp)])

    def per_source(key):
        """For each source: ('value', [numbers]) or ('dist', (SI option, SI values, SP values)); None if absent."""
        v = kw.get(key)
        if v is None:
            return None
        dm = re.fullmatch(r"FERG=D(\d+)", v)
        if not dm:
            return [("value", [float(x) for x in v.split()])] * k
        ds = cards.get(f"DS{dm.group(1)}", [])
        if not ds or ds[0] not in ("L", "S"):
            errors.append(f"SDEF {key}={v}: DS{dm.group(1)} must be L or S.")
            return [None] * k
        vals = ds[1:]
        if len(vals) % k:
            errors.append(f"DS{dm.group(1)} has {len(vals)} entries, not a multiple of the {k} sources.")
            return [None] * k
        g = len(vals) // k
        if ds[0] == "L":
            return [("value", [float(x) for x in vals[i * g:(i + 1) * g]]) for i in range(k)]
        if g != 1:
            errors.append(f"DS{dm.group(1)} S needs one distribution per source.")
            return [None] * k
        return [("dist", dist(vals[i])) for i in range(k)]

    def scalar(entry, what):
        """A dependent entry that must be one value: ('value', [v]) or a one-point discrete distribution."""
        if entry is None:
            return None
        kind, d = entry
        if kind == "value":
            return d[0] if len(d) == 1 else None
        opt, vals, probs = d
        return vals[0] if opt == "L" and len(vals) == 1 else None

    par, pos, rad, ext = per_source("PAR"), per_source("POS"), per_source("RAD"), per_source("EXT")
    axes = [per_source(a) for a in "XYZ"]
    vec, dirs = per_source("VEC"), per_source("DIR")
    shape = "box" if kw.get("X") else ("cylinder" if kw.get("AXS") else "sphere")
    for i, (s, e) in enumerate(zip(sources, si[1:])):
        tag = f"Source {i + 1}"
        want_par = 1 if getattr(s, "particle", "neutron") == "neutron" else 2
        if par is not None and scalar(par[i], "PAR") != want_par:
            errors.append(f"{tag}: MCNP particle {par[i]}, OpenMC {s.particle}.")
        space = s.space
        if isinstance(space, openmc.stats.Point) or space is None:
            p = tuple(space.xyz) if space is not None else (0.0, 0.0, 0.0)
            if shape == "box":
                got = [scalar(a[i], "XYZ") if a else None for a in axes]
                if None in got or not _close_all(got, p):
                    errors.append(f"{tag}: point {p} in OpenMC, but X/Y/Z give {[a[i] if a else None for a in axes]}.")
            else:
                if not (pos and pos[i] and pos[i][0] == "value" and _close_all(pos[i][1], p)):
                    errors.append(f"{tag}: point {p} in OpenMC, but POS gives {pos[i] if pos else None}.")
                for key, entry in (("RAD", rad), ("EXT", ext)):
                    if entry is not None and scalar(entry[i], key) not in (0.0,):
                        errors.append(f"{tag}: a point source needs {key} 0, found {entry[i]}.")
        elif isinstance(space, openmc.stats.Box):
            for a, lo, hi, name in zip(axes, space.lower_left, space.upper_right, "XYZ"):
                d = a[i][1] if a and a[i] and a[i][0] == "dist" else None
                if not (d and d[0] == "H" and _close_all(d[1], [lo, hi]) and _close_all(d[2], [0, 1])):
                    errors.append(f"{tag}: box {name} range {lo}..{hi} in OpenMC, but {name} gives {a[i] if a else None}.")
        else:
            cyl = isinstance(space, openmc.stats.CylindricalIndependent)
            if shape != ("cylinder" if cyl else "sphere"):
                errors.append(f"{tag}: a {'cylinder' if cyl else 'sphere'} source, but the SDEF keywords make a {shape}.")
                continue
            if not (pos and pos[i] and pos[i][0] == "value" and _close_all(pos[i][1], list(space.origin))):
                errors.append(f"{tag}: origin {tuple(space.origin)} in OpenMC, but POS gives {pos[i] if pos else None}.")
            d = rad[i][1] if rad and rad[i] and rad[i][0] == "dist" else None
            if not (d and _close_all(d[1], [space.r.a, space.r.b]) and _close_all(d[2], [-21, 1 if cyl else 2])):
                errors.append(f"{tag}: radius {space.r.a}..{space.r.b} in OpenMC, but RAD gives {rad[i] if rad else None}.")
            if cyl:
                d = ext[i][1] if ext and ext[i] and ext[i][0] == "dist" else None
                if not (d and _close_all(d[1], [space.z.a, space.z.b]) and _close_all(d[2], [-21, 0])):
                    errors.append(f"{tag}: height {space.z.a}..{space.z.b} in OpenMC, but EXT gives {ext[i] if ext else None}.")
        # energy: the i-th distribution on the ERG selector (eV in OpenMC, MeV in MCNP)
        opt, vals, probs = dist(e)
        en = s.energy if s.energy is not None else openmc.stats.Watt(a=0.988e6, b=2.249e-6)
        ok = False
        if isinstance(en, openmc.stats.Discrete):
            p = [float(v) / sum(en.p) for v in en.p]
            ok = opt == "L" and _close_all(vals, [x / 1e6 for x in en.x]) and (
                len(p) == 1 and probs == [1.0] or sum(probs) > 0 and _close_all([v / sum(probs) for v in probs], p))
        elif isinstance(en, openmc.stats.Watt):
            ok = not vals and _close_all(probs, [-3, en.a / 1e6, en.b * 1e6])
        elif isinstance(en, openmc.stats.Maxwell):
            ok = not vals and _close_all(probs, [-2, en.theta / 1e6])
        elif isinstance(en, openmc.stats.Uniform):
            ok = opt == "H" and _close_all(vals, [en.a / 1e6, en.b / 1e6]) and _close_all(probs, [0, 1])
        elif isinstance(en, openmc.stats.Tabular):
            ok = opt == "H" and _close_all(vals, [x / 1e6 for x in en.x])
        if not ok:
            errors.append(f"{tag}: energy {type(en).__name__} in OpenMC doesn't match distribution {e} "
                          f"(SI {opt} {vals}, SP {probs}).")
        # direction
        mono = isinstance(s.angle, openmc.stats.Monodirectional)
        if dirs is None:
            if mono:
                errors.append(f"{tag}: monodirectional in OpenMC, but the SDEF has no DIR.")
        else:
            d = dirs[i][1] if dirs[i] and dirs[i][0] == "dist" else None
            is_mono = bool(d) and d[0] == "L" and d[1] == [1.0]
            is_iso = bool(d) and d[0] == "H" and _close_all(d[1], [-1, 1]) and _close_all(d[2], [0, 1])
            if mono and not (is_mono and vec and vec[i] and _close_all(vec[i][1], list(s.angle.reference_uvw))):
                errors.append(f"{tag}: monodirectional along {tuple(s.angle.reference_uvw)} in OpenMC, but DIR/VEC give "
                              f"{dirs[i]} / {vec[i] if vec else None}.")
            if not mono and not is_iso:
                errors.append(f"{tag}: isotropic in OpenMC, but DIR gives {dirs[i]}.")
    return errors


def _tally_card_texts(raw_text):
    """[(name, bins text)] for every F and FMESH card, continuation lines (leading spaces) joined."""
    cards, cur = [], None
    for l in raw_text.splitlines():
        if l.startswith(" ") and l.strip():
            if cur is not None:
                cur[1].append(l.strip())
            continue
        cur = None
        w = l.split()
        if w and (w[0].startswith("FMESH") or (w[0][:1] == "F" and w[0][1:2].isdigit())):
            cur = (w[0], [" ".join(w[1:])])
            cards.append(cur)
    return [(name, " ".join(parts)) for name, parts in cards]


def _tally_bins(text):
    """Bins of an F card: a plain cell number, or a chain (c < L[i j k] < ... < c0) (manual p. 452-455) as
    [(cell, (i, j, k) or None)], top level first. Words that aren't cells (T, ...) are skipped."""
    bins = []
    for m in re.finditer(r"\(([^)]*)\)|(\S+)", text):
        if m.group(2) is not None:
            if m.group(2).isdigit():
                bins.append([(int(m.group(2)), None)])
            continue
        levels = []
        for part in m.group(1).split("<"):
            lm = re.fullmatch(r"\s*(\d+)\s*(?:\[\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\])?\s*", part)
            if lm is None:
                raise ValueError(f"can't read tally chain level '{part.strip()}'")
            levels.append((int(lm.group(1)), tuple(int(v) for v in lm.group(2, 3, 4)) if lm.group(2) else None))
        bins.append(levels[::-1])
    return bins


def validate_deck(deck_path, materials_path="materials.xml", model=None, geometry_samples=0):
    """Validate a deck. `model` is an openmc.Model (or None to use XML files in the cwd when present)."""
    print(f"--- Validating MCNP Deck: {deck_path} ---")

    if not os.path.exists(deck_path):
        print(f"FAIL: Deck file '{deck_path}' does not exist.")
        return False

    try:
        problem = _read_deck(deck_path)
    except Exception as e:
        print(f"FAIL: MontePy failed to parse deck: {e}")
        return False

    if model is None and all(os.path.exists(f) for f in ("geometry.xml", "settings.xml")) and os.path.exists(materials_path):
        try:
            model = _load_model(os.getcwd())
        except Exception as e:
            print(f"WARNING: Could not load the OpenMC model from this folder ({e}); running deck-only checks.")

    errors, passed = [], []
    lines = _data_lines(problem)
    raw_text = open(deck_path).read().upper()

    def starts(prefix):
        return any(l.startswith(prefix) for l in lines) or any(
            l.strip().startswith(prefix) for l in raw_text.splitlines())

    # 1. MODE
    particles = set()
    if problem.mode and hasattr(problem.mode, "particles"):
        particles |= {str(p).upper() for p in problem.mode.particles}
    for l in lines:
        if l.startswith("MODE"):
            particles |= set(l.split()[1:])
    has_n = bool(particles & {"N", "NEUTRON", "NEUTRONS", "PARTICLE.NEUTRON"})
    has_p = bool(particles & {"P", "PHOTON", "PHOTONS", "PARTICLE.PHOTON"})
    if not has_n:
        errors.append("MODE card missing or 'N' particle not specified.")
    photons = False
    if model is not None:
        from mcnp_cards import uses_photons
        photons = uses_photons(model.settings)
        if photons and not has_p:
            errors.append("The OpenMC model transports photons but MODE has no P.")
        if photons:
            no_imp_p = [c.number for c in problem.cells if c.importance.photon is None]
            if no_imp_p:
                errors.append(f"MODE includes photons but cells {no_imp_p[:10]} have no IMP:P.")

    # 2. run control
    has_kcode, has_ksrc, has_sdef, has_nps = starts("KCODE"), starts("KSRC"), starts("SDEF"), starts("NPS")
    run_mode = model.settings.run_mode if model is not None else None
    if run_mode == "fixed source":
        srcs = model.settings.source
        srcs = list(srcs) if isinstance(srcs, (list, tuple)) else ([srcs] if srcs is not None else [])
        if len(srcs) > 1 and has_sdef:
            src_errors = _check_sources(raw_text, srcs)
            errors += src_errors
            if not src_errors:
                passed.append(f"{len(srcs)} sources read back from the SDEF match OpenMC's (strength, particle, "
                              f"position and shape, energy, direction)")
        if has_kcode:
            errors.append("The OpenMC model is fixed source but the deck has a KCODE card.")
        if not has_sdef:
            errors.append("Fixed-source deck has no SDEF card.")
        if not has_nps:
            errors.append("Fixed-source deck has no NPS card.")
    else:
        if not has_kcode:
            errors.append("KCODE card missing.")
        elif not (has_ksrc or has_sdef):
            errors.append("KCODE is present but no initial source card (KSRC or SDEF) was found.")

    # 3. MT cards
    sab_by_mat = {}
    if model is not None:
        for mat in model.materials:
            names = [n for n, _ in getattr(mat, "_sab", [])]
            if names:
                sab_by_mat[mat.id] = (mat.name or f"Material {mat.id}", names)
    elif os.path.exists(materials_path):
        try:
            for m in ET.parse(materials_path).getroot().findall("material"):
                names = [s.get("name") for s in m.findall("sab") if s.get("name")]
                if names:
                    sab_by_mat[int(m.get("id"))] = (m.get("name", f"Material {m.get('id')}"), names)
        except Exception as e:
            print(f"WARNING: Could not parse {materials_path} for MT cross-validation: {e}")
    for mat_id, (mat_name, _names) in sab_by_mat.items():
        found = (mat_id in problem.materials and problem.materials[mat_id].thermal_scattering) or any(
            l.startswith(f"MT{mat_id} ") or l.startswith(f"MT {mat_id} ") for l in lines)
        if not found:
            errors.append(f"Material {mat_id} ({mat_name}) has S(a,b) thermal scattering in the OpenMC model, "
                          f"but no 'MT{mat_id}' card exists in the MCNP deck.")

    # 4. universe 0
    cells_in_u0, non_zero, filled = 0, set(), set()
    for cell in problem.cells:
        if cell.universe is None or cell.universe.number == 0:
            cells_in_u0 += 1
        else:
            non_zero.add(abs(cell.universe.number))
        fill = getattr(cell, "fill", None)
        if fill is not None:  # a single universe, or a lattice FILL array (any level)
            if getattr(fill, "multiple_universes", False) and getattr(fill, "universes", None) is not None:
                filled.update(abs(u.number) for u in fill.universes.ravel() if u is not None)
            elif getattr(fill, "universe", None) is not None and hasattr(fill.universe, "number"):
                filled.add(abs(fill.universe.number))
    if cells_in_u0 == 0:
        errors.append("Universe 0 hierarchy error: All cells are tagged with non-zero Universe IDs, "
                      "and no cell resides in Universe 0. MCNP will fail with 'no cells in universe 0'.")
    elif non_zero and not non_zero.issubset(filled):
        errors.append(f"Orphaned non-zero Universes detected {non_zero - filled}: no FILL (or lattice FILL array) uses them.")

    if model is not None:
        import openmc

        # 5. vacuum boundary -> graveyard
        vac = [s.id for s in model.geometry.get_all_surfaces().values() if s.boundary_type == "vacuum"]
        if vac and not any(c.importance.neutron == 0 for c in problem.cells):
            errors.append(f"The OpenMC geometry has vacuum boundaries (surfaces {vac}) but no MCNP cell has IMP:N=0; "
                          f"particles leaving the model would be lost.")

        # 6. tallies; chain bins (cells inside lattices) are checked against OpenMC's instances in step 7
        from mcnp_cards import current_bins
        expected = 0
        for t in model.tallies or []:
            if any(isinstance(f, openmc.SurfaceFilter) for f in t.filters):
                expected += len(current_bins(t, model.geometry)[0])  # one F1 per (surface, cell) bin that can score
            else:
                expected += len(t.scores)
        found_tallies = _tally_card_texts(raw_text)
        if len(found_tallies) != expected:
            errors.append(f"Expected {expected} tallies (one per OpenMC tally score) but found {len(found_tallies)}.")
        cells_by_num = {c.number: c for c in problem.cells}
        card_bins = []
        for head, rest in found_tallies:
            bins = None
            if head.startswith("F") and head.split(":")[0][-1:] == "1" and ":" in head:  # F1: surfaces, not cells
                card_bins.append((head, None))
                continue
            if not head.startswith("FMESH") and ":" in head:
                try:
                    bins = _tally_bins(rest)
                except ValueError as e:
                    errors.append(f"{head}: {e}.")
                    bins = []
                bad = [str(c) for c in sorted({c for b in bins for c, _ in b if c not in cells_by_num})]
                if bad:
                    errors.append(f"{head} refers to cells {bad} that aren't in the deck.")
                not_lat = sorted({c for b in bins for c, i in b if i is not None and c in cells_by_num
                                  and getattr(cells_by_num[c], "lattice_type", None) is None})
                if not_lat:
                    errors.append(f"{head} gives lattice indices for cells {not_lat}, which aren't lattice cells.")
            card_bins.append((head, bins))
        # current tallies: F1 on an OpenMC surface, C n 0 1, and the FC tag saying where OpenMC's value is
        faces = []
        surf_numbers = {s.number for s in problem.surfaces}
        current_pairs = set()
        for t in model.tallies or []:
            if any(isinstance(f, openmc.SurfaceFilter) for f in t.filters):
                current_pairs |= {(s.id if s is not None else None, c.id if c is not None else None)
                                  for s, c in current_bins(t, model.geometry)[0]}
        for head, rest in found_tallies:
            name = head.split(":")[0]
            if not (name[1:].isdigit() and name.endswith("1")):
                continue
            n = name[1:]
            words = rest.split()
            f1 = int(words[0]) if len(words) == 1 and words[0].isdigit() else None
            if f1 is None or f1 not in surf_numbers:
                errors.append(f"{head} must name one surface of the deck (found '{rest}').")
                continue
            fc = re.search(rf"^FC{n}\s.*\[S (\d+) (?:C (\d+) SEG (\d+)(?:-(\d+))? COS ([12]) X([+-]1)|NET)\]\s*$", raw_text, re.M)
            if fc is None:
                errors.append(f"{head}: no FC{n} card saying which bin holds OpenMC's value ([S s C c SEG a-b COS n X+1] or [S s NET]).")
                continue
            if not re.search(rf"^C{n}\s+0\s+1\s*$", raw_text, re.M):
                errors.append(f"{head}: needs the cosine card 'C{n} 0 1' to separate the two directions.")
            sid, cid = int(fc.group(1)), int(fc.group(2)) if fc.group(2) else None
            if (sid, cid) not in current_pairs:
                errors.append(f"{head}: no OpenMC current tally has the bin (surface {sid}, cell {cid}).")
                continue
            fsm = re.search(rf"^FS{n}\s+((?:.*\n?)(?:^\s+.*\n?)*)", raw_text, re.M)
            fs = [(abs(int(w)), -1 if w.startswith("-") else 1) for w in (fsm.group(1).split() if fsm else [])]
            if cid is None:
                faces.append((f"{head} (net current on surface {sid})", f1, sid, None, [], [], 0, 0))
            else:
                a = int(fc.group(3))
                segs = list(range(a, int(fc.group(4) or a) + 1))
                faces.append((f"{head} (surface {sid} leaving cell {cid})", f1, sid, cid, fs, segs,
                              int(fc.group(5)), int(fc.group(6))))

        chains = []
        if len(found_tallies) == expected:
            cards_iter = iter(card_bins)
            all_cells, pathed = model.geometry.get_all_cells(), False
            for t in model.tallies or []:
                inst = next((f for f in t.filters if isinstance(f, openmc.CellInstanceFilter)), None)
                current = any(isinstance(f, openmc.SurfaceFilter) for f in t.filters)
                for _card in range(len(current_bins(t, model.geometry)[0]) if current else len(t.scores)):
                    head, bins = next(cards_iter)
                    if inst is None or bins is None:
                        continue
                    if len(bins) != len(inst.bins):
                        errors.append(f"{head} has {len(bins)} bins but OpenMC tally '{t.name}' has {len(inst.bins)} cell instances.")
                        continue
                    if not pathed:
                        model.geometry.determine_paths()
                        pathed = True
                    for (cid, i), b in zip(inst.bins, bins):
                        cid, i = int(cid), int(i)
                        if b[-1][0] != cid:
                            errors.append(f"{head}: the bin for OpenMC cell {cid} (instance {i}) ends in MCNP cell {b[-1][0]}.")
                        elif cid in all_cells and 0 <= i < len(all_cells[cid].paths):
                            chains.append((f"{head} bin for cell {cid} instance {i}", tuple(b), all_cells[cid].paths[i]))

        # 7. geometry equivalence
        if geometry_samples:
            from geometry_check import check_geometry
            seen, uniq = set(), []
            for label, ch, op in chains:  # one entry per bin, even when several scores repeat the card
                if (ch, op) not in seen:
                    seen.add((ch, op))
                    uniq.append((label, ch, op))
            g = check_geometry(problem, model.geometry, n_samples=geometry_samples, chains=uniq or None)
            if g["reason"]:
                errors.append(f"Geometry check could not run: {g['reason']}.")
            elif not g["ok"]:
                errors.extend(f"Geometry: {e}" for e in g["errors"])
            else:
                passed.append(f"geometry matches OpenMC at {g['checked_points']} sampled points "
                              f"({g['skipped_near_surface']} on-surface points skipped), materials and densities match")
                if uniq:
                    passed.append(f"{len(uniq) - len(g['unhit_chains'])} of {len(uniq)} lattice tally bins hold the same "
                                  f"points as their OpenMC cell instances")
                for label in g["unhit_chains"]:
                    print(f"WARNING: {label}: no sampled point fell in it, so it wasn't compared with OpenMC.")
        elif chains:
            print("WARNING: lattice tally bins weren't compared with OpenMC (run with geometry samples to check them).")
        if geometry_samples and faces:
            from geometry_check import check_current_faces
            cf = check_current_faces(problem, model.geometry, faces)
            if cf["reason"]:
                errors.append(f"Current tally check could not run: {cf['reason']}.")
            elif cf["errors"]:
                errors.extend(f"Current tally: {e}" for e in cf["errors"])
            else:
                passed.append(f"{len(faces)} current tallies: F1 surface, FS face and direction match OpenMC at "
                              f"{cf['checked']} points on the surfaces")

    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  [ERROR] {err}")
        return False
    checks = "MODE, " + ("SDEF/NPS" if run_mode == "fixed source" else "KCODE/KSRC") + ", MT thermal scattering, Universe 0"
    if model is not None:
        checks += ", vacuum boundary, tallies"
    print(f"Validation PASSED: {checks}.")
    for p in passed:
        print(f"  [OK] {p}")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate an MCNP deck against its OpenMC model.")
    ap.add_argument("deck", nargs="?", default="pin_cell_runnable.mcnp")
    ap.add_argument("materials", nargs="?", default="materials.xml")
    ap.add_argument("--model", help="model.xml, or a folder with model.xml or geometry/materials/settings XML")
    ap.add_argument("--samples", type=int, default=20000, help="geometry check sample points (0 to skip)")
    args = ap.parse_args(argv)
    model = _load_model(args.model) if args.model else None
    if model is None and all(os.path.exists(f) for f in ("geometry.xml", "settings.xml", args.materials)):
        model = _load_model(os.getcwd())
    return validate_deck(args.deck, args.materials, model=model, geometry_samples=args.samples if model else 0)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
