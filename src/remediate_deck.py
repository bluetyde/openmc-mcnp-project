"""
Remediate an MCNPy-translated deck into a runnable one, for any OpenMC model.

MCNPy 0.0.7 gaps fixed here (see CLAUDE.md "Known quirks"):
1. Universe 0: MCNPy tags root-universe cells with `U <root id>` and never creates
   universe 0. Those tags are stripped.
2. Vacuum boundaries: MCNP has no vacuum surface type. When the OpenMC geometry has
   vacuum boundaries, an outside ("graveyard") cell with IMP:N=0 is added covering
   everything not in a root-universe cell.
3. Thermal scattering: MT cards from the materials' S(a,b) tables (SAB_MCNP_MAP).
   Unknown table names fail loudly instead of guessing an identifier.
4. Run control and source: eigenvalue -> KSRC (from the OpenMC source) + KCODE;
   fixed source -> SDEF (+ SI/SP) + NPS.
5. MODE N, or MODE N P with IMP:P on every cell when photons are transported.
6. Tallies: F4/E4/FM/SD for cell tallies, FMESH for regular and cylindrical mesh tallies.
7. Lattices: MCNPy's LAT/FILL cards are rewritten from the OpenMC RectLattices (src/lattice_cards.py).

Generated cards come from src/mcnp_cards.py (derived from the OpenMC objects), the
result is parsed and written by MontePy, and src/validate_deck.py checks it.
"""
import os
import re
import tempfile

import montepy
import openmc
from montepy.universe import Universe

import lattice_cards
import macrobody_cards
import mcnp_cards
from mcnp_cards import UnsupportedFeature
from deck_format import format_deck


ASCII_SUBS = {"\u00d7": "x", "\u00b0": " deg", "\u00b5": "u", "\u03bc": "u", "\u2013": "-", "\u2014": "-",
               "\u2212": "-", "\u00b2": "2", "\u00b3": "3"}


def _ascii(text):
    """MCNP input is ASCII (p. 24): common symbols become their plain spelling, anything else '?'. Studio part
    names carry things like 12x11 written with a multiplication sign, which must not reach the deck."""
    text = "".join(ASCII_SUBS.get(ch, ch) for ch in str(text))
    return text.encode("ascii", "replace").decode("ascii")


def load_model(path):
    """Load an OpenMC model from model.xml, or from a folder holding model.xml or the separate XML files.
    Hex lattices with one axial level are repaired after reading (lattice_cards.fix_loaded_hex_lattices)."""
    model = _load_model(path)
    lattice_cards.fix_loaded_hex_lattices(model)
    return model


def _load_model(path):
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "model.xml")):
            return openmc.Model.from_model_xml(os.path.join(path, "model.xml"))
        geo = os.path.join(path, "geometry.xml")
        mat = os.path.join(path, "materials.xml")
        st = os.path.join(path, "settings.xml")
        tal = os.path.join(path, "tallies.xml")
        for p in (geo, mat, st):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found (and no model.xml in {path}).")
        # from_xml reads tallies.xml only if that path exists
        return openmc.Model.from_xml(geometry=geo, materials=mat, settings=st, tallies=tal)
    if os.path.basename(path).endswith(".xml"):
        return openmc.Model.from_model_xml(path)
    raise FileNotFoundError(f"Don't know how to load a model from {path}.")


def vacuum_surfaces(geometry):
    return sorted(s.id for s in geometry.get_all_surfaces().values() if s.boundary_type == "vacuum")


def _root_cell_ids(geometry):
    return sorted(geometry.root_universe.cells)


def _graveyard_card(number, cell_ids):
    comps = [f"#{c}" for c in cell_ids]
    lines = [f"{number} 0"]
    for i in range(0, len(comps), 10):
        chunk = " ".join(comps[i:i + 10])
        if i == 0:
            lines[0] += " " + chunk
        else:
            lines.append("     " + chunk)
    lines[-1] += " IMP:N=0"  # photon importance is added through MontePy with the other cells
    return "\n".join(lines)


def remediate(source_deck, model, out_deck, sab_map=None, detector_responses=None, simplify_macrobodies=True):
    """Write a runnable deck to out_deck. Returns a report dict of what was added."""
    sab = dict(mcnp_cards.SAB_MCNP_MAP)
    sab.update(sab_map or {})
    geometry, materials, settings = model.geometry, model.materials, model.settings
    eigen = settings.run_mode == "eigenvalue"
    photons = mcnp_cards.uses_photons(settings)
    report = {"added": [], "notes": []}

    base_text = open(source_deck, "r").read().strip()
    blocks = base_text.split("\n\n")
    if len(blocks) < 3:
        raise ValueError(f"{source_deck} doesn't have cell, surface and data blocks separated by blank lines.")

    # 1b. lattices: MCNPy's LAT/FILL cards are rewritten from the OpenMC lattices (src/lattice_cards.py)
    lattice_maps = {}  # lattice id -> MCNP LAT cell and index map, for tally chains
    report["notes"] += lattice_cards.rewrite(blocks, model, lattice_maps)

    # 1c. macrobodies: simplify standalone primitives (box -> RPP, cylinder -> RCC)
    if simplify_macrobodies:
        report["notes"] += macrobody_cards.simplify_cells(blocks, model)

    # 2. graveyard cell for vacuum boundaries (inserted at the end of the cell block)
    vac = vacuum_surfaces(geometry)
    graveyard = None
    if vac:
        root_ids = _root_cell_ids(geometry)
        graveyard = max(999, max(geometry.get_all_cells()) + 1)
        blocks[0] = blocks[0].rstrip() + "\n" + _graveyard_card(graveyard, root_ids)
        report["added"].append(f"cell {graveyard}: outside region (vacuum boundary on surfaces {vac}), IMP:N=0")

    # 3. MT cards
    cards = []
    for mat in materials:
        sab_ids = []
        for name, _fraction in getattr(mat, "_sab", []):
            if name not in sab:
                raise UnsupportedFeature(
                    f"Material {mat.id} ({mat.name}) uses S(a,b) table '{name}', which has no MCNP identifier "
                    f"in SAB_MCNP_MAP. Add one (check your xsdir) or pass --sab {name}=<id>.")
            sab_ids.append(sab[name])
        if sab_ids:
            cards.append(f"MT{mat.id} {' '.join(sab_ids)}")

    # 4-6. source, mode, run control, tallies (order matches the original pin-cell remediation)
    if eigen:
        ksrc, kcode = mcnp_cards.eigenvalue_cards(settings)
        cards += [ksrc, mcnp_cards.mode_card(settings), kcode]
    else:
        cards.append(mcnp_cards.mode_card(settings))
        cards += mcnp_cards.fixed_source_cards(settings)
    t_cards, t_notes = mcnp_cards.tally_cards(model.tallies, geometry, model.materials, detector_responses,
                                              lattice_maps)
    report["notes"] += t_notes
    report["added"] += [c.split("\n")[0] for c in cards + t_cards]

    # tallies go on after MontePy has written the deck: MontePy 1.1.3 can't parse tally chains
    # (1 < 7[0 0 0] < 3), and it has nothing to change in the tally cards
    augmented = "\n\n".join(blocks).strip() + "\n" + "\n".join(cards) + "\n"

    fd, tmp = tempfile.mkstemp(suffix=".mcnp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(format_deck(augmented))
        problem = montepy.read_input(tmp)

        # 1. universe 0
        root = geometry.root_universe.id
        u0 = Universe(0)
        stripped = 0
        for cell in problem.cells:
            if cell.universe and cell.universe.number == root:
                cell.universe = u0
                stripped += 1
        if stripped:
            report["added"].append(f"universe 0: removed 'U {root}' from {stripped} cells")

        # 5. photon importances mirror neutron importances. MontePy 1.1.3 writes a duplicate
        # IMP:N if photon is set directly on a cell parsed with IMP:N only, so the neutron
        # importance is deleted and both are set again (written as IMP:n,p=...).
        if photons:
            for cell in problem.cells:
                value = cell.importance.neutron
                del cell.importance.neutron
                cell.importance.neutron = value
                cell.importance.photon = value
            report["added"].append("IMP:P on every cell (same as IMP:N)")

        problem.write_to_file(out_deck, overwrite=True)
        with open(out_deck) as f:
            deck_text = f.read()
        if t_cards:
            deck_text = deck_text.rstrip("\n") + "\n" + "\n".join(t_cards) + "\n"
        deck_text = decorate_deck(deck_text, model, graveyard_id=graveyard)
        with open(out_deck, "w") as f:
            f.write(format_deck(deck_text))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    report["graveyard_cell"] = graveyard
    report["run_mode"] = settings.run_mode
    return report


def decorate_deck(text, model, graveyard_id=None):
    """Add pedagogical comment cards and clear section dividers to an MCNP deck.

    Provides students and researchers with clear explanations of card syntax, cell names,
    material compositions, densities, and tally definitions without altering transport physics.
    """
    blocks = text.split("\n\n")
    if len(blocks) < 3:
        return text

    model_cells = model.geometry.get_all_cells() if model and getattr(model, "geometry", None) else {}
    model_surfs = model.geometry.get_all_surfaces() if model and getattr(model, "geometry", None) else {}
    model_mats = {m.id: m for m in (model.materials if model and getattr(model, "materials", None) else [])}

    # 1. Block 0: Title & Cells
    cell_lines = blocks[0].splitlines()
    title = cell_lines[0] if cell_lines else "OpenMC Studio MCNP 6.3 Input Deck"
    new_b0 = [title]
    new_b0.extend([
        "c ===================================================================",
        "c BLOCK 1: CELL CARDS",
        "c Format: <cell_id> <mat_id> <density> <surfaces> <parameters>",
        "c   - Material 0 = void (vacuum)",
        "c   - Negative density = mass density in g/cm3; positive = at/b-cm",
        "c   - Positive surface = outside (+); negative surface = inside (-)",
        "c ===================================================================",
    ])

    for line in cell_lines[1:]:
        sline = line.strip()
        if not sline or sline.startswith("c") or sline.startswith("$"):
            new_b0.append(line)
            continue
        parts = sline.split()
        if parts and parts[0].isdigit():
            cid = int(parts[0])
            if cid == graveyard_id or cid == 999:
                new_b0.append("c --- Cell 999: Outside World (vacuum boundary, particles terminated) ---")
            else:
                c = model_cells.get(cid)
                if c:
                    name = _ascii(getattr(c, "name", "") or f"Cell {cid}")
                    fill = getattr(c, "fill", None)
                    if fill is None:
                        fill_desc = "void"
                    elif isinstance(fill, openmc.Material):  # a lattice or universe has a name too, so test the type
                        rho = getattr(fill, "density", None)
                        rho_str = f", rho = -{rho:.4g} g/cm3" if rho else ""
                        fill_desc = f"Material {fill.id}: {_ascii(fill.name)}{rho_str}"
                    elif isinstance(fill, openmc.Lattice):
                        fill_desc = f"filled by lattice {fill.id}: {_ascii(fill.name)}" if fill.name else f"filled by lattice {fill.id}"
                    elif hasattr(fill, "id"):
                        fill_desc = f"filled by universe {fill.id}"
                    else:
                        fill_desc = str(fill)
                    new_b0.append(f"c --- Cell {cid}: {name} ({fill_desc}) ---")
        new_b0.append(line)

    # 2. Block 1: Surfaces
    surf_lines = blocks[1].splitlines()
    new_b1 = [
        "c ===================================================================",
        "c BLOCK 2: SURFACE CARDS",
        "c Format: <surf_id> <mnemonic> <parameters in cm>",
        "c ===================================================================",
    ]
    for line in surf_lines:
        sline = line.strip()
        if not sline or sline.startswith("c") or sline.startswith("$"):
            new_b1.append(line)
            continue
        parts = sline.split()
        if parts and parts[0].isdigit():
            sid = int(parts[0])
            s = model_surfs.get(sid)
            if s:
                stype = type(s).__name__
                sname = _ascii(getattr(s, "name", ""))
                sname_str = f" {sname}" if sname else ""
                if len(parts) > 1 and parts[1].upper() in ("RPP", "RCC", "SPH", "BOX"):
                    stype = f"{parts[1].upper()} Macrobody"
                new_b1.append(f"c --- Surface {sid}:{sname_str} ({stype}) ---")
        new_b1.append(line)

    # 3. Block 2: Data Cards
    data_lines = ("\n\n".join(blocks[2:])).splitlines()
    new_b2 = [
        "c ===================================================================",
        "c BLOCK 3: DATA CARDS (Materials, Physics, Source, Tallies)",
        "c ===================================================================",
    ]

    in_materials = False
    in_source = False
    in_tallies = False
    for line in data_lines:
        sline = line.strip()
        if not sline or sline.startswith("c") or sline.startswith("$"):
            new_b2.append(line)
            continue
        parts = sline.split()
        head = parts[0].upper()

        m_mat = re.match(r"^M(\d+)$", head)
        if m_mat:
            mid = int(m_mat.group(1))
            mat = model_mats.get(mid)
            mat_name = _ascii(getattr(mat, "name", "")) if mat else ""
            rho = getattr(mat, "density", None) if mat else None
            rho_str = f", rho = {rho:.4g} g/cm3" if rho else ""
            if not in_materials:
                new_b2.extend([
                    "c -------------------------------------------------------------------",
                    "c Materials & Thermal Scattering S(alpha, beta)",
                    "c Format: M<id> <zaid> <fraction> (negative = wt%, positive = at%)",
                    "c -------------------------------------------------------------------",
                ])
                in_materials = True
            new_b2.append(f"c --- Material {mid}: {mat_name}{rho_str} ---")
            new_b2.append(line)
            continue

        m_mt = re.match(r"^MT(\d+)$", head)
        if m_mt:
            mid = int(m_mt.group(1))
            new_b2.append(f"c MT{mid}: Thermal neutron scattering S(alpha, beta) for Material {mid}")
            new_b2.append(line)
            continue

        if head == "MODE":
            new_b2.extend([
                "c -------------------------------------------------------------------",
                "c Particle Transport Mode",
                "c -------------------------------------------------------------------",
            ])
            new_b2.append(line)
            continue

        if head in ("SDEF", "KCODE", "KSRC") and not in_source:
            new_b2.extend([
                "c -------------------------------------------------------------------",
                "c Source Definition & Run Control",
                "c -------------------------------------------------------------------",
            ])
            in_source = True
            if head == "KCODE":
                new_b2.append("c KCODE: particles/batch, initial keff guess, inactive batches, total batches")
            elif head == "KSRC":
                new_b2.append("c KSRC: Initial fission source spatial guess point(s)")
            elif head == "SDEF":
                new_b2.append("c SDEF: General particle source distribution")
            new_b2.append(line)
            continue

        if head == "NPS":
            new_b2.append("c NPS: Total particle histories to simulate")
            new_b2.append(line)
            continue

        if (head.startswith("FMESH") or (head.startswith("F") and len(head) > 1 and head[1].isdigit())) and not in_tallies:
            new_b2.extend([
                "c -------------------------------------------------------------------",
                "c Tallies",
                "c Format: F<n>:<p> <cells/surfaces> (e.g., F4: volume flux, F1: surface current)",
                "c -------------------------------------------------------------------",
            ])
            in_tallies = True
            new_b2.append(line)
            continue

        new_b2.append(line)

    return "\n\n".join(["\n".join(new_b0), "\n".join(new_b1), "\n".join(new_b2)]) + "\n"
