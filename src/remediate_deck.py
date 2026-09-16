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
6. Tallies: F4/E4/FM/SD for cell tallies, FMESH for regular-mesh tallies.

Generated cards come from src/mcnp_cards.py (derived from the OpenMC objects), the
result is parsed and written by MontePy, and src/validate_deck.py checks it.
"""
import os
import tempfile

import montepy
import openmc
from montepy.universe import Universe

import mcnp_cards
from mcnp_cards import UnsupportedFeature


def load_model(path):
    """Load an OpenMC model from model.xml, or from a folder holding model.xml or the separate XML files."""
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


def remediate(source_deck, model, out_deck, sab_map=None):
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
        for name, _fraction in getattr(mat, "_sab", []):
            if name not in sab:
                raise UnsupportedFeature(
                    f"Material {mat.id} ({mat.name}) uses S(a,b) table '{name}', which has no MCNP identifier "
                    f"in SAB_MCNP_MAP. Add one (check your xsdir) or pass --sab {name}=<id>.")
            cards.append(f"MT{mat.id} {sab[name]}")

    # 4-6. source, mode, run control, tallies (order matches the original pin-cell remediation)
    if eigen:
        ksrc, kcode = mcnp_cards.eigenvalue_cards(settings)
        cards += [ksrc, mcnp_cards.mode_card(settings), kcode]
    else:
        cards.append(mcnp_cards.mode_card(settings))
        cards += mcnp_cards.fixed_source_cards(settings)
    t_cards, t_notes = mcnp_cards.tally_cards(model.tallies, geometry)
    cards += t_cards
    report["notes"] += t_notes
    report["added"] += [c.split("\n")[0] for c in cards]

    augmented = "\n\n".join(blocks).strip() + "\n" + "\n".join(cards) + "\n"

    fd, tmp = tempfile.mkstemp(suffix=".mcnp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(augmented)
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
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    report["graveyard_cell"] = graveyard
    report["run_mode"] = settings.run_mode
    return report
