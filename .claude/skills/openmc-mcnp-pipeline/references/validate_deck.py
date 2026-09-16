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
import sys
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


def validate_deck(deck_path, materials_path="materials.xml", model=None, geometry_samples=0):
    """Validate a deck. `model` is an openmc.Model (or None to use XML files in the cwd when present)."""
    print(f"--- Validating MCNP Deck: {deck_path} ---")

    if not os.path.exists(deck_path):
        print(f"FAIL: Deck file '{deck_path}' does not exist.")
        return False

    try:
        problem = montepy.read_input(deck_path)
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
            non_zero.add(cell.universe.number)
        fill = getattr(cell, "fill", None)
        if fill:
            if hasattr(fill, "number"):
                filled.add(fill.number)
            elif getattr(fill, "universe", None) is not None and hasattr(fill.universe, "number"):
                filled.add(fill.universe.number)
    if cells_in_u0 == 0:
        errors.append("Universe 0 hierarchy error: All cells are tagged with non-zero Universe IDs, "
                      "and no cell resides in Universe 0. MCNP will fail with 'no cells in universe 0'.")
    elif non_zero and not non_zero.issubset(filled):
        errors.append(f"Orphaned non-zero Universes detected {non_zero - filled} without corresponding FILL cards in Universe 0.")

    if model is not None:
        # 5. vacuum boundary -> graveyard
        vac = [s.id for s in model.geometry.get_all_surfaces().values() if s.boundary_type == "vacuum"]
        if vac and not any(c.importance.neutron == 0 for c in problem.cells):
            errors.append(f"The OpenMC geometry has vacuum boundaries (surfaces {vac}) but no MCNP cell has IMP:N=0; "
                          f"particles leaving the model would be lost.")

        # 6. tallies
        expected = sum(len(t.scores) for t in (model.tallies or []))
        found_tallies = [l for l in raw_text.splitlines()
                         if l.strip() and not l.startswith(" ")
                         and (l.split()[0].startswith("FMESH") or (l.split()[0][:1] == "F" and l.split()[0][1:2].isdigit()))]
        if len(found_tallies) != expected:
            errors.append(f"Expected {expected} tallies (one per OpenMC tally score) but found {len(found_tallies)}.")
        cell_numbers = {c.number for c in problem.cells}
        for l in found_tallies:
            head, *rest = l.split()
            if head.startswith("F") and not head.startswith("FMESH") and ":" in head:
                bad = [w for w in rest if w.isdigit() and int(w) not in cell_numbers]
                if bad:
                    errors.append(f"{head} refers to cells {bad} that aren't in the deck.")

        # 7. geometry equivalence
        if geometry_samples:
            from geometry_check import check_geometry
            g = check_geometry(problem, model.geometry, n_samples=geometry_samples)
            if g["reason"]:
                errors.append(f"Geometry check could not run: {g['reason']}.")
            elif not g["ok"]:
                errors.extend(f"Geometry: {e}" for e in g["errors"])
            else:
                passed.append(f"geometry matches OpenMC at {g['checked_points']} sampled points "
                              f"({g['skipped_near_surface']} on-surface points skipped), materials and densities match")

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
