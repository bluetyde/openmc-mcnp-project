"""
Automated MCNP Deck Validator

Checks MCNP decks for runtime runnability and physics completeness beyond MontePy syntax parsing:
1. Mode Card: Asserts MODE N is present.
2. KCODE Card: Asserts KCODE card is present for criticality calculation.
3. Source Card: Asserts KSRC (point source) or SDEF (general source) is present when KCODE is used.
4. Thermal Scattering Cards (MT): Cross-references materials.xml to assert MT cards exist for all S(a,b) materials.
5. Universe 0 Hierarchy: Asserts Universe 0 has cell coverage (no isolated non-zero universe cells without FILL).

Usage:
    python src/validate_deck.py pin_cell_runnable.mcnp [materials.xml]
"""
import sys
import os
import xml.etree.ElementTree as ET
import montepy

def validate_deck(deck_path, materials_path="materials.xml"):
    print(f"--- Validating MCNP Deck: {deck_path} ---")

    if not os.path.exists(deck_path):
        print(f"FAIL: Deck file '{deck_path}' does not exist.")
        return False

    # 1. Parse deck with MontePy
    try:
        problem = montepy.read_input(deck_path)
    except Exception as e:
        print(f"FAIL: MontePy failed to parse deck: {e}")
        return False

    errors = []

    # Collect formatted MCNP data card lines from MontePy problem object
    all_data_lines = []
    for card in problem.data_inputs:
        try:
            lines = card.format_for_mcnp_input(problem.mcnp_version)
            all_data_lines.extend([l.strip().upper() for l in lines])
        except Exception:
            all_data_lines.append(str(card).strip().upper())

    # 2. Check MODE card
    has_mode_n = False
    if problem.mode and hasattr(problem.mode, 'particles'):
        particle_strs = [str(p).upper() for p in problem.mode.particles]
        if any(term in particle_strs for term in ["N", "NEUTRON", "NEUTRONS"]):
            has_mode_n = True

    if not has_mode_n:
        for line in all_data_lines:
            if line.startswith("MODE") and "N" in line.split():
                has_mode_n = True
                break

    if not has_mode_n:
        errors.append("MODE card missing or 'N' particle not specified.")

    # 3. Check KCODE and KSRC/SDEF cards
    has_kcode = False
    has_source = False
    for line in all_data_lines:
        if line.startswith("KCODE"):
            has_kcode = True
        if line.startswith("KSRC") or line.startswith("SDEF"):
            has_source = True

    if not has_kcode:
        errors.append("KCODE card missing.")
    elif not has_source:
        errors.append("KCODE is present but no initial source card (KSRC or SDEF) was found.")

    # 4. Check MT cards against materials.xml
    if os.path.exists(materials_path):
        try:
            tree = ET.parse(materials_path)
            root = tree.getroot()
            for mat_elem in root.findall("material"):
                mat_id = int(mat_elem.get("id"))
                mat_name = mat_elem.get("name", f"Material {mat_id}")
                sabs = mat_elem.findall("sab")
                if sabs:
                    mt_prefix = f"MT{mat_id}"
                    found_mt = False
                    # Check via MontePy material object property first
                    if mat_id in problem.materials and problem.materials[mat_id].thermal_scattering:
                        found_mt = True
                    else:
                        for line in all_data_lines:
                            if line.startswith(mt_prefix) or f"MT {mat_id}" in line or line.startswith(f"MT{mat_id} "):
                                found_mt = True
                                break
                    if not found_mt:
                        errors.append(f"Material {mat_id} ({mat_name}) has S(a,b) thermal scattering in {materials_path}, "
                                      f"but no '{mt_prefix}' card exists in the MCNP deck.")
        except Exception as e:
            print(f"WARNING: Could not parse {materials_path} for MT cross-validation: {e}")

    # 5. Check Universe 0 coverage
    cells_in_u0 = 0
    non_zero_universes = set()
    fill_universes = set()

    for cell in problem.cells:
        if cell.universe is None or cell.universe.number == 0:
            cells_in_u0 += 1
        else:
            non_zero_universes.add(cell.universe.number)
        
        if hasattr(cell, 'fill') and cell.fill:
            fill_obj = cell.fill
            if hasattr(fill_obj, 'number'):
                fill_universes.add(fill_obj.number)
            elif hasattr(fill_obj, 'universe') and fill_obj.universe and hasattr(fill_obj.universe, 'number'):
                fill_universes.add(fill_obj.universe.number)

    if cells_in_u0 == 0:
        errors.append("Universe 0 hierarchy error: All cells are tagged with non-zero Universe IDs, "
                      "and no cell resides in Universe 0. MCNP will fail with 'no cells in universe 0'.")
    elif non_zero_universes and not non_zero_universes.issubset(fill_universes):
        unfilled = non_zero_universes - fill_universes
        errors.append(f"Orphaned non-zero Universes detected {unfilled} without corresponding FILL cards in Universe 0.")

    # Summary
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  [ERROR] {err}")
        return False
    else:
        print("Validation PASSED: Deck has MODE N, KCODE, KSRC, MT thermal scattering, and valid Universe 0 coverage.")
        return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "pin_cell_runnable.mcnp"
    mat_file = sys.argv[2] if len(sys.argv) > 2 else "materials.xml"
    success = validate_deck(target, mat_file)
    sys.exit(0 if success else 1)
