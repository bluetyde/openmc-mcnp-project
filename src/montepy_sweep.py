"""
Phase 3: iterate on an existing MCNP deck with MontePy, preserving formatting
and comments (unlike regenerating the deck text from scratch each time).

Example: sweep fuel material density across a few values and write out one
deck per value. Adjust the loop for whatever parameter you're actually
sweeping (enrichment, pellet radius, pitch, etc.).

Run after make_runnable_deck.py has produced pin_cell_runnable.mcnp (not
pin_cell.mcnp — that one has no MODE/KCODE and sweeping it just produces
more non-runnable decks):
    python src/montepy_sweep.py
"""
import os
import montepy
from validate_deck import validate_deck

SOURCE_DECK = "pin_cell_runnable.mcnp"
OUT_DIR = "sweep_decks"
FUEL_MATERIAL_NUMBER = 1  # confirmed: material 1 = fuel (mass_density 10.4) in pin_cell_runnable.mcnp

densities_g_cm3 = [10.2, 10.4, 10.6, 10.8]

os.makedirs(OUT_DIR, exist_ok=True)

# First validate the source deck
if not validate_deck(SOURCE_DECK):
    raise RuntimeError(f"Source deck {SOURCE_DECK} failed validation. Fix source deck before running sweeps.")

for density in densities_g_cm3:
    problem = montepy.read_input(SOURCE_DECK)

    # Find the target cell(s) using this material and update density.
    for cell in problem.cells:
        if cell.material and cell.material.number == FUEL_MATERIAL_NUMBER:
            cell.mass_density = density

    out_path = os.path.join(OUT_DIR, f"pin_cell_density_{density:.1f}.mcnp")
    problem.write_to_file(out_path, overwrite=True)
    
    # Assert generated deck passes validation
    if not validate_deck(out_path):
        raise RuntimeError(f"Generated sweep deck {out_path} failed validation!")
    
    print(f"Wrote and validated {out_path}")

print(f"\n{len(densities_g_cm3)} decks written and validated in {OUT_DIR}/, "
      "formatting/comments preserved from the source deck.")
