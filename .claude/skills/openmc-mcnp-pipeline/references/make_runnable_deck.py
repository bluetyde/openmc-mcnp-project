"""
Phase 2b: remediate the MCNP deck produced by MCNPy and add run-control cards.

This is the pin-cell entry point for src/remediate_deck.py, which does the work for any
OpenMC model (see that file for the full list of MCNPy 0.0.7 gaps it fixes). For the pin
cell that means:
1. Cell Universe Assignment: strip MCNPy's `U 1` tags so the cells are in universe 0.
2. Thermal Scattering Cards (MT): inject MT cards for materials with S(a,b) in materials.xml
   (e.g. MT3 lwtr.01t for light water). Unknown S(a,b) names fail loudly.
3. Initial Source Point (KSRC): taken from the source's spatial distribution in settings.xml.
4. Run Control (MODE & KCODE): MODE N and KCODE from settings.xml.
The pin cell has reflective boundaries and no tallies, so no graveyard cell or tally cards
are added. For other models use src/export_mcnp.py.

Run after translate_to_mcnp.py has produced pin_cell.mcnp:
    python src/make_runnable_deck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from remediate_deck import load_model, remediate  # noqa: E402

SOURCE_DECK = "pin_cell.mcnp"
OUT_DECK = "pin_cell_runnable.mcnp"

model = load_model(os.getcwd())  # materials.xml, geometry.xml, settings.xml from openmc_model.py
report = remediate(SOURCE_DECK, model, OUT_DECK)

print(f"Wrote {OUT_DECK}")
for line in report["added"]:
    print(f"  + {line}")
for note in report["notes"]:
    print(f"  note: {note}")
