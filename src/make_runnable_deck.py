"""
Phase 2b: add MCNP run-control cards (MODE, KCODE) to the deck MCNPy produced.

MCNPy 0.0.7's openmc_to_mcnp() accepts an openmc_settings argument but does
nothing with it (`if openmc_settings is not None: pass` in
translate_mcnp_openmc.py) — no version of MCNPy released so far translates
openmc.Settings into MODE/KCODE cards. montepy 1.1.3 also has no dedicated
Kcode class (only Mode is wired into MCNP_Problem), so KCODE is added here
via montepy's generic catch-all DataInput class, which still round-trips
through montepy's own parser/writer rather than being hand-typed into the
deck text.

particles/inactive/batches are read from settings.xml — the same file
openmc_model.py exported — so there's one source of truth for these numbers
rather than retyping them.

Run after translate_to_mcnp.py has produced pin_cell.mcnp:
    python src/make_runnable_deck.py
"""
import os
import sys

import montepy
from montepy.data_inputs.data_input import DataInput
import openmc

SOURCE_DECK = "pin_cell.mcnp"
OUT_DECK = "pin_cell_runnable.mcnp"

# rkeff: initial guess for k-effective, not present in openmc.Settings —
# 1.0 is MCNP's own conventional default for this field.
INITIAL_KEFF_GUESS = 1.0

if not os.path.exists(SOURCE_DECK):
    sys.exit(
        f"{SOURCE_DECK} not found — run translate_to_mcnp.py first to "
        "produce it from geometry.xml/materials.xml."
    )

settings = openmc.Settings.from_xml("settings.xml")
particles = settings.particles
inactive = settings.inactive
batches = settings.batches

missing = [name for name, value in
           [("particles", particles), ("inactive", inactive), ("batches", batches)]
           if value is None]
if missing:
    sys.exit(
        f"settings.xml is missing {', '.join(missing)} — openmc_model.py must "
        "set these explicitly on the Settings object before exporting, "
        "otherwise this script would silently write a literal 'None' into "
        "the KCODE card."
    )

problem = montepy.read_input(SOURCE_DECK)

problem.mode.set("n")
problem.data_inputs.append(problem.mode)

kcode_line = f"KCODE {particles} {INITIAL_KEFF_GUESS} {inactive} {batches}"
problem.data_inputs.append(DataInput(kcode_line))

problem.write_to_file(OUT_DECK, overwrite=True)

print(f"Wrote {OUT_DECK}")
print(f"MODE:  MODE N")
print(f"KCODE: {kcode_line}")
print(f"(particles={particles}, inactive={inactive}, batches={batches} — from settings.xml)")
