"""
Phase 2: deterministic OpenMC -> MCNP translation via MCNPy.

Read CLAUDE.md "MCNPy install notes" before running this — MCNPy requires
Java 8 + MetaPy, installed outside pip. This script will fail loudly and
explain why rather than silently falling back to hand-written MCNP syntax.

Run after openmc_model.py has produced geometry.xml / materials.xml:
    python src/translate_to_mcnp.py
"""
import sys

try:
    import mcnpy
    from mcnpy.translate_mcnp_openmc import openmc_to_mcnp
except ImportError as e:
    sys.exit(
        "MCNPy is not importable in this environment.\n"
        "This is expected if Java 8 / MetaPy haven't been set up yet — "
        "see CLAUDE.md 'MCNPy install notes'.\n"
        f"Original error: {e}"
    )

import openmc

geometry = openmc.Geometry.from_xml("geometry.xml")
materials = openmc.Materials.from_xml("materials.xml")
settings = openmc.Settings.from_xml("settings.xml")

mcnp_deck = openmc_to_mcnp(geometry, materials, settings)
mcnp_deck.write("pin_cell.mcnp")

print("Wrote pin_cell.mcnp — inspect it, then hand it to MontePy for sweeps "
      "(see src/montepy_sweep.py).")
