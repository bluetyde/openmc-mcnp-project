# OpenMC → MCNP pipeline

Scaffold for: draft in OpenMC, translate deterministically with MCNPy,
iterate with MontePy, debug with Claude Code.

## Layout
```
CLAUDE.md               <- Claude Code reads this automatically for context
requirements.txt
src/openmc_model.py      Phase 1: build + validate the OpenMC pin-cell model
src/translate_to_mcnp.py Phase 2: OpenMC -> MCNP via MCNPy (deterministic)
src/montepy_sweep.py     Phase 3: parameter sweeps on the MCNP deck
```

## Quick start
```
pip install -r requirements.txt
python src/openmc_model.py
```
Then read CLAUDE.md's "MCNPy install notes" before running
`translate_to_mcnp.py` — that step needs Java 8 + MetaPy set up separately,
it's not just a pip install.
