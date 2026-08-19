# OpenMC → MCNP pipeline

Scaffold for: draft in OpenMC, translate deterministically with MCNPy, remediate & validate, iterate with MontePy.

## Layout
```
CLAUDE.md                   <- Claude Code reads this automatically for context
requirements.txt
src/openmc_model.py          Phase 1: build + validate the OpenMC pin-cell model
src/translate_to_mcnp.py     Phase 2: OpenMC -> MCNP via MCNPy (deterministic)
src/make_runnable_deck.py   Phase 2b: remediate MCNPy gaps (Universe 0, MT3, KSRC, MODE/KCODE)
src/validate_deck.py        Automated deck validator (MODE, KCODE, KSRC, MT cards, U 0)
src/montepy_sweep.py         Phase 3: parameter sweeps with automated deck validation
```

## Quick start
```bash
pip install -r requirements.txt
python src/openmc_model.py
python src/translate_to_mcnp.py
python src/make_runnable_deck.py
python src/validate_deck.py pin_cell_runnable.mcnp
python src/montepy_sweep.py
```
Read CLAUDE.md's "MCNPy install notes" before running `translate_to_mcnp.py` — that step needs Java 8 + MetaPy set up separately.

