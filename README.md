# OpenMC → MCNP pipeline

Scaffold for: draft in OpenMC, translate deterministically with MCNPy, remediate & validate, iterate with MontePy.

## Layout
```
CLAUDE.md                   <- Claude Code reads this automatically for context
requirements.txt
src/openmc_model.py          Phase 1: build + validate the OpenMC pin-cell model
src/translate_to_mcnp.py     Phase 2: OpenMC -> MCNP via MCNPy (deterministic)
src/make_runnable_deck.py   Phase 2b: remediate MCNPy gaps for the pin cell (wrapper over remediate_deck.py)
src/validate_deck.py        Automated deck validator (MODE, run control, MT, U 0, vacuum boundary, tallies, geometry)
src/montepy_sweep.py         Phase 3: parameter sweeps with automated deck validation

src/export_mcnp.py          Any OpenMC model -> translated deck -> runnable deck -> validation (one command)
src/remediate_deck.py        Remediation for any model: U 0, graveyard cell, MT, SDEF/NPS or KSRC/KCODE, MODE, tallies
src/mcnp_cards.py            Builds source, run-control and tally cards from OpenMC objects
src/geometry_check.py        Sampled-point comparison of deck geometry against the OpenMC model
tests/check_export_mcnp.py   Self-test: valid exports pass, and every validator check is seen failing
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

## Exporting any OpenMC model
```bash
python src/export_mcnp.py path/to/model.xml --name mymodel          # or a folder with the XML files
python src/export_mcnp.py model.xml --sab c_H_in_H2O=h-h2o.40t       # override an S(a,b) name for your xsdir
python tests/check_export_mcnp.py                                    # self-test after changing the export code
```
Writes `<name>.mcnp` (MCNPy output) and `<name>_runnable.mcnp`, and exits non-zero unless the
runnable deck validates. Supported: fixed-source (single point/box/sphere/cylinder source; line,
Watt, Maxwell or uniform energies; isotropic or beam) and eigenvalue runs; vacuum and reflective
boundaries; photon transport; cell tallies (flux and reaction rates) and regular-mesh flux tallies; rotated boxes
and cylinders (general P and GQ surfaces).
Anything else is refused with a reason instead of being exported approximately.

