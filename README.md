# openmc-mcnp-project: OpenMC models to runnable MCNP decks

Turns an OpenMC model into an MCNP 6.3 input deck that runs, and checks it before you do. It is the MCNP
back end of [OpenMC Studio](https://github.com/bluetyde/openmc-studio), and works on its own from the command
line for any OpenMC model.

The translation is deterministic: geometry and materials come from NuCoMP's MCNPy translator, and every card
MCNPy doesn't write (or writes wrong) is built from the OpenMC objects by code in `src/`, never typed by hand.
Anything that can't be translated faithfully is refused with a reason instead of exported approximately.

## What it exports

- **Geometry**: cells, universes and fills; rotated and translated parts (general `P` and `GQ` surfaces);
  standalone boxes and cylinders as `RPP` / `RCC` macrobodies; rectangular lattices as `LAT=1` and hexagonal
  lattices as `LAT=2`, rebuilt from the OpenMC lattice because MCNPy's lattice cards are wrong.
- **Boundaries**: vacuum (an `IMP:N=0` graveyard cell) and reflective.
- **Materials**: compositions and densities, with `MT` cards for thermal scattering (S(α,β)).
- **Run modes**: fixed source (`SDEF` + `NPS`) and eigenvalue (`KCODE` + `KSRC`); fission as capture (`NONU`)
  when OpenMC turns fission neutrons off; neutron and photon transport (`MODE N P`).
- **Sources**: point, box, sphere and cylinder sources; line, Watt, Maxwell, uniform, normal and tabulated
  energies; isotropic or beam directions; several independent sources in one `SDEF` (an `ERG` distribution
  picks the source).
- **Tallies**: cell flux and reaction rates (`F4`), detector responses as `FM` multipliers, energy bins (`E`),
  regular and cylindrical meshes (`FMESH`), tallies on parts inside a lattice (chains through the lattice),
  and surface currents (`F1` with `C 0 1` and `FS` to pick the face).
- **Dose rates** from OpenMC Studio: effective dose (ICRP-116 or ICRP-74) on cells (`F4:N`, `F4:P`) and
  meshes (`FMESH`), with `DE`/`DF` carrying exactly the table OpenMC used, `SD` with the cell volume Studio
  measured (so both codes divide by one volume), and the source rate as `FM` or `FACTOR`, so the deck prints
  Sv/h. From the command line, pass a Studio run's `dose.json` with `--dose`.
- **Readable decks**: section comments, lines wrapped within MCNP's 128 columns, and optional comments that
  link each card back to the Studio object it came from (`docs/studio-ids.md`).

## How a deck is checked

`src/validate_deck.py` runs on every export, and the export fails unless the deck passes:

- **Runnability**: `MODE`, run control that matches the OpenMC run mode, `MT` cards, universe 0, a vacuum
  boundary, importances, tally cards.
- **Geometry**: random points (plus points inside every cell's bounding box, so small cells aren't missed) are
  located in the OpenMC model and, independently, in the MCNP deck as parsed by MontePy, following fills and
  lattices down to each element. Every point must land in the same cell and material. A cell that no point
  reached is reported, so the check can't pass by never looking.

No MCNP executable is needed for any of this.

**Status**: every feature above is checked against OpenMC this way, and `tests/check_export_mcnp.py` also
confirms each validator check can fail. The decks have not yet been run in MCNP itself.

## Install

One conda environment holds everything: OpenMC, MontePy, Java 8 and MCNPy. OpenMC has no native Windows
build, so on Windows run everything inside WSL2 (Ubuntu); macOS and Linux run natively.

```bash
conda env create -f environment.yml    # Python 3.11, OpenMC, MontePy, Java 8
conda activate openmc-mcnp
```

**MCNPy** and its **MetaPy** dependency are not on PyPI. They come from RPI NuCoMP
([mcnpy](https://github.rpi.edu/NuCoMP/mcnpy), [metapy](https://github.rpi.edu/NuCoMP/metapy)), which may need
an RPI account. Install both from their wheel files (`pip install metapy` fetches an unrelated package); the
details are in [CLAUDE.md](CLAUDE.md), under "MCNPy install notes". Without MCNPy, the validator and geometry
check still run on existing decks, but nothing new can be translated.

OpenMC runs also need nuclear data: set `OPENMC_CROSS_SECTIONS` to a `cross_sections.xml` (ENDF/B-VIII.0 from
https://openmc.org/data/).

## Use

```bash
python src/export_mcnp.py path/to/model.xml --name mymodel         # or a folder with the XML files
python src/export_mcnp.py model.xml --sab c_H_in_H2O=h-h2o.40t      # override an S(a,b) name for your xsdir
python src/validate_deck.py deck.mcnp --model model.xml             # check an existing deck against its model
python src/export_mcnp.py model.xml --dose dose.json                  # a Studio run folder: dose tallies too
```

This writes `<name>.mcnp` (MCNPy's translation) and `<name>_runnable.mcnp` (the deck to run), and exits
non-zero unless the runnable deck validates. From Python, `model.export_to_model_xml()` writes the
`model.xml` it reads.

The original pin-cell walkthrough is still here: `src/openmc_model.py` (build the model),
`src/translate_to_mcnp.py`, `src/make_runnable_deck.py`, and `src/montepy_sweep.py` (parameter sweeps on a
finished deck with MontePy).

## Tests

```bash
python tests/check_export_mcnp.py        # exports and validates a set of models; every validator check is seen failing
python tests/test_geometry_coverage.py   # small rotated cells are sampled, and unsampled cells are reported
python tests/test_deck_format.py         # 128-column wrapping
python tests/test_studio_ids.py          # Studio ID comments survive a round trip
python tests/test_column_export.py       # a real translation stays within 128 columns (needs MCNPy)
python tests/test_dose_export.py         # dose cards: DE/DF numbers, SD volume, rate factor, photon F4:P (needs MCNPy)
```

MCNPy's Java bridge listens on a fixed port (25333), so run one export at a time on a machine.

## Layout

```
src/export_mcnp.py       any OpenMC model -> translated deck -> runnable deck -> validation (one command)
src/remediate_deck.py    fixes MCNPy's gaps: universe 0, graveyard cell, MT, run control, MODE, tallies
src/mcnp_cards.py        source, run-control and tally cards from OpenMC objects
src/lattice_cards.py     LAT=1 / LAT=2 lattice cards from the OpenMC lattice
src/macrobody_cards.py   RPP / RCC macrobodies for standalone boxes and cylinders
src/geometry_check.py    point-by-point geometry comparison of deck and model
src/validate_deck.py     the runnability checks, then the geometry check
src/deck_format.py       128-column formatting
src/studio_ids.py        optional Studio provenance comments
CLAUDE.md                environment notes, MCNPy install notes and known MCNPy quirks
```

Card rules cite pages of the MCNP 6.3.0 manual (LA-UR-22-30006 Rev. 1).

## License

Original code in this repository is licensed under the [MIT License](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for OpenMC and NuCoMP
MCNPy/MetaPy notices. External dependencies and third-party data retain their
own licenses; this license does not grant rights to MCNP itself.
