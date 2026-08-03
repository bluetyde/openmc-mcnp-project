# OpenMC → MCNP Workflow

## Environment: WSL2 only, one conda env
This project runs entirely inside WSL2 (Ubuntu), in a single conda env, in the
Linux filesystem — not split across Windows and WSL2, and not run from
`/mnt/c/...` (that's the Windows drive mounted into WSL2; it works but is
noticeably slower for conda/build-heavy work than the native Linux filesystem).

- `openmc` has no Windows build (not on PyPI at all; conda-forge only ships
  `linux-64`/`osx` builds) — this is the reason WSL2 is required in the first
  place, not a preference.
- Project lives at `~/openmc-mcnp-project` inside WSL2, cloned/copied there
  directly rather than edited from the Windows-side copy.
- One conda env (e.g. `openmc-mcnp`) holds `openmc`, `montepy`, `openjdk=8`,
  and MCNPy/MetaPy (once installed) together. Do not split MCNPy into a
  Windows-side env and openmc into a WSL2-side env — conda envs don't cross
  the Windows/WSL2 boundary, so a Python process in one can't import from the
  other, and Windows (`C:\...`) vs WSL2 (`/mnt/c/...`) paths don't match up
  cleanly either. Both problems reintroduce exactly what this pipeline is
  meant to avoid: hand-translating between disconnected environments.
- Claude Code (and any terminal work on this project) should run with its
  working directory inside WSL2, not a Windows shell — one shell, one env,
  one filesystem view.

## Purpose
Draft reactor/pin-cell models in OpenMC (Python), translate them to MCNP input
decks, then use MCNP-native tools for iteration and troubleshooting.

## Pipeline (follow this order)
1. `src/openmc_model.py` — build and validate the OpenMC model. Run it
   (`python src/openmc_model.py`) and confirm it exports XML / plots cleanly
   before moving on. Fix geometry/material errors here, where OpenMC's own
   error messages catch them immediately — do not carry a broken model
   forward into translation.
2. `src/translate_to_mcnp.py` — deterministic translation via MCNPy
   (`mcnpy.translate_mcnp_openmc.openmc_to_mcnp`). Prefer this over hand-writing
   MCNP cards from scratch. See "MCNPy install notes" below before running.
3. Once a valid deck exists, use `src/montepy_sweep.py` as a template for
   parameter sweeps (enrichment, dimensions, etc.) — MontePy preserves
   formatting/comments on edits, so run sweeps against the deck MCNPy produced,
   not by regenerating text by hand each time.
4. `src/make_runnable_deck.py` adds `MODE`/`KCODE` cards (MCNPy's translation
   doesn't — see "Known quirks") to produce `pin_cell_runnable.mcnp` from
   `pin_cell.mcnp`. These are two intentionally separate files, not
   redundant — `pin_cell.mcnp` (geometry+materials only) is
   `translate_to_mcnp.py`'s output, `pin_cell_runnable.mcnp` (adds
   `MODE N`/`KCODE`) is `make_runnable_deck.py`'s output. Don't merge them.
5. For MCNP runtime errors ("particle lost in cell X", lost-particle geometry
   errors), paste the surface/cell block and the error and ask for help tracing
   Boolean logic — that's a good use of conversational help since the deck
   text is already ground truth, not something being generated from memory.

## Hard rule
Never hand-author or hand-edit raw MCNP cell/surface cards from memory as a
substitute for running the actual translation/parsing tools. MCNP's
fixed-format 80-column FORTRAN-era syntax is easy to get subtly wrong in ways
that don't error loudly. If MCNPy can't handle a specific geometry feature,
say so explicitly and ask before improvising syntax.

## MCNPy install notes (do this before step 2)
MCNPy is NOT on PyPI and is not a pure-Python package. It requires:
- Java 8 (JRE) installed and on PATH — inside the WSL2 conda env, install via
  `conda install openjdk=8` rather than a system JDK, so it stays scoped to
  the same env as everything else.
- `py4j` (Java-Python bridge, pulled in as a normal pip dependency)
- `MetaPy` — must be manually downloaded and installed from a `.whl` file,
  `pip install metapy` fetches the wrong package. It's a **separate repo**,
  not bundled inside mcnpy despite the docs living under mcnpy's site:
  https://github.rpi.edu/NuCoMP/metapy (install instructions:
  https://pages.github.rpi.edu/NuCoMP/mcnpy_docs/build/html/getting_started.install.html)
- Then MCNPy itself: clone https://github.rpi.edu/NuCoMP/mcnpy.git and
  `pip install /path/to/dist/mcnpy-X.whl`

Check `java -version` first. If Java/MetaPy setup is too much friction for
this machine, flag it to the user rather than silently falling back to
freehand MCNP generation — that reintroduces the exact hallucination risk
this pipeline exists to avoid.

`mcnpy` (0.0.7 at least) has no `__version__` attribute — `import mcnpy;
mcnpy.__version__` raises `AttributeError`. Use `pip show mcnpy` to confirm
the installed version instead.

## Known quirks
- MontePy: `problem.mode` is not included in `write_to_file()` output by
  itself — `_write_to_stream()` only walks `cells`/`surfaces`/`data_inputs`,
  not `mode` directly. To get a `MODE` card written out, explicitly
  `problem.data_inputs.append(problem.mode)` before writing.
- Do not edit this project through the `\\wsl.localhost\` UNC path with
  Windows-side tools (editors, file browsers, Windows-side scripts) — creating a
  file that way has silently deleted a sibling directory, with no error raised.
  Edit from inside the WSL shell, or via `wsl -d Ubuntu -- <cmd>` from
  PowerShell. (Note also that Git Bash rewrites `/root/...` into
  `C:/Program Files/Git/root/...`, so prefer PowerShell for `wsl` calls.)

## Conventions
- All geometry dimensions as named variables at the top of each script, in cm.
- Keep OpenMC and MCNP units/conventions explicit in comments (OpenMC uses cm,
  MCNP also uses cm by default — but double check density units: g/cm3 vs
  atom/barn-cm differ between the two).
- Materials: natural Zircaloy-4 approximation is fine unless told otherwise.
