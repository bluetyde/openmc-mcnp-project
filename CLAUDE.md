# OpenMC → MCNP Workflow

## Environment: one conda env, platform-specific setup

`openmc` has no Windows build at all (not on PyPI; conda-forge ships only
`linux-64`/`osx-64`/`osx-arm64` — no `win-64`). That is the reason this
project needs a Linux-like environment on Windows specifically, not a
preference. Everything else (MetaPy/MCNPy wheels, the pipeline scripts) is
plain Python/JVM and has no platform restriction of its own — verified by
inspecting the wheels directly (`py3-none-any` tag, contents are `.jar`/`.class`
files, no `.so`/`.dylib`/`.dll`).

One conda env (e.g. `openmc-mcnp`) holds `openmc`, `montepy`, `openjdk=8`, and
MCNPy/MetaPy together, on every machine. Never split the pipeline across two
environments or two filesystems (e.g. openmc in one place, MCNPy in
another) — conda envs don't cross environment boundaries, so no single Python
process could import both, and mismatched path conventions between the two
sides don't line up cleanly either. That reintroduces exactly the
disconnected-environments problem this pipeline exists to avoid.

### Windows

Requires WSL2 (Ubuntu) — run everything inside it, in the Linux filesystem
(e.g. `~/openmc-mcnp-project`), not from `/mnt/c/...` (the Windows drive
mounted into WSL2; works, but noticeably slower for conda/build-heavy work
than the native Linux filesystem) and not split between a Windows-side copy
and the WSL2 copy. Claude Code (and any terminal work on this project) should
run with its working directory inside WSL2, not a Windows shell.

Do not edit this project through the `\\wsl.localhost\` UNC path with
Windows-side tools (editors, file browsers, Windows-side scripts) — see
"Known quirks".

### macOS

No WSL2 equivalent needed — run natively. But conda-forge has no `openmc`
build for `osx-arm64` (Apple Silicon) as of this writing, only `osx-64`
(confirmed by searching conda-forge directly for both platforms) — Java 8
itself does have native `osx-arm64` builds, so it's specifically `openmc`
that's missing, not the whole toolchain. On Apple Silicon, create the env
under Rosetta 2 with the x86_64 subdir instead of natively:

```bash
CONDA_SUBDIR=osx-64 conda env create -f environment.yml
conda activate openmc-mcnp
conda config --env --set subdir osx-64
```

The last line pins the env to `osx-64` for future installs into it (e.g. when
installing MetaPy/MCNPy), so it doesn't drift back to attempting native
arm64 resolution mid-setup. If a native `osx-arm64` `openmc` build appears on
conda-forge later, re-check before assuming this workaround is still needed.

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
4. `src/make_runnable_deck.py` remediates MCNPy translation gaps and adds `MODE`/`KCODE`/`KSRC` cards to produce `pin_cell_runnable.mcnp` from `pin_cell.mcnp`. Specifically, it:
   - Assigns cells to Universe 0 (strips `U 1` tags).
   - Injects missing $S(\alpha,\beta)$ thermal scattering `MT` cards from `materials.xml`.
   - Injects `KSRC` source point cards from `settings.xml`.
   - Appends `MODE N` and `KCODE` cards.
5. `src/validate_deck.py` — automated deck validator that asserts `MODE`, `KCODE`, `KSRC`, `MT` thermal scattering cards matching `materials.xml`, and Universe 0 hierarchy coverage before any deck is declared runnable.
6. Once a valid deck exists, use `src/montepy_sweep.py` as a template for parameter sweeps (enrichment, dimensions, etc.) — MontePy preserves formatting/comments on edits, and `montepy_sweep.py` validates all generated output decks.
7. For MCNP runtime errors ("particle lost in cell X", lost-particle geometry errors), paste the surface/cell block and the error and ask for help tracing Boolean logic — that's a good use of conversational help since the deck text is already ground truth.

## Hard rule
All deck modifications must be programmatically generated and validated via automated assertion checks (`src/validate_deck.py`). Never hand-author raw MCNP cell/surface cards from memory as a substitute for running translation and programmatic validation tools. Wrapping unparsed string buffers in generic classes (e.g. `DataInput`) satisfies the letter of tool mediation while providing none of its protection unless accompanied by automated deck validation checks.

## MCNPy install notes (do this before step 2)
MCNPy is NOT on PyPI and is not a pure-Python package. It requires:
- Java 8 (JRE) installed and on PATH — inside the project's conda env, install via
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
- MCNPy 0.0.7 translation gaps:
  - `openmc_to_mcnp()` ignores `settings.xml` (`if openmc_settings is not None: pass`), omitting `MODE`, `KCODE`, and `KSRC`.
  - `openmc_to_mcnp()` drops $S(\alpha,\beta)$ thermal scattering tags (`c_H_in_H2O` / `MT` cards).
  - `openmc_to_mcnp()` tags all cells with `U 1` without creating a Universe 0 container cell, causing MCNP to fail (`no cells in universe 0`).
  - All three are remediated programmatically in `make_runnable_deck.py` and asserted by `validate_deck.py`.
- MontePy: `problem.mode` is not included in `write_to_file()` output by
  itself — `_write_to_stream()` only walks `cells`/`surfaces`/`data_inputs`,
  not `mode` directly. To get a `MODE` card written out, explicitly
  `problem.data_inputs.append(problem.mode)` before writing.
- MontePy's parser is stricter than MCNP itself in multiple confirmed cases —
  MCNP infers/normalizes things MontePy requires explicit (e.g. MCNP accepts
  "lwtr.01" and infers the "t" suffix for MT thermal-scattering cards from
  context; MontePy's lexer requires "lwtr.01t" literally, confirmed against
  MCNP6.3.0 output on an unrelated deck). When a MontePy parse fails on a deck
  that MCNP itself ran successfully, check for MCNP-side leniency before
  assuming the input is actually wrong.
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

