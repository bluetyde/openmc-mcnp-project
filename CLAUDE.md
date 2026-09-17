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
4. `src/make_runnable_deck.py` remediates MCNPy translation gaps and adds `MODE`/`KCODE`/`KSRC` cards to produce `pin_cell_runnable.mcnp` from `pin_cell.mcnp`. It is the pin-cell entry point for `src/remediate_deck.py`, which does the work for any model. For the pin cell it:
   - Assigns cells to Universe 0 (strips `U 1` tags).
   - Injects missing $S(\alpha,\beta)$ thermal scattering `MT` cards from `materials.xml` (unknown table names fail loudly).
   - Injects `KSRC` from the source's spatial distribution in `settings.xml`.
   - Appends `MODE N` and `KCODE` cards.
5. `src/validate_deck.py` — automated deck validator. Asserts `MODE`, run control matching the OpenMC run mode (`KCODE` + `KSRC`/`SDEF` for eigenvalue; `SDEF` + `NPS` and no `KCODE` for fixed source), `MT` cards for every S(a,b) material, Universe 0 coverage, and — when the OpenMC model is available — an `IMP:N=0` cell for vacuum boundaries, one tally per OpenMC tally score, and geometry equivalence via `src/geometry_check.py` (sampled points must land in the same cell, material and density as in OpenMC, with no gaps or overlaps).

   **Any other OpenMC model** (e.g. from OpenMC Studio): `python src/export_mcnp.py model.xml --name <name>` runs translation, `remediate_deck.py` and validation in one step. It adds a graveyard cell for vacuum boundaries, `SDEF`/`SI`/`SP`/`NPS` or `KSRC`/`KCODE`, `MODE N [P]` with photon importances, and `F4`/`E4`/`FM`/`SD`/`FMESH` tallies. Unsupported features raise `UnsupportedFeature` with a reason rather than exporting something approximate. After changing any export or validation code, run `python tests/check_export_mcnp.py` — it proves each validator check still fires.
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
  - `openmc_to_mcnp()` drops vacuum boundary conditions. MCNP has no vacuum surface type, and no outside cell with `IMP:N=0` is created, so every particle leaving the model would be lost. `remediate_deck.py` adds a graveyard cell (`#` complement of every root-universe cell).
  - `openmc_to_mcnp()` translates no tallies or sources (MCNPy has tally/source classes, but the OpenMC translator never uses them).
  - All of these are remediated programmatically in `remediate_deck.py` (via `make_runnable_deck.py` for the pin cell, or `export_mcnp.py`) and asserted by `validate_deck.py`.
- Remediation details that are easy to get wrong:
  - `F4` divides by cell volume by default; OpenMC cell tallies don't. Exported cell tallies get `SDn 1 ...` so the numbers are comparable. `FMESH` results are still per cm² (no SD equivalent), which the export reports as a note.
  - MCNP's `-2` reaction in `FM` cards excludes fission, unlike OpenMC's `absorption`. MCNP's own fix (`-2:-6`) can't be used because MontePy 1.1.3 can't parse `:` in FM cards, so absorption is exported only for materials without actinides and refused for fuel.
  - `MT` identifiers depend on the MCNP data library (`lwtr.01t` in the classic libraries, names like `h-h2o.40t` in newer ENDF/B-VIII.0 releases). `SAB_MCNP_MAP` in `src/mcnp_cards.py` uses the classic names; check them against your xsdir and override with `--sab NAME=ID`. The old fallback that invented `<openmc_name>.01t` for unknown tables is gone — it produced identifiers MCNP can't find.
  - Before this generalization, `make_runnable_deck.py` wrote `KSRC 0.0 0.0 0.0` regardless of `settings.xml` despite its docstring; KSRC now comes from the source.
- MontePy 1.1.3 importances: setting `cell.importance.photon` on a cell that was parsed with only `IMP:N` writes a duplicate neutron importance (`IMP:N=1.0 IMP:p,n=1.0`), and the written deck then fails to parse. `problem.cells.set_equal_importance()` raises `KeyError: 'photon'` in the same situation. Working pattern: `v = cell.importance.neutron; del cell.importance.neutron; cell.importance.neutron = v; cell.importance.photon = v` (written as `IMP:n,p=`).
- MontePy 1.1.3 parses `SDEF` as a `ForbiddenDataInput`: it round-trips the text but can't be edited through the object model. `SI`/`SP`/`F`/`E`/`FM`/`SD`/`FMESH`/`NPS`/`KCODE`/`KSRC` parse as generic `DataInput`.
- The geometry check is sampling-based evidence, not a proof: small features can be missed, and it supports only P (4-constant), PX/PY/PZ, SO/S/SX/SY/SZ, CX/CY/CZ, C/X/C/Y/C/Z and GQ surfaces in universe 0 without transformations. Decks outside that are reported as "could not run", never silently passed.
- No MCNP executable is installed in this environment, so exported decks are validated by MontePy, `validate_deck.py` and the geometry check, but have not been run through MCNP itself.
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

