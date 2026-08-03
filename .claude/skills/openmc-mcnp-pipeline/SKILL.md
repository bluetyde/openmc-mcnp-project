---
name: openmc-mcnp-pipeline
description: >-
  Draft reactor/pin-cell models in OpenMC, translate them deterministically to MCNP input decks with
  MCNPy, and iterate on those decks with MontePy. Use this skill whenever a task touches OpenMC, MCNP,
  MCNPy, MetaPy, or MontePy — including translating an OpenMC model into an MCNP deck, installing or
  troubleshooting MCNPy/MetaPy (Java 8 + py4j bridge), writing or editing MCNP cell/surface/material
  cards, adding KCODE or MODE run-control cards, sweeping parameters like enrichment or density across
  decks, or tracing MCNP lost-particle and geometry errors. Reach for it as soon as MCNP deck text is
  involved at all — the whole point of this pipeline is to keep deck syntax coming from real parsing
  tools instead of being written from memory, so it applies even when the request sounds like a quick
  one-line edit to a deck.
---

# OpenMC → MCNP pipeline

Draft models in OpenMC (Python, good error messages, fast feedback), translate them to MCNP decks
mechanically, then use MCNP-native tools to iterate. The value is in *never* retyping deck syntax by
hand between those stages.

## The hard rule, and why it exists

Never hand-author or hand-edit raw MCNP cell/surface cards as a substitute for running the actual
translation/parsing tools. MCNP's fixed-format, 80-column, FORTRAN-era syntax fails quietly: a wrong
column or a misplaced sign produces a deck that parses and runs but models something different from
what was intended. There is no loud error to catch it.

So when a tool can't express something, say so explicitly and ask, rather than filling the gap with
improvised syntax. "MCNPy 0.0.7 can't translate settings, here are the options" is a good outcome.
Silently hand-writing a `KCODE` card because the translator skipped it is the failure mode this
pipeline exists to prevent.

This applies to edits too. Changing a density inside an existing deck goes through MontePy — it
parses the deck, mutates the object, and re-serializes, preserving formatting and comments. Editing
the text directly reintroduces exactly the risk above.

## Environment: one conda env, platform-specific setup

`openmc` has no Windows build at all — not on PyPI, and conda-forge ships only `linux-64`,
`osx-64`, and `osx-arm64` (no `win-64`). That's why Windows specifically needs a Linux-like layer;
it isn't a preference. MetaPy/MCNPy have no such restriction of their own — their wheels are
`py3-none-any` and contain only `.jar`/`.class` (JVM bytecode), no compiled native code — confirmed
by unzipping and checking, not by trusting the tag alone.

One conda env (e.g. `openmc-mcnp`) holds `openmc`, `montepy`, `openjdk=8`, plus MetaPy and MCNPy —
on every machine. Never split the pipeline across two environments or filesystems (e.g. openmc in
one place, MCNPy in another): conda envs don't cross that boundary, so no single Python process
could import both, and mismatched path conventions between the two sides don't line up either. That
recreates the disconnected-environments problem this pipeline exists to avoid.

**Windows** needs WSL2 (Ubuntu) — everything runs inside it, in the Linux filesystem (e.g.
`~/openmc-mcnp-project`), not `/mnt/c/...` (works, but slower for conda/build-heavy work) and not
split between a Windows-side copy and the WSL2 copy.

**macOS** runs natively, no WSL2 equivalent needed — but conda-forge has no `openmc` build for
`osx-arm64` (Apple Silicon), only `osx-64` (Java 8 itself does have native `osx-arm64` builds, so
it's specifically `openmc` that's missing). On Apple Silicon, build the env under Rosetta 2 instead:
`CONDA_SUBDIR=osx-64 conda env create -f environment.yml`, then
`conda config --env --set subdir osx-64` inside it so later installs (MetaPy/MCNPy) don't drift back
to native arm64 resolution. Re-check conda-forge before assuming this is still needed — a native
build may exist by the time this is read.

## Pipeline order

Each stage exists to catch a class of error where the error messages are best, so resist skipping
ahead — a broken model carried into translation produces confusing failures two layers away from
the actual mistake.

1. **`openmc_model.py`** — build and validate the OpenMC model; exports `materials.xml`,
   `geometry.xml`, `settings.xml`. Fix geometry/material problems here, where OpenMC reports them
   directly. This stage needs no cross-section data because it only builds and exports the model
   definition; `OPENMC_CROSS_SECTIONS` is only required once something actually runs transport
   (`openmc.run()` or the `openmc` executable).
2. **`translate_to_mcnp.py`** — deterministic translation via
   `mcnpy.translate_mcnp_openmc.openmc_to_mcnp(geometry, materials, settings)`. Produces a deck with
   cells, surfaces, and materials.
3. **`make_runnable_deck.py`** — adds `MODE N` and `KCODE` via MontePy, because MCNPy does not
   translate settings (see quirks). Reads particles/inactive/batches back out of `settings.xml` so
   those numbers have one source of truth rather than being retyped.
4. **`montepy_sweep.py`** — parameter sweeps (enrichment, density, dimensions). MontePy preserves
   formatting and comments, so sweep against the generated deck rather than regenerating text.
5. **Runtime errors** ("particle lost in cell X", lost-particle geometry errors) are a good fit for
   conversational debugging: the deck text is already ground truth, so reasoning about Boolean
   region logic isn't generating syntax from memory.

The geometry-only deck and the runnable deck are **intentionally separate files** — e.g.
`pin_cell.mcnp` (stage 2 output) and `pin_cell_runnable.mcnp` (stage 3 output). Keep both. Neither
is redundant, and merging them loses the clean boundary between "what the translator produced" and
"what was added on top".

## Installing MCNPy and MetaPy

MCNPy is not on PyPI and is not pure Python — it wraps Java classes over a py4j bridge.

- **Java 8** — inside the conda env, `conda install openjdk=8`, so it stays scoped to the env rather
  than depending on a system-wide JDK.
- **MetaPy** — install from its `.whl`. `pip install metapy` fetches an unrelated package. MetaPy
  lives in a **separate repo**, `https://github.rpi.edu/NuCoMP/metapy`, even though its install
  instructions are hosted on MCNPy's docs site — the docs layout suggests it's bundled with MCNPy,
  and it isn't. Clone it and install from `dist/metapy-X.whl`.
- **MCNPy** — clone `https://github.rpi.edu/NuCoMP/mcnpy.git`, then
  `pip install /path/to/dist/mcnpy-X.whl`. Install MetaPy first; it's a dependency. `py4j` comes in
  automatically.

Verify the bridge actually reaches the JVM, not just that the import succeeded. A healthy
`import mcnpy` prints `Metamodel Gateway Server Started` and later `Metamodel Gateway Server Killed`.
That round trip is the real check — Java/classpath problems surface there, not at install time.

If Java or MetaPy setup turns out to be too much friction on a given machine, flag it rather than
falling back to freehand MCNP generation.

## Known quirks

These were found by reading installed source, and they're version-specific — verify against the
installed versions before relying on them, and check whether a newer release has fixed them.

**MCNPy `openmc_to_mcnp()` ignores settings (confirmed 0.0.7).** The function accepts, type-hints,
and documents an `openmc_settings` argument, then does nothing with it. The body literally reads:

```python
if openmc_settings is not None:
    pass
```

So no `MODE`, no `KCODE`, no run-control cards — passing settings in changes the output not at all.
No released version (0.0.0–0.0.7) mentions settings translation in its changelog; this was never
built rather than being a regression. This is why stage 3 exists.

**MontePy doesn't serialize `problem.mode` unless it's in `data_inputs` (confirmed 1.1.3).**
`write_to_file()` → `_write_to_stream()` walks only `cells`, `surfaces`, and `data_inputs`. The
`problem.mode` accessor is separate, and a deck parsed from a file with no `MODE` card gets a
default `Mode` object that is not wired into `data_inputs`. Setting `problem.mode.set("n")` alone
writes nothing. Append it explicitly:

```python
problem.mode.set("n")
problem.data_inputs.append(problem.mode)
```

**MontePy has no `Kcode` class (1.1.3).** Only `Mode` is wired into `MCNP_Problem`. Add `KCODE`
through the generic catch-all `DataInput`, which still round-trips through MontePy's own parser and
writer — so it is tool-mediated, not hand-typed deck text:

```python
from montepy.data_inputs.data_input import DataInput
problem.data_inputs.append(DataInput(f"KCODE {particles} 1.0 {inactive} {batches}"))
```

The `1.0` is the initial k-effective guess. It has no counterpart in `openmc.Settings`, so it's a
deliberate MCNP-side default rather than something derived from the model — worth a comment wherever
it appears.

**Writing through the `\\wsl.localhost\` UNC path can silently delete directories.**
Creating a file that way — from a Windows-side editor, file browser, or script — has
removed a sibling directory during file creation, with no error raised, so the loss is
easy to miss unless the tree is re-checked afterward. Edit from inside the WSL shell, or
via `wsl -d Ubuntu -- <cmd>` from PowerShell. Git Bash is not a safe substitute for the
latter: it rewrites `/root/...` into `C:/Program Files/Git/root/...`.

**MCNPy has no `__version__` (0.0.7).** `mcnpy.__version__` raises `AttributeError`. Use
`pip show mcnpy`.

**Check mass vs atom density before editing cell densities.** MontePy exposes `cell.mass_density`
and `cell.atom_density` separately, and reading the wrong one raises `AttributeError: Cell N is in
mass density.` — a helpful guard, but only on read. Confirm which applies via `cell.is_atom_dens`
(`False` = mass density) before a sweep writes values. In deck text, a negative density like `-10.4`
is mass density in g/cm³; positive is atom density in atoms/barn-cm.

## Conventions

- Geometry dimensions as named variables at the top of each script, in cm.
- Keep units explicit in comments. OpenMC and MCNP both default to cm, but density conventions
  differ (g/cm³ vs atoms/barn-cm) and that's an easy silent mismatch.
- Natural Zircaloy-4 approximation is fine for cladding unless specified otherwise.
- Re-running `openmc_model.py` in a session where materials already exist may emit
  `IDWarning: Another Material instance already exists with id=N`. That's OpenMC's global ID
  registry noticing a re-run, not a translation problem.

## Reference scripts

Working versions of each stage, in `references/`. Read the one matching the current stage — they
show the actual API shapes, which matter more than the physics here since the physics is
model-specific and the API details are the part that's easy to get wrong.

| File | Stage |
|---|---|
| `references/openmc_model.py` | 1 — build model, export XML |
| `references/translate_to_mcnp.py` | 2 — OpenMC → MCNP via MCNPy |
| `references/make_runnable_deck.py` | 3 — add `MODE`/`KCODE` via MontePy |
| `references/montepy_sweep.py` | 4 — parameter sweep over a deck |

Verify a signature against the installed version before assuming a call matches — these scripts
were written against MCNPy 0.0.7 and MontePy 1.1.3, and `inspect.signature()` or reading the
installed source in `site-packages/` settles it quickly. That habit is what surfaced the
settings-ignored quirk above.
