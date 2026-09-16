"""
Export any OpenMC model to a runnable, validated MCNP deck.

    python src/export_mcnp.py path/to/model.xml
    python src/export_mcnp.py path/to/folder            (model.xml, or geometry/materials/settings[/tallies].xml)
    python src/export_mcnp.py model.xml --out-dir decks --name shielding --sab c_H_in_H2O=h-h2o.40t

Stages (same as the pin-cell pipeline, generalized):
  1. MCNPy translates geometry and materials  -> <name>.mcnp           (translator output, kept as-is)
  2. remediate_deck adds what MCNPy leaves out -> <name>_runnable.mcnp
  3. validate_deck checks the runnable deck, including the sampled geometry comparison

Exit code 0 only if the deck validated. --report writes a JSON summary (used by OpenMC Studio).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcnp_cards import UnsupportedFeature  # noqa: E402
from remediate_deck import load_model, remediate  # noqa: E402
from validate_deck import validate_deck  # noqa: E402


def translate(model, out_path):
    try:
        from mcnpy.translate_mcnp_openmc import openmc_to_mcnp
    except ImportError as e:
        raise RuntimeError("MCNPy is not importable in this environment. See CLAUDE.md 'MCNPy install notes'. "
                           f"Original error: {e}")
    # MCNPy prints progress and JVM gateway messages; keep them out of this script's output.
    with contextlib.redirect_stdout(io.StringIO()):
        deck = openmc_to_mcnp(model.geometry, model.materials, model.settings)
        deck.write(out_path)


def export(model_path, out_dir=None, name=None, sab=None, samples=20000):
    report = {"ok": False, "model": os.path.abspath(model_path), "stage": "load"}
    model = load_model(model_path)
    base = os.path.dirname(os.path.abspath(model_path)) if not os.path.isdir(model_path) else os.path.abspath(model_path)
    out_dir = os.path.abspath(out_dir or base)
    os.makedirs(out_dir, exist_ok=True)
    name = name or "model"
    translated = os.path.join(out_dir, f"{name}.mcnp")
    runnable = os.path.join(out_dir, f"{name}_runnable.mcnp")
    report.update(translated=translated, runnable=runnable)

    report["stage"] = "translate"
    translate(model, translated)

    report["stage"] = "remediate"
    report.update(remediate(translated, model, runnable, sab_map=sab))

    report["stage"] = "validate"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = validate_deck(runnable, model=model, geometry_samples=samples)
    report["validation"] = buf.getvalue()
    report["ok"] = bool(ok)
    report["stage"] = "done"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model.xml, or a folder with the model's XML files")
    ap.add_argument("--out-dir", help="where to write the decks (default: next to the model)")
    ap.add_argument("--name", default="model", help="deck file name stem (default: model)")
    ap.add_argument("--sab", action="append", default=[], metavar="OPENMC_NAME=MCNP_ID",
                    help="override an S(a,b) mapping, e.g. c_H_in_H2O=h-h2o.40t (repeatable)")
    ap.add_argument("--samples", type=int, default=20000, help="geometry check sample points (0 to skip)")
    ap.add_argument("--report", help="write a JSON report here")
    args = ap.parse_args(argv)

    sab = {}
    for item in args.sab:
        if "=" not in item:
            ap.error(f"--sab expects NAME=ID, got {item!r}")
        k, v = item.split("=", 1)
        sab[k.strip()] = v.strip()

    try:
        report = export(args.model, args.out_dir, args.name, sab, args.samples)
    except UnsupportedFeature as e:
        report = {"ok": False, "stage": "remediate", "error": f"Not exportable: {e}"}
    except Exception as e:
        report = {"ok": False, "stage": "error", "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
    if report.get("error"):
        print(report["error"])
        return 1
    print(f"Wrote {report['translated']}  (MCNPy translation)")
    print(f"Wrote {report['runnable']}  (runnable)")
    for a in report["added"]:
        print(f"  + {a}")
    for n in report["notes"]:
        print(f"  note: {n}")
    print(report["validation"].rstrip())
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
