"""
Self-test for src/export_mcnp.py and src/validate_deck.py.

1. Builds three small OpenMC models (fixed-source shielding with tallies, photon box
   source, eigenvalue fuel sphere), exports each, and asserts the decks validate.
2. Breaks a valid deck one way at a time and asserts the validator catches each break
   with the expected message. A validator that passes everything proves nothing, so
   every check has to be seen failing.
3. Asserts unsupported features are refused instead of exported.

Run from the project root, in the openmc-mcnp env:
    python tests/check_export_mcnp.py
Exit code 0 means every assertion held.
"""
import contextlib
import io
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import openmc  # noqa: E402

from export_mcnp import export  # noqa: E402
from mcnp_cards import UnsupportedFeature  # noqa: E402
from remediate_deck import load_model  # noqa: E402
from validate_deck import validate_deck  # noqa: E402

SAMPLES = 5000
failures = []


def check(cond, what):
    print(("  PASS " if cond else "  FAIL ") + what)
    if not cond:
        failures.append(what)


def shielding_model(photon=False, eigen=False, absorption_in_fuel=False):
    openmc.reset_auto_ids()
    poly = openmc.Material(name="Polyethylene")
    poly.set_density("g/cm3", 0.93)
    poly.add_element("H", 0.143716, "wo")
    poly.add_element("C", 0.856284, "wo")
    poly.add_s_alpha_beta("c_H_in_CH2")
    lead = openmc.Material(name="Lead")
    lead.set_density("g/cm3", 11.35)
    lead.add_element("Pb", 1.0)
    fuel = openmc.Material(name="UO2")
    fuel.set_density("g/cm3", 10.4)
    fuel.add_element("U", 1.0, enrichment=5.0)
    fuel.add_element("O", 2.0)
    mats = [fuel, poly, lead] if (eigen or absorption_in_fuel) else [poly, lead]

    R = 60.0
    planes = [openmc.XPlane(-R, boundary_type="vacuum"), openmc.XPlane(R, boundary_type="vacuum"),
              openmc.YPlane(-R, boundary_type="vacuum"), openmc.YPlane(R, boundary_type="vacuum"),
              openmc.ZPlane(-R, boundary_type="vacuum"), openmc.ZPlane(R, boundary_type="vacuum")]
    world = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
    s_in, s_out = openmc.Sphere(r=10.0), openmc.Sphere(r=25.0)
    det_cyl, det_lo, det_hi = openmc.ZCylinder(x0=40.0, r=3.0), openmc.ZPlane(-8.0), openmc.ZPlane(8.0)
    detector = -det_cyl & +det_lo & -det_hi
    inner = openmc.Cell(name="Inner", fill=fuel if (eigen or absorption_in_fuel) else None, region=-s_in & world)
    shell = openmc.Cell(name="Shell", fill=poly, region=+s_in & -s_out & world)
    det = openmc.Cell(name="Detector", fill=lead, region=detector & world)
    outside = openmc.Cell(name="World", region=world & +s_out & ~detector)
    geometry = openmc.Geometry([inner, shell, det, outside])

    settings = openmc.Settings()
    settings.particles, settings.batches, settings.seed = 1000, 5, 1
    if eigen:
        settings.run_mode, settings.inactive = "eigenvalue", 2
        settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    else:
        settings.run_mode = "fixed source"
        if photon:
            settings.photon_transport = True
            settings.source = openmc.IndependentSource(
                space=openmc.stats.Box((-2, -2, -2), (2, 2, 2)), particle="photon",
                energy=openmc.stats.Uniform(1e5, 2e6))
        else:
            settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)),
                                                       energy=openmc.stats.Discrete([14.1e6], [1.0]))
    t1 = openmc.Tally(name="Detector")
    t1.filters = [openmc.CellFilter([det]), openmc.EnergyFilter([0.0, 0.625, 1e5, 2e7])]
    t1.scores = ["flux", "(n,gamma)"]
    mesh = openmc.RegularMesh()
    mesh.dimension, mesh.lower_left, mesh.upper_right = [20, 1, 20], [-R, -1, -R], [R, 1, R]
    t2 = openmc.Tally(name="Map")
    t2.filters = [openmc.MeshFilter(mesh)]
    t2.scores = ["flux"]
    tallies = [t1, t2]
    if absorption_in_fuel:
        t3 = openmc.Tally(name="Fuel absorption")
        t3.filters = [openmc.CellFilter([inner])]
        t3.scores = ["absorption"]
        tallies.append(t3)
    return openmc.Model(geometry, openmc.Materials(mats), settings, openmc.Tallies(tallies))


def rotated_model():
    """A box rotated 30 degrees about z (general planes) and a cylinder tilted off-axis (GQ quadric)."""
    import numpy as np
    openmc.reset_auto_ids()
    lead = openmc.Material(name="Lead")
    lead.set_density("g/cm3", 11.35)
    lead.add_element("Pb", 1.0)
    water = openmc.Material(name="Water")
    water.set_density("g/cm3", 1.0)
    water.add_element("H", 2.0)
    water.add_element("O", 1.0)
    water.add_s_alpha_beta("c_H_in_H2O")
    R = 50.0
    planes = [openmc.XPlane(-R, boundary_type="vacuum"), openmc.XPlane(R, boundary_type="vacuum"),
              openmc.YPlane(-R, boundary_type="vacuum"), openmc.YPlane(R, boundary_type="vacuum"),
              openmc.ZPlane(-R, boundary_type="vacuum"), openmc.ZPlane(R, boundary_type="vacuum")]
    world = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
    a = np.radians(30)
    M = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    c, h = np.array([-15.0, 0.0, 0.0]), np.array([10.0, 4.0, 6.0])
    box = None
    for k in range(3):
        n = M[:, k]
        term = +openmc.Plane(a=n[0], b=n[1], c=n[2], d=float(n @ c - h[k])) & -openmc.Plane(a=n[0], b=n[1], c=n[2], d=float(n @ c + h[k]))
        box = term if box is None else box & term
    u = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    q, r, hh = np.array([20.0, 5.0, 0.0]), 5.0, 12.0
    Pm = np.eye(3) - np.outer(u, u)
    lin = -2 * Pm @ q
    quad = openmc.Quadric(a=Pm[0, 0], b=Pm[1, 1], c=Pm[2, 2], d=2 * Pm[0, 1], e=2 * Pm[1, 2], f=2 * Pm[0, 2],
                          g=lin[0], h=lin[1], j=lin[2], k=float(q @ Pm @ q - r * r))
    cyl = -quad & +openmc.Plane(a=u[0], b=u[1], c=u[2], d=float(u @ q - hh)) & -openmc.Plane(a=u[0], b=u[1], c=u[2], d=float(u @ q + hh))
    cells = [openmc.Cell(name="Box", fill=lead, region=box & world),
             openmc.Cell(name="Cylinder", fill=water, region=cyl & world & ~box),
             openmc.Cell(name="World", region=world & ~box & ~cyl)]
    settings = openmc.Settings(run_mode="fixed source", particles=1000, batches=5, seed=1)
    settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
    return openmc.Model(openmc.Geometry(cells), openmc.Materials([lead, water]), settings)


def export_model(model, work, name):
    d = os.path.join(work, name)
    os.makedirs(d)
    model.export_to_model_xml(os.path.join(d, "model.xml"))
    with contextlib.redirect_stdout(io.StringIO()):
        report = export(os.path.join(d, "model.xml"), d, name, samples=SAMPLES)
    return report


def validate_text(text, model, work):
    path = os.path.join(work, "mutant.mcnp")
    with open(path, "w") as f:
        f.write(text)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = validate_deck(path, model=model, geometry_samples=SAMPLES)
    return ok, buf.getvalue()


def main():
    work = tempfile.mkdtemp(prefix="export_mcnp_selftest_")
    try:
        print("1. Valid exports")
        reports = {}
        for name, kw in [("fixed", {}), ("photon", {"photon": True}), ("eigen", {"eigen": True})]:
            r = export_model(shielding_model(**kw), work, name)
            reports[name] = r
            check(r["ok"], f"{name}: exported deck validates")
            check("geometry matches OpenMC" in r["validation"], f"{name}: geometry check ran and matched")

        r = export_model(rotated_model(), work, "rotated")
        check(r["ok"], "rotated box + cylinder (general planes, GQ): exported deck validates")
        check("geometry matches OpenMC" in r["validation"], "rotated: geometry check ran and matched")
        rot_text = open(r["runnable"]).read()
        check(re.search(r"^\d+ GQ ", rot_text, re.M) is not None, "rotated: cylinder written as a GQ card")
        rot_model = load_model(os.path.join(work, "rotated", "model.xml"))
        lines = rot_text.split("\n")
        gi = next(i for i, l in enumerate(lines) if re.match(r"^\d+ GQ ", l))
        gj = gi + 1
        while gj < len(lines) and lines[gj].startswith("     "):
            gj += 1
        head, last = lines[gj - 1].rstrip().rsplit(" ", 1)
        lines[gj - 1] = head + " " + repr(float(last) * 0.8)  # scale the GQ constant K
        broken = "\n".join(lines)
        ok, out = validate_text(broken, rot_model, work)
        check(broken != rot_text and not ok and "Geometry:" in out, "[rotated: GQ constant changed] rejected by the geometry check")

        photon_text = open(reports["photon"]["runnable"]).read()
        check("MODE N P" in photon_text and re.search(r"IMP:n,p=", photon_text, re.I) is not None,
              "photon: MODE N P and combined IMP:n,p importances written")
        eigen_text = open(reports["eigen"]["runnable"]).read()
        check("KCODE 1000 1.0 2 5" in eigen_text and "KSRC 0.0 0.0 0.0" in eigen_text and "SDEF" not in eigen_text,
              "eigen: KCODE/KSRC from settings, no SDEF")

        tab_m = shielding_model()
        tab_m.settings.source = [openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Tabular([0.0, 1e6, 2e6, 5e6], [0.2, 0.5, 0.3], interpolation="histogram")
        )]
        tab_rep = export_model(tab_m, work, "tabular")
        check(tab_rep["ok"], "tabular source: exported deck validates")
        tab_text = open(tab_rep["runnable"]).read()
        check("SI1 H 0.0 1.0 2.0 5.0" in tab_text and "SP1 D 0 0.2 0.5 0.3" in tab_text,
              "tabular source: SI H and SP D cards written correctly")

        print("2. Each validator check fires on a broken deck")
        base_path = reports["fixed"]["runnable"]
        base = open(base_path).read()
        model = load_model(os.path.join(work, "fixed", "model.xml"))
        graveyard = reports["fixed"]["graveyard_cell"]

        def mutate(desc, fn, expect):
            text = fn(base)
            check(text != base, f"[{desc}] mutation changed the deck")
            ok, out = validate_text(text, model, work)
            check(not ok and re.search(expect, out) is not None, f"[{desc}] rejected with /{expect}/")
            if ok or re.search(expect, out) is None:
                print("      validator said:\n" + "\n".join("      " + l for l in out.splitlines()))

        def drop_line(pattern):
            return lambda t: "\n".join(l for l in t.splitlines() if not re.match(pattern, l)) + "\n"

        def sub(pattern, repl, count=1):
            return lambda t: re.sub(pattern, repl, t, count=count, flags=re.M)

        mutate("graveyard removed", drop_line(rf"^{graveyard} 0"), r"no MCNP cell has IMP:N=0|in no MCNP cell")
        mutate("sphere radius changed", sub(r"^(\d+) SO 25\.0", r"\1 SO 26.0"), r"Geometry: point .* but MCNP cell")
        mutate("density changed", sub(r"^(\d+ \d+) -0\.93 ", r"\1 -0.95 "), r"density 0\.93 g/cm3 in OpenMC")
        mutate("complement removed (overlap)", sub(r"^(\d+ 0 [^\n]*?)\(-?\d+:-?\d+:-?\d+\)", r"\1"),
               r"several MCNP cells|but MCNP cell")
        mutate("MT card removed", drop_line(r"^MT\d+ "), r"no 'MT\d+' card")
        mutate("KCODE added to fixed source", lambda t: t + "KCODE 1000 1.0 2 5\n", r"fixed source but the deck has a KCODE")
        mutate("SDEF removed", lambda t: re.sub(r"^SDEF[^\n]*\n(     [^\n]*\n)*", "", t, flags=re.M), r"no SDEF card")
        mutate("NPS removed", drop_line(r"^NPS "), r"no NPS card")
        mutate("tally removed", drop_line(r"^F14:N "), r"Expected 3 tallies")
        mutate("tally cell renumbered", sub(r"^F4:N \d+", "F4:N 777"), r"refers to cells \['777'\]")
        mutate("universe 0 not assigned", lambda t: re.sub(r"(IMP:N=[0-9.]+)", r"U 1 \1", t), r"Universe 0|universe 1")

        pbase = photon_text
        pmodel = load_model(os.path.join(work, "photon", "model.xml"))
        text = pbase.replace("MODE N P", "MODE N")
        ok, out = validate_text(text, pmodel, work)
        check(text != pbase and not ok and "MODE has no P" in out, "[photon: P removed from MODE] rejected")

        print("3. Unsupported features are refused")
        try:
            export_model(shielding_model(absorption_in_fuel=True), work, "absorb")
            check(False, "absorption in fuel refused")
        except UnsupportedFeature as e:
            check("absorption" in str(e) and "actinides" in str(e), "absorption in fuel refused with reason")
        m = shielding_model()
        m.settings.source = [m.settings.source[0], openmc.IndependentSource()]
        try:
            export_model(m, work, "twosrc")
            check(False, "two sources refused")
        except UnsupportedFeature as e:
            check("single source" in str(e), "two sources refused with reason")
        m = shielding_model()
        m.materials[0]._sab = [("c_made_up_table", 1.0)]
        try:
            export_model(m, work, "badsab")
            check(False, "unknown S(a,b) refused")
        except UnsupportedFeature as e:
            check("c_made_up_table" in str(e) and "--sab" in str(e), "unknown S(a,b) refused with how to fix")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
