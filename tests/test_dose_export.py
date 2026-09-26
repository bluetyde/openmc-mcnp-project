"""Dose tallies from OpenMC Studio exported to MCNP: DE/DF, SD, FM / FACTOR, and photon F4/FMESH.

A small model (a water tank around a D-T point source, a void detector sphere, photon transport on) carries
the tallies Studio writes for dose: flux x a log-log EnergyFunctionFilter of ICRP coefficients, one tally per
particle, on a cell and on a mesh. With Studio's dose description (dose.json's format: tallies, cell volumes,
source rate), the deck must carry
  - DEn LOG / DFn LOG with exactly the filter's numbers (energies in MeV),
  - SDn with Studio's cell volume, so MCNP divides by the same volume as Studio,
  - FMn (cells) or FACTOR= (FMESH) equal to source rate x 3600 x 1e-12, so the deck prints Sv/h,
  - F4:P / FMESH:P for the photon tallies,
and still validate. Without a volume, or with a non-log-log table, the export is refused with a reason.

Requires the MCNPy gateway (port 25333, claim it first where agents share a machine).
Run: python tests/test_dose_export.py
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import openmc
import openmc.data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_mcnp import export  # noqa: E402

RATE = 1e8


def dose_filter(particle, lowest):
    e, c = openmc.data.dose_coefficients(particle, geometry="AP", data_source="icrp116")
    e, c = [lowest] + list(e), [c[0]] + list(c)
    return openmc.EnergyFunctionFilter(e, c, interpolation="log-log")


def build(work, interpolation="log-log"):
    openmc.reset_auto_ids()
    water = openmc.Material(name="water")
    water.add_element("H", 2)
    water.add_element("O", 1)
    water.set_density("g/cm3", 1.0)
    world = openmc.model.RectangularParallelepiped(-40, 40, -40, 40, -40, 40, boundary_type="vacuum")
    tank = openmc.model.RectangularParallelepiped(-15, 15, -15, 15, -15, 15)
    det = openmc.Sphere(x0=25, r=2)
    c_tank = openmc.Cell(name="tank", fill=water, region=-tank)
    c_det = openmc.Cell(name="detector", region=-det)
    c_out = openmc.Cell(name="outside", region=-world & +tank & +det)
    geometry = openmc.Geometry([c_tank, c_det, c_out])
    s = openmc.Settings()
    s.run_mode, s.batches, s.particles, s.photon_transport = "fixed source", 10, 1000, True
    s.source = openmc.IndependentSource(space=openmc.stats.Point(), energy=openmc.stats.Discrete([14.1e6], [1]))
    fn = dose_filter("neutron", 1e-5)
    if interpolation != "log-log":
        fn.interpolation = interpolation
    t_n = openmc.Tally(name="Dose [neutron dose]")
    t_n.filters = [openmc.CellFilter([c_det]), openmc.ParticleFilter(["neutron"]), fn]
    t_n.scores = ["flux"]
    t_p = openmc.Tally(name="Dose [photon dose]")
    t_p.filters = [openmc.CellFilter([c_det]), openmc.ParticleFilter(["photon"]), dose_filter("photon", 1000.0)]
    t_p.scores = ["flux"]
    mesh = openmc.RegularMesh()
    mesh.dimension, mesh.lower_left, mesh.upper_right = (3, 3, 3), (15, -15, -15), (45, 15, 15)
    t_m = openmc.Tally(name="Map [neutron dose]")
    t_m.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter(["neutron"]), dose_filter("neutron", 1e-5)]
    t_m.scores = ["flux"]
    model = openmc.Model(geometry, openmc.Materials([water]), s, openmc.Tallies([t_n, t_p, t_m]))
    path = os.path.join(work, "model.xml")
    model.export_to_model_xml(path)
    meta = lambda p, name, mesh=False: {"studio": "t", "name": name, "particle": p, "data": "icrp116", "geometry": "AP",
                                        **({"mesh": True} if mesh else {})}
    dose = {"tallies": {str(t_n.id): meta("neutron", "Dose"), str(t_p.id): meta("photon", "Dose"),
                        str(t_m.id): meta("neutron", "Map", True)},
            "volumes": {str(c_det.id): [33.514, 0.071]}, "source_rate": RATE}
    return path, dose, {"n": t_n, "p": t_p, "m": t_m}, c_det


def numbers(deck, head):
    """The numbers on a card and its continuation lines (a card starting with `head`)."""
    lines = deck.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(head + " ") or l == head)
    words = lines[i].split()[1:]
    for l in lines[i + 1:]:
        if not l.startswith("     "):
            break
        words += l.split()
    return [float(w) for w in words if re.fullmatch(r"[-+0-9.eE]+", w)]


class DoseExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="dose-export-")
        path, cls.dose, cls.tallies, cls.det = build(cls.work)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = export(path, os.path.join(cls.work, "deck"), "dose", samples=4000, dose=cls.dose)
        cls.deck = Path(cls.report["runnable"]).read_text()

    def test_deck_validates(self):
        self.assertTrue(self.report["ok"], self.report.get("validation", "")[-2000:])

    def test_tally_cards(self):
        d = self.deck
        self.assertRegex(d, r"(?m)^F4:N " + str(self.det.id) + r"\b")
        self.assertRegex(d, r"(?m)^F14:P " + str(self.det.id) + r"\b")
        self.assertRegex(d, r"(?m)^FMESH24:N GEOM=XYZ")
        self.assertRegex(d, r"(?m)^FC4 Dose: neutron effective dose, Sv/h \(ICRP116 AP\)")
        self.assertRegex(d, r"(?m)^FC14 Dose: photon effective dose, Sv/h")

    def test_de_df_are_the_filters_numbers(self):
        for n, t in ((4, self.tallies["n"]), (14, self.tallies["p"]), (24, self.tallies["m"])):
            f = t.filters[-1]
            de, df = numbers(self.deck, f"DE{n}"), numbers(self.deck, f"DF{n}")
            self.assertEqual(len(de), len(f.energy))
            for a, b in zip(de, f.energy):
                self.assertAlmostEqual(a, b / 1e6, delta=abs(b / 1e6) * 1e-12)
            self.assertEqual(df, [float(y) for y in f.y])

    def test_volume_and_rate(self):
        self.assertEqual(numbers(self.deck, "SD4"), [33.514])
        self.assertEqual(numbers(self.deck, "SD14"), [33.514])
        factor = RATE * 3600e-12
        self.assertAlmostEqual(numbers(self.deck, "FM4")[0], factor, delta=factor * 1e-12)
        self.assertRegex(self.deck, r"(?m)^     FACTOR=" + re.escape(repr(factor)) + "$")
        self.assertNotRegex(self.deck, r"(?m)^SD24")  # MCNP divides FMESH results by voxel volume itself

    def test_lines_fit_128_columns(self):
        self.assertTrue(all(len(l) <= 128 for l in self.deck.splitlines()))

    def test_refusals(self):
        """No volume for a dosed cell, or a table that isn't log-log: refused with a reason, never guessed."""
        from mcnp_cards import UnsupportedFeature
        work = tempfile.mkdtemp(prefix="dose-export-bad-")
        path, dose, _, _ = build(work)
        dose["volumes"] = {}
        with self.assertRaisesRegex(UnsupportedFeature, "no volume for cell"), contextlib.redirect_stdout(io.StringIO()):
            export(path, os.path.join(work, "deck"), "bad", samples=0, dose=dose)
        work = tempfile.mkdtemp(prefix="dose-export-lin-")
        path, dose, _, _ = build(work, interpolation="linear-linear")
        with self.assertRaisesRegex(UnsupportedFeature, "LOG LOG"), contextlib.redirect_stdout(io.StringIO()):
            export(path, os.path.join(work, "deck"), "lin", samples=0, dose=dose)

    def test_log_keyword_left_implicit(self):
        """LOG is MCNP's default for DE/DF (p. 465) and MontePy 1.1.3 can't parse it, so it isn't written."""
        cards = [l for l in self.deck.splitlines() if l.startswith(("DE", "DF"))]
        self.assertEqual(len(cards), 6)
        self.assertFalse(any("LOG" in l for l in cards), cards)


    def test_command_line(self):
        """export_mcnp.py writes its JSON report with and without --dose (a Studio run's dose.json)."""
        import json
        import subprocess
        work = tempfile.mkdtemp(prefix="dose-export-cli-")
        path, dose, _, _ = build(work)
        Path(work, "dose.json").write_text(json.dumps(dose))
        cli = str(Path(__file__).resolve().parents[1] / "src" / "export_mcnp.py")
        for extra, cards in (([], False), (["--dose", str(Path(work, "dose.json"))], True)):
            report = Path(work, f"report{len(extra)}.json")
            subprocess.run([sys.executable, cli, path, "--out-dir", str(Path(work, f"deck{len(extra)}")), "--name", "m",
                            "--samples", "0", "--report", str(report)] + extra, capture_output=True, text=True, timeout=900)
            r = json.loads(report.read_text())
            if cards:
                self.assertTrue(r["ok"], r.get("error"))
                self.assertIn("SD4 33.514", Path(r["runnable"]).read_text())
            else:  # no dose description: the energy function has no meaning for MCNP, refused with a reason
                self.assertFalse(r["ok"])
                self.assertIn("Not exportable", r["error"])

def build_lattice(work):
    """Two void pins in a 2 x 1 RectLattice inside a water tank; a dose tally on each pin (CellInstanceFilter)."""
    openmc.reset_auto_ids()
    water = openmc.Material(name="water")
    water.add_element("H", 2)
    water.add_element("O", 1)
    water.set_density("g/cm3", 1.0)
    pin = openmc.ZCylinder(r=1.0)
    c_pin = openmc.Cell(name="pin", region=-pin)
    c_mod = openmc.Cell(name="mod", fill=water, region=+pin)
    u = openmc.Universe(cells=[c_pin, c_mod])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch, lat.universes = (-4, -2), (4, 4), [[u, u]]
    box = openmc.model.RectangularParallelepiped(-4, 4, -2, 2, -5, 5)
    world = openmc.model.RectangularParallelepiped(-20, 20, -20, 20, -20, 20, boundary_type="vacuum")
    c_lat = openmc.Cell(name="lattice", fill=lat, region=-box)
    c_out = openmc.Cell(name="outside", fill=water, region=-world & +box)
    geometry = openmc.Geometry([c_lat, c_out])
    geometry.determine_paths()
    s = openmc.Settings()
    s.run_mode, s.batches, s.particles = "fixed source", 10, 1000
    s.source = openmc.IndependentSource(space=openmc.stats.Point((0, 10, 0)), energy=openmc.stats.Discrete([14.1e6], [1]))
    t = openmc.Tally(name="Pins [neutron dose]")
    t.filters = [openmc.CellInstanceFilter([(c_pin, 0), (c_pin, 1)]), openmc.ParticleFilter(["neutron"]),
                 dose_filter("neutron", 1e-5)]
    t.scores = ["flux"]
    model = openmc.Model(geometry, openmc.Materials([water]), s, openmc.Tallies([t]))
    path = os.path.join(work, "model.xml")
    model.export_to_model_xml(path)
    dose = {"tallies": {str(t.id): {"studio": "t", "name": "Pins", "particle": "neutron", "data": "icrp116",
                                    "geometry": "AP"}},
            "volumes": {f"{c_pin.id}/0": [31.1, 0.1], f"{c_pin.id}/1": [31.7, 0.1]}, "source_rate": None}
    return path, dose


class DoseInLattice(unittest.TestCase):
    """A dose tally on parts inside a lattice: chain bins, and each element's own volume on SD (keyed
    "cell/instance" in Studio's dose description)."""

    def test_chain_bins_carry_each_instance_volume(self):
        work = tempfile.mkdtemp(prefix="dose-export-lat-")
        path, dose = build_lattice(work)
        with contextlib.redirect_stdout(io.StringIO()):
            report = export(path, os.path.join(work, "deck"), "lat", samples=2000, dose=dose)
        deck = Path(report["runnable"]).read_text()
        self.assertTrue(report["ok"], report.get("validation", "")[-2000:])
        f4 = next(l for l in deck.splitlines() if l.startswith("F4:N"))
        self.assertEqual(f4.count("<"), 4, f4)
        self.assertEqual(numbers(deck, "SD4"), [31.1, 31.7])


if __name__ == "__main__":
    unittest.main(verbosity=2)
