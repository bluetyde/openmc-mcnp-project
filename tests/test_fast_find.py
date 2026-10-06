"""The geometry check finds each sample point's OpenMC cell with openmc.lib (C++) instead of Geometry.find (Python).

That is only allowed to change how fast the answer comes, so these tests compare the two on every point:
  - lib_cell_ids against Geometry.find on a model with a lattice of pins, outside points and a nested universe;
  - check_geometry with and without `model` on the tilted-cube deck of test_geometry_coverage;
  - the fallback: with the nuclear data library missing, openmc.lib can't start; the check must still run in
    Python and say so (result["find"]), not fail and not pretend;
  - the comparison still bites: wrong cell ids from the fast path are reported as errors.

Needs openmc, MontePy and the nuclear data library (OPENMC_CROSS_SECTIONS), not MCNPy.
Run: python tests/test_fast_find.py
"""
import os
import pathlib
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import montepy  # noqa: E402
import openmc  # noqa: E402
import geometry_check  # noqa: E402
from geometry_check import check_geometry, lib_cell_ids  # noqa: E402
from test_geometry_coverage import model_and_deck  # noqa: E402


def lattice_model():
    """Water box holding a 3 x 3 lattice of fuel pins, in a vacuum-bounded world; the pin universe is nested."""
    openmc.reset_auto_ids()
    fuel = openmc.Material(material_id=1, name="fuel")
    fuel.set_density("g/cm3", 10.0)
    fuel.add_nuclide("U238", 1.0)
    water = openmc.Material(material_id=2, name="water")
    water.set_density("g/cm3", 1.0)
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    rod = openmc.ZCylinder(r=0.4)
    pin = openmc.Universe(cells=[openmc.Cell(fill=fuel, region=-rod), openmc.Cell(fill=water, region=+rod)])
    lat = openmc.RectLattice()
    lat.lower_left, lat.pitch, lat.universes = (-1.5, -1.5), (1.0, 1.0), [[pin] * 3] * 3
    box = [openmc.XPlane(-1.5), openmc.XPlane(1.5), openmc.YPlane(-1.5), openmc.YPlane(1.5),
           openmc.ZPlane(-2.0), openmc.ZPlane(2.0)]
    world = [openmc.XPlane(-5, boundary_type="vacuum"), openmc.XPlane(5, boundary_type="vacuum"),
             openmc.YPlane(-5, boundary_type="vacuum"), openmc.YPlane(5, boundary_type="vacuum"),
             openmc.ZPlane(-5, boundary_type="vacuum"), openmc.ZPlane(5, boundary_type="vacuum")]
    inner = +box[0] & -box[1] & +box[2] & -box[3] & +box[4] & -box[5]
    outer = +world[0] & -world[1] & +world[2] & -world[3] & +world[4] & -world[5]
    cells = [openmc.Cell(name="array", fill=lat, region=inner), openmc.Cell(name="water around", fill=water, region=outer & ~inner)]
    geometry = openmc.Geometry(cells)
    model = openmc.Model(geometry, openmc.Materials([fuel, water]))
    model.settings.run_mode = "fixed source"
    model.settings.particles, model.settings.batches = 10, 1
    return model


def python_ids(geometry, P):
    ids = []
    for p in P:
        found = geometry.find(tuple(p))
        ids.append(found[-1].id if found and isinstance(found[-1], openmc.Cell) else -1)
    return ids


class FastFind(unittest.TestCase):
    def test_lib_ids_equal_python_find_on_a_lattice_model(self):
        model = lattice_model()
        P = np.random.default_rng(7).uniform(-6, 6, size=(3000, 3))  # includes points outside the world
        ids, why = lib_cell_ids(model, P)
        self.assertIsNone(why)
        self.assertEqual(list(ids), python_ids(model.geometry, P))
        self.assertTrue((ids == -1).any() and (ids > 0).any(), "the points must cover outside and inside")

    def coverage_model(self, with_box):
        geometry, deck = model_and_deck(with_box)
        model = openmc.Model(geometry, openmc.Materials(list(geometry.get_all_materials().values())))
        model.settings.run_mode = "fixed source"
        model.settings.particles, model.settings.batches = 10, 1
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "deck.mcnp"
            path.write_text(deck)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem = montepy.read_input(str(path))
        return model, problem

    def test_check_geometry_gives_the_same_result_either_way(self):
        for with_box in (False, True):
            model, problem = self.coverage_model(with_box)
            slow = check_geometry(problem, model.geometry)
            fast = check_geometry(problem, model.geometry, model=model)
            self.assertEqual(slow["find"], "python")
            self.assertEqual(fast["find"], "openmc.lib")
            for key in ("ok", "checked_points", "errors", "skipped_near_surface", "unsampled_cells", "cell_hits"):
                self.assertEqual(slow[key], fast[key], key)
            self.assertGreater(fast["checked_points"], 20000)

    def test_missing_nuclear_data_falls_back_to_python_and_says_so(self):
        model, problem = self.coverage_model(True)
        old = os.environ.get("OPENMC_CROSS_SECTIONS")
        os.environ["OPENMC_CROSS_SECTIONS"] = os.path.join(tempfile.gettempdir(), "no-such-library", "cross_sections.xml")
        try:
            ids, why = lib_cell_ids(model, np.zeros((3, 3)))
            fast = check_geometry(problem, model.geometry, model=model)
        finally:
            if old is None:
                del os.environ["OPENMC_CROSS_SECTIONS"]
            else:
                os.environ["OPENMC_CROSS_SECTIONS"] = old
        self.assertIsNone(ids)
        self.assertIn("openmc.lib could not load the model", why)
        self.assertTrue(fast["find"].startswith("python (openmc.lib could not load the model"), fast["find"])
        self.assertTrue(fast["ok"], fast["errors"])  # the Python find still ran the whole check

    def test_wrong_cell_ids_from_the_fast_path_are_reported(self):
        model, problem = self.coverage_model(True)
        real = geometry_check.lib_cell_ids

        def swapped(m, P, timeout=1800):
            ids, why = real(m, P, timeout)
            return np.where(ids == 1, 2, np.where(ids == 2, 1, ids)), why
        geometry_check.lib_cell_ids = swapped
        try:
            g = check_geometry(problem, model.geometry, model=model)
        finally:
            geometry_check.lib_cell_ids = real
        self.assertEqual(g["find"], "openmc.lib")
        self.assertFalse(g["ok"])
        self.assertTrue(any("OpenMC cell" in e for e in g["errors"]), g["errors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
