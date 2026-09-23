"""The geometry check must say when a cell got no sample points, instead of passing silently.

OpenMC can't bound a region made of tilted planes or quadrics, so such a cell's "box" is the
whole domain and a small cell in a large world can get no points at all. Here a 2 cm cube,
turned 30 degrees, sits in a 200 cm world: check_geometry must list it as unsampled. The same
cube with an axis-aligned box added to its region (geometrically a no-op) must be compared.

Needs openmc and MontePy, not MCNPy. Run: python tests/test_geometry_coverage.py
"""
import math
import pathlib
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import montepy  # noqa: E402
import openmc  # noqa: E402
from geometry_check import check_geometry  # noqa: E402

C, S = math.cos(math.radians(30)), math.sin(math.radians(30))
# A cube of half-width 1 cm at the origin, turned 30 degrees about z: three pairs of planes.
CUBE = [("1", (C, S, 0, 1)), ("2", (C, S, 0, -1)), ("3", (-S, C, 0, 1)), ("4", (-S, C, 0, -1)),
        ("5", (0, 0, 1, 1)), ("6", (0, 0, 1, -1))]


def model_and_deck(with_box):
    openmc.reset_auto_ids()
    s = {n: openmc.Plane(a=a, b=b, c=c, d=d, surface_id=int(n)) for n, (a, b, c, d) in CUBE}
    world = [openmc.XPlane(-100, surface_id=7, boundary_type="vacuum"), openmc.XPlane(100, surface_id=8, boundary_type="vacuum"),
             openmc.YPlane(-100, surface_id=9, boundary_type="vacuum"), openmc.YPlane(100, surface_id=10, boundary_type="vacuum"),
             openmc.ZPlane(-100, surface_id=11, boundary_type="vacuum"), openmc.ZPlane(100, surface_id=12, boundary_type="vacuum")]
    box = [openmc.XPlane(-1.5, surface_id=13), openmc.XPlane(1.5, surface_id=14), openmc.YPlane(-1.5, surface_id=15),
           openmc.YPlane(1.5, surface_id=16), openmc.ZPlane(-1.5, surface_id=17), openmc.ZPlane(1.5, surface_id=18)]
    cube = -s["1"] & +s["2"] & -s["3"] & +s["4"] & -s["5"] & +s["6"]
    if with_box:
        cube = cube & +box[0] & -box[1] & +box[2] & -box[3] & +box[4] & -box[5]
    inside = +world[0] & -world[1] & +world[2] & -world[3] & +world[4] & -world[5]
    graphite = openmc.Material(material_id=1, name="graphite")
    graphite.set_density("g/cm3", 1.7)
    graphite.add_nuclide("C12", 1.0)
    cells = [openmc.Cell(cell_id=1, name="tilted cube", fill=graphite, region=cube),
             openmc.Cell(cell_id=2, name="World", region=inside & ~cube)]
    geometry = openmc.Geometry(cells)
    surf = "\n".join(f"{n} P {a:.15g} {b:.15g} {c:.15g} {d:.15g}" for n, (a, b, c, d) in CUBE)
    surf += "\n*7 PX -100\n*8 PX 100\n*9 PY -100\n*10 PY 100\n*11 PZ -100\n*12 PZ 100"
    boxcards = ""
    if with_box:
        surf += "\n13 PX -1.5\n14 PX 1.5\n15 PY -1.5\n16 PY 1.5\n17 PZ -1.5\n18 PZ 1.5"
        boxcards = " 13 -14 15 -16 17 -18"
    cube_mcnp = "-1 2 -3 4 -5 6" + boxcards
    deck = (f"coverage test\n1 1 -1.7 {cube_mcnp} IMP:N=1\n"
            f"2 0 7 -8 9 -10 11 -12 #1 IMP:N=1\n3 0 -7:8:-9:10:-11:12 IMP:N=0\n\n"
            f"{surf}\n\nM1 6012.80c 1\nMODE N\nNPS 10\n")
    return geometry, deck


class GeometryCoverage(unittest.TestCase):
    def run_check(self, with_box):
        geometry, deck = model_and_deck(with_box)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "deck.mcnp"
            path.write_text(deck)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem = montepy.read_input(str(path))
            return check_geometry(problem, geometry)

    def test_a_small_tilted_cell_in_a_big_world_is_reported_unsampled(self):
        g = self.run_check(with_box=False)
        self.assertTrue(g["ok"], g["errors"])            # nothing it looked at was wrong ...
        self.assertEqual(g["unsampled_cells"], [(1, "tilted cube")])  # ... but it never looked at the cube
        self.assertFalse(g["cell_hits"].get(1))

    def test_the_same_cell_with_an_axis_box_is_compared(self):
        g = self.run_check(with_box=True)
        self.assertTrue(g["ok"], g["errors"])
        self.assertEqual(g["unsampled_cells"], [])
        self.assertGreater(g["cell_hits"][1], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
