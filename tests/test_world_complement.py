"""The rest-of-the-world cell is written as #cell complements, which MCNPy translates in far fewer steps.

OpenMC Studio's world cell is `box & ~(part | part | ...)`; once written to XML and read back it is the box plus
one parenthesised union per cell. MCNPy takes a round trip to Java for every term (about 3.0 million calls for a
267-cell import, 0.9 million of them for that one cell), so world_complement.plan() finds the cell and the deck
gets `#1 #2 ...` instead. These tests check that it finds the right cells and nothing else, that the card text
is edited correctly, and (needs the MCNPy gateway, port 25333: claim it where agents share a machine) that a
model exports and validates, point by point against OpenMC, both ways.

Run: python tests/test_world_complement.py
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import openmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import world_complement  # noqa: E402
from export_mcnp import export  # noqa: E402
from remediate_deck import load_model  # noqa: E402


def grid_model(n_x=3, n_y=4, extra_clause=False):
    """n_x x n_y boxes of two materials in a vacuum-bounded world, built the way Studio writes it: every cell is its
    part's region and the world box, and the world cell is box & ~(union of the parts)."""
    openmc.reset_auto_ids()
    graphite = openmc.Material(name="graphite")
    graphite.set_density("g/cm3", 1.7)
    graphite.add_nuclide("C12", 1.0)
    water = openmc.Material(name="water")
    water.set_density("g/cm3", 1.0)
    water.add_nuclide("H1", 2.0)
    water.add_nuclide("O16", 1.0)
    box = [openmc.XPlane(-50, boundary_type="vacuum"), openmc.XPlane(50, boundary_type="vacuum"),
           openmc.YPlane(-50, boundary_type="vacuum"), openmc.YPlane(50, boundary_type="vacuum"),
           openmc.ZPlane(-50, boundary_type="vacuum"), openmc.ZPlane(50, boundary_type="vacuum")]
    world = +box[0] & -box[1] & +box[2] & -box[3] & +box[4] & -box[5]
    cells, parts = [], []
    for i in range(n_x):
        for j in range(n_y):
            x0, y0 = -40 + 25 * i, -40 + 20 * j
            planes = [openmc.XPlane(x0), openmc.XPlane(x0 + 20), openmc.YPlane(y0), openmc.YPlane(y0 + 15),
                      openmc.ZPlane(-10), openmc.ZPlane(10)]
            part = +planes[0] & -planes[1] & +planes[2] & -planes[3] & +planes[4] & -planes[5]
            parts.append(part)
            cells.append(openmc.Cell(name=f"box {i},{j}", fill=graphite if (i + j) % 2 else water, region=part & world))
    outside = world & ~openmc.Union(parts)
    if extra_clause:  # a clause that is not the complement of any cell: it must stay as it is
        outside = outside & (+openmc.XPlane(-45) | -openmc.YPlane(45))
    geometry = openmc.Geometry(cells + [openmc.Cell(name="outside", region=outside)])
    s = openmc.Settings()
    s.run_mode, s.batches, s.particles = "fixed source", 5, 1000
    s.source = openmc.IndependentSource(space=openmc.stats.Point(), energy=openmc.stats.Discrete([2e6], [1]))
    return openmc.Model(geometry, openmc.Materials([graphite, water]), s)


def reloaded(model, work):
    """The model as the exporter reads it: through model.xml, where the complement is expanded."""
    path = os.path.join(work, "model.xml")
    with contextlib.redirect_stdout(io.StringIO()):
        model.export_to_model_xml(path)
    return path, load_model(path)


class FindsTheWorldCell(unittest.TestCase):
    def test_every_part_is_complemented_and_the_box_is_kept(self):
        with tempfile.TemporaryDirectory() as work:
            _, m = reloaded(grid_model(), work)
        wc = world_complement.plan(m.geometry, min_terms=0)
        self.assertEqual(wc["cell"].name, "outside")
        self.assertEqual(wc["ids"], sorted(c.id for c in m.geometry.root_universe.cells.values() if c.name != "outside"))
        self.assertEqual(len(wc["ids"]), 12)
        self.assertEqual(wc["terms"], 12 * 6)
        self.assertEqual(sorted((h.surface.id, h.side) for h in wc["kept"]),
                         sorted((h.surface.id, h.side) for h in
                                [n for n in m.geometry.root_universe.cells[wc["cell"].id].region if isinstance(n, openmc.Halfspace)]))

    def test_small_models_keep_their_expanded_form(self):
        with tempfile.TemporaryDirectory() as work:
            _, m = reloaded(grid_model(), work)
        self.assertIsNone(world_complement.plan(m.geometry))  # 72 terms, below MIN_TERMS

    def test_a_clause_that_is_no_cells_complement_is_kept(self):
        with tempfile.TemporaryDirectory() as work:
            _, m = reloaded(grid_model(extra_clause=True), work)
        wc = world_complement.plan(m.geometry, min_terms=0)
        self.assertEqual(len(wc["ids"]), 12)
        kept = list(wc["kept"])
        self.assertEqual(sum(isinstance(n, openmc.Union) for n in kept), 1, "the extra clause stays for MCNPy")

    def test_a_cell_that_is_not_the_complement_of_the_rest_is_not_touched(self):
        with tempfile.TemporaryDirectory() as work:
            _, m = reloaded(grid_model(), work)
            cells = m.geometry.root_universe.cells
            world = next(c for c in cells.values() if c.name == "outside")
            victim = next(c for c in cells.values() if c.name == "box 1,1")
            # a clause for a region that matches no cell: shrink one cell so its complement differs
            victim.region = victim.region & +openmc.XPlane(-100.5)
            victim.region = victim.region & -openmc.XPlane(-5.5)
            wc = world_complement.plan(m.geometry, min_terms=0)
        self.assertNotIn(victim.id, wc["ids"])
        self.assertEqual(len(wc["ids"]), 11)
        self.assertIs(wc["cell"], world)


class PatchesTheCard(unittest.TestCase):
    TEXT = ("title\n1 1 -1.7 1 -2 3 -4 5 -6 U 1\n     IMP:N=1.0 \n"
            "7 0 1 -2 3 -4 5 -6\n     (-7:8:-9:10)\n     IMP:N=1.0 \n8 0 9 -10 IMP:N=1\n\n1 PX 1\n")

    def test_complements_go_before_the_keywords_and_lines_stay_short(self):
        wc = {"cell": SimpleNamespace(id=7), "ids": list(range(100, 160))}
        out = world_complement.patch_deck(self.TEXT, wc).split("\n")
        self.assertEqual(out[:3], self.TEXT.split("\n")[:3], "other cells are untouched")
        self.assertEqual(out[-4:], self.TEXT.split("\n")[-4:])
        start = out.index(next(l for l in out if l.startswith("7 0 ")))
        end = next(i for i in range(start + 1, len(out)) if out[i].startswith("8 0 "))
        card = out[start:end]
        self.assertTrue(all(len(l) <= world_complement.LINE for l in card), card)
        self.assertTrue(all(l.startswith("     ") for l in card[1:]))
        tokens = " ".join(card).split()
        self.assertEqual(tokens[:11], ["7", "0", "1", "-2", "3", "-4", "5", "-6", "(-7:8:-9:10)", "#100", "#101"])
        self.assertEqual([t for t in tokens if t.startswith("#")], [f"#{i}" for i in range(100, 160)])
        self.assertEqual(tokens[-1], "IMP:N=1.0")
        self.assertGreater(tokens.index("IMP:N=1.0"), tokens.index("#159"))

    def test_a_cell_missing_from_the_deck_is_an_error(self):
        with self.assertRaises(ValueError):
            world_complement.patch_deck(self.TEXT, {"cell": SimpleNamespace(id=99), "ids": [1]})


class ExportsBothWays(unittest.TestCase):
    """Needs MCNPy. The same model with and without the complements must validate against OpenMC."""

    def export(self, min_terms):
        old = world_complement.MIN_TERMS
        world_complement.MIN_TERMS = min_terms
        try:
            with tempfile.TemporaryDirectory() as work:
                path, _ = reloaded(grid_model(), work)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = export(path, os.path.join(work, "deck"), "wc", samples=3000)
                return r, Path(r["runnable"]).read_text()
        finally:
            world_complement.MIN_TERMS = old

    def test_deck_validates_with_and_without_complements(self):
        with_wc, deck_wc = self.export(0)
        plain, deck_plain = self.export(10 ** 9)
        self.assertTrue(with_wc["ok"], with_wc.get("validation", "")[-1500:])
        self.assertTrue(plain["ok"], plain.get("validation", "")[-1500:])
        self.assertTrue(any("#cell" in n for n in with_wc["notes"]), with_wc["notes"])
        self.assertFalse(any("#cell" in n for n in plain["notes"]))
        cells_wc, cells_plain = deck_wc.split("\n\n")[0], deck_plain.split("\n\n")[0]
        # remediate's own graveyard cell (IMP:N=0) is already written as #cell complements in both decks, so the
        # difference is the world cell's 12 complements
        self.assertEqual(len(re.findall(r"#\d+", cells_wc)) - len(re.findall(r"#\d+", cells_plain)), 12)
        self.assertEqual(len(re.findall(r"(?m)^\d+ ", cells_wc)), len(re.findall(r"(?m)^\d+ ", cells_plain)))
        self.assertIn("geometry matches OpenMC", with_wc["validation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
