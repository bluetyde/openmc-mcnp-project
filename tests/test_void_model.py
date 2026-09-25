"""A model with no materials (everything void) exports: MCNPy writes no data block for it, and remediation must
start one rather than refuse ("doesn't have cell, surface and data blocks") or run the data cards into the
surface block. Found 2026-09-25 with OpenMC Studio's all-void dose test models.

Requires the MCNPy gateway (port 25333, claim it first where agents share a machine).
Run: python tests/test_void_model.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import openmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_mcnp import export  # noqa: E402


class VoidModel(unittest.TestCase):
    def test_all_void_model_exports_and_validates(self):
        openmc.reset_auto_ids()
        world = openmc.model.RectangularParallelepiped(-20, 20, -20, 20, -20, 20, boundary_type="vacuum")
        ball = openmc.Sphere(x0=10, r=2)
        c_ball = openmc.Cell(name="detector", region=-ball)
        geometry = openmc.Geometry([c_ball, openmc.Cell(name="rest", region=-world & +ball)])
        s = openmc.Settings()
        s.run_mode, s.batches, s.particles = "fixed source", 5, 1000
        s.source = openmc.IndependentSource(space=openmc.stats.Point(), energy=openmc.stats.Discrete([2e6], [1]))
        t = openmc.Tally(name="flux in the detector")
        t.filters = [openmc.CellFilter([c_ball])]
        t.scores = ["flux"]
        work = tempfile.mkdtemp(prefix="void-model-")
        path = os.path.join(work, "model.xml")
        openmc.Model(geometry, openmc.Materials(), s, openmc.Tallies([t])).export_to_model_xml(path)
        with contextlib.redirect_stdout(io.StringIO()):
            r = export(path, os.path.join(work, "deck"), "void", samples=2000)
        self.assertTrue(r["ok"], r.get("validation", "")[-1500:])
        deck = Path(r["runnable"]).read_text()
        self.assertRegex(deck, r"(?m)^MODE N$")
        self.assertRegex(deck, r"(?m)^SDEF ")
        self.assertRegex(deck, r"(?m)^F4:N \d+$")
        self.assertNotRegex(deck, r"(?m)^M\d+\s", "no materials to write")
        blocks = [b for b in deck.strip().split("\n\n")]
        self.assertEqual(len(blocks), 3, "cells, surfaces, data: two blank lines")


if __name__ == "__main__":
    unittest.main(verbosity=2)
