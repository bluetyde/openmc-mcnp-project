"""OpenMC tori (XTorus, YTorus, ZTorus) are written as MCNP TX, TY and TZ cards; MCNPy cannot translate them itself.

MCNPy fails on a torus (no MCNP card for the class: an AttributeError on None), so torus_cards.py hands it a stand-in sphere and rewrites the card.
The pure tests need only OpenMC. The export tests need the MCNPy gateway (port 25333: claim it first where agents share a machine). They check the card, the point-by-point
comparison with OpenMC (the torus cell must actually receive sample points), and three deliberate breaks of the written deck that the comparison must catch: B and C swapped
(the two minor radii, which a whole-torus volume cannot tell apart), the wrong axis, and the wrong centre.

Run: python tests/test_torus_cards.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torus_cards  # noqa: E402
from export_mcnp import export  # noqa: E402
from validate_deck import validate_deck  # noqa: E402


def torus_model(cls=openmc.ZTorus, centre=(0.0, 0.0, 0.0), a=10.0, b=4.0, c=3.0, two=False):
    openmc.reset_auto_ids()
    iron = openmc.Material(name="Iron")
    iron.set_density("g/cm3", 7.87)
    iron.add_element("Fe", 1.0)
    world = openmc.model.RectangularParallelepiped(-25, 25, -25, 25, -25, 25, boundary_type="vacuum")
    t = cls(x0=centre[0], y0=centre[1], z0=centre[2], a=a, b=b, c=c)
    cells = [openmc.Cell(name="ring", fill=iron, region=-t)]
    if two:
        t2 = openmc.ZTorus(x0=0, y0=0, z0=0, a=a, b=1.0, c=1.0)
        cells = [openmc.Cell(name="inner", fill=iron, region=-t2), openmc.Cell(name="ring", fill=iron, region=-t & +t2)]
    rest = openmc.Cell(name="rest", region=-world & +t)
    s = openmc.Settings()
    s.run_mode, s.batches, s.particles = "fixed source", 5, 1000
    s.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)), energy=openmc.stats.Discrete([2e6], [1]))
    return openmc.Model(openmc.Geometry(cells + [rest]), openmc.Materials([iron]), s, openmc.Tallies()), [t]


class Pure(unittest.TestCase):
    def test_plan_finds_each_torus_with_its_card(self):
        for cls, card in ((openmc.XTorus, "TX"), (openmc.YTorus, "TY"), (openmc.ZTorus, "TZ")):
            model, _ = torus_model(cls, centre=(1.0, -2.0, 3.0), a=10.0, b=4.0, c=3.0)
            tp = torus_cards.plan(model.geometry)
            self.assertEqual(len(tp), 1)
            self.assertEqual(tp[0]["card"], card)
            self.assertEqual(tp[0]["values"], [1.0, -2.0, 3.0, 10.0, 4.0, 3.0])

    def test_a_zero_or_negative_radius_is_refused(self):
        for a, b, c in ((10.0, 0.0, 3.0), (10.0, -4.0, 3.0), (10.0, 4.0, 0.0), (10.0, 4.0, -3.0), (0.0, 4.0, 3.0), (-10.0, 4.0, 3.0)):
            with self.assertRaises(ValueError, msg=(a, b, c)):       # OpenMC itself may already refuse some of these; either way no deck is written
                model, _ = torus_model(a=a, b=b, c=c)
                torus_cards.plan(model.geometry)

    def test_no_torus_no_plan(self):
        openmc.reset_auto_ids()
        s = openmc.Sphere(r=3, boundary_type="vacuum")
        self.assertIsNone(torus_cards.plan(openmc.Geometry([openmc.Cell(region=-s)])))

    def test_a_torus_that_closes_on_itself_is_refused(self):
        model, _ = torus_model(a=2.0, b=1.0, c=3.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            torus_cards.plan(model.geometry)

    def test_the_view_swaps_in_a_sphere_and_puts_the_torus_back_even_after_an_error(self):
        model, (t,) = torus_model()
        tp = torus_cards.plan(model.geometry)
        ring = next(c for c in model.geometry.get_all_cells().values() if c.name == "ring")
        with torus_cards.mcnpy_view(tp, model.geometry):
            h = next(iter(ring.region.get_surfaces().values()))
            self.assertIsInstance(h, openmc.Sphere)
            self.assertEqual(h.id, t.id, "the stand-in keeps the torus's surface number")
        self.assertIs(next(iter(ring.region.get_surfaces().values())), t)
        with self.assertRaises(RuntimeError):
            with torus_cards.mcnpy_view(tp, model.geometry):
                raise RuntimeError("MCNPy failed")
        self.assertIs(next(iter(ring.region.get_surfaces().values())), t)

    def test_patch_rewrites_only_the_stand_in_card(self):
        model, (t,) = torus_model(openmc.YTorus, centre=(1.0, 2.0, 3.0))
        tp = torus_cards.plan(model.geometry)
        r = tp[0]["radius"]
        deck = f"c cells\n1 1 -7.87 -{t.id}\n\n98 SO {r}\n99 S 1.0 2.0 3.0 {r + 5}\n{t.id} S 1.0 2.0 3.0 {r}\n\nMODE N\n"   # the decoy with the same radius comes first
        out = torus_cards.patch_deck(deck, tp)
        self.assertIn(f"{t.id} TY 1.0 2.0 3.0 10.0 4.0 3.0", out)
        self.assertIn(f"99 S 1.0 2.0 3.0 {r + 5}", out)
        self.assertIn(f"98 SO {r}", out, "another surface with the same radius but another number is left alone")
        self.assertNotIn(f"{t.id} S ", out)

    def test_patch_names_a_missing_stand_in(self):
        model, _ = torus_model()
        with self.assertRaisesRegex(ValueError, "not in the translated deck"):
            torus_cards.patch_deck("1 0 -1\n\n1 SO 5.0\n\n", torus_cards.plan(model.geometry))


class Export(unittest.TestCase):
    def export(self, model, name):
        work = tempfile.mkdtemp(prefix="torus-")
        path = os.path.join(work, "model.xml")
        model.export_to_model_xml(path)
        with contextlib.redirect_stdout(io.StringIO()):
            r = export(path, os.path.join(work, "deck"), name, samples=20000)
        return r

    def test_each_axis_exports_validates_and_the_torus_cell_receives_points(self):
        for cls, card, centre in ((openmc.ZTorus, "TZ", (0.0, 0.0, 0.0)), (openmc.XTorus, "TX", (2.0, -3.0, 1.0)), (openmc.YTorus, "TY", (-1.0, 4.0, 2.0))):
            model, _ = torus_model(cls, centre=centre)
            r = self.export(model, card)
            self.assertTrue(r["ok"], r.get("validation", "")[-1500:])
            deck = Path(r["runnable"]).read_text()
            x, y, z = ("0" if v == 0 else repr(v) for v in centre)       # how the deck writes a number: 0 bare, others as floats
            self.assertRegex(deck, rf"(?m)^\d+ {card} {x} {y} {z} 10\.0 4\.0 3\.0$")
            self.assertNotIn("no sampled point fell in it", r["validation"], "the torus cell must be compared, not skipped")
            self.assertTrue(any("torus surface" in n for n in r["notes"]), r["notes"])

    def test_two_tori_in_one_model(self):
        model, _ = torus_model(two=True)
        r = self.export(model, "two")
        self.assertTrue(r["ok"], r.get("validation", "")[-1500:])
        self.assertEqual(len(re.findall(r"(?m)^\d+ TZ ", Path(r["runnable"]).read_text())), 2)

    def check_break(self, change, what):
        model, _ = torus_model(openmc.ZTorus, centre=(1.0, 2.0, 0.0))
        r = self.export(model, "brk")
        self.assertTrue(r["ok"], "the unbroken deck must validate first: " + r.get("validation", "")[-800:])
        deck = Path(r["runnable"]).read_text()
        card = re.search(r"(?m)^\d+ TZ 1\.0 2\.0 0 10\.0 4\.0 3\.0$", deck)
        self.assertTrue(card, deck[:3000])
        broken = deck.replace(card.group(0), change(card.group(0)))
        self.assertNotEqual(broken, deck)
        path = Path(r["runnable"]).with_name("broken.mcnp")
        path.write_text(broken)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = validate_deck(str(path), model=model, geometry_samples=20000)
        self.assertFalse(ok, f"the check missed {what}:\n" + buf.getvalue()[-800:])

    def test_the_comparison_catches_the_two_minor_radii_swapped(self):
        self.check_break(lambda s: s.replace("4.0 3.0", "3.0 4.0"), "B and C swapped")

    def test_the_comparison_catches_the_wrong_axis(self):
        self.check_break(lambda s: s.replace(" TZ ", " TX "), "the wrong axis")

    def test_the_comparison_catches_the_wrong_centre(self):
        self.check_break(lambda s: s.replace("TZ 1.0 2.0 0", "TZ 2.0 2.0 0"), "a centre moved by 1 cm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
