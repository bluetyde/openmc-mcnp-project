"""Plain cells get their regions written directly; MCNPy translates a short placeholder instead.

MCNPy's cost is a Java round trip per region term, and its "Making Universes" phase was 292 s of 311 on the 267-cell
graphite-pile import. cell_regions gives every cell that is only a material (or void) and half-spaces a placeholder of
about one term while MCNPy translates, then writes the real region into the card. These tests check the region text
(operator precedence), which cells are touched, that the placeholders still make MCNPy write every surface, the card
patching and its refusals, and (needs the MCNPy gateway, port 25333: claim it where agents share a machine) that a
model exports and validates, point by point against OpenMC, with and without the direct cards.

Run: python tests/test_cell_regions.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_regions  # noqa: E402
import world_complement  # noqa: E402
from export_mcnp import export  # noqa: E402
from test_world_complement import grid_model, reloaded  # noqa: E402


def planes(n=6):
    openmc.reset_auto_ids()
    return [openmc.XPlane(float(i), surface_id=i + 1) for i in range(n)]


class RegionText(unittest.TestCase):
    def test_signs_and_operators(self):
        s = planes()
        self.assertEqual(cell_regions.region_text(+s[0] & -s[1] & -s[2]), "1 -2 -3")
        self.assertEqual(cell_regions.region_text(+s[0] & -s[1] & (+s[2] | -s[3])), "1 -2 (3:-4)")
        self.assertEqual(cell_regions.region_text((+s[0] & -s[1]) | (+s[2] & -s[3])), "(1 -2):(3 -4)")
        self.assertEqual(cell_regions.region_text(+s[0] | -s[1] | +s[2]), "1:-2:3")

    def test_blank_binds_tighter_than_colon(self):
        """MCNP reads `1 2:3` as (1 2):3, so a union inside an intersection must be parenthesised, and the other way
        round the text keeps the groups explicit."""
        s = planes()
        self.assertEqual(cell_regions.region_text(+s[0] & (-s[1] | (+s[2] & -s[3]))), "1 (-2:(3 -4))")

    def test_a_long_union_is_cut_after_its_colons_and_nothing_else_changes(self):
        s = planes(40)
        union = openmc.Union([-x for x in s])
        text = cell_regions.region_text(+s[0] & union)
        tokens = cell_regions._tokens(text)
        self.assertTrue(all(len(t) <= cell_regions.LONG_UNION for t in tokens), tokens)
        self.assertEqual("".join(tokens), text.replace(" ", ""))


def model_with_cells(extra=None):
    """Four material cells of planes plus the world box, as reloaded from XML; `extra` adds a cell."""
    with tempfile.TemporaryDirectory() as work:
        _, m = reloaded(grid_model(2, 2), work)
    return m


class Plan(unittest.TestCase):
    def test_placeholders_name_every_surface_and_stay_short(self):
        m = model_with_cells()
        wc = world_complement.plan(m.geometry, min_terms=0)
        cp = cell_regions.plan(m.geometry, skip={wc["cell"].id}, extra_seen=[h.surface.id for h in cell_regions._literals(wc["kept"])],
                               min_terms=0)
        self.assertEqual(sorted(cp["texts"]), sorted(wc["ids"]))
        named = {h.surface.id for p in cp["placeholders"].values() for h in cell_regions._literals(p)}
        used = {h.surface.id for c in cp["cells"].values() for h in cell_regions._literals(c.region)}
        box = {h.surface.id for h in cell_regions._literals(wc["kept"])}
        self.assertEqual(named | box, used | box, "every surface a plain cell uses is named by some placeholder or the box")
        self.assertEqual(cp["surfaces"], used)
        total = sum(cell_regions._terms(p) for p in cp["placeholders"].values())
        self.assertLessEqual(total, len(used - box) + len(cp["cells"]), "about one term per cell, not the full regions")
        self.assertLess(total, sum(cell_regions._terms(c.region) for c in cp["cells"].values()))

    def test_below_the_threshold_nothing_changes(self):
        m = model_with_cells()
        self.assertIsNone(cell_regions.plan(m.geometry, skip={max(m.geometry.get_all_cells())}))

    def test_a_cell_filled_with_a_universe_or_moved_is_left_to_mcnpy(self):
        m = model_with_cells()
        cells = list(m.geometry.get_all_cells().values())
        world = max(cells, key=lambda c: c.id)
        moved = next(c for c in cells if c is not world)
        moved.translation = (1.0, 0.0, 0.0)
        cp = cell_regions.plan(m.geometry, skip={world.id}, min_terms=0)
        self.assertNotIn(moved.id, cp["texts"])
        self.assertEqual(len(cp["texts"]), len(cells) - 2)

    def test_a_complement_in_another_cell_disables_it(self):
        m = model_with_cells()
        cells = list(m.geometry.get_all_cells().values())
        world = max(cells, key=lambda c: c.id)
        other = next(c for c in cells if c is not world)
        other.region = openmc.Complement(other.region)
        self.assertIsNone(cell_regions.plan(m.geometry, skip={world.id}, min_terms=0))

    def test_the_regions_are_put_back(self):
        m = model_with_cells()
        world = max(m.geometry.get_all_cells())
        cp = cell_regions.plan(m.geometry, skip={world}, min_terms=0)
        before = {cid: str(c.region) for cid, c in cp["cells"].items()}
        with cell_regions.mcnpy_view(cp):
            during = sum(cell_regions._terms(c.region) for c in cp["cells"].values())
            self.assertLess(during, sum(cell_regions._terms(r) for r in cp["regions"].values()))
        self.assertEqual({cid: str(c.region) for cid, c in cp["cells"].items()}, before)


class Patch(unittest.TestCase):
    def cp(self, **kw):
        cp = {"texts": {1: "1 -2 (3:-4)", 2: "5 -6"}, "place": {1: "1 -2 3", 2: "5"}, "surfaces": {1, 2, 3, 4, 5, 6}}
        cp.update(kw)
        return cp

    TEXT = ("$ title\n1 1 -1.7 1 -2 3 U 1\n     IMP:N=1.0 \n2 0 5 IMP:N=1.0\nc comment\n3 0 9 IMP:N=0\n\n"
            + "".join(f"{i} PX {i}\n" for i in range(1, 7)) + "\nM1 6012 1\n")

    def test_regions_replace_placeholders_and_everything_else_is_untouched(self):
        out = cell_regions.patch_deck(self.TEXT, self.cp()).split("\n")
        # a patched card is re-wrapped as a whole, so one that fits on a line comes out on one line
        self.assertEqual(out[:6], ["$ title", "1 1 -1.7 1 -2 (3:-4) U 1 IMP:N=1.0", "2 0 5 -6 IMP:N=1.0", "c comment",
                                   "3 0 9 IMP:N=0", ""])
        self.assertEqual(out[6:], self.TEXT.split("\n")[7:], "the surface and data blocks are as they were")

    def test_long_cards_wrap_within_the_line_width(self):
        s = planes(60)
        union = openmc.Union([-x for x in s])
        cp = self.cp(texts={1: cell_regions.region_text(+s[0] & union), 2: "5"}, place={1: "1 -2 3", 2: "5"})
        out = cell_regions.patch_deck(self.TEXT, cp).split("\n")
        card = out[1:next(i for i in range(2, len(out)) if out[i].startswith("2 0"))]
        self.assertGreater(len(card), 3)
        self.assertTrue(all(len(line) <= cell_regions.LINE for line in card), [len(x) for x in card])
        self.assertTrue(all(line.startswith("     ") for line in card[1:]))
        self.assertEqual(card[-1].split()[-3:], ["U", "1", "IMP:N=1.0"])

    def test_an_unexpected_placeholder_is_refused(self):
        with self.assertRaisesRegex(cell_regions.DirectCardsMismatch, "cell 1: expected the placeholder"):
            cell_regions.patch_deck(self.TEXT, self.cp(place={1: "9", 2: "5"}))

    def test_a_cell_missing_from_the_deck_is_refused(self):
        cp = self.cp(texts={1: "1", 2: "5", 7: "4"}, place={1: "1 -2 3", 2: "5", 7: "4"})
        with self.assertRaisesRegex(cell_regions.DirectCardsMismatch, r"cells \[7\]"):
            cell_regions.patch_deck(self.TEXT, cp)

    def test_a_surface_without_a_card_is_refused(self):
        with self.assertRaisesRegex(cell_regions.DirectCardsMismatch, r"surfaces \[99\] have no card"):
            cell_regions.patch_deck(self.TEXT, self.cp(surfaces={1, 2, 3, 4, 5, 6, 99}))


class ExportsBothWays(unittest.TestCase):
    """Needs MCNPy. The same model with and without the direct cards must validate against OpenMC."""

    def export(self, min_terms):
        old = (cell_regions.MIN_TERMS, world_complement.MIN_TERMS)
        cell_regions.MIN_TERMS = world_complement.MIN_TERMS = min_terms
        try:
            with tempfile.TemporaryDirectory() as work:
                path, _ = reloaded(grid_model(4, 4), work)
                with contextlib.redirect_stdout(io.StringIO()):
                    r = export(path, os.path.join(work, "deck"), "cr", samples=3000)
                return r, Path(r["runnable"]).read_text()
        finally:
            cell_regions.MIN_TERMS, world_complement.MIN_TERMS = old

    def test_deck_validates_with_and_without_direct_cards(self):
        direct, deck_direct = self.export(0)
        plain, deck_plain = self.export(10 ** 9)
        self.assertTrue(direct["ok"], direct.get("validation", "")[-1500:])
        self.assertTrue(plain["ok"], plain.get("validation", "")[-1500:])
        self.assertTrue(any("plain cells are written directly" in n for n in direct["notes"]), direct["notes"])
        self.assertFalse(any("written directly" in n for n in plain["notes"]))
        blocks_d, blocks_p = deck_direct.split("\n\n"), deck_plain.split("\n\n")
        surf = lambda b: sorted(int(m.group(1)) for m in re.finditer(r"(?m)^[*+]?(\d+)\s+[A-Za-z]", b))
        self.assertEqual(surf(blocks_d[1]), surf(blocks_p[1]), "the same surface cards")
        ids = lambda b: sorted(int(m.group(1)) for m in re.finditer(r"(?m)^(\d+)\s+\d+\s", b))
        self.assertEqual(ids(blocks_d[0]), ids(blocks_p[0]), "the same cells")
        self.assertIn("geometry matches OpenMC", direct["validation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
