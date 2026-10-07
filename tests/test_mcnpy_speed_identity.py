"""The MCNPy speedups may only change how many round trips are made, never the deck MCNPy writes.

mcnpy_speed.fast_add and cached_reflection are argued exact from MCNPy's source (tests/test_mcnpy_speed.py runs that
source against a fake gateway). This is the check on the real thing, the one that decides whether they may stay on:
the same model is translated with MCNPY_SPEEDUPS=0 and =1 and the two decks must be byte for byte the same, on a
model with a lattice of universes (where cells are filed under universes other than 0), on the Studio-shaped grid
with the world cell as #cell complements and the direct cell regions, and on the same grid with MCNPy's own text.
It also checks that the round trips really drop, so a speedup that silently switched itself off is noticed.

Each translation runs in its own process: MCNPy numbers the cells and surfaces it creates for a lattice from a
counter that lives as long as the process, so a second translation in the same process writes different numbers
(5 0 -14 ... then 6 0 -15 ..., with the same settings), which would look like a difference between the two modes.

Requires the MCNPy gateway (port 25333, claim it first where agents share a machine).
Run: python tests/test_mcnpy_speed_identity.py
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

MODELS = ("lattice", "grid_mcnpy_text", "grid_direct")


def build(name, work):
    import cell_regions
    import world_complement
    from test_fast_find import lattice_model
    from test_world_complement import grid_model, reloaded
    if name == "lattice":
        return reloaded(lattice_model(), work)[1]
    cell_regions.MIN_TERMS = world_complement.MIN_TERMS = 10 ** 9 if name == "grid_mcnpy_text" else 0
    return reloaded(grid_model(4, 4), work)[1]


def worker(flag, name, out):
    """One translation in this process; writes the deck and the number of py4j round trips."""
    os.environ["MCNPY_SPEEDUPS"] = flag
    from export_mcnp import translate
    from mcnpy.translate_mcnp_openmc import openmc_to_mcnp  # noqa: F401  (starts MCNPy's gateway)
    import py4j.java_gateway as jg
    calls, original = [0], jg.GatewayClient.send_command

    def counting(client, command, retry=True, binary=False):
        calls[0] += 1
        return original(client, command, retry, binary)
    jg.GatewayClient.send_command = counting
    with tempfile.TemporaryDirectory() as work:
        model = build(name, work)
        with contextlib.redirect_stdout(io.StringIO()):
            notes = translate(model, out)
    Path(out + ".info").write_text(f"{calls[0]}\n" + "\n".join(notes))


class SameDeckEitherWay(unittest.TestCase):
    def run_in_a_fresh_process(self, flag, name):
        with tempfile.TemporaryDirectory() as work:
            out = os.path.join(work, "deck.mcnp")
            r = subprocess.run([sys.executable, "-W", "ignore", __file__, "--worker", flag, name, out],
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
            info = Path(out + ".info").read_text().split("\n")
            return Path(out).read_bytes(), int(info[0]), info[1:]

    def check(self, name, fewer_calls_by=1.5):
        fast, calls_fast, notes_fast = self.run_in_a_fresh_process("1", name)
        slow, calls_slow, notes_slow = self.run_in_a_fresh_process("0", name)
        self.assertEqual(fast, slow, "the speedups changed the deck MCNPy writes")
        self.assertFalse(any("speedups left off" in n for n in notes_fast), notes_fast)
        self.assertTrue(any("fast_add off (MCNPY_SPEEDUPS=0)" in n for n in notes_slow), notes_slow)
        self.assertLess(calls_fast * fewer_calls_by, calls_slow, f"{calls_fast} calls with the speedups, {calls_slow} without")
        self.assertGreater(len(fast), 300)

    def test_a_lattice_of_universes(self):
        self.check("lattice")

    def test_the_studio_shaped_grid_with_mcnpy_s_own_text(self):
        self.check("grid_mcnpy_text")

    def test_the_studio_shaped_grid_with_complements_and_direct_cells(self):
        self.check("grid_direct")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(*sys.argv[2:5])
    else:
        unittest.main(verbosity=2)
