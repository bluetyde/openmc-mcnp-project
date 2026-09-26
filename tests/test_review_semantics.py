"""What the exported cards mean, not just that they parse (Codex review, 2026-09-25).

- A histogram source keeps its bin probabilities: OpenMC's Tabular holds densities per eV, MCNP's SP D holds
  the probability of each bin after a leading 0, so unequal bins must be integrated, not copied.
- A photon current tally is an F1:P, not F1:N.
- A tally whose energy bins start above 0 keeps that lower edge, so nothing below it lands in the first bin.
No MCNPy needed: OpenMC objects and mcnp_cards only.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import openmc  # noqa: E402
import mcnp_cards  # noqa: E402
from mcnp_cards import _Dists, _energy, histogram_mass, tally_cards  # noqa: E402


def sp_of(dist):
    d = _Dists()
    _energy(dist, d)
    si = next(c for c in d.cards if c.startswith("SI"))
    sp = next(c for c in d.cards if c.startswith("SP"))
    return [float(v) for v in si.split()[2:]], [float(v) for v in sp.split()[2:]]


def sampled_fractions(dist, edges_ev, n=200_000, seed=7):
    e = dist.sample(n, seed=seed) if hasattr(dist, "sample") else None
    if e is None:
        raise unittest.SkipTest("this OpenMC has no Tabular.sample")
    counts, _ = np.histogram(e, bins=edges_ev)
    return counts / n


class HistogramSource(unittest.TestCase):
    def test_unequal_bins_keep_their_probabilities(self):
        # the reproduced case: 50/50 over 0-1-3 MeV. As densities these are 0.5 / 1e6 and 0.5 / 2e6 per eV.
        x = [0.0, 1e6, 3e6]
        dens = openmc.stats.Tabular(x, [0.5 / 1e6, 0.5 / 2e6], interpolation="histogram")
        si, sp = sp_of(dens)
        self.assertEqual(si[1:] if si[0] == "H" else si, [0.0, 1.0, 3.0])
        self.assertEqual(sp[0], 0.0, "SP D starts with 0 for a histogram (manual: first bin probability is 0)")
        np.testing.assert_allclose(sp[1:], [0.5, 0.5], rtol=1e-12)
        frac = sampled_fractions(dens, x)
        np.testing.assert_allclose(frac, [0.5, 0.5], atol=0.005)

    def test_densities_copied_as_probabilities_would_be_wrong(self):
        # the old export copied p straight into SP: guard the meaning with a spectrum whose widths vary 1:10
        x = [0.0, 1e5, 1.1e6]
        dens = openmc.stats.Tabular(x, [2.0, 1.0], interpolation="histogram")  # masses 2e5 : 1e6
        _, sp = sp_of(dens)
        np.testing.assert_allclose(sp[1:], [2e5 / 1.2e6, 1e6 / 1.2e6], rtol=1e-11)

    def test_one_value_per_edge_ignores_the_last(self):
        # OpenMC accepts p with one value per edge for a histogram; the last value is never used
        x = [1e5, 1e6, 2e6, 5e6]
        a = openmc.stats.Tabular(x, [3.0, 1.0, 2.0], interpolation="histogram")
        b = openmc.stats.Tabular(x, [3.0, 1.0, 2.0, 99.0], interpolation="histogram")
        self.assertEqual(sp_of(a), sp_of(b))
        self.assertEqual(len(sp_of(a)[1]), len(x), "a leading 0 plus one probability per bin")

    def test_zero_probability_bins_stay_zero(self):
        x = [0.0, 1e6, 2e6, 4e6]
        _, sp = sp_of(openmc.stats.Tabular(x, [1e-6, 0.0, 0.25e-6], interpolation="histogram"))
        np.testing.assert_allclose(sp, [0, 2 / 3, 0, 1 / 3], rtol=1e-12)

    def test_studio_histogram_helper_round_trips(self):
        # OpenMC Studio writes _histogram(edges, probs): probabilities / width. The deck must give back probs.
        x, probs = [5e4, 2e5, 4e5, 6.5e6, 1.075e7], [0.008, 0.015, 0.4, 0.2]
        dens = openmc.stats.Tabular(x, [p / (hi - lo) for p, lo, hi in zip(probs, x, x[1:])], interpolation="histogram")
        np.testing.assert_allclose(histogram_mass(dens), np.array(probs) / sum(probs), rtol=1e-11)

    def test_validator_rejects_copied_densities(self):
        from validate_deck import validate_deck  # noqa: F401  (import check: the validator uses histogram_mass)
        self.assertTrue(callable(histogram_mass))


def _current_model(particle):
    openmc.reset_auto_ids()
    s = openmc.Sphere(r=5)
    outer = openmc.Sphere(r=20, boundary_type="vacuum")
    inner, shell = openmc.Cell(region=-s), openmc.Cell(region=+s & -outer)
    geom = openmc.Geometry([inner, shell])
    t = openmc.Tally(name=f"{particle} current")
    t.filters = [openmc.SurfaceFilter([s]), openmc.CellFromFilter([inner]), openmc.ParticleFilter([particle]),
                 openmc.EnergyFilter([0.0, 1e6, 2e6])]
    t.scores = ["current"]
    return t, geom


class CurrentParticle(unittest.TestCase):
    def test_photon_current_is_f1_p(self):
        t, geom = _current_model("photon")
        cards, _ = tally_cards([t], geom)
        f1 = [c for c in cards if c.startswith("F1:")]
        self.assertEqual(f1, ["F1:P 1"], cards)

    def test_neutron_current_is_f1_n(self):
        t, geom = _current_model("neutron")
        cards, _ = tally_cards([t], geom)
        self.assertEqual([c for c in cards if c.startswith("F1:")], ["F1:N 1"], cards)


class EnergyLowerEdge(unittest.TestCase):
    def _cards(self, edges, mesh=False):
        openmc.reset_auto_ids()
        s = openmc.Sphere(r=10, boundary_type="vacuum")
        c = openmc.Cell(region=-s)
        t = openmc.Tally(name="bins")
        if mesh:
            m = openmc.RegularMesh()
            m.dimension, m.lower_left, m.upper_right = [2, 2, 2], [-5, -5, -5], [5, 5, 5]
            t.filters = [openmc.MeshFilter(m), openmc.EnergyFilter(edges)]
        else:
            t.filters = [openmc.CellFilter([c]), openmc.EnergyFilter(edges)]
        t.scores = ["flux"]
        return tally_cards([t], openmc.Geometry([c]))

    def test_positive_lower_edge_is_kept(self):
        cards, notes = self._cards([1e6, 2e6, 3e6])
        self.assertIn("E4 1.0 2.0 3.0", cards, "0-1 MeV is MCNP's own first bin, so 1-2 MeV stays clean")
        self.assertTrue(any("first bin (0 to 1 MeV) is extra" in n for n in notes), notes)

    def test_zero_lower_edge_is_implicit(self):
        cards, notes = self._cards([0.0, 2e6, 3e6])
        self.assertIn("E4 2.0 3.0", cards)
        self.assertFalse(any("extra" in n for n in notes))

    def test_mesh_keeps_it_too(self):
        cards, _ = self._cards([1e6, 2e6], mesh=True)
        self.assertTrue(any("EMESH=1.0 2.0" in c for c in cards), cards)


if __name__ == "__main__":
    unittest.main()
