"""Real translation/remediation/validation regression; requires the MCNPy gateway claim."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import openmc
from export_mcnp import export
from validate_deck import validate_deck, _cards_by_name
from deck_format import overlong_lines


class ColumnExport(unittest.TestCase):
    def test_long_labels_energy_cards_and_validator(self):
        openmc.reset_auto_ids()
        mat = openmc.Material(name='Hydrogen ' * 35)
        mat.set_density('g/cm3', 0.001)
        mat.add_nuclide('H1', 1)
        surface = openmc.Sphere(r=10, boundary_type='vacuum', name='Boundary ' * 35)
        cell = openmc.Cell(fill=mat, region=-surface, name='Detector cell ' * 35)
        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.particles, settings.batches = 100, 2
        settings.source = openmc.IndependentSource(space=openmc.stats.Point((0, 0, 0)))
        tally = openmc.Tally(name='Flux')
        edges = [i * 100000.0 for i in range(101)]
        tally.filters = [openmc.CellFilter([cell]), openmc.EnergyFilter(edges)]
        tally.scores = ['flux']
        model = openmc.Model(openmc.Geometry([cell]), openmc.Materials([mat]), settings, openmc.Tallies([tally]))
        with tempfile.TemporaryDirectory(prefix='column_export_') as folder:
            xml = Path(folder) / 'model.xml'
            model.export_to_model_xml(xml)
            report = export(str(xml), out_dir=folder, samples=500)
            self.assertTrue(report['ok'], report['validation'])
            deck_path = Path(report['runnable'])
            text = deck_path.read_text()
            self.assertFalse(overlong_lines(text))
            cards = _cards_by_name(text.upper())
            self.assertEqual([float(x) for x in cards['E4']], [e / 1e6 for e in edges[1:]])
            # Even an overlong comment must fail before parser/physics validation.
            deck_path.write_text(text + 'c ' + 'x' * 127 + '\n')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                valid = validate_deck(str(deck_path), model=model)
            self.assertFalse(valid)
            self.assertIn('128 columns', buf.getvalue())


if __name__ == '__main__':
    unittest.main()
