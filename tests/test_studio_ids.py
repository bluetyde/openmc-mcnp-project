"""Provenance comments preserve MCNP text and the 128-column boundary."""
import json
from pathlib import Path
import sys
import unittest
from urllib.parse import unquote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from studio_ids import annotate, comment_cards
from deck_format import format_deck, overlong_lines


class StudioIds(unittest.TestCase):
    def test_comments_preserve_cards_and_derived_owners(self):
        deck = 'Title\n1 0 -8 $ ordinary comment\n     imp:n=1\n7 0 -9 lat=1\n\n8 RPP 0 1 0 1 0 1\n9 RPP 0 2 0 2 0 2\n\nF4:N 1\n     7\nFC4 Duplicate name\n'
        owner = {'kind':'part', 'id':'long / $ c Unicode λ ' * 30, 'slot':0}
        ids = {'cell':{1:owner}, 'surface':{8:owner}, 'lattice':{3:{'kind':'group','id':'array'}},
               'tally':{99:{'kind':'tally','id':'stable-tally'}}}
        result = annotate(deck, ids, {3:{'cell':7,'surfaces':[9]}}, {4:99})
        self.assertEqual('\n'.join(l for l in result.split('\n') if not l.startswith('c @studio')),deck)
        self.assertFalse(overlong_lines(result))
        self.assertEqual(format_deck(result),result)
        self.assertIn('c @studio-v1 surface 9',result)
        self.assertIn('c @studio-v1 tally 4',result)
        chunks = [line.split()[-1] for line in result.splitlines() if line.startswith('c @studio-v1 surface 8 ')]
        self.assertNotIn('slot',json.loads(unquote(''.join(chunks))))
        self.assertEqual(annotate(deck,None),deck)

    def test_serialization_is_lossless(self):
        owner = {'kind':'tally','id':'a\n$ c λ "' * 50}
        cards = comment_cards('tally',44,owner)
        self.assertEqual(json.loads(unquote(''.join(c.split()[-1] for c in cards))),owner)
        self.assertTrue(all(len(c) <= 128 and c.isascii() for c in cards))


if __name__ == '__main__':
    unittest.main()
