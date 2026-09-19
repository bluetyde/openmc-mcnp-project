"""Dependency-free boundary and syntax-preservation tests for generated decks."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from deck_format import format_deck, overlong_lines


class DeckFormatting(unittest.TestCase):
    def test_boundary_tabs_and_blank_blocks(self):
        text = 'title\r\nc ' + 'x' * 126 + '\r\n\r\n' + 'SI1 ' + ' 1' * 100 + '\r\n'
        out = format_deck(text)
        self.assertFalse(overlong_lines(out))
        self.assertIn('c ' + 'x' * 126 + '\n\n', out)
        self.assertEqual(format_deck(out), out)
        self.assertEqual(overlong_lines('x' * 120 + '\tX'), [(1, 129)])
        self.assertEqual(format_deck('title\nc\ttext\n'), 'title\nc       text\n')

    def test_preserve_data_tokens_and_continuation(self):
        card = 'SI1 L ' + ' '.join(str(i) for i in range(100))
        out = format_deck('title\n' + card + '\n')
        lines = out.splitlines()[1:]
        self.assertEqual(' '.join(lines).split(), card.split())
        self.assertTrue(all(l.startswith('     ') for l in lines[1:]))
        self.assertFalse(overlong_lines(out))

    def test_comments_and_title(self):
        out = format_deck('t' * 180 + '\nc ' + 'word ' * 80 + '\n')
        self.assertEqual(len(out.splitlines()[0]), 128)
        self.assertTrue(all(l.startswith('c ') for l in out.splitlines()[1:]))
        self.assertIn('t' * 52, out)
        self.assertFalse(overlong_lines(out))

    def test_inline_comment_and_ampersand(self):
        data = 'SDEF ' + ' '.join('X=' + str(i) for i in range(35))
        out = format_deck('title\n' + data + ' & $ ' + 'note ' * 35 + '\n     Y=1\n')
        code = [l for l in out.splitlines()[1:] if not l.startswith('c ')]
        self.assertEqual(' '.join(code).split(), (data + ' & Y=1').split())
        self.assertTrue(code[-2].endswith(' &'))
        self.assertFalse(overlong_lines(out))

    def test_chain_and_no_token_splitting(self):
        card = 'F4:N ' + ' '.join('(1 < 7[0 0 0] < 3)' for _ in range(20))
        out = format_deck('title\n' + card)
        self.assertEqual(out.split(), ('title\n' + card).split())
        with self.assertRaisesRegex(ValueError, 'data token'):
            format_deck('title\nSI1 ' + '1' * 130)
        with self.assertRaisesRegex(ValueError, 'quoted'):
            format_deck('title\nFILE="' + 'long name ' * 20 + '"')


if __name__ == '__main__':
    unittest.main()
