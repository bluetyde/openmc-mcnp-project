"""The MontePy version is pinned, and the pins agree with each other and with the environment the tests run in.

requirements.txt and environment.yml both pin `montepy` to one version, and that version is the one installed here. Upgrading MontePy is a decision (a newer one needs Python 3.12): this test
fails until the pins and the installed version are changed together, so the exporter never runs on a MontePy nobody chose.

    python tests/test_pins.py
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def pin(path, pattern):
    hits = re.findall(pattern, (ROOT / path).read_text(encoding="utf-8"), re.M)
    return hits


class MontePyPin(unittest.TestCase):
    def test_both_files_pin_the_same_version(self):
        pip = pin("requirements.txt", r"^montepy==([0-9][0-9.]*)\s*$")
        conda = pin("environment.yml", r"^\s*- montepy=([0-9][0-9.]*)\s*$")
        self.assertEqual(len(pip), 1, "requirements.txt must pin montepy with ==")
        self.assertEqual(len(conda), 1, "environment.yml must pin montepy with =")
        self.assertEqual(pip, conda)

    def test_the_pin_is_the_installed_version(self):
        import montepy
        pinned = pin("requirements.txt", r"^montepy==([0-9][0-9.]*)\s*$")
        self.assertEqual(pinned, [montepy.__version__], "the installed MontePy is not the pinned one: change the pins and the environment together")


if __name__ == "__main__":
    unittest.main(verbosity=2)
