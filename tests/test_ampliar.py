import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ampliar", ROOT / "src" / "ampliar.py")
AMPLIAR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AMPLIAR)


class UpscalePresetTests(unittest.TestCase):
    def test_fiel_maps_to_zero_softness(self):
        with tempfile.TemporaryDirectory() as folder:
            image = pathlib.Path(folder) / "in.png"
            output = pathlib.Path(folder) / "out.png"
            image.write_bytes(b"input")
            seen = {}

            def fake_seedvr2(source, destination, scale, softness):
                seen.update(source=source, destination=destination,
                            scale=scale, softness=softness)
                pathlib.Path(destination).write_bytes(b"output")
                return 0

            argv = ["ampliar.py", str(image), "--out", str(output),
                    "--modo", "fiel"]
            with (
                mock.patch.object(AMPLIAR, "seedvr2", side_effect=fake_seedvr2),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(AMPLIAR.main(), 0)

            self.assertEqual(seen["softness"], 0.0)
            self.assertEqual(seen["scale"], 2.0)

    def test_explicit_softness_overrides_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            image = pathlib.Path(folder) / "in.png"
            output = pathlib.Path(folder) / "out.png"
            image.write_bytes(b"input")
            seen = {}

            def fake_seedvr2(_source, destination, _scale, softness):
                seen["softness"] = softness
                pathlib.Path(destination).write_bytes(b"output")
                return 0

            argv = ["ampliar.py", str(image), "--out", str(output),
                    "--modo", "criativo", "--softness", "0.2"]
            with (
                mock.patch.object(AMPLIAR, "seedvr2", side_effect=fake_seedvr2),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(AMPLIAR.main(), 0)

            self.assertEqual(seen["softness"], 0.2)


if __name__ == "__main__":
    unittest.main()
