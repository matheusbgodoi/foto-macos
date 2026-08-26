import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("krea2", ROOT / "src" / "krea2.py")
KREA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KREA)


class KreaRunnerTests(unittest.TestCase):
    def test_atomic_output_replaces_stale_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            binary = root / "mflux-generate-krea2"
            binary.write_text("fake")
            lora = root / "famegrid.safetensors"
            lora.write_text("fake")
            model = root / "model"
            model.mkdir()
            output = root / "result.png"
            output.write_bytes(b"stale")

            def fake_run(command):
                temporary = pathlib.Path(command[command.index("--output") + 1])
                temporary.write_bytes(b"fresh")
                temporary.with_suffix(".metadata.json").write_text("{}")
                return types.SimpleNamespace(returncode=0)

            argv = ["krea2.py", "a real photograph", "--saida", str(output)]
            with (
                mock.patch.object(KREA, "MFLUX", str(binary)),
                mock.patch.object(KREA, "LORA", str(lora)),
                mock.patch.object(KREA, "model_path", return_value=str(model)),
                mock.patch.object(KREA.subprocess, "run", side_effect=fake_run),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(KREA.main(), 0)

            self.assertEqual(output.read_bytes(), b"fresh")
            self.assertTrue(output.with_suffix(".metadata.json").is_file())
            self.assertFalse(list(root.glob("*.foto-macos-*")))


if __name__ == "__main__":
    unittest.main()
