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

    def test_named_identity_rewrites_prompt_and_stacks_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            binary = root / "mflux-generate-krea2"
            binary.write_text("fake")
            famegrid = root / "famegrid.safetensors"
            famegrid.write_text("fake")
            identity_lora = root / "matheus.safetensors"
            identity_lora.write_text("fake")
            registry = root / "identities.json"
            registry.write_text(
                '{"Pessoa":{"token":"pessoa_test_token","lora":"%s","scale":0.85}}'
                % identity_lora
            )
            model = root / "model"
            model.mkdir()
            output = root / "result.png"
            seen = {}

            def fake_run(command):
                seen["command"] = command
                temporary = pathlib.Path(command[command.index("--output") + 1])
                temporary.write_bytes(b"fresh")
                return types.SimpleNamespace(returncode=0)

            argv = ["krea2.py", "Pessoa giving a talk", "--saida", str(output)]
            with (
                mock.patch.object(KREA, "MFLUX", str(binary)),
                mock.patch.object(KREA, "LORA", str(famegrid)),
                mock.patch.object(KREA, "IDENTITIES_FILE", str(registry)),
                mock.patch.object(KREA, "model_path", return_value=str(model)),
                mock.patch.object(KREA.subprocess, "run", side_effect=fake_run),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(KREA.main(), 0)

            command = seen["command"]
            self.assertEqual(command.count("--lora"), 2)
            famegrid_position = command.index(str(famegrid))
            self.assertEqual(command[famegrid_position + 1], "0.3")
            prompt = command[command.index("--prompt") + 1]
            self.assertIn("pessoa_test_token giving a talk", prompt)
            self.assertIn(str(identity_lora), command)

    def test_explicit_famegrid_weight_overrides_identity_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            binary = root / "mflux-generate-krea2"
            binary.write_text("fake")
            famegrid = root / "famegrid.safetensors"
            famegrid.write_text("fake")
            identity_lora = root / "matheus.safetensors"
            identity_lora.write_text("fake")
            registry = root / "identities.json"
            registry.write_text(
                '{"Pessoa":{"token":"pessoa_test_token","lora":"%s"}}'
                % identity_lora
            )
            model = root / "model"
            model.mkdir()
            output = root / "result.png"
            seen = {}

            def fake_run(command):
                seen["command"] = command
                temporary = pathlib.Path(command[command.index("--output") + 1])
                temporary.write_bytes(b"fresh")
                return types.SimpleNamespace(returncode=0)

            argv = [
                "krea2.py", "Pessoa giving a talk", "--saida", str(output),
                "--peso", "0.55",
            ]
            with (
                mock.patch.object(KREA, "MFLUX", str(binary)),
                mock.patch.object(KREA, "LORA", str(famegrid)),
                mock.patch.object(KREA, "IDENTITIES_FILE", str(registry)),
                mock.patch.object(KREA, "model_path", return_value=str(model)),
                mock.patch.object(KREA.subprocess, "run", side_effect=fake_run),
                mock.patch.object(sys, "argv", argv),
            ):
                self.assertEqual(KREA.main(), 0)

            command = seen["command"]
            famegrid_position = command.index(str(famegrid))
            self.assertEqual(command[famegrid_position + 1], "0.55")


if __name__ == "__main__":
    unittest.main()
