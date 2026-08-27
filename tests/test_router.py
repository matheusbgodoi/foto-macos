import importlib.util
import pathlib
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gerar_coringa", ROOT / "src" / "gerar_coringa.py")
ROUTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER)


class RouterTests(unittest.TestCase):
    def test_detects_explicit_styles(self):
        cases = {
            "um castelo em pixel art 8-bit": "pixel-art",
            "um cachorro em desenho animado": "cartoon",
            "personagem de anime": "anime",
            "foto de celular na praia": "iphone",
            "qualidade máxima, indistinguível de real": "famegrid",
            "Famegrid em uma cozinha": "famegrid",
            "uma pessoa na rua": "foto-natural",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(ROUTER.detect_style(prompt), expected)

    def test_every_public_style_has_a_prompt_or_preset(self):
        for style in ("foto-natural", "iphone", "profissional", "produto"):
            self.assertIn(style, ROUTER.PHOTO_PRESETS)
            self.assertIn(style, ROUTER.PHOTO_PROMPTS)
        for style in ("cartoon", "pixel-art", "ilustracao", "anime"):
            self.assertIn(style, ROUTER.STYLE_PROMPTS)

    def test_identity_forces_krea_without_overwriting_style(self):
        self.assertEqual(ROUTER.detect_style(
            "Pessoa em uma palestra, foto de iPhone"), "iphone")
        self.assertEqual(ROUTER.select_engine(
            "auto", "iphone", True, [], True), "krea2")

    def test_explicit_engine_still_wins(self):
        self.assertEqual(ROUTER.select_engine(
            "flux2", "iphone", True, [], True), "flux2")

    def test_router_fallbacks(self):
        self.assertEqual(ROUTER.select_engine(
            "auto", "foto-natural", False, [], True), "drawthings")
        self.assertEqual(ROUTER.select_engine(
            "auto", "foto-natural", False, [], False), "flux2")
        self.assertEqual(ROUTER.select_engine(
            "auto", "cartoon", False, ["style.safetensors"], True), "sdxl")

    def test_krea_route_passes_disable_famegrid(self):
        args = types.SimpleNamespace(
            prompt="Pessoa at a desk", tamanho="768x1024", seed=42,
            sem_famegrid=True,
        )
        with mock.patch.object(ROUTER, "run", return_value=0) as run:
            self.assertEqual(ROUTER.mlx_krea2(
                args, "/tmp/result.png", "foto-natural"), 0)
        command = run.call_args.args[0]
        self.assertIn("--sem-famegrid", command)


if __name__ == "__main__":
    unittest.main()
