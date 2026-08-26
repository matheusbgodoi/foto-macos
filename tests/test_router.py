import importlib.util
import pathlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
