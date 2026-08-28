import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import krea2_reid


class Krea2ReIDGraphTests(unittest.TestCase):
    def test_matches_official_runtime_settings(self):
        graph = krea2_reid.build(
            "same person in a new scene", "reference.png", 123,
            (768, 1024), "test/reid")

        self.assertEqual(graph["2"]["inputs"]["unet_name"],
                         "krea2_turbo_int8_convrot.safetensors")
        self.assertEqual(graph["32"]["inputs"]["strength_model"], 1.0)
        self.assertIs(graph["33"]["inputs"]["kv_cache"], True)
        self.assertEqual(graph["7"]["inputs"]["steps"], 8)
        self.assertEqual(graph["7"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["7"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["7"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["6"]["inputs"]["width"], 768)
        self.assertEqual(graph["6"]["inputs"]["height"], 1024)

    def test_reference_reaches_positive_and_negative_conditioning(self):
        graph = krea2_reid.build("prompt", "ref.png", 1,
                                 (512, 512), "test/reid")
        for node in ("34", "35"):
            self.assertEqual(graph[node]["inputs"]["image1"], ["28", 0])
            self.assertEqual(graph[node]["inputs"]["vae"], ["1", 0])
        self.assertEqual(
            graph["21"]["inputs"]["reference_latents_method"],
            "index_timestep_zero")


if __name__ == "__main__":
    unittest.main()
