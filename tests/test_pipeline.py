import importlib.util
import pathlib
import tempfile
import unittest

from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pipeline", ROOT / "src" / "pipeline.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class OutputValidationTests(unittest.TestCase):
    def test_rejects_black_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "black.png"
            Image.new("RGB", (128, 128), "black").save(path)
            self.assertFalse(PIPELINE.imagem_valida(path))

    def test_accepts_real_varying_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "varying.png"
            image = Image.new("RGB", (128, 128), (30, 60, 90))
            for x in range(64, 128):
                for y in range(128):
                    image.putpixel((x, y), (180, 140, 100))
            image.save(path)
            self.assertTrue(PIPELINE.imagem_valida(path))


if __name__ == "__main__":
    unittest.main()
