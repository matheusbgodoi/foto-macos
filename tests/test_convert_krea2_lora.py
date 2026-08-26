import importlib.util
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_krea2_lora", ROOT / "src" / "convert_krea2_fused_qkv_lora.py"
)
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


class KreaFusedQKVTests(unittest.TestCase):
    def test_transformer_gqa_split_preserves_delta(self):
        prefix = "transformer.transformer_blocks.0.attn."
        rank = 2
        matrix_a = np.arange(rank * 6144, dtype=np.float32).reshape(rank, 6144)
        matrix_b = np.arange(9216 * rank, dtype=np.float32).reshape(9216, rank)
        result = CONVERT.convert_tensors({
            prefix + "to_qkv.lora_A.weight": matrix_a,
            prefix + "to_qkv.lora_B.weight": matrix_b,
        })

        sizes = (6144, 1536, 1536)
        start = 0
        for name, size in zip(("q", "k", "v"), sizes):
            np.testing.assert_array_equal(
                result[prefix + f"to_{name}.lora_A.weight"], matrix_a
            )
            split_b = result[prefix + f"to_{name}.lora_B.weight"]
            np.testing.assert_array_equal(split_b @ matrix_a, matrix_b[start:start + size] @ matrix_a)
            start += size

    def test_text_fusion_splits_equally_and_keeps_other_layers(self):
        prefix = "transformer.text_fusion.refiner_blocks.1.attn."
        matrix_a = np.ones((1, 2560), dtype=np.float32)
        matrix_b = np.arange(7680, dtype=np.float32).reshape(7680, 1)
        other = np.array([[7]], dtype=np.float32)
        result = CONVERT.convert_tensors({
            prefix + "to_qkv.lora_A.weight": matrix_a,
            prefix + "to_qkv.lora_B.weight": matrix_b,
            prefix + "to_out.0.lora_A.weight": other,
        })

        self.assertEqual(result[prefix + "to_q.lora_B.weight"].shape, (2560, 1))
        self.assertEqual(result[prefix + "to_k.lora_B.weight"].shape, (2560, 1))
        self.assertEqual(result[prefix + "to_v.lora_B.weight"].shape, (2560, 1))
        np.testing.assert_array_equal(result[prefix + "to_out.0.lora_A.weight"], other)

    def test_rejects_unknown_architecture(self):
        with self.assertRaisesRegex(ValueError, "modulo Krea 2 conhecido"):
            CONVERT.convert_tensors({
                "transformer.unknown.attn.to_qkv.lora_A.weight": np.zeros((1, 2)),
                "transformer.unknown.attn.to_qkv.lora_B.weight": np.zeros((6, 1)),
            })


if __name__ == "__main__":
    unittest.main()
