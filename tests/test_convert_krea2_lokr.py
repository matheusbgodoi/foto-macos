import importlib.util
import pathlib
import unittest

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "convert_krea2_lokr", ROOT / "src" / "convert_krea2_lycoris_lokr.py"
)
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)


class KreaLoKrConversionTests(unittest.TestCase):
    def test_splits_fused_qkv_without_approximation(self):
        prefix = "lycoris_transformer_blocks_3_attn_to_"
        source = {}
        for name, rows in (("q", 4), ("k", 1), ("v", 1)):
            source[f"{prefix}{name}.lokr_w1"] = torch.zeros(2, 2)
            source[f"{prefix}{name}.lokr_w2"] = torch.zeros(rows, 2)
            source[f"{prefix}{name}.alpha"] = torch.tensor(1.0)
        w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        w2 = torch.arange(12.0).reshape(6, 2)
        source[f"{prefix}qkv.lokr_w1"] = w1
        source[f"{prefix}qkv.lokr_w2"] = w2
        source[f"{prefix}qkv.alpha"] = torch.tensor(1.0)
        gate = f"{prefix}gate"
        source[f"{gate}.lokr_w1"] = torch.ones(2, 2)
        source[f"{gate}.lokr_w2"] = torch.ones(2, 2)
        source[f"{gate}.alpha"] = torch.tensor(1.0)

        result = CONVERT.convert_tensors(source)
        base = "transformer_blocks.3.attn.to_"
        self.assertTrue(torch.equal(result[f"{base}q.lokr_w1"], w1))
        self.assertTrue(torch.equal(result[f"{base}q.lokr_w2"], w2[:4]))
        self.assertTrue(torch.equal(result[f"{base}k.lokr_w2"], w2[4:5]))
        self.assertTrue(torch.equal(result[f"{base}v.lokr_w2"], w2[5:]))
        self.assertIn(f"{base}gate.lokr_w1", result)
        self.assertNotIn(f"{base}qkv.lokr_w1", result)

    def test_rejects_adapter_with_zero_effective_factor(self):
        prefix = "lycoris_text_fusion_layerwise_blocks_0_attn_to_"
        source = {}
        for name in ("q", "k", "v"):
            source[f"{prefix}{name}.lokr_w1"] = torch.zeros(2, 2)
            source[f"{prefix}{name}.lokr_w2"] = torch.zeros(1, 2)
        source[f"{prefix}qkv.lokr_w1"] = torch.zeros(2, 2)
        source[f"{prefix}qkv.lokr_w2"] = torch.ones(3, 2)
        with self.assertRaisesRegex(ValueError, "sem efeito"):
            CONVERT.convert_tensors(source)

    def test_keeps_unfused_qkv_targets(self):
        source = {}
        for name in ("q", "k", "v"):
            base = f"lycoris_transformer_blocks_1_attn_to_{name}"
            source[f"{base}.lokr_w1"] = torch.ones(2, 2)
            source[f"{base}.lokr_w2"] = torch.ones(2, 2)
            source[f"{base}.alpha"] = torch.tensor(1.0)
        result = CONVERT.convert_tensors(source)
        for name in ("q", "k", "v"):
            self.assertIn(
                f"transformer_blocks.1.attn.to_{name}.lokr_w1", result
            )


if __name__ == "__main__":
    unittest.main()
