import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_photos_identity", ROOT / "src" / "prepare_photos_identity.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class IdentityCaptionTests(unittest.TestCase):
    def test_draft_does_not_caption_stable_identity_features(self):
        item = {
            "_bucket": "close",
            "_target_face": {"has_smile": True, "glasses_type": 1},
        }

        result = PREPARE.caption(item, "rare_person_token")

        self.assertIn("rare_person_token", result)
        self.assertIn("wearing glasses", result)
        self.assertNotIn("hair", result.lower())
        self.assertNotIn("skin", result.lower())


if __name__ == "__main__":
    unittest.main()
