import importlib.util
import unittest
from pathlib import Path

from model_manifest import build_safetensors_manifest, discover_safetensors_weight_files, sha256_file

_HAS_NUMPY = importlib.util.find_spec("numpy") is not None
_HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


class ModelManifestHashTests(unittest.TestCase):
    def test_sha256_file_stable(self) -> None:
        p = Path(__file__).resolve().parent / "test_model_manifest.py"
        a = sha256_file(p)
        b = sha256_file(p)
        self.assertEqual(len(a), 64)
        self.assertEqual(a, b)

    def test_discover_empty_dir(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(discover_safetensors_weight_files(Path(d)), [])


@unittest.skipUnless(_HAS_NUMPY and _HAS_SAFETENSORS, "requires numpy and safetensors")
class ModelManifestSafetensorsTests(unittest.TestCase):
    def test_discover_single_model_safetensors(self) -> None:
        import tempfile

        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            save_file(
                {"layer.weight": np.zeros((2, 3), dtype=np.float32)},
                str(root / "model.safetensors"),
            )
            files = discover_safetensors_weight_files(root)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "model.safetensors")

    def test_build_manifest_keys_and_sha(self) -> None:
        import tempfile

        import numpy as np
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            save_file(
                {"a": np.ones(1, dtype=np.float32), "b": np.zeros(1, dtype=np.float32)},
                str(root / "model.safetensors"),
            )
            m = build_safetensors_manifest(root, hf_model_id="dummy/Test")
            self.assertIsNone(m.get("error"))
            self.assertEqual(m["safetensors_key_count"], 2)
            self.assertEqual(set(m["safetensors_keys"]), {"a", "b"})
            self.assertTrue(m["model_safetensors_sha256"])


if __name__ == "__main__":
    unittest.main()
