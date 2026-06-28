import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.detector import _candidate_model_paths, _ensure_ultralytics_config_dir


class DetectorConfigTest(unittest.TestCase):
    def test_candidate_model_path_defaults_to_onnx(self):
        self.assertEqual(
            _candidate_model_paths("yolo/epoch51"),
            ["yolo/epoch51.onnx"],
        )

    def test_candidate_model_path_keeps_explicit_onnx(self):
        self.assertEqual(
            _candidate_model_paths("yolo/epoch51.onnx"),
            ["yolo/epoch51.onnx"],
        )

    def test_ultralytics_config_dir_is_created_before_import(self):
        old_value = os.environ.pop("YOLO_CONFIG_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = _ensure_ultralytics_config_dir(Path(temp_dir))
                self.assertTrue(path.exists())
                self.assertEqual(os.environ["YOLO_CONFIG_DIR"], str(path))
        finally:
            if old_value is None:
                os.environ.pop("YOLO_CONFIG_DIR", None)
            else:
                os.environ["YOLO_CONFIG_DIR"] = old_value


if __name__ == "__main__":
    unittest.main()
