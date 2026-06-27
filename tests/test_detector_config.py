import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitle.detector import (
    _candidate_model_paths,
    _ensure_ultralytics_config_dir,
    _inject_tensorrt_runtime_paths,
)


class DetectorConfigTest(unittest.TestCase):
    def test_gpu_candidates_default_to_onnx(self):
        self.assertEqual(
            _candidate_model_paths("yolo/epoch51", "gpu"),
            ["yolo/epoch51.onnx"],
        )

    def test_tensorrt_candidates_are_opt_in(self):
        self.assertEqual(
            _candidate_model_paths("yolo/epoch51", "gpu", backend="tensorrt"),
            ["yolo/epoch51.engine", "yolo/epoch51.onnx"],
        )

    def test_cpu_candidates_skip_engine(self):
        self.assertEqual(
            _candidate_model_paths("yolo/epoch51.engine", "cpu", backend="tensorrt"),
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

    def test_tensorrt_home_lib_is_added_to_path(self):
        old_home = os.environ.get("TENSORRT_HOME")
        old_path = os.environ.get("PATH", "")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                lib_dir = Path(temp_dir) / "lib"
                lib_dir.mkdir()
                os.environ["TENSORRT_HOME"] = temp_dir

                added = _inject_tensorrt_runtime_paths()

                self.assertIn(str(lib_dir), added)
                self.assertIn(str(lib_dir), os.environ["PATH"])
        finally:
            if old_home is None:
                os.environ.pop("TENSORRT_HOME", None)
            else:
                os.environ["TENSORRT_HOME"] = old_home
            os.environ["PATH"] = old_path


if __name__ == "__main__":
    unittest.main()
