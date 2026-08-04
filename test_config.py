from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ui import CONFIG_PATH, Config, DEFAULT_LOCAL_BASE_URL, _flatten_saved_config


class ConfigTests(unittest.TestCase):
    def test_flatten_nested_schema(self) -> None:
        raw = {
            "debug": {"enabled": True},
            "terrain": {
                "source": "kinect",
                "colour_scheme": "heat",
                "show_contours": False,
                "depth": {"min_mm": 350, "max_mm": 900},
                "smoothing": {
                    "blend": 0.2,
                    "threshold_mm": 20,
                    "delay_frames": 3,
                    "reject_mm": 80,
                },
            },
            "guide": {
                "enabled": True,
                "llm": {
                    "enabled": True,
                    "provider": {
                        "location": "local",
                        "base_url": "http://localhost:12434/v1",
                        "model": "demo",
                    },
                },
            },
            "vision": {
                "enabled": True,
                "detection": {
                    "enabled": True,
                    "backend": "yolo",
                    "reasoner": {
                        "base_url": "http://localhost:12434/engines/v1/chat/completions",
                        "model": "vision-demo",
                    },
                },
            },
        }
        flat = _flatten_saved_config(raw)
        self.assertTrue(flat["debug_mode"])
        self.assertEqual(flat["depth_source"], "kinect")
        self.assertEqual(flat["colour_scheme"], "heat")
        self.assertEqual(flat["min_depth_mm"], 350)
        self.assertEqual(flat["llm_model"], "demo")
        self.assertEqual(flat["cv_detection_api_model"], "vision-demo")

    def test_load_normalizes_urls_and_defaults(self) -> None:
        payload = {
            "terrain": {"source": "simulator", "colour_scheme": "terrain"},
            "guide": {
                "enabled": False,
                "llm": {
                    "enabled": False,
                    "provider": {
                        "base_url": "http://localhost:12434/v1/",
                        "model": "",
                    },
                },
            },
            "vision": {
                "enabled": False,
                "detection": {
                    "backend": "yolo",
                    "reasoner": {"base_url": "http://127.0.0.1:12434/v1"},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sandcam-settings.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("ui.CONFIG_PATH", path):
                config = Config.load()
        self.assertEqual(config.depth_source, "simulator")
        self.assertEqual(config.llm_base_url, DEFAULT_LOCAL_BASE_URL)
        self.assertEqual(config.cv_detection_api_url, "http://127.0.0.1:12434/engines/v1")
        self.assertFalse(config.ai_enabled)
        self.assertFalse(config.vision_enabled)
        self.assertEqual(config.cv_detection_backend, "yolo")

    def test_save_round_trip_preserves_source_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sandcam-settings.json"
            with patch("ui.CONFIG_PATH", path):
                config = Config()
                config.depth_source = "kinect"
                config.debug_mode = True
                config.cv_detection_backend = "yolo"
                config.save()
                reloaded = Config.load()
                saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(reloaded.depth_source, "kinect")
            self.assertTrue(reloaded.debug_mode)
            self.assertEqual(saved["terrain"]["source"], "kinect")
            self.assertIn("reasoner", saved["vision"]["detection"])


class CalibrationAssetTests(unittest.TestCase):
    def test_calibration_filenames_match_corner_ids(self) -> None:
        from webcam_observer import CORNER_MARKER_IDS

        calibration_dir = Path(__file__).with_name("calibration")
        expected = {
            100: "TL-100.svg",
            101: "TR-101.svg",
            102: "BL-102.svg",
            103: "BR-103.svg",
        }
        self.assertEqual(CORNER_MARKER_IDS[100], "top_left")
        self.assertEqual(CORNER_MARKER_IDS[102], "bottom_left")
        self.assertEqual(CORNER_MARKER_IDS[103], "bottom_right")
        for marker_id, filename in expected.items():
            path = calibration_dir / filename
            self.assertTrue(path.exists(), f"missing {filename}")
            self.assertIn(str(marker_id), filename)


if __name__ == "__main__":
    unittest.main()
