from __future__ import annotations

import json
import time
import unittest

import numpy as np

from ai_guide import (
    AsyncNarrator,
    GuideEngine,
    GuideMessage,
    OpenAICompatibleNarrator,
    ProviderConfig,
    TemplateNarrator,
    WorldAnalyzer,
    WorldEvent,
    test_provider_connection,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BadProvider:
    def generate(self, **kwargs):
        raise RuntimeError("boom")


class GuideTests(unittest.TestCase):
    def test_world_analyzer_detects_water_growth(self) -> None:
        analyzer = WorldAnalyzer()
        dry = np.full((40, 40), 0.75, dtype=np.float32)
        wet = dry.copy()
        wet[:, :20] = 0.2

        first = analyzer.analyze(dry, 1.0, shark_count=0, dinosaur_count=1)
        second = analyzer.with_deltas(
            analyzer.analyze(wet, 2.0, shark_count=2, dinosaur_count=1),
            first,
        )

        self.assertGreater(second.deltas["water_ratio"], 0.08)
        self.assertGreaterEqual(second.shark_count, 2)

    def test_local_template_mode_generates_message_and_challenge(self) -> None:
        engine = GuideEngine(template_narrator=TemplateNarrator(), event_cooldown=0.01)
        try:
            frame = np.full((48, 48), 0.6, dtype=np.float32)
            message = engine.update(
                frame,
                time.monotonic(),
                shark_count=0,
                dinosaur_count=3,
                creatures_enabled=True,
                guide_enabled=True,
                challenges_enabled=True,
                verbosity="normal",
            )
            self.assertIsNotNone(message)
            self.assertIsNotNone(engine.active_challenge)
        finally:
            engine.close()

    def test_openai_compatible_narrator_accepts_local_endpoint(self) -> None:
        provider = OpenAICompatibleNarrator(
            ProviderConfig(
                backend="local_openai_compatible",
                base_url="http://127.0.0.1:1234/v1",
                model="test-model",
            ),
            opener=lambda req, timeout: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"title": "Tiny Ocean", "body": "A bright blue lake is growing."}
                                )
                            }
                        }
                    ]
                }
            ),
        )
        world = WorldAnalyzer().analyze(np.full((20, 20), 0.2, dtype=np.float32), 1.0, shark_count=2, dinosaur_count=0)
        message = GuideMessage("observation", "Lake", "A lake is forming.")
        result = provider.generate(
            world_state=world,
            message=message,
            challenge=None,
            verbosity="normal",
        )
        self.assertEqual(result["title"], "Tiny Ocean")

    def test_openai_compatible_narrator_accepts_cloud_endpoint(self) -> None:
        provider = OpenAICompatibleNarrator(
            ProviderConfig(
                backend="cloud_openai_compatible",
                base_url="https://example.com/v1",
                model="cloud-model",
            ),
            opener=lambda req, timeout: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"title": "Sky Ridge", "body": "A mountain ridge is pushing upward."}
                                )
                            }
                        }
                    ]
                }
            ),
        )
        world = WorldAnalyzer().analyze(np.full((20, 20), 0.85, dtype=np.float32), 1.0, shark_count=0, dinosaur_count=2)
        message = GuideMessage("observation", "Peak", "A peak is rising.")
        result = provider.generate(
            world_state=world,
            message=message,
            challenge=None,
            verbosity="lively",
        )
        self.assertEqual(result["body"], "A mountain ridge is pushing upward.")

    def test_bad_llm_provider_falls_back_to_template_copy(self) -> None:
        async_narrator = AsyncNarrator(_BadProvider())
        engine = GuideEngine(
            template_narrator=TemplateNarrator(),
            async_narrator=async_narrator,
            event_cooldown=0.01,
        )
        try:
            frame = np.full((48, 48), 0.6, dtype=np.float32)
            message = engine.update(
                frame,
                time.monotonic(),
                shark_count=0,
                dinosaur_count=2,
                creatures_enabled=True,
                guide_enabled=True,
                challenges_enabled=True,
                verbosity="normal",
            )
            self.assertIsNotNone(message)
            self.assertEqual(message.title, "Sandbox Ready")
        finally:
            engine.close()

    def test_creature_loss_event_is_suppressed_when_creatures_disabled(self) -> None:
        engine = GuideEngine(template_narrator=TemplateNarrator())
        prev = WorldAnalyzer().analyze(np.full((20, 20), 0.2, dtype=np.float32), 1.0, shark_count=2, dinosaur_count=0)
        current = WorldAnalyzer().with_deltas(
            WorldAnalyzer().analyze(np.full((20, 20), 0.2, dtype=np.float32), 2.0, shark_count=0, dinosaur_count=0),
            prev,
        )
        event = engine._pick_event(current, prev, creatures_enabled=False)
        self.assertNotEqual(event, WorldEvent("shark_habitat_lost", "shark_habitat_lost", {}))

    def test_template_connection_test_reports_ready(self) -> None:
        result = test_provider_connection(ProviderConfig(backend="template"))
        self.assertTrue(result.ok)
        self.assertIn("ready", result.summary.lower())

    def test_connection_test_reports_success_for_openai_compatible_backend(self) -> None:
        result = test_provider_connection(
            ProviderConfig(
                backend="local_openai_compatible",
                base_url="http://127.0.0.1:1234/v1",
                model="test-model",
            ),
            opener=lambda req, timeout: _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {"title": "Connection Passed", "body": "The model answered."}
                                )
                            }
                        }
                    ]
                }
            ),
        )
        self.assertTrue(result.ok)
        self.assertIn("Connection Passed", result.summary)


if __name__ == "__main__":
    unittest.main()
