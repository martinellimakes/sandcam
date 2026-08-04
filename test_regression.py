from __future__ import annotations

import time
import unittest

import numpy as np
import pygame

from creatures import CreatureManager
from cv_interaction import CVInteractionEngine
from ui import Config, draw_guide_overlay
from webcam_observer import CVTrackedObject


class CreatureTrimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((64, 64))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_set_targets_trims_excess_creatures(self) -> None:
        manager = CreatureManager(n_sharks=3, n_dinos=2)
        # Mixed terrain so both sharks (water) and dinosaurs (land) can spawn.
        frame = np.full((80, 80), 0.8, dtype=np.float32)
        frame[0:40, :] = 0.2
        for _ in range(5):
            manager.update(frame, 0.016)
        before = manager.counts()
        self.assertGreaterEqual(before["sharks"], 1)
        self.assertGreaterEqual(before["dinosaurs"], 1)

        manager.set_targets(sharks=1, dinosaurs=0)
        manager.update(frame, 0.016)
        trimmed = manager.counts()
        self.assertLessEqual(trimmed["sharks"], 1)
        self.assertEqual(trimmed["dinosaurs"], 0)


class CVInteractionTests(unittest.TestCase):
    def test_template_event_without_llm(self) -> None:
        engine = CVInteractionEngine()
        engine.disable_llm()
        frame = np.full((40, 40), 0.8, dtype=np.float32)
        obj = CVTrackedObject(
            track_id=7,
            label="toy car",
            confidence=0.9,
            bbox=(10, 10, 30, 30),
            camera_pos=(20.0, 20.0),
            sandbox_pos=(15.0, 15.0),
            biome="land",
            stable_for_seconds=1.0,
        )
        engine.update([obj], frame, time.monotonic(), interactions_enabled=True)
        events = engine.pop_events()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].title)
        self.assertTrue(events[0].body)


class VisionToastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        cls.surface = pygame.display.set_mode((400, 300))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_force_overlay_draws_without_ai_features(self) -> None:
        config = Config()
        config.ai_enabled = False
        config.guide_enabled = False

        gated = pygame.Surface((400, 300))
        gated.fill((0, 0, 0))
        draw_guide_overlay(
            gated,
            config,
            title="Boat Afloat",
            body="A toy boat is sailing on the lake.",
            challenge_text=None,
            force=False,
        )
        self.assertEqual(gated.get_at((30, 30))[:3], (0, 0, 0))

        toast = pygame.Surface((400, 300))
        toast.fill((0, 0, 0))
        draw_guide_overlay(
            toast,
            config,
            title="Boat Afloat",
            body="A toy boat is sailing on the lake.",
            challenge_text=None,
            force=True,
        )
        # Card is drawn near top-left; sample inside the expected card bounds.
        self.assertNotEqual(toast.get_at((40, 40))[:3], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
