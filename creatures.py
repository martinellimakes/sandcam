"""
Animated creatures that react to the terrain height map.

Sharks swim in water (elevation < WATER_LEVEL).
Dinos walk on land (elevation >= WATER_LEVEL).

The sprites are drawn as chunky top-down pixel art, then rotated and blitted
at runtime. Creatures are culled if terrain changes under them and no longer
matches their biome.
"""

from __future__ import annotations

import math
import random
from typing import ClassVar

import numpy as np
import pygame

from renderer import WATER_LEVEL

# Tuning
_ARRIVAL_RADIUS = 24
_STEER_RATE = 6.0
_WANDER_NOISE = 0.08


class Creature:
    """
    Base class for terrain-aware animated creatures.

    Subclasses must implement:
        is_valid(elevation) -> bool
        draw(surface) -> None
    """

    def __init__(self, x: float, y: float, speed: float) -> None:
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.angle = random.uniform(0, math.tau)
        self.speed = speed
        self._target: np.ndarray | None = None
        self._wander = 0.0

    def is_valid(self, elevation: float) -> bool:
        raise NotImplementedError

    def draw(self, surface: pygame.Surface) -> None:
        raise NotImplementedError

    def alive_on(self, frame: np.ndarray) -> bool:
        """Return True while the creature is still standing in valid habitat."""
        return self._valid_pos(frame, self.pos)

    def update(self, frame: np.ndarray, dt: float) -> None:
        h, w = frame.shape

        if (
            self._target is None
            or np.linalg.norm(self._target - self.pos) < _ARRIVAL_RADIUS
            or not self._valid_pos(frame, self._target)
        ):
            self._target = self._pick_target(frame, w, h)

        if self._target is None:
            return

        desired = self._target - self.pos
        dist = np.linalg.norm(desired)
        if dist > 0:
            desired /= dist

        self._wander += random.gauss(0, _WANDER_NOISE)
        self._wander = float(np.clip(self._wander, -0.6, 0.6))
        wander_vec = np.array([math.cos(self._wander), math.sin(self._wander)])
        direction = desired * 0.88 + wander_vec * 0.12
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm

        target_angle = math.atan2(direction[1], direction[0])
        diff = (target_angle - self.angle + math.pi) % math.tau - math.pi
        self.angle += diff * min(1.0, _STEER_RATE * dt)

        move = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed * dt
        new_pos = self.pos + move
        if self._valid_pos(frame, new_pos):
            self.pos = new_pos
            self.vel = move / dt if dt > 0 else self.vel
        else:
            self.angle += math.pi + random.uniform(-0.4, 0.4)
            self._target = None

    def _sample(self, frame: np.ndarray, pos: np.ndarray) -> float:
        h, w = frame.shape
        x = int(np.clip(pos[0], 0, w - 1))
        y = int(np.clip(pos[1], 0, h - 1))
        return float(frame[y, x])

    def _valid_pos(self, frame: np.ndarray, pos: np.ndarray) -> bool:
        h, w = frame.shape
        if pos[0] < 0 or pos[0] >= w or pos[1] < 0 or pos[1] >= h:
            return False
        return self.is_valid(self._sample(frame, pos))

    def _pick_target(self, frame: np.ndarray, w: int, h: int) -> np.ndarray | None:
        for radius in (150, 350, max(w, h)):
            for _ in range(20):
                angle = random.uniform(0, math.tau)
                distance = random.uniform(40, radius)
                tx = float(np.clip(self.pos[0] + math.cos(angle) * distance, 0, w - 1))
                ty = float(np.clip(self.pos[1] + math.sin(angle) * distance, 0, h - 1))
                candidate = np.array([tx, ty])
                if self._valid_pos(frame, candidate):
                    return candidate
        return None

    @staticmethod
    def _shadow(surface: pygame.Surface, pos: np.ndarray, rx: int, ry: int) -> None:
        shadow = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 45), shadow.get_rect())
        surface.blit(shadow, (int(pos[0]) - rx, int(pos[1]) - ry + ry // 2))

    @staticmethod
    def _surface(size: tuple[int, int]) -> pygame.Surface:
        return pygame.Surface(size, pygame.SRCALPHA)

    @staticmethod
    def _points(
        points: list[tuple[float, float]], center: tuple[float, float]
    ) -> list[tuple[int, int]]:
        cx, cy = center
        return [(int(cx + px), int(cy + py)) for px, py in points]

    def _blit_sprite(self, surface: pygame.Surface, sprite: pygame.Surface) -> None:
        rotated = pygame.transform.rotozoom(sprite, -math.degrees(self.angle), 1.0)
        rect = rotated.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        surface.blit(rotated, rect)

    @staticmethod
    def _scale_pixel_art(sprite: pygame.Surface, scale: int) -> pygame.Surface:
        width, height = sprite.get_size()
        return pygame.transform.scale(sprite, (width * scale, height * scale))


class Shark(Creature):
    """Swims in water using a top-down pixel-art sprite loop."""

    _BODY_COLOR: ClassVar = (75, 125, 155)
    _DARK_COLOR: ClassVar = (50, 90, 115)
    _FIN_COLOR: ClassVar = (55, 100, 130)
    _FRAME_SWEEPS: ClassVar = (-1.0, -0.35, 0.35, 1.0)

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y, speed=random.uniform(55, 95))
        self._phase = random.uniform(0, math.tau)
        self._scale = random.uniform(0.85, 1.2)
        self._sprites = self._build_sprites()

    def is_valid(self, elevation: float) -> bool:
        return elevation < WATER_LEVEL

    def update(self, frame: np.ndarray, dt: float) -> None:
        self._phase += dt * 3.2
        super().update(frame, dt)

    def draw(self, surface: pygame.Surface) -> None:
        scale = self._scale
        self._shadow(surface, self.pos, int(22 * scale), int(8 * scale))
        self._blit_sprite(surface, self._sprites[self._frame_index()])

    def _frame_index(self) -> int:
        cycle = (self._phase % math.tau) / math.tau
        return int(cycle * len(self._sprites)) % len(self._sprites)

    def _build_sprites(self) -> list[pygame.Surface]:
        pixel_scale = max(3, int(round(self._scale * 3)))
        sprites: list[pygame.Surface] = []

        for sweep in self._FRAME_SWEEPS:
            surf = self._surface((18, 12))
            tail_shift = int(round(sweep * 2))

            body = [
                (2, 5),
                (4, 3),
                (8, 2),
                (12, 3),
                (15, 5),
                (12, 7),
                (8, 8),
                (4, 7),
            ]
            pygame.draw.polygon(surf, self._BODY_COLOR, body)
            pygame.draw.polygon(surf, self._DARK_COLOR, body, 1)

            dorsal_fin = [
                (7, 5),
                (9, 2),
                (10, 5),
            ]
            pygame.draw.polygon(surf, self._FIN_COLOR, dorsal_fin)

            left_fin = [
                (8, 3),
                (10, 1),
                (11, 3),
            ]
            right_fin = [
                (8, 7),
                (10, 9),
                (11, 7),
            ]
            pygame.draw.polygon(surf, self._FIN_COLOR, left_fin)
            pygame.draw.polygon(surf, self._FIN_COLOR, right_fin)

            tail_top = [
                (1, 5),
                (0, 2 + tail_shift),
                (3, 4),
            ]
            tail_bottom = [
                (1, 5),
                (0, 8 - tail_shift),
                (3, 6),
            ]
            pygame.draw.polygon(surf, self._FIN_COLOR, tail_top)
            pygame.draw.polygon(surf, self._FIN_COLOR, tail_bottom)

            pygame.draw.rect(surf, (220, 230, 240), pygame.Rect(12, 4, 1, 1))
            pygame.draw.rect(surf, (220, 230, 240), pygame.Rect(12, 6, 1, 1))
            pygame.draw.rect(surf, (10, 10, 10), pygame.Rect(12, 4, 1, 1), 1)
            pygame.draw.rect(surf, (10, 10, 10), pygame.Rect(12, 6, 1, 1), 1)

            sprites.append(self._scale_pixel_art(surf, pixel_scale))

        return sprites


class Dinosaur(Creature):
    """Walks on land using a top-down pixel-art sprite loop."""

    _PALETTES: ClassVar = [
        ((75, 140, 60), (45, 100, 35)),
        ((140, 95, 55), (100, 65, 30)),
        ((155, 75, 75), (110, 45, 45)),
        ((80, 120, 140), (50, 85, 105)),
    ]
    _FRAME_SWINGS: ClassVar = (-1.0, -0.35, 0.35, 1.0)

    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y, speed=random.uniform(35, 65))
        self._body_col, self._dark_col = random.choice(self._PALETTES)
        self._phase = random.uniform(0, math.tau)
        self._scale = random.uniform(0.8, 1.15)
        self._sprites = self._build_sprites()

    def is_valid(self, elevation: float) -> bool:
        return elevation >= WATER_LEVEL

    def update(self, frame: np.ndarray, dt: float) -> None:
        self._phase += dt * self.speed * 0.07
        super().update(frame, dt)

    def draw(self, surface: pygame.Surface) -> None:
        scale = self._scale
        self._shadow(surface, self.pos, int(20 * scale), int(7 * scale))
        self._blit_sprite(surface, self._sprites[self._frame_index()])

    def _frame_index(self) -> int:
        cycle = (self._phase % math.tau) / math.tau
        return int(cycle * len(self._sprites)) % len(self._sprites)

    def _build_sprites(self) -> list[pygame.Surface]:
        pixel_scale = max(3, int(round(self._scale * 3)))
        sprites: list[pygame.Surface] = []

        for swing_ratio in self._FRAME_SWINGS:
            surf = self._surface((22, 18))
            leg_shift = int(round(swing_ratio * 2))
            tail_shift = int(round(swing_ratio))

            tail = [
                (3, 8),
                (0, 6 - tail_shift),
                (1, 8),
                (0, 10 + tail_shift),
            ]
            pygame.draw.polygon(surf, self._body_col, tail)
            pygame.draw.polygon(surf, self._dark_col, tail, 1)

            body = [
                (4, 5),
                (9, 3),
                (13, 3),
                (16, 5),
                (17, 8),
                (16, 11),
                (13, 13),
                (9, 13),
                (4, 11),
                (2, 8),
            ]
            pygame.draw.polygon(surf, self._body_col, body)
            pygame.draw.polygon(surf, self._dark_col, body, 1)

            stripe = [
                (7, 5),
                (11, 4),
                (13, 8),
                (11, 12),
                (7, 11),
                (5, 8),
            ]
            pygame.draw.polygon(surf, self._dark_col, stripe, 1)

            head = [
                (15, 5),
                (19, 4),
                (21, 6),
                (21, 10),
                (19, 12),
                (15, 11),
                (17, 8),
            ]
            pygame.draw.polygon(surf, self._body_col, head)
            pygame.draw.polygon(surf, self._dark_col, head, 1)

            for y, sign in ((5, 1), (11, -1)):
                pygame.draw.line(surf, self._dark_col, (5, y), (2, y + leg_shift * sign), 1)
                pygame.draw.line(surf, self._dark_col, (2, y + leg_shift * sign), (0, y + leg_shift * sign), 1)

            pygame.draw.rect(surf, (240, 240, 200), pygame.Rect(18, 6, 1, 1))
            pygame.draw.rect(surf, (240, 240, 200), pygame.Rect(18, 9, 1, 1))
            pygame.draw.rect(surf, (20, 20, 20), pygame.Rect(18, 6, 1, 1), 1)
            pygame.draw.rect(surf, (20, 20, 20), pygame.Rect(18, 9, 1, 1), 1)

            sprites.append(self._scale_pixel_art(surf, pixel_scale))

        return sprites


class CreatureManager:
    """
    Spawns and updates all creatures.

    Creatures are spawned lazily on the first frame that has valid terrain for
    them, so it is safe to create the manager before depth data arrives.
    """

    def __init__(self, n_sharks: int = 4, n_dinos: int = 5) -> None:
        self._targets = {Shark: n_sharks, Dinosaur: n_dinos}
        self._creatures: list[Creature] = []

    def update(self, frame: np.ndarray, dt: float) -> None:
        h, w = frame.shape
        self._creatures = [creature for creature in self._creatures if creature.alive_on(frame)]
        self._spawn_missing(frame, w, h)
        for creature in self._creatures:
            creature.update(frame, dt)

    def draw(self, surface: pygame.Surface) -> None:
        for creature in self._creatures:
            creature.draw(surface)

    def counts(self) -> dict[str, int]:
        counts = {"sharks": 0, "dinosaurs": 0}
        for creature in self._creatures:
            if isinstance(creature, Shark):
                counts["sharks"] += 1
            elif isinstance(creature, Dinosaur):
                counts["dinosaurs"] += 1
        return counts

    def _spawn_missing(self, frame: np.ndarray, w: int, h: int) -> None:
        counts = {cls: 0 for cls in self._targets}
        for creature in self._creatures:
            counts[type(creature)] = counts.get(type(creature), 0) + 1

        for cls, target in self._targets.items():
            while counts[cls] < target:
                creature = self._try_spawn(cls, frame, w, h)
                if creature is None:
                    break
                self._creatures.append(creature)
                counts[cls] += 1

    @staticmethod
    def _try_spawn(
        cls: type[Creature], frame: np.ndarray, w: int, h: int
    ) -> Creature | None:
        for _ in range(120):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            creature = cls(x, y)
            if creature._valid_pos(frame, np.array([x, y])):
                return creature
        return None
