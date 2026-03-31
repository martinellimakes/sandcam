"""
Animated creatures that react to the terrain height map.

Rendering upgrades:
- asset-backed pixel art loaded from JSON files in assets/creatures
- 16-direction pre-baked facing sprites to avoid runtime rotozoom blur
- terrain-aware compositing such as ripples, shoreline tint, and stronger
  ground contact shadows
"""

from __future__ import annotations

import json
import math
import pathlib
import random
from typing import ClassVar

import numpy as np
import pygame

from renderer import WATER_LEVEL

_ARRIVAL_RADIUS = 24
_STEER_RATE = 6.0
_WANDER_NOISE = 0.08
_DIRECTION_COUNT = 16
_SHORE_SAMPLE_RADIUS = 14
_ASSET_DIR = pathlib.Path(__file__).with_name("assets") / "creatures"
_CARDINAL_DIRECTION_MAP = {
    0: 2,   # right
    1: 2,
    2: 0,   # up-right leans toward the upward-facing row
    3: 0,
    4: 0,   # up
    5: 0,
    6: 1,   # up-left leans toward left art
    7: 1,
    8: 1,   # left
    9: 1,
    10: 3,  # down-left leans toward the downward-facing row
    11: 3,
    12: 3,  # down
    13: 3,
    14: 2,  # down-right leans toward right art
    15: 2,
}


class Creature:
    """
    Base class for terrain-aware animated creatures.

    Subclasses must implement:
        is_valid(elevation) -> bool
        _frame_index() -> int
        _asset_name() -> str
        _fallback_frames() -> list[pygame.Surface]
        _draw_environment(surface, frame, sprite) -> None
    """

    def __init__(self, x: float, y: float, speed: float) -> None:
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.angle = random.uniform(0, math.tau)
        self.speed = speed
        self._target: np.ndarray | None = None
        self._wander = 0.0
        self._directional_frames = self._prepare_directional_frames()

    def is_valid(self, elevation: float) -> bool:
        raise NotImplementedError

    def draw(self, surface: pygame.Surface, frame: np.ndarray) -> None:
        sprite = self._render_sprite(self._current_sprite())
        self._draw_environment(surface, frame, sprite)
        self._blit_sprite(surface, sprite)

    def alive_on(self, frame: np.ndarray) -> bool:
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
    def _shadow(
        surface: pygame.Surface,
        pos: np.ndarray,
        rx: int,
        ry: int,
        *,
        alpha: int = 45,
        colour: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        shadow = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (*colour, alpha), shadow.get_rect())
        surface.blit(shadow, (int(pos[0]) - rx, int(pos[1]) - ry + ry // 2))

    @staticmethod
    def _surface(size: tuple[int, int]) -> pygame.Surface:
        return pygame.Surface(size, pygame.SRCALPHA)

    @staticmethod
    def _scale_pixel_art(sprite: pygame.Surface, scale: int) -> pygame.Surface:
        width, height = sprite.get_size()
        return pygame.transform.scale(sprite, (width * scale, height * scale))

    def _blit_sprite(self, surface: pygame.Surface, sprite: pygame.Surface) -> None:
        rect = sprite.get_rect(center=(int(self.pos[0]), int(self.pos[1])))
        surface.blit(sprite, rect)

    def _terrain_context(self, frame: np.ndarray) -> dict[str, float]:
        h, w = frame.shape
        cx = int(np.clip(round(self.pos[0]), 0, w - 1))
        cy = int(np.clip(round(self.pos[1]), 0, h - 1))
        radius = _SHORE_SAMPLE_RADIUS
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return {"elevation": 0.5, "water_ratio": 0.0, "shore_mix": 0.0}
        water_ratio = float(np.mean(patch < WATER_LEVEL))
        shore_mix = float(np.clip(min(water_ratio, 1.0 - water_ratio) * 2.6, 0.0, 1.0))
        elevation = float(frame[cy, cx])
        return {
            "elevation": elevation,
            "water_ratio": water_ratio,
            "shore_mix": shore_mix,
        }

    def _prepare_directional_frames(self) -> list[list[pygame.Surface]]:
        directional_frames = self._load_sheet_directions(self._asset_name())
        if directional_frames is not None:
            return directional_frames

        base_frames = self._load_asset_frames(self._asset_name())
        directions: list[list[pygame.Surface]] = []
        for angle_idx in range(_DIRECTION_COUNT):
            direction_frames: list[pygame.Surface] = []
            rotation = -360.0 * angle_idx / _DIRECTION_COUNT
            for frame in base_frames:
                rotated = pygame.transform.rotate(frame, rotation)
                direction_frames.append(rotated)
            directions.append(direction_frames)
        return directions

    def _current_sprite(self) -> pygame.Surface:
        frame_idx = self._frame_index()
        direction_idx = self._direction_index()
        return self._directional_frames[direction_idx][frame_idx]

    def _direction_index(self) -> int:
        direction = int(round((self.angle % math.tau) / math.tau * _DIRECTION_COUNT))
        return direction % _DIRECTION_COUNT

    def _render_sprite(self, sprite: pygame.Surface) -> pygame.Surface:
        return sprite

    def _load_asset_frames(self, name: str) -> list[pygame.Surface]:
        asset_path = _ASSET_DIR / f"{name}.json"
        if not asset_path.exists():
            return self._fallback_frames()

        try:
            payload = json.loads(asset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._fallback_frames()

        palette = {
            key: tuple(value)
            for key, value in payload.get("palette", {}).items()
        }
        scale = int(payload.get("scale", 3))
        frames: list[pygame.Surface] = []
        for rows in payload.get("frames", []):
            if not rows:
                continue
            width = max(len(row) for row in rows)
            height = len(rows)
            surf = self._surface((width, height))
            for y, row in enumerate(rows):
                for x, token in enumerate(row):
                    colour = palette.get(token)
                    if colour is None or len(colour) < 4 or colour[3] == 0:
                        continue
                    surf.set_at((x, y), pygame.Color(*colour))
            frames.append(self._scale_pixel_art(surf, scale))
        return frames or self._fallback_frames()

    def _load_sheet_directions(self, name: str) -> list[list[pygame.Surface]] | None:
        sheet_path = _ASSET_DIR / f"{name}.png"
        if not sheet_path.exists():
            return None

        try:
            sheet = pygame.image.load(str(sheet_path)).convert_alpha()
        except pygame.error:
            return None

        rows = self._sheet_rows()
        cols = self._sheet_cols()
        if rows <= 0 or cols <= 0:
            return None

        cell_w = sheet.get_width() // cols
        cell_h = sheet.get_height() // rows
        if cell_w <= 0 or cell_h <= 0:
            return None

        base_directions: list[list[pygame.Surface]] = []
        for row in range(rows):
            frames: list[pygame.Surface] = []
            for col in range(cols):
                rect = pygame.Rect(col * cell_w, row * cell_h, cell_w, cell_h)
                frame = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
                frame.blit(sheet, (0, 0), rect)
                frames.append(self._scale_sheet_frame(frame))
            base_directions.append(frames)

        if len(base_directions) == _DIRECTION_COUNT:
            return base_directions

        expanded: list[list[pygame.Surface]] = []
        for direction_idx in range(_DIRECTION_COUNT):
            source_idx = _CARDINAL_DIRECTION_MAP[direction_idx] % len(base_directions)
            expanded.append(base_directions[source_idx])
        return expanded

    def _sheet_rows(self) -> int:
        return 0

    def _sheet_cols(self) -> int:
        return 0

    def _scale_sheet_frame(self, frame: pygame.Surface) -> pygame.Surface:
        return frame

    def _asset_name(self) -> str:
        raise NotImplementedError

    def _fallback_frames(self) -> list[pygame.Surface]:
        raise NotImplementedError

    def _frame_index(self) -> int:
        raise NotImplementedError

    def _draw_environment(
        self,
        surface: pygame.Surface,
        frame: np.ndarray,
        sprite: pygame.Surface,
    ) -> None:
        raise NotImplementedError


class Shark(Creature):
    _BODY_COLOR: ClassVar = (75, 125, 155)
    _DARK_COLOR: ClassVar = (50, 90, 115)
    _FIN_COLOR: ClassVar = (55, 100, 130)
    _FRAME_SWEEPS: ClassVar = (-1.0, -0.35, 0.35, 1.0)
    _VISIBILITY_PHASES: ClassVar = ("body", "subsurface", "deep")
    _VISIBILITY_DURATIONS: ClassVar = {
        "body": (4.0, 7.0),
        "subsurface": (2.5, 4.5),
        "deep": (3.5, 6.0),
    }
    _SUBSURFACE_MIN_ANIMATION_FRAMES: ClassVar = 2
    _PHASE_ASSET_CANDIDATES: ClassVar = {
        "body": ("hai", "shark"),
        "subsurface": ("hai-fin-shadow", "shark_fin_shadow", "shark-fin-shadow"),
        "deep": ("hai-fin_only", "shark_fin", "shark-fin"),
    }
    _ROW_BASE_ANGLES: ClassVar = {
        0: math.pi / 2,
        1: math.pi,
        2: 0.0,
        3: math.pi * 1.5,
    }

    def __init__(self, x: float, y: float) -> None:
        self._phase = random.uniform(0, math.tau)
        self._scale = random.uniform(0.85, 1.2)
        super().__init__(x, y, speed=random.uniform(28, 48))
        self._phase_frames = self._load_phase_frames()
        self._visibility_index = random.randrange(len(self._VISIBILITY_PHASES))
        self._visibility_direction = random.choice((-1, 1))
        self._visibility_duration_seconds = self._roll_visibility_duration()
        self._visibility_time = random.uniform(0.0, self._visibility_duration_seconds)

    def is_valid(self, elevation: float) -> bool:
        return elevation < WATER_LEVEL

    def update(self, frame: np.ndarray, dt: float) -> None:
        self._phase += dt * 3.2
        self._visibility_time += dt
        if self._visibility_time >= self._visibility_duration_seconds:
            self._visibility_time = 0.0
            self._advance_visibility_phase()
            self._visibility_duration_seconds = self._roll_visibility_duration()
        super().update(frame, dt)

    def _frame_index(self) -> int:
        cycle = (self._phase % math.tau) / math.tau
        count = len(self._current_frames()[0])
        return int(cycle * count) % count

    def _asset_name(self) -> str:
        return "hai"

    def _sheet_rows(self) -> int:
        return 4

    def _sheet_cols(self) -> int:
        return 4

    def _scale_sheet_frame(self, frame: pygame.Surface) -> pygame.Surface:
        width = max(42, int(round(frame.get_width() * 0.42 * self._scale)))
        height = max(40, int(round(frame.get_height() * 0.42 * self._scale)))
        return pygame.transform.smoothscale(frame, (width, height))

    def _current_sprite(self) -> pygame.Surface:
        frame_idx = self._frame_index()
        direction_idx = self._direction_index()
        return self._current_frames()[direction_idx][frame_idx]

    def _render_sprite(self, sprite: pygame.Surface) -> pygame.Surface:
        direction_idx = self._direction_index()
        row_idx = _CARDINAL_DIRECTION_MAP[direction_idx]
        base_angle = self._ROW_BASE_ANGLES[row_idx]
        rotation = math.degrees(base_angle - self.angle)
        if abs(rotation) < 0.5:
            return sprite
        return pygame.transform.rotate(sprite, rotation)

    def _current_frames(self) -> list[list[pygame.Surface]]:
        return self._phase_frames[self._current_visibility_phase()]

    def _current_visibility_phase(self) -> str:
        return self._VISIBILITY_PHASES[self._visibility_index]

    def _roll_visibility_duration(self) -> float:
        phase = self._current_visibility_phase()
        low, high = self._VISIBILITY_DURATIONS[phase]
        duration = random.uniform(low, high)
        if phase == "subsurface":
            duration = max(duration, self._minimum_subsurface_duration())
        return duration

    def _advance_visibility_phase(self) -> None:
        phase = self._current_visibility_phase()
        if phase == "body":
            self._visibility_direction = 1
            self._visibility_index = 1
            return
        if phase == "deep":
            self._visibility_direction = -1
            self._visibility_index = 1
            return
        self._visibility_index = max(
            0,
            min(len(self._VISIBILITY_PHASES) - 1, self._visibility_index + self._visibility_direction),
        )

    def _minimum_subsurface_duration(self) -> float:
        frame_count = len(self._phase_frames["subsurface"][0])
        if frame_count <= 0:
            return 0.0
        seconds_per_animation_frame = math.tau / (3.2 * frame_count)
        return seconds_per_animation_frame * self._SUBSURFACE_MIN_ANIMATION_FRAMES

    def _load_phase_frames(self) -> dict[str, list[list[pygame.Surface]]]:
        body_frames = self._directional_frames
        frames = {"body": body_frames}

        for phase in ("subsurface", "deep"):
            frames[phase] = self._load_phase_variant(phase) or body_frames
        return frames

    def _load_phase_variant(self, phase: str) -> list[list[pygame.Surface]] | None:
        for name in self._PHASE_ASSET_CANDIDATES[phase]:
            variant = self._load_sheet_directions(name)
            if variant is not None:
                return variant
        return None

    def _fallback_frames(self) -> list[pygame.Surface]:
        pixel_scale = max(3, int(round(self._scale * 3)))
        sprites: list[pygame.Surface] = []

        for sweep in self._FRAME_SWEEPS:
            surf = self._surface((18, 12))
            tail_shift = int(round(sweep * 2))

            body = [(2, 5), (4, 3), (8, 2), (12, 3), (15, 5), (12, 7), (8, 8), (4, 7)]
            pygame.draw.polygon(surf, self._BODY_COLOR, body)
            pygame.draw.polygon(surf, self._DARK_COLOR, body, 1)
            pygame.draw.polygon(surf, self._FIN_COLOR, [(7, 5), (9, 2), (10, 5)])
            pygame.draw.polygon(surf, self._FIN_COLOR, [(8, 3), (10, 1), (11, 3)])
            pygame.draw.polygon(surf, self._FIN_COLOR, [(8, 7), (10, 9), (11, 7)])
            pygame.draw.polygon(surf, self._FIN_COLOR, [(1, 5), (0, 2 + tail_shift), (3, 4)])
            pygame.draw.polygon(surf, self._FIN_COLOR, [(1, 5), (0, 8 - tail_shift), (3, 6)])
            pygame.draw.rect(surf, (220, 230, 240), pygame.Rect(12, 4, 1, 1))
            pygame.draw.rect(surf, (220, 230, 240), pygame.Rect(12, 6, 1, 1))
            sprites.append(self._scale_pixel_art(surf, pixel_scale))

        return sprites

    def _draw_environment(
        self,
        surface: pygame.Surface,
        frame: np.ndarray,
        sprite: pygame.Surface,
    ) -> None:
        context = self._terrain_context(frame)
        scale = self._scale
        phase = self._current_visibility_phase()
        shadow_alpha = 38
        shadow_colour = (15, 35, 55)
        shadow_rx = int(22 * scale)
        shadow_ry = int(8 * scale)

        if phase == "subsurface":
            shadow_alpha = 26
            shadow_rx = int(20 * scale)
            shadow_ry = int(7 * scale)
        elif phase == "deep":
            shadow_alpha = 18
            shadow_colour = (35, 60, 90)
            shadow_rx = int(16 * scale)
            shadow_ry = int(6 * scale)

        self._shadow(
            surface,
            self.pos,
            shadow_rx,
            shadow_ry,
            alpha=shadow_alpha,
            colour=shadow_colour,
        )

        if phase != "deep" and context["shore_mix"] > 0.18:
            shimmer = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            shimmer.fill((185, 225, 255, int(36 * context["shore_mix"])))
            surface.blit(
                shimmer,
                shimmer.get_rect(center=(int(self.pos[0]), int(self.pos[1]))),
                special_flags=pygame.BLEND_RGBA_ADD,
            )


class Dinosaur(Creature):
    _PALETTES: ClassVar = [
        ((75, 140, 60), (45, 100, 35)),
        ((140, 95, 55), (100, 65, 30)),
        ((155, 75, 75), (110, 45, 45)),
        ((80, 120, 140), (50, 85, 105)),
    ]
    _FRAME_SWINGS: ClassVar = (-1.0, -0.35, 0.35, 1.0)

    def __init__(self, x: float, y: float) -> None:
        self._body_col, self._dark_col = random.choice(self._PALETTES)
        self._phase = random.uniform(0, math.tau)
        self._scale = random.uniform(0.8, 1.15)
        super().__init__(x, y, speed=random.uniform(35, 65))

    def is_valid(self, elevation: float) -> bool:
        return elevation >= WATER_LEVEL

    def update(self, frame: np.ndarray, dt: float) -> None:
        self._phase += dt * self.speed * 0.07
        super().update(frame, dt)

    def _frame_index(self) -> int:
        cycle = (self._phase % math.tau) / math.tau
        count = len(self._directional_frames[0])
        return int(cycle * count) % count

    def _asset_name(self) -> str:
        return "dinosaur"

    def _fallback_frames(self) -> list[pygame.Surface]:
        pixel_scale = max(3, int(round(self._scale * 3)))
        sprites: list[pygame.Surface] = []

        for swing_ratio in self._FRAME_SWINGS:
            surf = self._surface((22, 18))
            leg_shift = int(round(swing_ratio * 2))
            tail_shift = int(round(swing_ratio))

            tail = [(3, 8), (0, 6 - tail_shift), (1, 8), (0, 10 + tail_shift)]
            body = [(4, 5), (9, 3), (13, 3), (16, 5), (17, 8), (16, 11), (13, 13), (9, 13), (4, 11), (2, 8)]
            head = [(15, 5), (19, 4), (21, 6), (21, 10), (19, 12), (15, 11), (17, 8)]
            pygame.draw.polygon(surf, self._body_col, tail)
            pygame.draw.polygon(surf, self._dark_col, tail, 1)
            pygame.draw.polygon(surf, self._body_col, body)
            pygame.draw.polygon(surf, self._dark_col, body, 1)
            pygame.draw.polygon(surf, self._body_col, head)
            pygame.draw.polygon(surf, self._dark_col, head, 1)
            for y, sign in ((5, 1), (11, -1)):
                pygame.draw.line(surf, self._dark_col, (5, y), (2, y + leg_shift * sign), 1)
                pygame.draw.line(surf, self._dark_col, (2, y + leg_shift * sign), (0, y + leg_shift * sign), 1)
            sprites.append(self._scale_pixel_art(surf, pixel_scale))

        return sprites

    def _draw_environment(
        self,
        surface: pygame.Surface,
        frame: np.ndarray,
        sprite: pygame.Surface,
    ) -> None:
        context = self._terrain_context(frame)
        scale = self._scale
        shore_mix = context["shore_mix"]

        self._shadow(
            surface,
            self.pos,
            int(20 * scale),
            int(7 * scale),
            alpha=58,
            colour=(0, 0, 0),
        )

        if np.linalg.norm(self.vel) > 12:
            dust = pygame.Surface((36, 18), pygame.SRCALPHA)
            pygame.draw.ellipse(
                dust,
                (180, 165, 120, 34 + int(20 * (1.0 - shore_mix))),
                dust.get_rect(),
            )
            surface.blit(dust, (int(self.pos[0]) - 18, int(self.pos[1]) + 4))

        if shore_mix > 0.18:
            coast = pygame.Surface(sprite.get_size(), pygame.SRCALPHA)
            coast.fill((235, 205, 130, int(32 * shore_mix)))
            surface.blit(
                coast,
                coast.get_rect(center=(int(self.pos[0]), int(self.pos[1]))),
                special_flags=pygame.BLEND_RGBA_ADD,
            )


class CreatureManager:
    """
    Spawns and updates all creatures.

    Creatures are spawned lazily on the first frame that has valid terrain for
    them, so it is safe to create the manager before depth data arrives.
    """

    def __init__(self, n_sharks: int = 4, n_dinos: int = 5) -> None:
        self._targets = {Shark: n_sharks, Dinosaur: n_dinos}
        self._creatures: list[Creature] = []

    def set_targets(self, *, sharks: int | None = None, dinosaurs: int | None = None) -> None:
        if sharks is not None:
            self._targets[Shark] = max(0, int(sharks))
        if dinosaurs is not None:
            self._targets[Dinosaur] = max(0, int(dinosaurs))

    def update(self, frame: np.ndarray, dt: float) -> None:
        h, w = frame.shape
        self._creatures = [creature for creature in self._creatures if creature.alive_on(frame)]
        self._spawn_missing(frame, w, h)
        for creature in self._creatures:
            creature.update(frame, dt)

    def draw(self, surface: pygame.Surface, frame: np.ndarray) -> None:
        for creature in self._creatures:
            creature.draw(surface, frame)

    def counts(self) -> dict[str, int]:
        counts = {"sharks": 0, "dinosaurs": 0}
        for creature in self._creatures:
            if isinstance(creature, Shark):
                counts["sharks"] += 1
            elif isinstance(creature, Dinosaur):
                counts["dinosaurs"] += 1
        return counts

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
    def _try_spawn(cls: type[Creature], frame: np.ndarray, w: int, h: int) -> Creature | None:
        for _ in range(120):
            x = random.uniform(0, w)
            y = random.uniform(0, h)
            creature = cls(x, y)
            if creature._valid_pos(frame, np.array([x, y])):
                return creature
        return None
