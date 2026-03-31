"""
Optional local interaction rules for webcam-detected sandbox objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import pygame

from webcam_observer import VisionEvent, VisionObject


@dataclass
class _Effect:
    pos: tuple[float, float]
    colour: tuple[int, int, int]
    started_at: float
    duration: float
    radius: float
    style: str


class InteractionEngine:
    def __init__(self) -> None:
        self._last_state_by_marker: dict[int, str] = {}
        self._effects: list[_Effect] = []
        self._pending_events: list[VisionEvent] = []

    def update(
        self,
        objects: list[VisionObject],
        frame: np.ndarray,
        now: float,
        *,
        reactions_enabled: bool,
    ) -> None:
        visible_markers = {obj.marker_id for obj in objects}
        self._last_state_by_marker = {
            marker_id: state
            for marker_id, state in self._last_state_by_marker.items()
            if marker_id in visible_markers
        }

        self._effects = [effect for effect in self._effects if now - effect.started_at <= effect.duration]
        if not reactions_enabled:
            return

        for obj in objects:
            state = self._interaction_state(obj, frame)
            last_state = self._last_state_by_marker.get(obj.marker_id)
            if state != last_state:
                event = self._event_for_state(obj, state)
                if event is not None:
                    self._pending_events.append(event)
                    self._spawn_effect(obj, state, now)
            self._last_state_by_marker[obj.marker_id] = state

    def pop_events(self) -> list[VisionEvent]:
        events = self._pending_events[:]
        self._pending_events.clear()
        return events

    def draw(self, surface: pygame.Surface) -> None:
        now = time.monotonic()
        for effect in self._effects:
            age = (now - effect.started_at) / max(effect.duration, 1e-6)
            if age >= 1.0:
                continue
            radius = int(effect.radius * (1.0 + age * 0.8))
            alpha = max(18, int(140 * (1.0 - age)))
            ring = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
            colour = (*effect.colour, alpha)
            center = (ring.get_width() // 2, ring.get_height() // 2)
            if effect.style == "ripple":
                pygame.draw.circle(ring, colour, center, radius, width=2)
                pygame.draw.circle(ring, (*effect.colour, max(10, alpha // 2)), center, max(4, radius - 10), width=2)
            elif effect.style == "warning":
                pygame.draw.circle(ring, colour, center, radius, width=3)
                pygame.draw.circle(ring, (*effect.colour, max(10, alpha // 3)), center, max(4, radius - 12), width=1)
            else:
                pygame.draw.circle(ring, colour, center, radius, width=2)
            surface.blit(ring, (int(effect.pos[0]) - center[0], int(effect.pos[1]) - center[1]))

    @staticmethod
    def _interaction_state(obj: VisionObject, frame: np.ndarray) -> str:
        if obj.kind == "boat":
            return "boat_in_water" if obj.biome_under_object == "water" else "boat_on_land"
        if obj.kind == "dino_toy":
            return "dino_on_land" if obj.biome_under_object == "land" else "dino_in_water"
        if obj.kind == "house":
            return "house_near_coast" if _near_coast(frame, obj.sandbox_pos) else "house_inland"
        if obj.kind == "tree":
            return "tree_near_coast" if _near_coast(frame, obj.sandbox_pos) else "tree_on_land"
        if obj.kind == "volcano":
            return "volcano_marker"
        return "object_seen"

    @staticmethod
    def _event_for_state(obj: VisionObject, state: str) -> VisionEvent | None:
        x, y = obj.sandbox_pos
        if state == "boat_in_water":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Boat On The Water",
                body="The boat found water, so the sandbox starts to ripple around it.",
                sandbox_pos=(x, y),
            )
        if state == "boat_on_land":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Boat Stranded",
                body="The boat is resting on dry land and needs a lake or river.",
                sandbox_pos=(x, y),
            )
        if state == "dino_on_land":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Dinosaur Territory",
                body="The dinosaur toy has reached solid ground, right where it belongs.",
                sandbox_pos=(x, y),
            )
        if state == "dino_in_water":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Dinosaur In Trouble",
                body="That dinosaur toy is standing in water, so the habitat looks unsafe.",
                sandbox_pos=(x, y),
            )
        if state == "house_near_coast":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Coastal Village",
                body="The house is close to the shoreline, where flooding and trade both matter.",
                sandbox_pos=(x, y),
            )
        if state == "house_inland":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Inland Settlement",
                body="The house sits safely inland, away from the edge of the water.",
                sandbox_pos=(x, y),
            )
        if state == "tree_near_coast":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Waterside Tree",
                body="A tree marker near the water can anchor a lush riverside habitat.",
                sandbox_pos=(x, y),
            )
        if state == "tree_on_land":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Forest Patch",
                body="A tree marker adds the feeling of a small woodland on the land.",
                sandbox_pos=(x, y),
            )
        if state == "volcano_marker":
            return VisionEvent(
                event_key=f"{state}:{obj.marker_id}",
                kind=state,
                title="Volcano Marker Detected",
                body="A volcano toy changes the mood of the landscape into a hazard zone.",
                sandbox_pos=(x, y),
            )
        return None

    def _spawn_effect(self, obj: VisionObject, state: str, now: float) -> None:
        colour = (110, 180, 255)
        style = "pulse"
        radius = 18.0
        if state == "boat_in_water":
            colour = (110, 180, 255)
            style = "ripple"
            radius = 24.0
        elif state == "boat_on_land":
            colour = (190, 150, 90)
            style = "warning"
            radius = 20.0
        elif state == "dino_on_land":
            colour = (110, 210, 120)
            style = "pulse"
            radius = 18.0
        elif state == "dino_in_water":
            colour = (240, 160, 70)
            style = "warning"
            radius = 18.0
        elif state.startswith("house") or state.startswith("tree"):
            colour = (220, 200, 100) if "coast" in state else (120, 210, 120)
            style = "pulse"
            radius = 22.0
        elif state == "volcano_marker":
            colour = (235, 100, 60)
            style = "warning"
            radius = 26.0
        self._effects.append(
            _Effect(
                pos=obj.sandbox_pos,
                colour=colour,
                started_at=now,
                duration=1.4,
                radius=radius,
                style=style,
            )
        )


def _near_coast(frame: np.ndarray, sandbox_pos: tuple[float, float]) -> bool:
    h, w = frame.shape
    cx = int(np.clip(round(sandbox_pos[0]), 0, w - 1))
    cy = int(np.clip(round(sandbox_pos[1]), 0, h - 1))
    radius = 18
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    water_ratio = float(np.mean(patch < 0.46))
    return 0.08 <= water_ratio <= 0.45
