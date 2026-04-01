"""
LLM-driven interaction engine for CV-detected sandbox objects.

For each new detection (or when an object crosses a biome boundary) this engine:
  1. Fires a template VisionEvent immediately so the guide overlay always has text.
  2. Optionally enriches that text asynchronously via the configured LLM.
  3. Spawns a visual effect at the object's sandbox position.

Objects are tracked by track_id.  When a track disappears the state is cleaned
up automatically on the next update() call.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from urllib import request as urllib_request

import numpy as np
import pygame

from webcam_observer import CVTrackedObject, VisionEvent


_SYSTEM = (
    "You describe how a physical toy or object placed in an augmented-reality sandbox "
    "interacts with its terrain. Keep it short, vivid, and appropriate for children. "
    "Return compact JSON only — no prose, no markdown: "
    '{"title":"3-6 word heading","body":"1-2 sentence description",'
    '"effect":"ripple|pulse|warning|glow"}'
)

_EFFECT_COLOUR: dict[str, tuple[int, int, int]] = {
    "water": (80, 160, 255),
    "land": (120, 210, 120),
    "unknown": (180, 180, 220),
}


# ── visual effects ─────────────────────────────────────────────────────────────

@dataclass
class _Effect:
    pos: tuple[float, float]
    colour: tuple[int, int, int]
    started_at: float
    duration: float
    radius: float
    style: str  # ripple | pulse | warning | glow


# ── per-object state ───────────────────────────────────────────────────────────

@dataclass
class _TrackState:
    track_id: int
    label: str
    biome: str
    sandbox_pos: tuple[float, float]
    last_event_at: float


# ── helpers ────────────────────────────────────────────────────────────────────

def _template(label: str, biome: str) -> tuple[str, str, str]:
    """Instant fallback narration when LLM is disabled or still loading."""
    if biome == "water":
        return (
            f"{label.title()} In The Water",
            f"A {label} appeared near the water, disturbing the calm surface.",
            "ripple",
        )
    if biome == "land":
        return (
            f"{label.title()} On The Land",
            f"A {label} has been placed on the terrain, exploring the landscape.",
            "pulse",
        )
    return (
        f"{label.title()} Detected",
        f"Something new has entered the sandbox — a {label}.",
        "pulse",
    )


def _terrain_context(frame: np.ndarray | None, pos: tuple[float, float]) -> str:
    """Brief terrain description around a sandbox position for the LLM prompt."""
    if frame is None:
        return ""
    h, w = frame.shape
    cx = int(max(0, min(w - 1, round(pos[0]))))
    cy = int(max(0, min(h - 1, round(pos[1]))))
    r = 30
    patch = frame[max(0, cy - r):min(h, cy + r + 1), max(0, cx - r):min(w, cx + r + 1)]
    if patch.size == 0:
        return ""
    water_ratio = float(np.mean(patch < 0.46))
    elevation = float(frame[cy, cx])
    parts: list[str] = []
    if water_ratio > 0.6:
        parts.append("surrounded by water")
    elif water_ratio > 0.25:
        parts.append("near the shoreline")
    else:
        parts.append("on dry land")
    if elevation > 0.75:
        parts.append("at mountain-peak elevation")
    elif elevation > 0.6:
        parts.append("on elevated ground")
    return (", ".join(parts) + ".") if parts else ""


# ── async LLM worker ───────────────────────────────────────────────────────────

class _LLMWorker:
    """Fire-and-forget LLM queries with result polling.  Thread-safe."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._results: dict[str, dict[str, str]] = {}
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def request(self, key: str, label: str, biome: str, terrain: str) -> None:
        with self._lock:
            if key in self._pending or key in self._results:
                return
            self._pending.add(key)
        threading.Thread(
            target=self._run, args=(key, label, biome, terrain), daemon=True
        ).start()

    def poll(self, key: str) -> dict[str, str] | None:
        with self._lock:
            return self._results.pop(key, None)

    def _run(self, key: str, label: str, biome: str, terrain: str) -> None:
        result = self._query(label, biome, terrain)
        with self._lock:
            self._pending.discard(key)
            self._results[key] = result

    def _query(self, label: str, biome: str, terrain: str) -> dict[str, str]:
        msg = f"A '{label}' has been placed in the sandbox ({biome}). {terrain} Describe the interaction."
        payload = json.dumps({
            "model": self._model,
            "max_tokens": 120,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": msg},
            ],
        }).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib_request.Request(
            f"{self._base_url}/chat/completions", data=payload, headers=headers
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
            s, e = text.find("{"), text.rfind("}") + 1
            if s != -1 and e > 0:
                return json.loads(text[s:e])
        except Exception:
            pass
        return {}


# ── main engine ───────────────────────────────────────────────────────────────

class CVInteractionEngine:
    """
    Tracks CV-detected objects and generates VisionEvents + visual effects.

    Template responses are immediate; when LLM is configured, an enriched
    response replaces the template asynchronously (typically < a few seconds).
    """

    def __init__(self) -> None:
        self._states: dict[int, _TrackState] = {}
        self._effects: list[_Effect] = []
        self._pending_events: list[VisionEvent] = []
        self._llm: _LLMWorker | None = None
        self._waiting: dict[int, str] = {}  # track_id → event_key

    def configure_llm(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 6.0,
    ) -> None:
        self._llm = _LLMWorker(base_url, model, api_key, timeout)

    def disable_llm(self) -> None:
        self._llm = None

    def update(
        self,
        objects: list[CVTrackedObject],
        frame: np.ndarray | None,
        now: float,
        *,
        interactions_enabled: bool = True,
    ) -> None:
        visible = {obj.track_id for obj in objects}

        # Clean up removed objects
        for tid in list(self._states):
            if tid not in visible:
                self._states.pop(tid)
                self._waiting.pop(tid, None)

        # Expire old effects
        self._effects = [e for e in self._effects if now - e.started_at <= e.duration]

        if not interactions_enabled:
            return

        for obj in objects:
            state = self._states.get(obj.track_id)
            if state is None:
                # New object appeared
                self._states[obj.track_id] = _TrackState(
                    track_id=obj.track_id,
                    label=obj.label,
                    biome=obj.biome,
                    sandbox_pos=obj.sandbox_pos,
                    last_event_at=now,
                )
                self._fire(obj, frame, now, kind="cv_new")
            else:
                state.sandbox_pos = obj.sandbox_pos
                # Biome change: object moved between water and land
                if obj.biome != state.biome and now - state.last_event_at > 3.0:
                    state.biome = obj.biome
                    state.last_event_at = now
                    self._fire(obj, frame, now, kind="cv_move")

        # Poll completed LLM requests and emit enriched events
        if self._llm is not None:
            for tid, event_key in list(self._waiting.items()):
                result = self._llm.poll(event_key)
                if result is None:
                    continue
                del self._waiting[tid]
                if not result.get("title") or not result.get("body"):
                    continue
                state = self._states.get(tid)
                pos = state.sandbox_pos if state else (0.0, 0.0)
                biome = state.biome if state else "unknown"
                style = str(result.get("effect", "pulse"))
                colour = _EFFECT_COLOUR.get(biome, (180, 180, 220))
                self._pending_events.append(VisionEvent(
                    event_key=event_key + ":llm",
                    kind="cv_llm",
                    title=str(result["title"]),
                    body=str(result["body"]),
                    sandbox_pos=pos,
                ))
                self._spawn(pos, colour, style, now)

    def pop_events(self) -> list[VisionEvent]:
        events = self._pending_events[:]
        self._pending_events.clear()
        return events

    def draw(self, surface: pygame.Surface) -> None:
        now = time.monotonic()
        for eff in self._effects:
            age = (now - eff.started_at) / max(eff.duration, 1e-6)
            if age >= 1.0:
                continue
            r = int(eff.radius * (1.0 + age * 0.8))
            alpha = max(18, int(140 * (1.0 - age)))
            sz = r * 2 + 8
            ring = pygame.Surface((sz, sz), pygame.SRCALPHA)
            c = (*eff.colour, alpha)
            cx, cy = sz // 2, sz // 2
            if eff.style == "ripple":
                pygame.draw.circle(ring, c, (cx, cy), r, width=2)
                pygame.draw.circle(ring, (*eff.colour, max(10, alpha // 2)), (cx, cy), max(4, r - 10), width=2)
            elif eff.style == "warning":
                pygame.draw.circle(ring, c, (cx, cy), r, width=3)
                pygame.draw.circle(ring, (*eff.colour, max(10, alpha // 3)), (cx, cy), max(4, r - 12), width=1)
            elif eff.style == "glow":
                glow = pygame.Surface((sz, sz), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*eff.colour, max(8, alpha // 4)), (cx, cy), r)
                surface.blit(glow, (int(eff.pos[0]) - cx, int(eff.pos[1]) - cy))
                pygame.draw.circle(ring, c, (cx, cy), r, width=3)
            else:  # pulse
                pygame.draw.circle(ring, c, (cx, cy), r, width=2)
            surface.blit(ring, (int(eff.pos[0]) - cx, int(eff.pos[1]) - cy))

    # ── private ────────────────────────────────────────────────────────────────

    def _fire(
        self,
        obj: CVTrackedObject,
        frame: np.ndarray | None,
        now: float,
        kind: str,
    ) -> None:
        event_key = f"{kind}:{obj.track_id}:{obj.label}:{obj.biome}"
        terrain = _terrain_context(frame, obj.sandbox_pos)
        colour = _EFFECT_COLOUR.get(obj.biome, (180, 180, 220))

        # Always emit a template event first for instant feedback
        title, body, style = _template(obj.label, obj.biome)
        self._pending_events.append(VisionEvent(
            event_key=event_key,
            kind=kind,
            title=title,
            body=body,
            sandbox_pos=obj.sandbox_pos,
        ))

        if self._llm is not None:
            # Queue LLM enrichment; show a subtle scan pulse while waiting
            self._llm.request(event_key, obj.label, obj.biome, terrain)
            self._waiting[obj.track_id] = event_key
            self._spawn(obj.sandbox_pos, (180, 180, 220), "pulse", now, radius=14.0, duration=0.9)
        else:
            self._spawn(obj.sandbox_pos, colour, style, now)

    def _spawn(
        self,
        pos: tuple[float, float],
        colour: tuple[int, int, int],
        style: str,
        now: float,
        *,
        radius: float = 22.0,
        duration: float = 1.4,
    ) -> None:
        valid = {"ripple", "pulse", "warning", "glow"}
        self._effects.append(_Effect(
            pos=pos,
            colour=colour,
            started_at=now,
            duration=duration,
            radius=radius,
            style=style if style in valid else "pulse",
        ))
