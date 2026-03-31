"""
Optional AI guide system for the AR sandbox.

The guide always reasons about terrain locally. LLMs only refine the wording of
messages and are entirely optional.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
import queue
import threading
import time
from typing import Any, Protocol
from urllib import error, request

import numpy as np
from scipy.ndimage import label

from renderer import WATER_LEVEL

ANALYSIS_INTERVAL_SECONDS = 0.5
EVENT_COOLDOWN_SECONDS = 8.0
CHALLENGE_COOLDOWN_SECONDS = 15.0
_SYSTEM_PROMPT = (
    "You write short, kid-friendly museum captions for a live augmented-reality "
    "sandbox. Return compact JSON with keys title, body, and optional "
    "challenge_text. Keep it positive, vivid, and under two short sentences."
)


@dataclass(frozen=True)
class ProviderConfig:
    backend: str = "template"
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = ""
    timeout_seconds: float = 2.0
    api_key: str | None = None


@dataclass(frozen=True)
class ConnectionTestResult:
    ok: bool
    summary: str


@dataclass(frozen=True)
class WorldState:
    timestamp: float
    water_ratio: float
    land_ratio: float
    coastline_ratio: float
    water_regions: int
    land_regions: int
    lakes: int
    islands: int
    largest_water_ratio: float
    largest_land_ratio: float
    highest_peak: float
    mean_height: float
    steep_ratio: float
    shark_count: int
    dinosaur_count: int
    shark_habitat_ratio: float
    dinosaur_habitat_ratio: float
    features: tuple[str, ...]
    deltas: dict[str, float]


@dataclass(frozen=True)
class WorldEvent:
    key: str
    kind: str
    payload: dict[str, float | int | str]


@dataclass(frozen=True)
class GuideMessage:
    kind: str
    title: str
    body: str
    challenge_id: str | None = None
    completed: bool = False
    expires_at: float = 0.0
    event_key: str = ""


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    title: str
    prompt: str


class NarrationProvider(Protocol):
    def generate(
        self,
        *,
        world_state: WorldState,
        message: GuideMessage,
        challenge: Challenge | None,
        verbosity: str,
    ) -> dict[str, str]:
        ...


def _component_stats(mask: np.ndarray) -> tuple[int, int, int]:
    if not np.any(mask):
        return 0, 0, 0

    labels, count = label(mask)
    if count == 0:
        return 0, 0, 0

    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :]))
    border_labels |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)

    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    largest = int(sizes[1:].max()) if count else 0
    enclosed = sum(1 for idx in range(1, count + 1) if idx not in border_labels)
    return int(count), enclosed, largest


class WorldAnalyzer:
    def analyze(
        self,
        frame: np.ndarray,
        now: float,
        *,
        shark_count: int,
        dinosaur_count: int,
    ) -> WorldState:
        water = frame < WATER_LEVEL
        land = ~water

        water_ratio = float(np.mean(water))
        land_ratio = 1.0 - water_ratio
        coastline_edges = (
            (water[:, 1:] != water[:, :-1]).sum()
            + (water[1:, :] != water[:-1, :]).sum()
        )
        coastline_ratio = float(coastline_edges / max(1, frame.size))

        water_regions, lakes, largest_water = _component_stats(water)
        land_regions, islands, largest_land = _component_stats(land)

        grad_y, grad_x = np.gradient(frame)
        slope = np.hypot(grad_x, grad_y)
        steep_ratio = float(np.mean(slope > 0.03))

        largest_water_ratio = largest_water / max(1, frame.size)
        largest_land_ratio = largest_land / max(1, frame.size)
        highest_peak = float(np.max(frame))
        mean_height = float(np.mean(frame))

        features: list[str] = []
        if largest_water_ratio >= 0.14:
            features.append("large_lake")
        if islands >= 1:
            features.append("isolated_island")
        if land_regions >= 2 and largest_land_ratio < 0.92:
            features.append("split_landmass")
        if highest_peak >= 0.78 and steep_ratio >= 0.05:
            features.append("mountain_range")
        if (
            0.08 <= water_ratio <= 0.40
            and coastline_ratio >= 0.13
            and water_regions >= 1
            and largest_water_ratio < 0.55
        ):
            features.append("river_like_channel")

        return WorldState(
            timestamp=now,
            water_ratio=water_ratio,
            land_ratio=land_ratio,
            coastline_ratio=coastline_ratio,
            water_regions=water_regions,
            land_regions=land_regions,
            lakes=lakes,
            islands=islands,
            largest_water_ratio=largest_water_ratio,
            largest_land_ratio=largest_land_ratio,
            highest_peak=highest_peak,
            mean_height=mean_height,
            steep_ratio=steep_ratio,
            shark_count=shark_count,
            dinosaur_count=dinosaur_count,
            shark_habitat_ratio=water_ratio,
            dinosaur_habitat_ratio=land_ratio,
            features=tuple(features),
            deltas={},
        )

    @staticmethod
    def with_deltas(current: WorldState, previous: WorldState | None) -> WorldState:
        if previous is None:
            return current

        deltas = {
            "water_ratio": current.water_ratio - previous.water_ratio,
            "land_ratio": current.land_ratio - previous.land_ratio,
            "highest_peak": current.highest_peak - previous.highest_peak,
            "coastline_ratio": current.coastline_ratio - previous.coastline_ratio,
            "shark_count": float(current.shark_count - previous.shark_count),
            "dinosaur_count": float(current.dinosaur_count - previous.dinosaur_count),
            "islands": float(current.islands - previous.islands),
        }
        return replace(current, deltas=deltas)


class TemplateNarrator:
    def generate(
        self,
        *,
        world_state: WorldState,
        message: GuideMessage,
        challenge: Challenge | None,
        verbosity: str,
    ) -> dict[str, str]:
        del world_state
        if message.kind == "challenge" and challenge is not None:
            body = challenge.prompt
        elif message.kind == "celebration":
            body = message.body
        elif verbosity == "quiet":
            body = message.body.split(".")[0].strip()
        elif verbosity == "lively":
            body = message.body
            if not body.endswith("!"):
                body = body.rstrip(".") + "!"
        else:
            body = message.body

        payload = {"title": message.title, "body": body}
        if challenge is not None:
            payload["challenge_text"] = challenge.prompt
        return payload


class OpenAICompatibleNarrator:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        opener: Any = None,
    ) -> None:
        self._config = config
        self._opener = opener if opener is not None else request.urlopen

    def generate(
        self,
        *,
        world_state: WorldState,
        message: GuideMessage,
        challenge: Challenge | None,
        verbosity: str,
    ) -> dict[str, str]:
        endpoint = self._config.base_url.rstrip("/") + "/chat/completions"
        challenge_prompt = challenge.prompt if challenge is not None else ""
        payload = {
            "model": self._config.model or "default",
            "temperature": 0.4 if verbosity == "quiet" else 0.7,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": {
                                "kind": message.kind,
                                "title": message.title,
                                "body": message.body,
                            },
                            "challenge_text": challenge_prompt,
                            "world_state": {
                                "water_ratio": round(world_state.water_ratio, 3),
                                "land_ratio": round(world_state.land_ratio, 3),
                                "coastline_ratio": round(world_state.coastline_ratio, 3),
                                "highest_peak": round(world_state.highest_peak, 3),
                                "islands": world_state.islands,
                                "lakes": world_state.lakes,
                                "features": list(world_state.features),
                                "shark_count": world_state.shark_count,
                                "dinosaur_count": world_state.dinosaur_count,
                            },
                            "verbosity": verbosity,
                        }
                    ),
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(req, timeout=self._config.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except (error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Narration request failed") from exc

        try:
            outer = json.loads(raw)
            content = outer["choices"][0]["message"]["content"]
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    text = item.get("text") or item.get("content") or item.get("output_text")
                    if text:
                        parts.append(str(text))
                content = "".join(parts)
            inner = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Narration response was not valid JSON") from exc

        title = str(inner.get("title", message.title)).strip() or message.title
        body = str(inner.get("body", message.body)).strip() or message.body
        output = {"title": title, "body": body}
        if challenge is not None:
            challenge_text = str(inner.get("challenge_text", challenge.prompt)).strip()
            output["challenge_text"] = challenge_text or challenge.prompt
        return output


class AsyncNarrator:
    def __init__(self, provider: NarrationProvider) -> None:
        self._provider = provider
        self._tasks: queue.Queue[tuple[str, WorldState, GuideMessage, Challenge | None, str]] = queue.Queue()
        self._lock = threading.Lock()
        self._results: dict[str, dict[str, str]] = {}
        self._cache: dict[str, dict[str, str]] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def request(
        self,
        event_key: str,
        world_state: WorldState,
        message: GuideMessage,
        challenge: Challenge | None,
        verbosity: str,
    ) -> None:
        if self._closed or not event_key or event_key in self._cache:
            return
        self._tasks.put((event_key, world_state, message, challenge, verbosity))

    def poll(self, event_key: str) -> dict[str, str] | None:
        with self._lock:
            if event_key in self._results:
                result = self._results.pop(event_key)
                self._cache[event_key] = result
                return result
            return self._cache.get(event_key)

    def close(self) -> None:
        self._closed = True
        self._tasks.put(("", WorldState(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, (), {}), GuideMessage("", "", ""), None, ""))
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while not self._closed:
            event_key, world_state, message, challenge, verbosity = self._tasks.get()
            if self._closed or not event_key:
                return
            try:
                result = self._provider.generate(
                    world_state=world_state,
                    message=message,
                    challenge=challenge,
                    verbosity=verbosity,
                )
            except Exception:
                continue
            with self._lock:
                self._results[event_key] = result


class AsyncConnectionTester:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending = False
        self._result: ConnectionTestResult | None = None
        self._closed = False

    def start(self, config: ProviderConfig) -> bool:
        with self._lock:
            if self._closed or self._pending:
                return False
            self._pending = True
            self._result = None
            self._thread = threading.Thread(
                target=self._run,
                args=(config,),
                daemon=True,
            )
            self._thread.start()
            return True

    def poll(self) -> ConnectionTestResult | None:
        with self._lock:
            if self._result is None:
                return None
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.2)

    def _run(self, config: ProviderConfig) -> None:
        try:
            result = test_provider_connection(config)
        except Exception as exc:
            result = ConnectionTestResult(False, f"Connection test failed: {exc}")
        with self._lock:
            self._pending = False
            if not self._closed:
                self._result = result


def _challenge_definitions() -> list[Challenge]:
    return [
        Challenge("shark_lake", "Shark Challenge", "Shape a wide lake so sharks have room to swim."),
        Challenge("mountain_peak", "Mountain Challenge", "Raise a steep mountain ridge that towers over the sand."),
        Challenge("island_chain", "Island Challenge", "Split the land into islands surrounded by water."),
        Challenge("river_channel", "River Challenge", "Carve a winding water channel through the terrain."),
        Challenge("dinosaur_land", "Dinosaur Challenge", "Build a broad stretch of dry land for the dinosaurs."),
    ]


def _resolve_api_key(backend: str) -> str | None:
    specific = (
        "SANDCAM_LOCAL_LLM_API_KEY"
        if backend == "local_openai_compatible"
        else "SANDCAM_CLOUD_LLM_API_KEY"
    )
    return (
        os.getenv(specific)
        or os.getenv("SANDCAM_LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def build_async_narrator(config: ProviderConfig) -> AsyncNarrator | None:
    if config.backend == "template":
        return None
    enriched = replace(config, api_key=config.api_key or _resolve_api_key(config.backend))
    return AsyncNarrator(OpenAICompatibleNarrator(enriched))


def test_provider_connection(config: ProviderConfig, *, opener: Any = None) -> ConnectionTestResult:
    if config.backend == "template":
        return ConnectionTestResult(True, "Template mode is local and ready.")

    enriched = replace(config, api_key=config.api_key or _resolve_api_key(config.backend))
    narrator = OpenAICompatibleNarrator(enriched, opener=opener)
    world = WorldState(
        timestamp=time.monotonic(),
        water_ratio=0.32,
        land_ratio=0.68,
        coastline_ratio=0.15,
        water_regions=1,
        land_regions=1,
        lakes=1,
        islands=0,
        largest_water_ratio=0.22,
        largest_land_ratio=0.68,
        highest_peak=0.81,
        mean_height=0.54,
        steep_ratio=0.08,
        shark_count=2,
        dinosaur_count=3,
        shark_habitat_ratio=0.32,
        dinosaur_habitat_ratio=0.68,
        features=("large_lake", "mountain_range"),
        deltas={},
    )
    message = GuideMessage(
        kind="observation",
        title="Connection Check",
        body="The sandbox is testing its narrator connection.",
    )
    response = narrator.generate(
        world_state=world,
        message=message,
        challenge=None,
        verbosity="normal",
    )
    title = response.get("title", "").strip() or "Narrator"
    return ConnectionTestResult(True, f"Connected: {title}")


class GuideEngine:
    def __init__(
        self,
        *,
        template_narrator: TemplateNarrator,
        async_narrator: AsyncNarrator | None = None,
        analysis_interval: float = ANALYSIS_INTERVAL_SECONDS,
        event_cooldown: float = EVENT_COOLDOWN_SECONDS,
        challenge_cooldown: float = CHALLENGE_COOLDOWN_SECONDS,
    ) -> None:
        self._analyzer = WorldAnalyzer()
        self._template = template_narrator
        self._async = async_narrator
        self._analysis_interval = analysis_interval
        self._event_cooldown = event_cooldown
        self._challenge_cooldown = challenge_cooldown
        self._last_analysis_at = -math.inf
        self._last_message_at = -math.inf
        self._last_challenge_at = -math.inf
        self._last_challenge_by_id: dict[str, float] = {}
        self._previous_state: WorldState | None = None
        self._message: GuideMessage | None = None
        self._active_challenge: Challenge | None = None
        self._challenge_cycle = 0

    @property
    def current_message(self) -> GuideMessage | None:
        return self._message

    @property
    def active_challenge(self) -> Challenge | None:
        return self._active_challenge

    def close(self) -> None:
        if self._async is not None:
            self._async.close()

    def push_external_event(
        self,
        *,
        event_key: str,
        title: str,
        body: str,
        now: float,
        verbosity: str,
    ) -> GuideMessage | None:
        if now - self._last_message_at < self._event_cooldown:
            return self._poll_async()
        state = self._previous_state
        if state is None:
            state = WorldState(
                timestamp=now,
                water_ratio=0.0,
                land_ratio=1.0,
                coastline_ratio=0.0,
                water_regions=0,
                land_regions=1,
                lakes=0,
                islands=0,
                largest_water_ratio=0.0,
                largest_land_ratio=1.0,
                highest_peak=0.0,
                mean_height=0.0,
                steep_ratio=0.0,
                shark_count=0,
                dinosaur_count=0,
                shark_habitat_ratio=0.0,
                dinosaur_habitat_ratio=1.0,
                features=(),
                deltas={},
            )
        message = GuideMessage(
            kind="observation",
            title=title,
            body=body,
            challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
            event_key=event_key,
        )
        self._publish(message, state, verbosity)
        return self._message

    def update(
        self,
        frame: np.ndarray,
        now: float,
        *,
        shark_count: int,
        dinosaur_count: int,
        creatures_enabled: bool,
        guide_enabled: bool,
        challenges_enabled: bool,
        verbosity: str,
    ) -> GuideMessage | None:
        if now - self._last_analysis_at < self._analysis_interval:
            return self._poll_async()

        self._last_analysis_at = now
        current = self._analyzer.analyze(
            frame,
            now,
            shark_count=shark_count if creatures_enabled else 0,
            dinosaur_count=dinosaur_count if creatures_enabled else 0,
        )
        current = self._analyzer.with_deltas(current, self._previous_state)

        if challenges_enabled:
            if self._active_challenge is None or self._challenge_complete(current, self._active_challenge):
                if self._active_challenge is not None and self._challenge_complete(current, self._active_challenge):
                    self._publish(
                        GuideMessage(
                            kind="celebration",
                            title="Challenge Complete",
                            body="You reshaped the world and solved the guide's challenge.",
                            challenge_id=self._active_challenge.challenge_id,
                            completed=True,
                            expires_at=now + self._event_cooldown,
                            event_key=f"celebrate:{self._active_challenge.challenge_id}:{int(now)}",
                        ),
                        current,
                        verbosity,
                    )
                self._choose_challenge(current, now)
        else:
            self._active_challenge = None

        event = self._pick_event(current, self._previous_state, creatures_enabled)
        self._previous_state = current

        if not guide_enabled:
            return None

        if event is not None and now - self._last_message_at >= self._event_cooldown:
            message = self._message_from_event(event, current)
            self._publish(message, current, verbosity)
            return self._message

        if self._message is None and self._active_challenge is not None:
            self._publish(
                GuideMessage(
                    kind="challenge",
                    title=self._active_challenge.title,
                    body=self._active_challenge.prompt,
                    challenge_id=self._active_challenge.challenge_id,
                    expires_at=now + self._event_cooldown,
                    event_key=f"challenge:{self._active_challenge.challenge_id}:{int(now)}",
                ),
                current,
                verbosity,
            )

        return self._poll_async()

    def _poll_async(self) -> GuideMessage | None:
        if self._async is None or self._message is None or not self._message.event_key:
            return self._message
        enriched = self._async.poll(self._message.event_key)
        if not enriched:
            return self._message
        self._message = replace(
            self._message,
            title=enriched.get("title", self._message.title),
            body=enriched.get("body", self._message.body),
        )
        return self._message

    def _publish(self, message: GuideMessage, state: WorldState, verbosity: str) -> None:
        challenge = self._active_challenge if not message.completed else None
        local = self._template.generate(
            world_state=state,
            message=message,
            challenge=challenge,
            verbosity=verbosity,
        )
        self._message = replace(
            message,
            title=local["title"],
            body=local["body"],
        )
        self._last_message_at = state.timestamp
        if self._async is not None:
            self._async.request(message.event_key, state, self._message, challenge, verbosity)

    def _choose_challenge(self, state: WorldState, now: float) -> None:
        challenges = _challenge_definitions()
        for offset in range(len(challenges)):
            idx = (self._challenge_cycle + offset) % len(challenges)
            candidate = challenges[idx]
            if self._challenge_complete(state, candidate):
                continue
            if now - self._last_challenge_by_id.get(candidate.challenge_id, -math.inf) < self._challenge_cooldown:
                continue
            self._active_challenge = candidate
            self._last_challenge_at = now
            self._last_challenge_by_id[candidate.challenge_id] = now
            self._challenge_cycle = idx + 1
            return
        self._active_challenge = None

    def _challenge_complete(self, state: WorldState, challenge: Challenge) -> bool:
        if challenge.challenge_id == "shark_lake":
            return state.largest_water_ratio >= 0.18
        if challenge.challenge_id == "mountain_peak":
            return state.highest_peak >= 0.82 and "mountain_range" in state.features
        if challenge.challenge_id == "island_chain":
            return state.islands >= 2
        if challenge.challenge_id == "river_channel":
            return "river_like_channel" in state.features and state.coastline_ratio >= 0.14
        if challenge.challenge_id == "dinosaur_land":
            return state.land_ratio >= 0.72
        return False

    def _pick_event(
        self,
        current: WorldState,
        previous: WorldState | None,
        creatures_enabled: bool,
    ) -> WorldEvent | None:
        if previous is None:
            return WorldEvent(
                key="world_ready",
                kind="world_ready",
                payload={},
            )

        delta_water = current.deltas.get("water_ratio", 0.0)
        if delta_water >= 0.08:
            return WorldEvent("water_gained", "water_gained", {"delta": round(delta_water, 3)})
        if delta_water <= -0.08:
            return WorldEvent("water_lost", "water_lost", {"delta": round(delta_water, 3)})
        if current.deltas.get("highest_peak", 0.0) >= 0.06 and current.highest_peak >= 0.8:
            return WorldEvent("mountain_raised", "mountain_raised", {"peak": round(current.highest_peak, 3)})
        if current.islands > previous.islands:
            return WorldEvent("island_created", "island_created", {"islands": current.islands})
        if "river_like_channel" in current.features and "river_like_channel" not in previous.features:
            return WorldEvent("channel_opened", "channel_opened", {})
        if creatures_enabled and current.shark_count < previous.shark_count:
            return WorldEvent("shark_habitat_lost", "shark_habitat_lost", {})
        if creatures_enabled and current.dinosaur_count < previous.dinosaur_count:
            return WorldEvent("dinosaur_habitat_lost", "dinosaur_habitat_lost", {})
        return None

    def _message_from_event(self, event: WorldEvent, state: WorldState) -> GuideMessage:
        if event.kind == "world_ready":
            return GuideMessage(
                kind="observation",
                title="Sandbox Ready",
                body="The world is awake. Shape the sand to make lakes, ridges, and islands.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=event.key,
            )
        if event.kind == "water_gained":
            return GuideMessage(
                kind="observation",
                title="Water Is Spreading",
                body="A new watery region is opening up. You might be carving a lake or a channel.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        if event.kind == "water_lost":
            return GuideMessage(
                kind="observation",
                title="Land Is Rising",
                body="The water is shrinking and more dry ground is appearing.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        if event.kind == "mountain_raised":
            return GuideMessage(
                kind="observation",
                title="Mountain Growing",
                body="That ridge is getting taller and steeper. The landscape is starting to tower.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        if event.kind == "island_created":
            return GuideMessage(
                kind="observation",
                title="Islands Forming",
                body="You split the land into separated pieces, like a tiny archipelago.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        if event.kind == "channel_opened":
            return GuideMessage(
                kind="observation",
                title="River Path Found",
                body="A winding water path has appeared through the terrain.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        if event.kind == "shark_habitat_lost":
            return GuideMessage(
                kind="observation",
                title="Sharks Need Water",
                body="Part of the water habitat disappeared, so the sharks retreated.",
                challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
                event_key=f"{event.key}:{int(state.timestamp)}",
            )
        return GuideMessage(
            kind="observation",
            title="Dinosaurs Need Land",
            body="Some dry ground vanished, so the dinosaurs lost part of their world.",
            challenge_id=self._active_challenge.challenge_id if self._active_challenge else None,
            event_key=f"{event.key}:{int(state.timestamp)}",
        )
