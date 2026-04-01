"""
Computer-vision object detection for the AR sandbox.

Supports two backends:
  yolo          — local YOLO model via ultralytics (pip install ultralytics)
  openai_vision — any OpenAI-compatible vision API (GPT-4V, local models, etc.)

CVDetectionLayer runs on a background thread, rate-limits API calls to avoid
saturating slow endpoints, and provides thread-safe tracked results via
get_raw_tracks().  Sandbox-space mapping is handled by webcam_observer, which
owns calibration state.
"""
from __future__ import annotations

import base64
import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


# ── raw detection ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CVRawDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in camera pixels

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def iou(self, other: "CVRawDetection") -> float:
        ax1, ay1, ax2, ay2 = self.bbox
        bx1, by1, bx2, by2 = other.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / (union + 1e-9)


# ── backends ──────────────────────────────────────────────────────────────────

class CVDetectionBackend(Protocol):
    def detect(self, frame: np.ndarray) -> list[CVRawDetection]: ...
    def is_available(self) -> bool: ...


class YOLODetector:
    """Local YOLO detection via ultralytics (pip install ultralytics)."""

    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self._model_path = model_path
        self._model: Any = None
        self._load_error: str = ""

    def is_available(self) -> bool:
        try:
            import ultralytics  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error:
            return False
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self._model_path)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def detect(self, frame: np.ndarray) -> list[CVRawDetection]:
        if not self._load():
            return []
        try:
            results = self._model(frame, verbose=False)
        except Exception:
            return []
        out: list[CVRawDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names.get(cls, str(cls))
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(CVRawDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return out


class OpenAIVisionDetector:
    """
    Object detection via any OpenAI-compatible vision API.
    Sends a compressed JPEG and parses a JSON detection list from the response.
    Works with cloud APIs (GPT-4o, Claude) and local servers (LM Studio, Ollama).
    """

    _SYSTEM = (
        "You are a compact object-detection assistant for an augmented-reality sandbox. "
        "Given an image identify every distinct physical object — toys, figurines, "
        "vehicles, animals, buildings, natural items. Ignore background sand, walls, "
        "and the camera frame itself. "
        "Return ONLY a JSON array, no prose: "
        '[{"label":"toy car","confidence":0.9,"bbox":[x1,y1,x2,y2]}] '
        "bbox values are integer pixel coordinates in the original (unscaled) image."
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "gpt-4o",
        api_key: str = "",
        timeout: float = 8.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._custom_suffix = ""  # appended to system prompt when custom objects exist

    def is_available(self) -> bool:
        return bool(self._base_url and self._model)

    def set_custom_objects(self, objects: list) -> None:
        """Inject known object names into the detection system prompt."""
        if objects:
            lines = "\n".join(f'- "{o.label}": {o.description}' for o in objects)
            self._custom_suffix = (
                f"\nKnown sandbox objects — use these exact labels when you see them:\n{lines}"
            )
        else:
            self._custom_suffix = ""

    def detect(self, frame: np.ndarray) -> list[CVRawDetection]:
        try:
            import cv2  # type: ignore
        except ImportError:
            return []
        from urllib import request as urllib_request

        h, w = frame.shape[:2]
        max_side = 640
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            sx, sy = w / small.shape[1], h / small.shape[0]
        else:
            small, sx, sy = frame, 1.0, 1.0

        _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()

        payload = json.dumps({
            "model": self._model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": self._SYSTEM + self._custom_suffix},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Detect all objects."},
                ]},
            ],
        }).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib_request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers=headers,
        )
        try:
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"].strip()
        except Exception:
            return []

        start, end = text.find("["), text.rfind("]") + 1
        if start == -1 or end <= 0:
            return []
        try:
            raw = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []

        out: list[CVRawDetection] = []
        for item in raw:
            try:
                label = str(item["label"])
                conf = float(item.get("confidence", 0.8))
                b = item["bbox"]
                x1 = int(b[0] * sx)
                y1 = int(b[1] * sy)
                x2 = int(b[2] * sx)
                y2 = int(b[3] * sy)
                out.append(CVRawDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return out


# ── tracker ───────────────────────────────────────────────────────────────────

@dataclass
class CVRawTrack:
    """
    Tracked detection in camera-space coordinates.
    webcam_observer maps camera_pos to sandbox_pos using calibration.
    """
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    camera_pos: tuple[float, float]
    first_seen: float
    last_seen: float
    stable_since: float


_id_lock = threading.Lock()
_next_id = 0


def _new_id() -> int:
    global _next_id
    with _id_lock:
        _next_id += 1
        return _next_id


class CVObjectTracker:
    """Greedy IoU-based multi-object tracker.  Assigns stable IDs across frames."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        stale_seconds: float = 1.0,
        stable_movement_px: float = 15.0,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._stale_seconds = stale_seconds
        self._stable_px = stable_movement_px
        self._tracks: dict[int, CVRawTrack] = {}

    def update(self, detections: list[CVRawDetection], now: float) -> list[CVRawTrack]:
        # Cull tracks not seen recently
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if now - t.last_seen <= self._stale_seconds
        }

        used: set[int] = set()
        unmatched: list[CVRawDetection] = []

        for det in detections:
            best_tid: int | None = None
            best_iou = self._iou_threshold

            for tid, track in self._tracks.items():
                if tid in used or track.label != det.label:
                    continue
                track_det = CVRawDetection(track.label, track.confidence, track.bbox)
                score = det.iou(track_det)
                if score > best_iou:
                    best_iou, best_tid = score, tid

            if best_tid is not None:
                used.add(best_tid)
                old = self._tracks[best_tid]
                movement = math.dist(old.camera_pos, det.center)
                self._tracks[best_tid] = CVRawTrack(
                    track_id=best_tid,
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    camera_pos=det.center,
                    first_seen=old.first_seen,
                    last_seen=now,
                    stable_since=old.stable_since if movement < self._stable_px else now,
                )
            else:
                unmatched.append(det)

        for det in unmatched:
            tid = _new_id()
            self._tracks[tid] = CVRawTrack(
                track_id=tid,
                label=det.label,
                confidence=det.confidence,
                bbox=det.bbox,
                camera_pos=det.center,
                first_seen=now,
                last_seen=now,
                stable_since=now,
            )

        return list(self._tracks.values())

    def clear(self) -> None:
        self._tracks.clear()


# ── per-track custom-object identifier (YOLO only) ───────────────────────────

class _AsyncTrackIdentifier:
    """
    For YOLO tracks: when a new track_id appears, crops its bbox and asks the
    vision API which custom object it is.  One API call per track, cached.
    Results are applied back to CVRawTrack.label on the next loop iteration.
    """

    def __init__(
        self,
        objects: list,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float,
    ) -> None:
        self._objects = objects
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._seen: set[int] = set()
        self._results: dict[int, str] = {}
        self._lock = threading.Lock()

    def submit(self, track_id: int, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        with self._lock:
            if track_id in self._seen:
                return
            self._seen.add(track_id)
        threading.Thread(
            target=self._run, args=(track_id, frame.copy(), bbox), daemon=True
        ).start()

    def pop_results(self) -> dict[int, str]:
        with self._lock:
            r = dict(self._results)
            self._results.clear()
            return r

    def _run(self, track_id: int, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return
        from cv_object_store import match_object

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        pad = 20
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            return
        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf.tobytes()).decode()
        label = match_object(
            b64,
            self._objects,
            base_url=self._base_url,
            model=self._model,
            api_key=self._api_key,
            timeout=self._timeout,
        )
        if label:
            print(f"[CV] track #{track_id} matched custom object '{label}'")
            with self._lock:
                self._results[track_id] = label


# ── detection layer ───────────────────────────────────────────────────────────

class CVDetectionLayer:
    """
    Background-threaded detection + tracking.

    Call submit_frame() from the webcam capture thread (non-blocking).
    Call get_raw_tracks() from the main thread to read the latest results.
    Rate-limiting via min_interval_seconds prevents flooding slow API backends.
    """

    # Labels always ignored regardless of confidence, case-insensitive.
    # Tuned for overhead/downward-facing camera setups where hands and heads
    # are common false positives.
    DEFAULT_IGNORE: frozenset[str] = frozenset({
        "person", "people",
        "hand", "hands",
        "head", "face",
        "arm", "finger",
    })

    def __init__(
        self,
        backend: CVDetectionBackend,
        min_confidence: float = 0.5,
        min_interval_seconds: float = 0.5,
        ignore_labels: set[str] | None = None,
    ) -> None:
        self._backend = backend
        self._min_confidence = min_confidence
        self._min_interval = min_interval_seconds
        self._ignore: set[str] = (
            {lbl.lower() for lbl in ignore_labels}
            if ignore_labels is not None
            else set(self.DEFAULT_IGNORE)
        )
        self._tracker = CVObjectTracker()
        self._track_identifier: _AsyncTrackIdentifier | None = None
        self._frame_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._tracks: list[CVRawTrack] = []
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        """Non-blocking.  Drops any queued frame and replaces with the latest."""
        try:
            self._frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get_raw_tracks(self) -> list[CVRawTrack]:
        with self._lock:
            return list(self._tracks)

    def set_custom_objects(
        self,
        objects: list,
        *,
        api_url: str = "",
        api_model: str = "",
        api_key: str = "",
        timeout: float = 8.0,
    ) -> None:
        """
        Inject custom object knowledge.
        - For the Vision API backend: enriches the detection system prompt (free).
        - For YOLO: sets up a per-new-track identifier that calls the API once
          per track to remap generic labels to custom ones.
        """
        if hasattr(self._backend, "set_custom_objects"):
            self._backend.set_custom_objects(objects)

        if objects and isinstance(self._backend, YOLODetector) and api_url and api_model:
            self._track_identifier = _AsyncTrackIdentifier(
                objects,
                base_url=api_url,
                model=api_model,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            self._track_identifier = None

    def close(self) -> None:
        self._running = False
        try:
            self._frame_q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _loop(self) -> None:
        last_run = 0.0
        prev_track_ids: set[int] = set()
        while self._running:
            try:
                frame = self._frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            now = time.monotonic()
            if now - last_run < self._min_interval:
                continue
            last_run = now
            try:
                raw = self._backend.detect(frame)
            except Exception as exc:
                print(f"[CV] detection error: {exc}")
                raw = []
            above_threshold = [d for d in raw if d.confidence >= self._min_confidence]
            ignored = [d for d in above_threshold if d.label.lower() in self._ignore]
            filtered = [d for d in above_threshold if d.label.lower() not in self._ignore]

            ts = time.strftime("%H:%M:%S")
            if ignored:
                counts: dict[str, int] = {}
                for d in ignored:
                    counts[d.label] = counts.get(d.label, 0) + 1
                summary = ", ".join(f"{lbl}×{n}" for lbl, n in counts.items())
                print(f"[CV {ts}] ignored ({summary})")
            if filtered:
                for d in filtered:
                    x1, y1, x2, y2 = d.bbox
                    print(f"[CV {ts}] {d.label:20s}  conf={d.confidence:.2f}  bbox=({x1},{y1})-({x2},{y2})")
            else:
                print(f"[CV {ts}] no detections above threshold {self._min_confidence:.2f}")

            tracks = self._tracker.update(filtered, now)

            current_ids = {t.track_id for t in tracks}
            appeared = current_ids - prev_track_ids
            disappeared = prev_track_ids - current_ids
            for t in tracks:
                if t.track_id in appeared:
                    print(f"[CV] +track #{t.track_id}  '{t.label}'  conf={t.confidence:.2f}")
                    if self._track_identifier is not None:
                        self._track_identifier.submit(t.track_id, frame, t.bbox)
            for tid in disappeared:
                print(f"[CV] -track #{tid}  removed")
            prev_track_ids = current_ids

            # Apply any completed custom-object identifications
            if self._track_identifier is not None:
                for tid, custom_label in self._track_identifier.pop_results().items():
                    for t in tracks:
                        if t.track_id == tid:
                            t.label = custom_label

            with self._lock:
                self._tracks = tracks


# ── factory ───────────────────────────────────────────────────────────────────

def build_cv_layer(
    backend_name: str,
    *,
    model: str = "",
    api_url: str = "",
    api_key: str = "",
    api_model: str = "",
    timeout: float = 8.0,
    min_confidence: float = 0.5,
    ignore_labels: set[str] | None = None,
) -> "CVDetectionLayer | None":
    """
    Build and start a CVDetectionLayer.  Returns None if the backend is unavailable
    (e.g. ultralytics not installed, or no model configured for vision API).
    Rate-limiting is set automatically: 0.15 s for YOLO, 2.5 s for API backends.
    """
    if backend_name == "yolo":
        backend: CVDetectionBackend = YOLODetector(model_path=model or "yolo11n.pt")
        interval = 0.15
    elif backend_name == "openai_vision":
        backend = OpenAIVisionDetector(
            base_url=api_url or "http://127.0.0.1:1234/v1",
            model=api_model or "gpt-4o",
            api_key=api_key,
            timeout=timeout,
        )
        interval = 2.5
    else:
        return None

    if not backend.is_available():
        return None

    return CVDetectionLayer(
        backend,
        min_confidence=min_confidence,
        min_interval_seconds=interval,
        ignore_labels=ignore_labels,
    )
