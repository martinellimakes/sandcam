"""
Optional webcam-based object observation for the AR sandbox.

The vision subsystem is fully optional and should not affect the base sandbox
when disabled or when OpenCV is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import math
import sys
import threading
import time
from typing import Any

import numpy as np

CORNER_MARKER_IDS = {
    100: "top_left",
    101: "top_right",
    102: "bottom_left",
    103: "bottom_right",
}
OBJECT_KIND_BY_MARKER = {
    1: "boat",
    2: "dino_toy",
    3: "house",
    4: "tree",
    5: "volcano",
}
_DEST_POINT_ORDER = {
    "top_left": 0,
    "top_right": 1,
    "bottom_right": 2,
    "bottom_left": 3,
}


@dataclass(frozen=True)
class CameraTestResult:
    ok: bool
    summary: str


@dataclass(frozen=True)
class CameraInfo:
    index: int
    summary: str


@dataclass(frozen=True)
class CalibrationData:
    camera_points: tuple[tuple[float, float], ...] = ()

    @property
    def ready(self) -> bool:
        return len(self.camera_points) == 4


@dataclass(frozen=True)
class VisionObject:
    marker_id: int
    kind: str
    camera_pos: tuple[float, float]
    sandbox_pos: tuple[float, float]
    confidence: float
    stable_for_seconds: float
    biome_under_object: str


@dataclass(frozen=True)
class VisionEvent:
    event_key: str
    kind: str
    title: str
    body: str
    sandbox_pos: tuple[float, float]


@dataclass
class _TrackedMarker:
    marker_id: int
    kind: str
    camera_pos: tuple[float, float]
    corners: tuple[tuple[float, float], ...]
    first_seen: float
    last_seen: float
    stable_since: float
    confidence: float


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def cv2_available() -> bool:
    return _import_cv2() is not None


def _backend_candidates(cv2: Any) -> list[tuple[str, int | None]]:
    if sys.platform == "win32":
        if hasattr(cv2, "CAP_DSHOW"):
            candidates: list[tuple[str, int | None]] = [("dshow", cv2.CAP_DSHOW)]
        else:
            candidates = []
        if hasattr(cv2, "CAP_MSMF"):
            candidates.append(("msmf", cv2.CAP_MSMF))
        return candidates
    return [("default", None)]


@contextmanager
def _quiet_opencv_logs(cv2: Any):
    logging = getattr(getattr(cv2, "utils", object()), "logging", None)
    if logging is None or not hasattr(logging, "setLogLevel"):
        yield
        return

    previous = logging.LOG_LEVEL_WARNING
    try:
        logging.setLogLevel(logging.LOG_LEVEL_SILENT)
        yield
    finally:
        logging.setLogLevel(previous)


def _probe_capture(capture: Any, attempts: int = 4) -> tuple[bool, Any | None]:
    cv2 = _import_cv2()
    manager = _quiet_opencv_logs(cv2) if cv2 is not None else nullcontext()
    with manager:
        for _ in range(attempts):
            ok, frame = capture.read()
            if ok and frame is not None:
                return True, frame
            time.sleep(0.03)
    return False, None


def _open_capture(
    cv2: Any,
    camera_index: int,
    *,
    start_at: int = 0,
) -> tuple[Any | None, str | None, Any | None, int | None]:
    candidates = _backend_candidates(cv2)
    for backend_idx in range(start_at, len(candidates)):
        name, backend = candidates[backend_idx]
        with _quiet_opencv_logs(cv2):
            capture = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        ok, frame = _probe_capture(capture)
        if ok:
            return capture, name, frame, backend_idx
        capture.release()
    return None, None, None, None


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a_rows: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        a_rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        a_rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])
    _, _, vh = np.linalg.svd(np.asarray(a_rows, dtype=np.float64))
    h = vh[-1, :].reshape(3, 3)
    if abs(h[2, 2]) < 1e-9:
        return h
    return h / h[2, 2]


def map_camera_to_sandbox(
    point: tuple[float, float],
    calibration: CalibrationData,
    scene_size: tuple[int, int],
) -> tuple[float, float]:
    if not calibration.ready:
        return point

    w, h = scene_size
    src = np.asarray(calibration.camera_points, dtype=np.float64)
    dst = np.asarray([(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)], dtype=np.float64)
    homography = _solve_homography(src, dst)
    vec = homography @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    if abs(vec[2]) < 1e-9:
        return point
    mapped = vec[:2] / vec[2]
    return float(mapped[0]), float(mapped[1])


def _homography_for_scene(
    calibration: CalibrationData,
    scene_size: tuple[int, int],
) -> np.ndarray | None:
    if not calibration.ready:
        return None
    w, h = scene_size
    src = np.asarray(calibration.camera_points, dtype=np.float64)
    dst = np.asarray(
        [(0.0, 0.0), (w - 1.0, 0.0), (w - 1.0, h - 1.0), (0.0, h - 1.0)],
        dtype=np.float64,
    )
    return _solve_homography(src, dst)


def calibration_from_corner_markers(
    detections: list[tuple[int, tuple[float, float]]],
) -> CalibrationData | None:
    ordered: list[tuple[float, float] | None] = [None, None, None, None]
    for marker_id, point in detections:
        label = CORNER_MARKER_IDS.get(marker_id)
        if label is None:
            continue
        ordered[_DEST_POINT_ORDER[label]] = point
    if any(point is None for point in ordered):
        return None
    return CalibrationData(camera_points=tuple(point for point in ordered if point is not None))


class AsyncCameraTester:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending = False
        self._result: CameraTestResult | None = None
        self._closed = False

    def start(self, camera_index: int) -> bool:
        with self._lock:
            if self._closed or self._pending:
                return False
            self._pending = True
            self._result = None
            self._thread = threading.Thread(
                target=self._run,
                args=(camera_index,),
                daemon=True,
            )
            self._thread.start()
            return True

    def poll(self) -> CameraTestResult | None:
        with self._lock:
            if self._result is None:
                return None
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def _run(self, camera_index: int) -> None:
        result = test_camera(camera_index)
        with self._lock:
            self._pending = False
            if not self._closed:
                self._result = result


class AsyncCameraDiscovery:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._pending = False
        self._result: list[CameraInfo] | None = None
        self._closed = False

    def start(self, max_index: int = 6) -> bool:
        with self._lock:
            if self._closed or self._pending:
                return False
            self._pending = True
            self._result = None
            self._thread = threading.Thread(
                target=self._run,
                args=(max_index,),
                daemon=True,
            )
            self._thread.start()
            return True

    def poll(self) -> list[CameraInfo] | None:
        with self._lock:
            if self._result is None:
                return None
            result = self._result
            self._result = None
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def _run(self, max_index: int) -> None:
        result = list_cameras(max_index=max_index)
        with self._lock:
            self._pending = False
            if not self._closed:
                self._result = result


def test_camera(camera_index: int) -> CameraTestResult:
    cv2 = _import_cv2()
    if cv2 is None:
        return CameraTestResult(False, "OpenCV is not installed, so webcam vision is unavailable.")

    capture, backend_name, frame, _backend_idx = _open_capture(cv2, camera_index)
    if capture is None:
        return CameraTestResult(False, f"Camera {camera_index} could not be opened.")
    capture.release()
    if frame is None:
        return CameraTestResult(False, f"Camera {camera_index} opened but did not return a frame.")
    h, w = frame.shape[:2]
    backend_note = f" via {backend_name}" if backend_name else ""
    return CameraTestResult(True, f"Camera {camera_index} is working at {w}x{h}{backend_note}.")


def list_cameras(max_index: int = 6) -> list[CameraInfo]:
    cameras: list[CameraInfo] = []
    for index in range(max(1, max_index)):
        result = test_camera(index)
        if result.ok:
            cameras.append(CameraInfo(index=index, summary=result.summary))
    return cameras


class WebcamObserver:
    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._cv2 = _import_cv2()
        self._capture = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tracked: dict[int, _TrackedMarker] = {}
        self._latest_frame: np.ndarray | None = None
        self._error_message = ""
        self._backend_index: int | None = None
        self._backend_name: str | None = None

        if self._cv2 is None:
            self._error_message = "OpenCV is not installed."
            return

        self._start_capture()

    @property
    def available(self) -> bool:
        return self._cv2 is not None and self._capture is not None and not self._error_message

    @property
    def error_message(self) -> str:
        return self._error_message

    def _start_capture(self, *, start_at: int = 0) -> None:
        assert self._cv2 is not None
        capture, backend_name, first_frame, backend_idx = _open_capture(
            self._cv2,
            self._camera_index,
            start_at=start_at,
        )
        if capture is None:
            self._error_message = f"Could not open camera {self._camera_index}."
            return
        self._capture = capture
        self._backend_index = backend_idx
        self._backend_name = backend_name
        self._error_message = ""
        if first_frame is not None:
            self._latest_frame = first_frame
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._backend_name = None

    def _loop(self) -> None:
        assert self._cv2 is not None
        failed_reads = 0
        while self._running and self._capture is not None:
            with _quiet_opencv_logs(self._cv2):
                ok, frame = self._capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= 3:
                    next_backend = 0 if self._backend_index is None else self._backend_index + 1
                    current = self._capture
                    self._capture = None
                    current.release()
                    if next_backend < len(_backend_candidates(self._cv2)):
                        self._start_capture(start_at=next_backend)
                    else:
                        self._running = False
                        self._error_message = (
                            f"Camera {self._camera_index}"
                            + (f" via {self._backend_name}" if self._backend_name else "")
                            + " stopped returning frames."
                        )
                    return
                time.sleep(0.05)
                continue
            failed_reads = 0
            detections = self._detect_markers(frame)
            now = time.monotonic()
            with self._lock:
                self._latest_frame = frame
                self._update_tracks(detections, now)

    def _detect_markers(self, frame: np.ndarray) -> list[tuple[int, tuple[float, float], tuple[tuple[float, float], ...]]]:
        assert self._cv2 is not None
        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "getPredefinedDictionary"):
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            if hasattr(cv2.aruco, "DetectorParameters"):
                detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
                corners, ids, _rejected = detector.detectMarkers(gray)
            else:
                parameters = cv2.aruco.DetectorParameters_create()
                corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        else:
            return []

        results: list[tuple[int, tuple[float, float], tuple[tuple[float, float], ...]]] = []
        if ids is None:
            return results
        for marker_id, corner_set in zip(ids.flatten(), corners, strict=True):
            pts = corner_set.reshape(-1, 2)
            center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            point_list = tuple((float(px), float(py)) for px, py in pts)
            results.append((int(marker_id), center, point_list))
        return results

    def _update_tracks(
        self,
        detections: list[tuple[int, tuple[float, float], tuple[tuple[float, float], ...]]],
        now: float,
    ) -> None:
        seen: set[int] = set()
        for marker_id, center, corners in detections:
            if marker_id not in OBJECT_KIND_BY_MARKER and marker_id not in CORNER_MARKER_IDS:
                continue
            seen.add(marker_id)
            existing = self._tracked.get(marker_id)
            if existing is None:
                kind = OBJECT_KIND_BY_MARKER.get(marker_id, CORNER_MARKER_IDS.get(marker_id, "marker"))
                self._tracked[marker_id] = _TrackedMarker(
                    marker_id=marker_id,
                    kind=kind,
                    camera_pos=center,
                    corners=corners,
                    first_seen=now,
                    last_seen=now,
                    stable_since=now,
                    confidence=1.0,
                )
                continue

            movement = math.dist(existing.camera_pos, center)
            stable_since = existing.stable_since if movement < 12.0 else now
            self._tracked[marker_id] = _TrackedMarker(
                marker_id=marker_id,
                kind=existing.kind,
                camera_pos=center,
                corners=corners,
                first_seen=existing.first_seen,
                last_seen=now,
                stable_since=stable_since,
                confidence=1.0,
            )

        stale = [marker_id for marker_id, marker in self._tracked.items() if marker_id not in seen and now - marker.last_seen > 0.75]
        for marker_id in stale:
            self._tracked.pop(marker_id, None)

    def auto_calibrate(self) -> CalibrationData | None:
        with self._lock:
            points = [
                (marker.marker_id, marker.camera_pos)
                for marker in self._tracked.values()
                if marker.marker_id in CORNER_MARKER_IDS
            ]
        return calibration_from_corner_markers(points)

    def get_objects(
        self,
        calibration: CalibrationData,
        scene_size: tuple[int, int],
        *,
        min_stable_seconds: float = 0.6,
        frame: np.ndarray | None = None,
    ) -> list[VisionObject]:
        with self._lock:
            tracked = list(self._tracked.values())
        now = time.monotonic()
        objects: list[VisionObject] = []
        for marker in tracked:
            if marker.marker_id not in OBJECT_KIND_BY_MARKER:
                continue
            stable_for = now - marker.stable_since
            if stable_for < min_stable_seconds:
                continue
            sandbox_pos = map_camera_to_sandbox(marker.camera_pos, calibration, scene_size)
            biome = _sample_biome(frame, sandbox_pos) if frame is not None else "unknown"
            objects.append(
                VisionObject(
                    marker_id=marker.marker_id,
                    kind=marker.kind,
                    camera_pos=marker.camera_pos,
                    sandbox_pos=sandbox_pos,
                    confidence=marker.confidence,
                    stable_for_seconds=stable_for,
                    biome_under_object=biome,
                )
            )
        return objects

    def get_warped_rgb(
        self,
        calibration: CalibrationData,
        scene_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self._cv2 is None:
            return None
        homography = _homography_for_scene(calibration, scene_size)
        if homography is None:
            return None
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()

        w, h = scene_size
        warped = self._cv2.warpPerspective(frame, homography, (w, h))
        rgb = self._cv2.cvtColor(warped, self._cv2.COLOR_BGR2RGB)
        return rgb

    def get_preview_rgb(
        self,
        scene_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self._cv2 is None:
            return None
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()

        w, h = scene_size
        resized = self._cv2.resize(frame, (w, h), interpolation=self._cv2.INTER_LINEAR)
        rgb = self._cv2.cvtColor(resized, self._cv2.COLOR_BGR2RGB)
        return rgb

    def debug_points(
        self,
        calibration: CalibrationData,
        scene_size: tuple[int, int],
    ) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        with self._lock:
            tracked = list(self._tracked.values())
        debug: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for marker in tracked:
            mapped = map_camera_to_sandbox(marker.camera_pos, calibration, scene_size)
            debug.append((marker.kind, marker.camera_pos, mapped))
        return debug


def _sample_biome(frame: np.ndarray | None, sandbox_pos: tuple[float, float]) -> str:
    if frame is None:
        return "unknown"
    h, w = frame.shape
    x = int(np.clip(round(sandbox_pos[0]), 0, w - 1))
    y = int(np.clip(round(sandbox_pos[1]), 0, h - 1))
    return "water" if float(frame[y, x]) < 0.46 else "land"
