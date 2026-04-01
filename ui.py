"""
Sidebar settings panel and guide overlay.

Toggle open / closed with Tab. All settings live in Config, which is the
single source of truth read by main.py each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
from typing import Iterable

import pygame

# ── palette ───────────────────────────────────────────────────────────────────
_BG         = (22, 24, 34)
_DIVIDER    = (44, 48, 66)
_ACCENT     = (60, 130, 210)
_TEXT       = (210, 215, 225)
_SUBTEXT    = (120, 126, 148)
_TRACK      = (48, 52, 72)
_FILL       = (55, 110, 185)
_THUMB      = (85, 160, 230)
_BTN        = (42, 46, 65)
_BTN_HOV    = (58, 65, 88)
_BTN_ACT    = (50, 115, 195)
_INPUT_BG   = (28, 33, 46)
_INPUT_ACT  = (30, 68, 112)
_SCROLL_TRK = (34, 38, 52)
_SCROLL_BAR = (90, 110, 145)
_CARD_BG    = (16, 22, 32, 220)
_CARD_BORDER = (72, 124, 188)
_SUCCESS    = (90, 180, 120)

SIDEBAR_W   = 310
PAD         = 14
ROW         = 30
ANIM_SPEED  = 1680.0

SCHEMES = ["terrain", "heat", "greyscale", "desert"]
VERBOSITIES = ["quiet", "normal", "lively"]
LLM_BACKENDS = [
    ("template", "Template"),
    ("local_openai_compatible", "Local"),
    ("cloud_openai_compatible", "Cloud"),
]
CV_BACKENDS = [
    ("yolo", "YOLO"),
    ("openai_vision", "Vision API"),
]
CONFIG_PATH = pathlib.Path(__file__).with_name("sandcam-settings.json")
_GUIDE_FONTS: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font] | None = None
SLIDER_SPECS = {
    "min": {"label": "Min", "lo": 100, "hi": 2000, "kind": "int"},
    "max": {"label": "Max", "lo": 200, "hi": 3000, "kind": "int"},
    "alpha": {"label": "Blend", "lo": 0.05, "hi": 1.0, "kind": "float"},
    "threshold": {"label": "Threshold", "lo": 5, "hi": 120, "kind": "int"},
    "persist": {"label": "Delay", "lo": 1, "hi": 12, "kind": "int"},
    "reject": {"label": "Reject", "lo": 0, "hi": 400, "kind": "int"},
    "sharks": {"label": "Sharks", "lo": 0, "hi": 12, "kind": "int"},
    "dinosaurs": {"label": "Dinosaurs", "lo": 0, "hi": 12, "kind": "int"},
    "cv_confidence": {"label": "Min Confidence", "lo": 0.1, "hi": 1.0, "kind": "float"},
}


@dataclass
class Config:
    min_depth_mm: int = 400
    max_depth_mm: int = 1100
    temporal_alpha: float = 0.35
    change_threshold_mm: int = 35
    persistence_frames: int = 4
    foreground_reject_mm: int = 120
    colour_scheme: str = "terrain"
    show_contours: bool = True
    show_creatures: bool = True
    shark_count: int = 4
    dinosaur_count: int = 5
    display_index: int = 0
    fullscreen: bool = False
    windowed_w: int = 900
    windowed_h: int = 650
    ai_enabled: bool = False
    guide_enabled: bool = True
    guide_challenges_enabled: bool = True
    llm_enabled: bool = False
    llm_backend: str = "template"
    llm_base_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = ""
    llm_timeout_seconds: float = 2.0
    guide_verbosity: str = "normal"
    guide_update_seconds: float = 8.0
    vision_enabled: bool = False
    object_reactions_enabled: bool = True
    camera_index: int = 0
    available_cameras: list[dict[str, str | int]] | None = None
    vision_debug_enabled: bool = False
    vision_calibration_points: list[list[float]] | None = None
    vision_calibrating: bool = False
    # ── CV object detection ───────────────────────────────────────────────────
    cv_detection_enabled: bool = False
    cv_detection_backend: str = "yolo"       # "yolo" | "openai_vision"
    cv_detection_model: str = "yolo11n.pt"   # YOLO model file
    cv_detection_confidence: float = 0.5
    cv_detection_api_url: str = ""           # vision API base URL (openai_vision)
    cv_detection_api_key: str = ""           # vision API key
    cv_detection_api_model: str = ""         # vision model name
    cv_llm_interactions_enabled: bool = True
    cv_ignore_labels: str = "person,hand,hands,head,face,arm,finger"
    cv_detection_status: str = ""            # transient: error / info message
    # ── CV training mode ─────────────────────────────────────────────────────
    cv_training_mode: bool = False
    cv_capture_requested: bool = False       # transient
    cv_capture_status: str = ""              # transient: "Identifying..." / "Found: red car"
    cv_capture_label: str = ""              # transient: pending label before save
    cv_capture_description: str = ""        # transient: pending description before save
    cv_capture_thumbnail_b64: str = ""      # transient
    cv_capture_save_requested: bool = False  # transient
    cv_custom_objects: list = None           # transient: loaded from store for display  # type: ignore[assignment]
    # ── transient flags ───────────────────────────────────────────────────────
    display_changed: bool = False
    depth_range_changed: bool = False
    filter_changed: bool = False
    ai_changed: bool = False
    vision_changed: bool = False
    cv_detection_changed: bool = False
    camera_scan_requested: bool = False
    camera_scan_status: str = "idle"
    camera_scan_message: str = ""
    camera_test_requested: bool = False
    camera_test_status: str = "idle"
    camera_test_message: str = ""
    calibration_message: str = ""
    llm_test_requested: bool = False
    llm_test_status: str = "idle"
    llm_test_message: str = ""
    save_requested: bool = False

    @classmethod
    def load(cls) -> "Config":
        config = cls()
        if not CONFIG_PATH.exists():
            return config

        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return config

        transient = {
            "display_changed",
            "depth_range_changed",
            "filter_changed",
            "ai_changed",
            "vision_changed",
            "cv_detection_changed",
            "cv_detection_status",
            "cv_capture_requested",
            "cv_capture_status",
            "cv_capture_label",
            "cv_capture_description",
            "cv_capture_thumbnail_b64",
            "cv_capture_save_requested",
            "cv_custom_objects",
            "available_cameras",
            "camera_scan_requested",
            "camera_scan_status",
            "camera_scan_message",
            "camera_test_requested",
            "camera_test_status",
            "camera_test_message",
            "calibration_message",
            "llm_test_requested",
            "llm_test_status",
            "llm_test_message",
            "save_requested",
        }
        for key, value in raw.items():
            if hasattr(config, key) and key not in transient:
                setattr(config, key, value)

        if config.colour_scheme not in SCHEMES:
            config.colour_scheme = "terrain"
        if config.guide_verbosity not in VERBOSITIES:
            config.guide_verbosity = "normal"
        if config.llm_backend not in {backend for backend, _ in LLM_BACKENDS}:
            config.llm_backend = "template"
        if config.cv_detection_backend not in {b for b, _ in CV_BACKENDS}:
            config.cv_detection_backend = "yolo"

        config.min_depth_mm = int(config.min_depth_mm)
        config.max_depth_mm = max(int(config.max_depth_mm), config.min_depth_mm + 50)
        config.temporal_alpha = float(config.temporal_alpha)
        config.change_threshold_mm = int(config.change_threshold_mm)
        config.persistence_frames = int(config.persistence_frames)
        config.foreground_reject_mm = int(config.foreground_reject_mm)
        config.display_index = max(0, int(config.display_index))
        config.shark_count = max(0, min(12, int(config.shark_count)))
        config.dinosaur_count = max(0, min(12, int(config.dinosaur_count)))
        config.windowed_w = max(320, int(config.windowed_w))
        config.windowed_h = max(240, int(config.windowed_h))
        config.llm_timeout_seconds = max(0.5, float(config.llm_timeout_seconds))
        config.guide_update_seconds = max(2.0, float(config.guide_update_seconds))
        config.llm_base_url = str(config.llm_base_url).strip() or "http://127.0.0.1:1234/v1"
        config.llm_model = str(config.llm_model).strip()
        config.camera_index = max(0, int(config.camera_index))
        config.cv_detection_confidence = max(0.1, min(1.0, float(config.cv_detection_confidence)))
        config.cv_detection_model = str(config.cv_detection_model).strip() or "yolo11n.pt"
        config.cv_detection_api_url = str(config.cv_detection_api_url).strip()
        config.cv_detection_api_key = str(config.cv_detection_api_key).strip()
        config.cv_detection_api_model = str(config.cv_detection_api_model).strip()
        config.cv_ignore_labels = str(config.cv_ignore_labels).strip()
        if config.available_cameras is None:
            config.available_cameras = []
        if not config.vision_calibration_points or len(config.vision_calibration_points) != 4:
            config.vision_calibration_points = []
        return config

    def save(self) -> None:
        payload = {
            "min_depth_mm": self.min_depth_mm,
            "max_depth_mm": self.max_depth_mm,
            "temporal_alpha": self.temporal_alpha,
            "change_threshold_mm": self.change_threshold_mm,
            "persistence_frames": self.persistence_frames,
            "foreground_reject_mm": self.foreground_reject_mm,
            "colour_scheme": self.colour_scheme,
            "show_contours": self.show_contours,
            "show_creatures": self.show_creatures,
            "shark_count": self.shark_count,
            "dinosaur_count": self.dinosaur_count,
            "display_index": self.display_index,
            "fullscreen": self.fullscreen,
            "windowed_w": self.windowed_w,
            "windowed_h": self.windowed_h,
            "ai_enabled": self.ai_enabled,
            "guide_enabled": self.guide_enabled,
            "guide_challenges_enabled": self.guide_challenges_enabled,
            "llm_enabled": self.llm_enabled,
            "llm_backend": self.llm_backend,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "guide_verbosity": self.guide_verbosity,
            "guide_update_seconds": self.guide_update_seconds,
            "vision_enabled": self.vision_enabled,
            "object_reactions_enabled": self.object_reactions_enabled,
            "camera_index": self.camera_index,
            "vision_debug_enabled": self.vision_debug_enabled,
            "vision_calibration_points": self.vision_calibration_points or [],
            "cv_detection_enabled": self.cv_detection_enabled,
            "cv_detection_backend": self.cv_detection_backend,
            "cv_detection_model": self.cv_detection_model,
            "cv_detection_confidence": self.cv_detection_confidence,
            "cv_detection_api_url": self.cv_detection_api_url,
            "cv_detection_api_key": self.cv_detection_api_key,
            "cv_detection_api_model": self.cv_detection_api_model,
            "cv_llm_interactions_enabled": self.cv_llm_interactions_enabled,
            "cv_ignore_labels": self.cv_ignore_labels,
            "cv_training_mode": self.cv_training_mode,
        }
        CONFIG_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def request_save(self) -> None:
        self.save_requested = True

    def request_ai_refresh(self) -> None:
        self.ai_changed = True
        self.llm_test_status = "idle"
        self.llm_test_message = ""
        self.request_save()

    def request_cv_detection_refresh(self) -> None:
        self.cv_detection_changed = True
        self.cv_detection_status = ""
        self.request_save()

    def request_vision_refresh(self) -> None:
        self.vision_changed = True
        self.camera_scan_requested = self.vision_enabled
        self.camera_scan_status = "scanning" if self.vision_enabled else "idle"
        self.camera_scan_message = "Scanning webcams..." if self.vision_enabled else ""
        self.camera_test_status = "idle"
        self.camera_test_message = ""
        self.request_save()

    def request_camera_scan(self) -> None:
        self.camera_scan_requested = True
        self.camera_scan_status = "scanning"
        self.camera_scan_message = "Scanning webcams..."

    def request_camera_test(self) -> None:
        self.camera_test_requested = True
        self.camera_test_status = "running"
        self.camera_test_message = "Testing camera..."

    def request_llm_test(self) -> None:
        self.llm_test_requested = True
        self.llm_test_status = "running"
        self.llm_test_message = "Testing LLM connection..."


class Sidebar:
    """Animated right-hand settings panel."""

    def __init__(self) -> None:
        self._open = False
        self._offset = SIDEBAR_W
        self._drag: str | None = None
        self._scroll = 0
        self._content_h = 0
        self._layout: dict[str, pygame.Rect] = {}
        self._text_input: str | None = None
        self._font_sm: pygame.font.Font | None = None
        self._font_md: pygame.font.Font | None = None
        self._text_cache: dict[tuple[str, str, tuple[int, ...]], pygame.Surface] = {}
        self._bg_cache: dict[int, pygame.Surface] = {}

    def toggle(self) -> None:
        self._open = not self._open

    @property
    def visible(self) -> bool:
        return self._open or self._offset < SIDEBAR_W

    def handle_event(self, event: pygame.event.Event, config: Config) -> bool:
        if not self.visible:
            return False

        if event.type in (
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEMOTION,
            pygame.MOUSEWHEEL,
        ):
            mx = pygame.mouse.get_pos()[0]
            sw = pygame.display.get_surface().get_width()
            sidebar_x = sw - SIDEBAR_W + self._offset
            if mx < sidebar_x:
                if event.type == pygame.MOUSEMOTION and self._drag:
                    return self._apply_drag(config, mx)
                if event.type == pygame.MOUSEBUTTONUP and self._drag:
                    self._drag = None
                    return True
                return False

        if event.type == pygame.KEYDOWN and self._text_input:
            return self._handle_text_key(event, config)

        layout = self._layout
        mx, my = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._text_input = None

            for key in SLIDER_SPECS:
                if layout.get(f"{key}_track", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    self._drag = key
                    return self._apply_drag(config, mx)

            for scheme in SCHEMES:
                if layout.get(f"scheme_{scheme}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    if config.colour_scheme != scheme:
                        config.colour_scheme = scheme
                        config.request_save()
                    return True

            for i in range(len(pygame.display.get_desktop_sizes())):
                if layout.get(f"disp_{i}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    if config.display_index != i:
                        config.display_index = i
                        config.display_changed = True
                        config.request_save()
                    return True

            if layout.get("fullscreen", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.fullscreen = not config.fullscreen
                config.display_changed = True
                config.request_save()
                return True

            if layout.get("contours", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.show_contours = not config.show_contours
                config.request_save()
                return True

            if layout.get("creatures", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.show_creatures = not config.show_creatures
                config.request_save()
                return True

            if layout.get("ai_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.ai_enabled = not config.ai_enabled
                config.request_ai_refresh()
                return True

            if layout.get("vision_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.vision_enabled = not config.vision_enabled
                if not config.vision_enabled:
                    config.vision_calibrating = False
                    config.calibration_message = ""
                config.request_vision_refresh()
                return True

            if config.ai_enabled:
                if layout.get("guide_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.guide_enabled = not config.guide_enabled
                    config.request_ai_refresh()
                    return True

                if layout.get("guide_challenges_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.guide_challenges_enabled = not config.guide_challenges_enabled
                    config.request_ai_refresh()
                    return True

                if layout.get("llm_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.llm_enabled = not config.llm_enabled
                    config.request_ai_refresh()
                    return True

                for backend, _label in LLM_BACKENDS:
                    if layout.get(f"backend_{backend}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.llm_backend != backend:
                            config.llm_backend = backend
                            config.request_ai_refresh()
                        return True

                for verbosity in VERBOSITIES:
                    if layout.get(f"verbosity_{verbosity}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.guide_verbosity != verbosity:
                            config.guide_verbosity = verbosity
                            config.request_ai_refresh()
                        return True

                for field in ("llm_model", "llm_base_url"):
                    if layout.get(field, pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        self._text_input = field
                        return True

                if layout.get("llm_test", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.request_llm_test()
                    return True

            if config.vision_enabled:
                if layout.get("object_reactions_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.object_reactions_enabled = not config.object_reactions_enabled
                    config.request_vision_refresh()
                    return True

                if layout.get("vision_debug_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.vision_debug_enabled = not config.vision_debug_enabled
                    config.request_save()
                    return True

                if layout.get("vision_calibrating", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.vision_calibrating = not config.vision_calibrating
                    if config.vision_calibrating:
                        config.calibration_message = (
                            "Show corner tags 100, 101, 102, and 103 to calibrate the sandbox."
                        )
                    config.request_save()
                    return True

                if layout.get("camera_test", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.request_camera_test()
                    return True

                if layout.get("camera_scan", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.request_camera_scan()
                    return True

                if layout.get("camera_minus", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.camera_index = max(0, config.camera_index - 1)
                    config.request_vision_refresh()
                    return True

                if layout.get("camera_plus", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.camera_index += 1
                    config.request_vision_refresh()
                    return True

                for camera in config.available_cameras or []:
                    camera_index = int(camera["index"])
                    if layout.get(f"camera_pick_{camera_index}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.camera_index != camera_index:
                            config.camera_index = camera_index
                            config.request_vision_refresh()
                        return True

            if layout.get("cv_detection_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.cv_detection_enabled = not config.cv_detection_enabled
                config.request_cv_detection_refresh()
                return True

            if config.cv_detection_enabled:
                for backend, _label in CV_BACKENDS:
                    if layout.get(f"cv_backend_{backend}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.cv_detection_backend != backend:
                            config.cv_detection_backend = backend
                            config.request_cv_detection_refresh()
                        return True

                if layout.get("cv_llm_interactions_enabled", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.cv_llm_interactions_enabled = not config.cv_llm_interactions_enabled
                    config.request_cv_detection_refresh()
                    return True

                for field in ("cv_detection_model", "cv_detection_api_url", "cv_detection_api_key", "cv_detection_api_model", "cv_ignore_labels"):
                    if layout.get(field, pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        self._text_input = field
                        return True

                if layout.get("cv_training_mode", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    config.cv_training_mode = not config.cv_training_mode
                    config.cv_capture_status = ""
                    config.cv_capture_label = ""
                    config.request_save()
                    return True

                if config.cv_training_mode:
                    if layout.get("cv_capture", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.cv_capture_status != "Identifying...":
                            config.cv_capture_requested = True
                            config.cv_capture_status = "Identifying..."
                            config.cv_capture_label = ""
                        return True

                    if layout.get("cv_capture_save", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        if config.cv_capture_label:
                            config.cv_capture_save_requested = True
                        return True

                    if layout.get("cv_capture_label", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                        self._text_input = "cv_capture_label"
                        return True

                    for i, obj in enumerate(config.cv_custom_objects or []):
                        if layout.get(f"cv_obj_del_{i}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                            objects = list(config.cv_custom_objects or [])
                            objects.pop(i)
                            config.cv_custom_objects = objects
                            from cv_object_store import save_objects
                            save_objects(objects)
                            config.cv_detection_changed = True
                            config.request_save()
                            return True

            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._drag:
                self._drag = None
                return True

        if event.type == pygame.MOUSEMOTION and self._drag:
            return self._apply_drag(config, mx)

        if event.type == pygame.MOUSEWHEEL:
            self._scroll_by(-event.y * 24)
            return True

        return False

    def update(self, dt: float) -> None:
        target = 0 if self._open else SIDEBAR_W
        step = ANIM_SPEED * max(0.0, dt)
        if self._offset < target:
            self._offset = min(self._offset + step, target)
        elif self._offset > target:
            self._offset = max(self._offset - step, target)

    def draw(self, surface: pygame.Surface, config: Config) -> None:
        if self._offset >= SIDEBAR_W:
            return

        if self._font_sm is None:
            self._font_sm = pygame.font.SysFont("segoeui", 13)
            self._font_md = pygame.font.SysFont("segoeui", 15, bold=True)

        sw, sh = surface.get_size()
        sx = sw - SIDEBAR_W + self._offset
        self._scroll = min(self._scroll, self._max_scroll(sh))

        surface.set_clip(pygame.Rect(sx, 0, SIDEBAR_W, sh))

        bg = self._bg_cache.get(sh)
        if bg is None:
            bg = pygame.Surface((SIDEBAR_W, sh), pygame.SRCALPHA)
            bg.fill((*_BG, 225))
            self._bg_cache[sh] = bg
        surface.blit(bg, (sx, 0))
        pygame.draw.line(surface, _DIVIDER, (sx, 0), (sx, sh))

        layout: dict[str, pygame.Rect] = {}
        ix = sx + PAD
        iw = SIDEBAR_W - PAD * 2
        y = PAD - self._scroll

        self._text(surface, "Settings", ix, y, self._font_md, _TEXT)
        hint = self._font_sm.render("Tab", True, _SUBTEXT)
        surface.blit(hint, (sx + SIDEBAR_W - PAD - hint.get_width(), y + 2))
        y += 28

        y = self._section_header(surface, "Depth Range", sx, y)
        for key in ("min", "max"):
            y = self._draw_slider(surface, config, layout, key, ix, iw, y)

        y = self._section_header(surface, "Smoothing", sx, y + 4)
        for key in ("alpha", "threshold", "persist", "reject"):
            y = self._draw_slider(surface, config, layout, key, ix, iw, y)

        y = self._section_header(surface, "Colour Scheme", sx, y + 4)
        btn_w = (iw - 6) // 2
        for i, scheme in enumerate(SCHEMES):
            bx = ix + (i % 2) * (btn_w + 6)
            by = y + (i // 2) * (ROW + 2)
            rect = pygame.Rect(bx, by, btn_w, ROW - 4)
            layout[f"scheme_{scheme}"] = rect
            self._button(surface, rect, scheme.capitalize(), active=config.colour_scheme == scheme)
        y += ((len(SCHEMES) - 1) // 2 + 1) * (ROW + 2) + 6

        y = self._section_header(surface, "Display", sx, y)
        displays = pygame.display.get_desktop_sizes()
        for i, (dw, dh) in enumerate(displays):
            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout[f"disp_{i}"] = rect
            active = config.display_index == i and config.fullscreen
            self._button(surface, rect, f"Display {i + 1}  {dw}x{dh}", active=active)
            y += ROW

        y += 4
        rect = pygame.Rect(ix, y, iw, ROW)
        layout["fullscreen"] = rect
        fs_label = "Borderless ON" if config.fullscreen else "Borderless"
        self._button(surface, rect, fs_label, active=config.fullscreen)
        y += ROW + 10

        y = self._section_header(surface, "View", sx, y)
        rect = pygame.Rect(ix, y, iw, ROW - 4)
        layout["contours"] = rect
        self._button(surface, rect, "Contour Lines", active=config.show_contours)
        y += ROW

        rect = pygame.Rect(ix, y, iw, ROW - 4)
        layout["creatures"] = rect
        creatures_label = "Creatures ON" if config.show_creatures else "Creatures OFF"
        self._button(surface, rect, creatures_label, active=config.show_creatures)
        y += ROW + 8
        y = self._section_header(surface, "Creatures", sx, y)
        for key in ("sharks", "dinosaurs"):
            y = self._draw_slider(surface, config, layout, key, ix, iw, y)

        y = self._section_header(surface, "AI Guide", sx, y)
        rect = pygame.Rect(ix, y, iw, ROW - 4)
        layout["ai_enabled"] = rect
        ai_label = "AI Features ON" if config.ai_enabled else "AI Features OFF"
        self._button(surface, rect, ai_label, active=config.ai_enabled)
        y += ROW

        if config.ai_enabled:
            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["guide_enabled"] = rect
            self._button(surface, rect, "Guide Overlay", active=config.guide_enabled)
            y += ROW

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["guide_challenges_enabled"] = rect
            self._button(surface, rect, "Challenges", active=config.guide_challenges_enabled)
            y += ROW

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["llm_enabled"] = rect
            llm_label = "LLM Wording ON" if config.llm_enabled else "LLM Wording OFF"
            self._button(surface, rect, llm_label, active=config.llm_enabled)
            y += ROW + 2

            y = self._labelled_buttons(
                surface,
                layout,
                prefix="verbosity",
                label="Verbosity",
                options=[(item, item.capitalize()) for item in VERBOSITIES],
                active_key=config.guide_verbosity,
                ix=ix,
                iw=iw,
                y=y,
                columns=3,
            )

            if config.llm_enabled:
                y = self._labelled_buttons(
                    surface,
                    layout,
                    prefix="backend",
                    label="LLM Backend",
                    options=LLM_BACKENDS,
                    active_key=config.llm_backend,
                    ix=ix,
                    iw=iw,
                    y=y + 2,
                    columns=3,
                )

                if config.llm_backend != "template":
                    y = self._draw_text_input(
                        surface,
                        layout,
                        key="llm_model",
                        label="LLM Model",
                        value=config.llm_model or "(provider default)",
                        ix=ix,
                        iw=iw,
                        y=y + 2,
                    )
                    y = self._draw_text_input(
                        surface,
                        layout,
                        key="llm_base_url",
                        label="Endpoint URL",
                        value=config.llm_base_url,
                        ix=ix,
                        iw=iw,
                        y=y + 2,
                    )
                    rect = pygame.Rect(ix, y + 6, iw, ROW - 4)
                    layout["llm_test"] = rect
                    label = "Testing..." if config.llm_test_status == "running" else "Test LLM Connection"
                    self._button(
                        surface,
                        rect,
                        label,
                        active=config.llm_test_status == "running",
                    )
                    y += ROW + 2
                    if config.llm_test_message:
                        y = self._draw_status_text(
                            surface,
                            config.llm_test_message,
                            ix,
                            iw,
                            y,
                            ok=config.llm_test_status == "ok",
                            bad=config.llm_test_status == "error",
                        )

        y = self._section_header(surface, "Vision", sx, y + 6)
        rect = pygame.Rect(ix, y, iw, ROW - 4)
        layout["vision_enabled"] = rect
        vision_label = "Vision ON" if config.vision_enabled else "Vision OFF"
        self._button(surface, rect, vision_label, active=config.vision_enabled)
        y += ROW

        if config.vision_enabled:
            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["object_reactions_enabled"] = rect
            reactions_label = "Object Reactions ON" if config.object_reactions_enabled else "Object Reactions OFF"
            self._button(surface, rect, reactions_label, active=config.object_reactions_enabled)
            y += ROW

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["vision_debug_enabled"] = rect
            debug_label = "Vision Debug ON" if config.vision_debug_enabled else "Vision Debug OFF"
            self._button(surface, rect, debug_label, active=config.vision_debug_enabled)
            y += ROW + 2

            self._label(surface, "Camera Index", ix, y)
            y += 17
            btn_w = (iw - 12) // 3
            left = pygame.Rect(ix, y, btn_w, ROW - 4)
            mid = pygame.Rect(ix + btn_w + 6, y, btn_w, ROW - 4)
            right = pygame.Rect(ix + (btn_w + 6) * 2, y, btn_w, ROW - 4)
            layout["camera_minus"] = left
            layout["camera_plus"] = right
            self._button(surface, left, "-", active=False)
            self._button(surface, mid, f"{config.camera_index}", active=True)
            self._button(surface, right, "+", active=False)
            y += ROW

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["camera_scan"] = rect
            scan_label = "Scanning Cameras..." if config.camera_scan_status == "scanning" else "Scan Cameras"
            self._button(surface, rect, scan_label, active=config.camera_scan_status == "scanning")
            y += ROW
            if config.camera_scan_message:
                y = self._draw_status_text(
                    surface,
                    config.camera_scan_message,
                    ix,
                    iw,
                    y,
                    ok=config.camera_scan_status == "ok",
                    bad=config.camera_scan_status == "error",
                )

            cameras = config.available_cameras or []
            if cameras:
                y = self._section_header(surface, "Detected Cameras", sx, y + 2)
                for camera in cameras:
                    camera_index = int(camera["index"])
                    rect = pygame.Rect(ix, y, iw, ROW - 4)
                    layout[f"camera_pick_{camera_index}"] = rect
                    label = str(camera.get("label", f"Camera {camera_index}"))
                    self._button(surface, rect, label, active=config.camera_index == camera_index)
                    y += ROW

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["camera_test"] = rect
            camera_test_label = "Testing Camera..." if config.camera_test_status == "running" else "Test Camera"
            self._button(surface, rect, camera_test_label, active=config.camera_test_status == "running")
            y += ROW
            if config.camera_test_message:
                y = self._draw_status_text(
                    surface,
                    config.camera_test_message,
                    ix,
                    iw,
                    y,
                    ok=config.camera_test_status == "ok",
                    bad=config.camera_test_status == "error",
                )

            rect = pygame.Rect(ix, y + 2, iw, ROW - 4)
            layout["vision_calibrating"] = rect
            calibrate_label = "Calibration Mode ON" if config.vision_calibrating else "Calibration Mode"
            self._button(surface, rect, calibrate_label, active=config.vision_calibrating)
            y += ROW + 2
            if config.calibration_message:
                y = self._draw_status_text(
                    surface,
                    config.calibration_message,
                    ix,
                    iw,
                    y,
                    ok=bool(config.vision_calibration_points),
                    bad=False,
                )

        y = self._section_header(surface, "Object Detection", sx, y + 6)
        rect = pygame.Rect(ix, y, iw, ROW - 4)
        layout["cv_detection_enabled"] = rect
        cv_label = "CV Detection ON" if config.cv_detection_enabled else "CV Detection OFF"
        self._button(surface, rect, cv_label, active=config.cv_detection_enabled)
        y += ROW

        if config.cv_detection_enabled:
            y = self._labelled_buttons(
                surface,
                layout,
                prefix="cv_backend",
                label="Backend",
                options=CV_BACKENDS,
                active_key=config.cv_detection_backend,
                ix=ix,
                iw=iw,
                y=y,
                columns=2,
            )

            y = self._draw_slider(surface, config, layout, "cv_confidence", ix, iw, y)

            if config.cv_detection_backend == "yolo":
                y = self._draw_text_input(
                    surface,
                    layout,
                    key="cv_detection_model",
                    label="YOLO Model",
                    value=config.cv_detection_model,
                    ix=ix,
                    iw=iw,
                    y=y,
                )
            else:  # openai_vision
                y = self._draw_text_input(
                    surface,
                    layout,
                    key="cv_detection_api_url",
                    label="Vision API URL",
                    value=config.cv_detection_api_url or "(uses LLM URL)",
                    ix=ix,
                    iw=iw,
                    y=y,
                )
                y = self._draw_text_input(
                    surface,
                    layout,
                    key="cv_detection_api_model",
                    label="Vision Model",
                    value=config.cv_detection_api_model or "gpt-4o",
                    ix=ix,
                    iw=iw,
                    y=y,
                )
                y = self._draw_text_input(
                    surface,
                    layout,
                    key="cv_detection_api_key",
                    label="API Key",
                    value="*" * min(len(config.cv_detection_api_key), 12) if config.cv_detection_api_key else "(none)",
                    ix=ix,
                    iw=iw,
                    y=y,
                )

            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["cv_llm_interactions_enabled"] = rect
            llm_int_label = "LLM Interactions ON" if config.cv_llm_interactions_enabled else "LLM Interactions OFF"
            self._button(surface, rect, llm_int_label, active=config.cv_llm_interactions_enabled)
            y += ROW

            y = self._draw_text_input(
                surface,
                layout,
                key="cv_ignore_labels",
                label="Ignore Labels",
                value=config.cv_ignore_labels or "(none)",
                ix=ix,
                iw=iw,
                y=y,
            )

            if config.cv_detection_status:
                y = self._draw_status_text(
                    surface,
                    config.cv_detection_status,
                    ix,
                    iw,
                    y,
                    ok=False,
                    bad=config.cv_detection_status.startswith("Error"),
                )

            y += 4
            rect = pygame.Rect(ix, y, iw, ROW - 4)
            layout["cv_training_mode"] = rect
            train_label = "Training Mode ON" if config.cv_training_mode else "Training Mode"
            self._button(surface, rect, train_label, active=config.cv_training_mode)
            y += ROW

            if config.cv_training_mode:
                rect = pygame.Rect(ix, y, iw, ROW - 4)
                layout["cv_capture"] = rect
                capturing = config.cv_capture_status == "Identifying..."
                self._button(surface, rect, "Identifying..." if capturing else "Capture Object", active=capturing)
                y += ROW

                if config.cv_capture_status and config.cv_capture_status != "Identifying...":
                    y = self._draw_status_text(surface, config.cv_capture_status, ix, iw, y, ok=bool(config.cv_capture_label))

                if config.cv_capture_label:
                    y = self._draw_text_input(
                        surface, layout,
                        key="cv_capture_label",
                        label="Label (rename if needed)",
                        value=config.cv_capture_label,
                        ix=ix, iw=iw, y=y,
                    )
                    rect = pygame.Rect(ix, y, iw, ROW - 4)
                    layout["cv_capture_save"] = rect
                    self._button(surface, rect, "Save Object", active=False)
                    y += ROW

                objects = config.cv_custom_objects or []
                if objects:
                    self._label(surface, f"Trained objects ({len(objects)})", ix, y)
                    y += 17
                    del_w = 28
                    for i, obj in enumerate(objects):
                        name_rect = pygame.Rect(ix, y, iw - del_w - 4, ROW - 4)
                        del_rect = pygame.Rect(ix + iw - del_w, y, del_w, ROW - 4)
                        layout[f"cv_obj_del_{i}"] = del_rect
                        self._button(surface, name_rect, obj.label[:28], active=True)
                        self._button(surface, del_rect, "✕", active=False)
                        y += ROW

        self._content_h = y + ROW + PAD + self._scroll
        self._draw_scrollbar(surface, sx, sh)
        self._layout = layout
        surface.set_clip(None)

    def _handle_text_key(self, event: pygame.event.Event, config: Config) -> bool:
        field = self._text_input
        if field is None:
            return False
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_TAB):
            self._text_input = None
            return True

        value = getattr(config, field)
        _refresh = config.request_cv_detection_refresh if field.startswith("cv_") else config.request_ai_refresh
        if event.key == pygame.K_BACKSPACE:
            setattr(config, field, value[:-1])
            _refresh()
            return True
        if event.key == pygame.K_DELETE:
            setattr(config, field, "")
            _refresh()
            return True

        if event.unicode and event.unicode.isprintable():
            if len(value) < 140:
                setattr(config, field, value + event.unicode)
                if field.startswith("cv_"):
                    config.request_cv_detection_refresh()
                else:
                    config.request_ai_refresh()
            return True
        return False

    def _apply_drag(self, config: Config, mx: int) -> bool:
        key = self._drag
        if not key:
            return False
        track = self._layout.get(f"{key}_track")
        if not track:
            return False
        spec = SLIDER_SPECS[key]
        t = max(0.0, min(1.0, (mx - track.left) / track.width))
        value = spec["lo"] + t * (spec["hi"] - spec["lo"])
        value = int(round(value)) if spec["kind"] == "int" else round(float(value), 2)

        if key == "min":
            config.min_depth_mm = int(value)
            config.min_depth_mm = min(config.min_depth_mm, config.max_depth_mm - 50)
            config.depth_range_changed = True
            config.request_save()
        elif key == "max":
            config.max_depth_mm = int(value)
            config.max_depth_mm = max(config.max_depth_mm, config.min_depth_mm + 50)
            config.depth_range_changed = True
            config.request_save()
        elif key == "alpha":
            config.temporal_alpha = float(value)
            config.filter_changed = True
            config.request_save()
        elif key == "threshold":
            config.change_threshold_mm = int(value)
            config.filter_changed = True
            config.request_save()
        elif key == "persist":
            config.persistence_frames = int(value)
            config.filter_changed = True
            config.request_save()
        elif key == "reject":
            config.foreground_reject_mm = int(value)
            config.filter_changed = True
            config.request_save()
        elif key == "sharks":
            config.shark_count = int(value)
            config.request_save()
        elif key == "dinosaurs":
            config.dinosaur_count = int(value)
            config.request_save()
        elif key == "cv_confidence":
            config.cv_detection_confidence = round(float(value), 2)
            config.cv_detection_changed = True
            config.request_save()
        return True

    def _draw_slider(
        self,
        surface: pygame.Surface,
        config: Config,
        layout: dict[str, pygame.Rect],
        key: str,
        ix: int,
        iw: int,
        y: int,
    ) -> int:
        spec = SLIDER_SPECS[key]
        value = self._slider_value(config, key)
        self._label(surface, spec["label"], ix, y)
        self._label(surface, self._slider_value_text(key, value), ix + iw, y, align_right=True)
        y += 17
        track_visual = pygame.Rect(ix, y + 6, iw, 6)
        layout[f"{key}_track"] = pygame.Rect(ix, y, iw, 20)
        self._slider(surface, track_visual, float(value), float(spec["lo"]), float(spec["hi"]))
        return y + ROW

    def _labelled_buttons(
        self,
        surface: pygame.Surface,
        layout: dict[str, pygame.Rect],
        *,
        prefix: str,
        label: str,
        options: Iterable[tuple[str, str]],
        active_key: str,
        ix: int,
        iw: int,
        y: int,
        columns: int,
    ) -> int:
        self._label(surface, label, ix, y)
        y += 17
        items = list(options)
        gap = 6
        btn_w = (iw - gap * (columns - 1)) // columns
        for index, (key, text) in enumerate(items):
            bx = ix + (index % columns) * (btn_w + gap)
            by = y + (index // columns) * (ROW + 2)
            rect = pygame.Rect(bx, by, btn_w, ROW - 4)
            layout[f"{prefix}_{key}"] = rect
            self._button(surface, rect, text, active=key == active_key)
        rows = ((len(items) - 1) // columns) + 1
        return y + rows * (ROW + 2)

    def _draw_text_input(
        self,
        surface: pygame.Surface,
        layout: dict[str, pygame.Rect],
        *,
        key: str,
        label: str,
        value: str,
        ix: int,
        iw: int,
        y: int,
    ) -> int:
        self._label(surface, label, ix, y)
        y += 17
        rect = pygame.Rect(ix, y, iw, ROW)
        layout[key] = rect
        active = self._text_input == key
        colour = _INPUT_ACT if active else _INPUT_BG
        pygame.draw.rect(surface, colour, rect, border_radius=4)
        pygame.draw.rect(surface, _CARD_BORDER if active else _DIVIDER, rect, width=1, border_radius=4)
        text = value
        max_chars = max(8, (iw - 18) // 7)
        if len(text) > max_chars:
            text = "..." + text[-(max_chars - 3):]
        txt = self._font_sm.render(text, True, _TEXT)
        surface.blit(txt, (rect.x + 8, rect.y + 8))
        return y + ROW + 4

    def _draw_status_text(
        self,
        surface: pygame.Surface,
        text: str,
        ix: int,
        iw: int,
        y: int,
        *,
        ok: bool = False,
        bad: bool = False,
    ) -> int:
        colour = _SUBTEXT
        if ok:
            colour = _SUCCESS
        elif bad:
            colour = (205, 120, 120)
        for line in _wrap_text(self._font_sm, text, iw):
            surface.blit(self._font_sm.render(line, True, colour), (ix, y))
            y += 16
        return y + 4

    def _slider_value(self, config: Config, key: str) -> float | int:
        return {
            "min": config.min_depth_mm,
            "max": config.max_depth_mm,
            "alpha": config.temporal_alpha,
            "threshold": config.change_threshold_mm,
            "persist": config.persistence_frames,
            "reject": config.foreground_reject_mm,
            "sharks": config.shark_count,
            "dinosaurs": config.dinosaur_count,
            "cv_confidence": config.cv_detection_confidence,
        }[key]

    def _slider_value_text(self, key: str, value: float | int) -> str:
        return {
            "min": f"{int(value)} mm",
            "max": f"{int(value)} mm",
            "alpha": f"{float(value):.2f}",
            "threshold": f"{int(value)} mm",
            "persist": f"{int(value)} fr",
            "reject": f"{int(value)} mm",
            "sharks": str(int(value)),
            "dinosaurs": str(int(value)),
            "cv_confidence": f"{float(value):.2f}",
        }[key]

    def _section_header(self, surface, label, sx, y):
        pygame.draw.line(surface, _DIVIDER, (sx, y), (sx + SIDEBAR_W, y))
        lbl = self._font_sm.render(label.upper(), True, _SUBTEXT)
        surface.blit(lbl, (sx + PAD, y + 5))
        return y + 22

    def _max_scroll(self, viewport_h: int) -> int:
        return max(0, self._content_h - viewport_h)

    def _scroll_by(self, delta: int) -> None:
        viewport = pygame.display.get_surface()
        if viewport is None:
            return
        self._scroll = max(0, min(self._scroll + delta, self._max_scroll(viewport.get_height())))

    def _draw_scrollbar(self, surface: pygame.Surface, sx: int, sh: int) -> None:
        max_scroll = self._max_scroll(sh)
        if max_scroll <= 0:
            return
        track = pygame.Rect(sx + SIDEBAR_W - 8, 8, 4, sh - 16)
        thumb_h = max(28, int(track.height * sh / self._content_h))
        travel = track.height - thumb_h
        thumb_y = track.top + int(travel * (self._scroll / max_scroll))
        thumb = pygame.Rect(track.left, thumb_y, track.width, thumb_h)
        pygame.draw.rect(surface, _SCROLL_TRK, track, border_radius=2)
        pygame.draw.rect(surface, _SCROLL_BAR, thumb, border_radius=2)

    def _label(self, surface, text, x, y, align_right=False):
        lbl = self._render_cached("sm", text, _SUBTEXT)
        blit_x = (x - lbl.get_width()) if align_right else x
        surface.blit(lbl, (blit_x, y))

    def _text(self, surface, text, x, y, font, colour):
        font_key = "md" if font is self._font_md else "sm"
        surface.blit(self._render_cached(font_key, text, colour), (x, y))

    def _slider(self, surface, track: pygame.Rect, val: float, lo: float, hi: float):
        t = (val - lo) / (hi - lo)
        tx = int(track.left + t * track.width)
        pygame.draw.rect(surface, _TRACK, track, border_radius=3)
        filled = pygame.Rect(track.left, track.top, tx - track.left, track.height)
        if filled.width > 0:
            pygame.draw.rect(surface, _FILL, filled, border_radius=3)
        pygame.draw.circle(surface, _THUMB, (tx, track.centery), 7)
        pygame.draw.circle(surface, _BG, (tx, track.centery), 3)

    def _button(self, surface, rect: pygame.Rect, label: str, active: bool = False):
        mx, my = pygame.mouse.get_pos()
        hover = rect.collidepoint(mx, my)
        colour = _BTN_ACT if active else (_BTN_HOV if hover else _BTN)
        pygame.draw.rect(surface, colour, rect, border_radius=4)
        txt = self._render_cached("sm", label, _TEXT)
        surface.blit(txt, txt.get_rect(center=rect.center))

    def _render_cached(
        self,
        font_key: str,
        text: str,
        colour: tuple[int, ...],
    ) -> pygame.Surface:
        key = (font_key, text, tuple(colour))
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        font = self._font_md if font_key == "md" else self._font_sm
        assert font is not None
        rendered = font.render(text, True, colour)
        self._text_cache[key] = rendered
        return rendered


def draw_guide_overlay(
    surface: pygame.Surface,
    config: Config,
    *,
    title: str | None,
    body: str | None,
    challenge_text: str | None,
    challenge_done: bool = False,
) -> None:
    if not (config.ai_enabled and config.guide_enabled and title and body):
        return

    global _GUIDE_FONTS
    if _GUIDE_FONTS is None:
        _GUIDE_FONTS = (
            pygame.font.SysFont("segoeui", 19, bold=True),
            pygame.font.SysFont("segoeui", 16),
            pygame.font.SysFont("segoeui", 13, bold=True),
        )
    title_font, body_font, label_font = _GUIDE_FONTS

    width = min(380, max(260, surface.get_width() // 3))
    x = 18
    y = 18
    inner = width - 24
    lines = _wrap_text(body_font, body, inner)
    challenge_lines = _wrap_text(body_font, challenge_text or "", inner) if challenge_text else []
    height = 22 + 12 + len(lines) * 19 + 10
    if challenge_lines:
        height += 22 + len(challenge_lines) * 19 + 8

    rect = pygame.Rect(x, y, width, height)
    card = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(card, _CARD_BG, card.get_rect(), border_radius=12)
    pygame.draw.rect(card, _CARD_BORDER, card.get_rect(), width=2, border_radius=12)
    surface.blit(card, rect.topleft)

    cy = y + 12
    surface.blit(title_font.render(title, True, _TEXT), (x + 12, cy))
    cy += 28

    for line in lines:
        surface.blit(body_font.render(line, True, _TEXT), (x + 12, cy))
        cy += 19

    if challenge_lines:
        cy += 6
        colour = _SUCCESS if challenge_done else _ACCENT
        label = "Challenge Complete" if challenge_done else "Current Challenge"
        surface.blit(label_font.render(label, True, colour), (x + 12, cy))
        cy += 20
        for line in challenge_lines:
            surface.blit(body_font.render(line, True, _TEXT), (x + 12, cy))
            cy += 19


def _wrap_text(font: pygame.font.Font, text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.size(trial)[0] <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines
