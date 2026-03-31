"""
Sidebar settings panel.

Toggle open / closed with Tab.  All settings live in Config, which is the
single source of truth read by main.py and renderer.py each frame.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import pathlib
import pygame

# ── palette ───────────────────────────────────────────────────────────────────
_BG         = (22,  24,  34)
_DIVIDER    = (44,  48,  66)
_ACCENT     = (60, 130, 210)
_TEXT       = (210, 215, 225)
_SUBTEXT    = (120, 126, 148)
_TRACK      = (48,  52,  72)
_FILL       = (55, 110, 185)
_THUMB      = (85, 160, 230)
_BTN        = (42,  46,  65)
_BTN_HOV    = (58,  65,  88)
_BTN_ACT    = (50, 115, 195)
_SCROLL_TRK = (34,  38,  52)
_SCROLL_BAR = (90, 110, 145)

SIDEBAR_W   = 270
PAD         = 14
ROW         = 30
ANIM_SPEED  = 28        # pixels per frame

SCHEMES = ["terrain", "heat", "greyscale", "desert"]
CONFIG_PATH = pathlib.Path(__file__).with_name("sandcam-settings.json")
SLIDER_SPECS = {
    "min": {"label": "Min", "lo": 100, "hi": 2000, "kind": "int", "section": "depth"},
    "max": {"label": "Max", "lo": 200, "hi": 3000, "kind": "int", "section": "depth"},
    "alpha": {"label": "Blend", "lo": 0.05, "hi": 1.0, "kind": "float", "section": "filter"},
    "threshold": {"label": "Threshold", "lo": 5, "hi": 120, "kind": "int", "section": "filter"},
    "persist": {"label": "Delay", "lo": 1, "hi": 12, "kind": "int", "section": "filter"},
    "reject": {"label": "Reject", "lo": 0, "hi": 400, "kind": "int", "section": "filter"},
}


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # depth calibration
    min_depth_mm:  int  = 400
    max_depth_mm:  int  = 1100
    # depth filtering
    temporal_alpha:       float = 0.35
    change_threshold_mm:  int   = 35
    persistence_frames:   int   = 4
    foreground_reject_mm: int   = 120
    # rendering
    colour_scheme: str  = "terrain"
    show_contours: bool = True
    show_creatures: bool = True
    # display
    display_index: int  = 0
    fullscreen:    bool = False
    windowed_w:    int  = 900
    windowed_h:    int  = 650
    # change flags — main.py clears these after acting on them
    display_changed:     bool = False
    depth_range_changed: bool = False
    filter_changed:      bool = False
    save_requested:      bool = False

    @classmethod
    def load(cls) -> "Config":
        """Load persisted settings from disk if available."""
        config = cls()
        if not CONFIG_PATH.exists():
            return config

        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return config

        for key, value in raw.items():
            if hasattr(config, key) and key not in {
                "display_changed",
                "depth_range_changed",
                "filter_changed",
                "save_requested",
            }:
                setattr(config, key, value)

        if config.colour_scheme not in SCHEMES:
            config.colour_scheme = "terrain"
        config.min_depth_mm = int(config.min_depth_mm)
        config.max_depth_mm = max(int(config.max_depth_mm), config.min_depth_mm + 50)
        config.temporal_alpha = float(config.temporal_alpha)
        config.change_threshold_mm = int(config.change_threshold_mm)
        config.persistence_frames = int(config.persistence_frames)
        config.foreground_reject_mm = int(config.foreground_reject_mm)
        config.display_index = max(0, int(config.display_index))
        config.windowed_w = max(320, int(config.windowed_w))
        config.windowed_h = max(240, int(config.windowed_h))
        return config

    def save(self) -> None:
        """Persist user-configurable settings to disk."""
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
            "display_index": self.display_index,
            "fullscreen": self.fullscreen,
            "windowed_w": self.windowed_w,
            "windowed_h": self.windowed_h,
        }
        CONFIG_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def request_save(self) -> None:
        self.save_requested = True


# ── sidebar ───────────────────────────────────────────────────────────────────

class Sidebar:
    """Animated right-hand settings panel."""

    def __init__(self) -> None:
        self._open   = False
        self._offset = SIDEBAR_W      # px right of final position; SIDEBAR_W = hidden
        self._drag   : str | None = None
        self._scroll = 0
        self._content_h = 0
        self._layout : dict[str, pygame.Rect] = {}
        self._font_sm: pygame.font.Font | None = None
        self._font_md: pygame.font.Font | None = None

    # ── public ────────────────────────────────────────────────────────────────

    def toggle(self) -> None:
        self._open = not self._open

    @property
    def visible(self) -> bool:
        return self._open or self._offset < SIDEBAR_W

    def handle_event(self, event: pygame.event.Event, config: Config) -> bool:
        """Process one pygame event.  Returns True if the event was consumed."""
        if not self.visible:
            return False

        # Block mouse events that land inside the sidebar from reaching main.py
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                          pygame.MOUSEMOTION, pygame.MOUSEWHEEL):
            mx = pygame.mouse.get_pos()[0]
            sw = pygame.display.get_surface().get_width()
            sidebar_x = sw - SIDEBAR_W + self._offset
            if mx < sidebar_x:
                # Outside the sidebar — only block if we're mid-drag
                if event.type == pygame.MOUSEMOTION and self._drag:
                    return self._apply_drag(config, mx)
                if event.type == pygame.MOUSEBUTTONUP and self._drag:
                    self._drag = None
                    return True
                return False

        layout = self._layout
        mx, my = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Sliders
            for key in SLIDER_SPECS:
                if layout.get(f"{key}_track", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    self._drag = key
                    return self._apply_drag(config, mx)

            # Scheme buttons
            for scheme in SCHEMES:
                if layout.get(f"scheme_{scheme}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    if config.colour_scheme != scheme:
                        config.colour_scheme = scheme
                        config.request_save()
                    return True

            # Display buttons
            for i in range(len(pygame.display.get_desktop_sizes())):
                if layout.get(f"disp_{i}", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                    if config.display_index != i:
                        config.display_index = i
                        config.display_changed = True
                        config.request_save()
                    return True

            # Fullscreen toggle
            if layout.get("fullscreen", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.fullscreen = not config.fullscreen
                config.display_changed = True
                config.request_save()
                return True

            # Contours toggle
            if layout.get("contours", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.show_contours = not config.show_contours
                config.request_save()
                return True

            # Creatures toggle
            if layout.get("creatures", pygame.Rect(0, 0, 0, 0)).collidepoint(mx, my):
                config.show_creatures = not config.show_creatures
                config.request_save()
                return True

            return True   # consumed — click was inside sidebar

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._drag:
                self._drag = None
                return True

        elif event.type == pygame.MOUSEMOTION and self._drag:
            return self._apply_drag(config, mx)

        elif event.type == pygame.MOUSEWHEEL:
            self._scroll_by(-event.y * 24)
            return True

        return False

    def update(self) -> None:
        """Advance slide animation — call once per frame."""
        target = 0 if self._open else SIDEBAR_W
        if self._offset < target:
            self._offset = min(self._offset + ANIM_SPEED, target)
        elif self._offset > target:
            self._offset = max(self._offset - ANIM_SPEED, target)

    def draw(self, surface: pygame.Surface, config: Config) -> None:
        if self._offset >= SIDEBAR_W:
            return

        if self._font_sm is None:
            self._font_sm = pygame.font.SysFont("segoeui", 13)
            self._font_md = pygame.font.SysFont("segoeui", 15, bold=True)

        sw, sh = surface.get_size()
        sx = sw - SIDEBAR_W + self._offset    # left edge of sidebar
        self._scroll = min(self._scroll, self._max_scroll(sh))

        # Clip rendering to the sidebar area so nothing bleeds out during animation
        surface.set_clip(pygame.Rect(sx, 0, SIDEBAR_W, sh))

        # Background
        bg = pygame.Surface((SIDEBAR_W, sh), pygame.SRCALPHA)
        bg.fill((*_BG, 225))
        surface.blit(bg, (sx, 0))
        pygame.draw.line(surface, _DIVIDER, (sx, 0), (sx, sh))

        layout: dict[str, pygame.Rect] = {}
        ix = sx + PAD          # inner left edge
        iw = SIDEBAR_W - PAD * 2

        y = PAD - self._scroll

        # Title
        self._text(surface, "Settings", ix, y, self._font_md, _TEXT)
        hint = self._font_sm.render("Tab", True, _SUBTEXT)
        surface.blit(hint, (sx + SIDEBAR_W - PAD - hint.get_width(), y + 2))
        y += 28

        # ── Depth Range ───────────────────────────────────────────────────────
        y = self._section_header(surface, "Depth Range", sx, y)

        for key in ("min", "max"):
            y = self._draw_slider(surface, config, layout, key, ix, iw, y)

        # ── Smoothing ────────────────────────────────────────────────────────
        y = self._section_header(surface, "Smoothing", sx, y + 4)
        for key in ("alpha", "threshold", "persist", "reject"):
            y = self._draw_slider(surface, config, layout, key, ix, iw, y)

        # ── Colour Scheme ─────────────────────────────────────────────────────
        y = self._section_header(surface, "Colour Scheme", sx, y + 4)

        btn_w = (iw - 6) // 2
        for i, scheme in enumerate(SCHEMES):
            bx = ix + (i % 2) * (btn_w + 6)
            by = y + (i // 2) * (ROW + 2)
            r = pygame.Rect(bx, by, btn_w, ROW - 4)
            layout[f"scheme_{scheme}"] = r
            self._button(surface, r, scheme.capitalize(),
                         active=(config.colour_scheme == scheme))
        y += ((len(SCHEMES) - 1) // 2 + 1) * (ROW + 2) + 6

        # ── Display ───────────────────────────────────────────────────────────
        y = self._section_header(surface, "Display", sx, y)

        displays = pygame.display.get_desktop_sizes()
        for i, (dw, dh) in enumerate(displays):
            r = pygame.Rect(ix, y, iw, ROW - 4)
            layout[f"disp_{i}"] = r
            active = (config.display_index == i) and config.fullscreen
            self._button(surface, r, f"Display {i + 1}  {dw}×{dh}", active=active)
            y += ROW

        y += 4
        r = pygame.Rect(ix, y, iw, ROW)
        layout["fullscreen"] = r
        fs_label = "Borderless ON" if config.fullscreen else "Borderless"
        self._button(surface, r, fs_label, active=config.fullscreen)
        y += ROW + 10

        # ── View ──────────────────────────────────────────────────────────────
        y = self._section_header(surface, "View", sx, y)

        r = pygame.Rect(ix, y, iw, ROW - 4)
        layout["contours"] = r
        self._button(surface, r, "Contour Lines", active=config.show_contours)
        y += ROW

        r = pygame.Rect(ix, y, iw, ROW - 4)
        layout["creatures"] = r
        creatures_label = "Creatures ON" if config.show_creatures else "Creatures OFF"
        self._button(surface, r, creatures_label, active=config.show_creatures)

        self._content_h = y + ROW + PAD + self._scroll
        self._draw_scrollbar(surface, sx, sh)
        self._layout = layout
        surface.set_clip(None)

    # ── private ───────────────────────────────────────────────────────────────

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
        if spec["kind"] == "int":
            value = int(round(value))
        else:
            value = round(float(value), 2)

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

    def _slider_value(self, config: Config, key: str) -> float | int:
        return {
            "min": config.min_depth_mm,
            "max": config.max_depth_mm,
            "alpha": config.temporal_alpha,
            "threshold": config.change_threshold_mm,
            "persist": config.persistence_frames,
            "reject": config.foreground_reject_mm,
        }[key]

    def _slider_value_text(self, key: str, value: float | int) -> str:
        return {
            "min": f"{int(value)} mm",
            "max": f"{int(value)} mm",
            "alpha": f"{float(value):.2f}",
            "threshold": f"{int(value)} mm",
            "persist": f"{int(value)} fr",
            "reject": f"{int(value)} mm",
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
        lbl = self._font_sm.render(text, True, _SUBTEXT)
        blit_x = (x - lbl.get_width()) if align_right else x
        surface.blit(lbl, (blit_x, y))

    def _text(self, surface, text, x, y, font, colour):
        surface.blit(font.render(text, True, colour), (x, y))

    def _slider(self, surface, track: pygame.Rect, val, lo, hi):
        t  = (val - lo) / (hi - lo)
        tx = int(track.left + t * track.width)
        pygame.draw.rect(surface, _TRACK, track, border_radius=3)
        filled = pygame.Rect(track.left, track.top, tx - track.left, track.height)
        if filled.width > 0:
            pygame.draw.rect(surface, _FILL, filled, border_radius=3)
        pygame.draw.circle(surface, _THUMB, (tx, track.centery), 7)
        pygame.draw.circle(surface, _BG,    (tx, track.centery), 3)

    def _button(self, surface, rect: pygame.Rect, label: str, active: bool = False):
        mx, my = pygame.mouse.get_pos()
        hover  = rect.collidepoint(mx, my)
        colour = _BTN_ACT if active else (_BTN_HOV if hover else _BTN)
        pygame.draw.rect(surface, colour, rect, border_radius=4)
        txt = self._font_sm.render(label, True, _TEXT)
        surface.blit(txt, txt.get_rect(center=rect.center))
