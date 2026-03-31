"""
AR Sandbox POC

Controls
--------
Left-drag   raise terrain (simulator only)
Right-drag  lower terrain (simulator only)
Scroll      resize brush  (simulator only)
C           toggle contour lines
G           toggle creatures
R           reset terrain (simulator only)
Tab         open / close settings sidebar
Esc / Q     quit
"""

import sys

import pygame

from creatures import CreatureManager
# from depth_source import MouseSimulator
from depth_source import KinectV1Source
from renderer import Renderer
from ui import Config, Sidebar

FPS           = 60
BRUSH_MIN     = 8
BRUSH_MAX     = 180
BRUSH_DEFAULT = 50
BRUSH_DELTA   = 0.012
MAX_RENDER_W  = 960
MAX_RENDER_H  = 720


def _fallback_display_bounds() -> list[tuple[int, int, int, int]]:
    x = 0
    bounds: list[tuple[int, int, int, int]] = []
    for w, h in pygame.display.get_desktop_sizes():
        bounds.append((x, 0, w, h))
        x += w
    return bounds


def _get_display_bounds() -> list[tuple[int, int, int, int]]:
    desktops = pygame.display.get_desktop_sizes()
    if sys.platform != "win32":
        return _fallback_display_bounds()

    try:
        import ctypes
        from ctypes import wintypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", wintypes.LONG),
                ("top", wintypes.LONG),
                ("right", wintypes.LONG),
                ("bottom", wintypes.LONG),
            ]

        bounds: list[tuple[int, int, int, int]] = []
        callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HMONITOR,
            wintypes.HDC,
            ctypes.POINTER(RECT),
            wintypes.LPARAM,
        )

        def _callback(_monitor, _hdc, rect_ptr, _lparam):
            rect = rect_ptr.contents
            bounds.append(
                (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
            )
            return True

        ctypes.windll.user32.EnumDisplayMonitors(
            0, 0, callback_type(_callback), 0
        )
        if len(bounds) == len(desktops):
            return bounds
    except Exception:
        pass

    return _fallback_display_bounds()


def _move_window_to_display(
    screen: pygame.Surface,
    display_bounds: tuple[int, int, int, int],
    *,
    borderless: bool,
) -> None:
    if sys.platform != "win32":
        return

    info = pygame.display.get_wm_info()
    hwnd = info.get("window")
    if not hwnd:
        return

    import ctypes
    from ctypes import wintypes

    left, top, display_w, display_h = display_bounds
    win_w, win_h = screen.get_size()
    if not borderless:
        left += max(0, (display_w - win_w) // 2)
        top += max(0, (display_h - win_h) // 2)

    flags = 0x0004 | 0x0010 | 0x0040 | 0x0020  # no z-order, no activate, show, refresh frame
    ctypes.windll.user32.SetWindowPos(
        wintypes.HWND(hwnd),
        wintypes.HWND(0),
        int(left),
        int(top),
        int(win_w),
        int(win_h),
        flags,
    )


def _make_display(config: Config) -> tuple[pygame.Surface, int, int]:
    """Create (or recreate) the pygame display surface from current config."""
    displays = _get_display_bounds()
    idx = min(config.display_index, len(displays) - 1)
    left, top, dw, dh = displays[idx]

    if config.fullscreen:
        # Borderless windowed fullscreen is more reliable than exclusive
        # fullscreen on Windows/projector setups.
        if sys.platform == "win32":
            screen = pygame.display.set_mode((dw, dh), pygame.NOFRAME)
        else:
            screen = pygame.display.set_mode((dw, dh), pygame.NOFRAME, display=idx)
        _move_window_to_display(screen, (left, top, dw, dh), borderless=True)
        w, h = screen.get_size()
    else:
        w, h = config.windowed_w, config.windowed_h
        screen = pygame.display.set_mode((w, h))
        _move_window_to_display(screen, (left, top, dw, dh), borderless=False)
    return screen, w, h


def _compute_render_size(display_w: int, display_h: int) -> tuple[int, int]:
    scale = min(1.0, min(MAX_RENDER_W / display_w, MAX_RENDER_H / display_h))
    render_w = max(320, int(display_w * scale))
    render_h = max(240, int(display_h * scale))
    return render_w, render_h


def _screen_to_scene(
    x: int, y: int, screen_w: int, screen_h: int, scene_w: int, scene_h: int
) -> tuple[int, int]:
    sx = int(x * scene_w / max(1, screen_w))
    sy = int(y * scene_h / max(1, screen_h))
    return sx, sy


def main() -> int:
    pygame.init()
    pygame.display.set_caption("AR Sandbox")

    config  = Config.load()
    screen, screen_w, screen_h = _make_display(config)
    render_w, render_h = _compute_render_size(screen_w, screen_h)
    scene = pygame.Surface((render_w, render_h))

    # source  = MouseSimulator(render_w, render_h)
    source   = KinectV1Source(render_w, render_h,
                              min_depth_mm=config.min_depth_mm,
                              max_depth_mm=config.max_depth_mm,
                              temporal_alpha=config.temporal_alpha,
                              change_threshold_mm=config.change_threshold_mm,
                              persistence_frames=config.persistence_frames,
                              foreground_reject_mm=config.foreground_reject_mm)
    renderer = Renderer(render_w, render_h)
    creatures = CreatureManager()
    sidebar  = Sidebar()
    hud_font = pygame.font.SysFont(None, 22)
    clock    = pygame.time.Clock()

    brush_radius = BRUSH_DEFAULT

    try:
        running = True
        while running:
            dt = clock.tick(FPS) / 1000.0

            # ── events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue

                # Sidebar gets first look at every event
                consumed = sidebar.handle_event(event, config)
                if consumed:
                    continue

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_TAB:
                        sidebar.toggle()
                    elif event.key == pygame.K_r:
                        if hasattr(source, "reset"):
                            source.reset()
                    elif event.key == pygame.K_c:
                        config.show_contours = not config.show_contours
                        config.request_save()
                    elif event.key == pygame.K_g:
                        config.show_creatures = not config.show_creatures
                        config.request_save()
                elif event.type == pygame.MOUSEWHEEL:
                    brush_radius = max(BRUSH_MIN, min(BRUSH_MAX,
                                                      brush_radius + event.y * 4))

            # ── react to config changes ───────────────────────────────────────
            if config.display_changed:
                config.display_changed = False
                screen, screen_w, screen_h = _make_display(config)
                render_w, render_h = _compute_render_size(screen_w, screen_h)
                scene = pygame.Surface((render_w, render_h))
                renderer = Renderer(render_w, render_h)
                source.resize(render_w, render_h)

            if config.depth_range_changed:
                config.depth_range_changed = False
                if hasattr(source, "set_depth_range"):
                    source.set_depth_range(config.min_depth_mm, config.max_depth_mm)

            if config.filter_changed:
                config.filter_changed = False
                if hasattr(source, "set_filter_params"):
                    source.set_filter_params(
                        temporal_alpha=config.temporal_alpha,
                        change_threshold_mm=config.change_threshold_mm,
                        persistence_frames=config.persistence_frames,
                        foreground_reject_mm=config.foreground_reject_mm,
                    )

            if config.save_requested:
                config.save_requested = False
                config.save()

            # ── sculpt (simulator only) ───────────────────────────────────────
            buttons = pygame.mouse.get_pressed()
            mx, my  = pygame.mouse.get_pos()
            scene_mx, scene_my = _screen_to_scene(
                mx, my, screen.get_width(), screen.get_height(), render_w, render_h
            )
            scene_brush_radius = max(
                1,
                int(round(brush_radius * min(render_w / screen.get_width(), render_h / screen.get_height()))),
            )
            if buttons[0]:
                source.sculpt(scene_mx, scene_my, scene_brush_radius, +BRUSH_DELTA)
            elif buttons[2]:
                source.sculpt(scene_mx, scene_my, scene_brush_radius, -BRUSH_DELTA)

            # ── render ────────────────────────────────────────────────────────
            frame = source.get_frame()
            if config.show_creatures:
                creatures.update(frame, dt)
            renderer.draw(scene, frame,
                          show_contours=config.show_contours,
                          colour_scheme=config.colour_scheme)
            if config.show_creatures:
                creatures.draw(scene)

            if scene.get_size() == screen.get_size():
                screen.blit(scene, (0, 0))
            else:
                scaled = pygame.transform.scale(scene, screen.get_size())
                screen.blit(scaled, (0, 0))

            pygame.draw.circle(screen, (255, 255, 255), (mx, my), brush_radius, 1)

            creatures_state = "on" if config.show_creatures else "off"
            hud = (f"Tab settings  C contours  G creatures:{creatures_state}  R reset  "
                   f"brush {brush_radius}px  |  {config.colour_scheme}")
            label = hud_font.render(hud, True, (200, 200, 200))
            screen.blit(label, (10, screen.get_height() - 24))

            sidebar.update()
            sidebar.draw(screen, config)

            pygame.display.flip()
    finally:
        try:
            config.save()
        except Exception:
            pass
        try:
            if hasattr(source, "close"):
                source.close()
        finally:
            pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
