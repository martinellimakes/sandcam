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
import time

import pygame

from ai_guide import (
    AsyncConnectionTester,
    GuideEngine,
    ProviderConfig,
    TemplateNarrator,
    build_async_narrator,
)
from creatures import CreatureManager
# from depth_source import MouseSimulator
from depth_source import KinectV1Source
from interaction_engine import InteractionEngine
from renderer import Renderer
from ui import Config, Sidebar, draw_guide_overlay
from webcam_observer import (
    AsyncCameraDiscovery,
    AsyncCameraTester,
    CalibrationData,
    WebcamObserver,
)

FPS           = 60
BRUSH_MIN     = 8
BRUSH_MAX     = 180
BRUSH_DEFAULT = 50
BRUSH_DELTA   = 0.012
MAX_RENDER_W  = 960
MAX_RENDER_H  = 720


def _draw_startup_screen(
    screen: pygame.Surface,
    *,
    title: str,
    message: str,
    progress: float,
) -> None:
    progress = max(0.0, min(1.0, progress))
    screen.fill((14, 18, 26))

    title_font = pygame.font.SysFont("segoeui", 30, bold=True)
    body_font = pygame.font.SysFont("segoeui", 18)
    small_font = pygame.font.SysFont("segoeui", 14)

    panel_w = min(520, screen.get_width() - 80)
    panel_h = 170
    panel = pygame.Rect(
        (screen.get_width() - panel_w) // 2,
        (screen.get_height() - panel_h) // 2,
        panel_w,
        panel_h,
    )

    card = pygame.Surface(panel.size, pygame.SRCALPHA)
    pygame.draw.rect(card, (22, 28, 40, 238), card.get_rect(), border_radius=16)
    pygame.draw.rect(card, (72, 124, 188), card.get_rect(), width=2, border_radius=16)
    screen.blit(card, panel.topleft)

    screen.blit(title_font.render(title, True, (220, 228, 238)), (panel.x + 20, panel.y + 18))
    screen.blit(body_font.render(message, True, (170, 182, 198)), (panel.x + 20, panel.y + 66))

    track = pygame.Rect(panel.x + 20, panel.y + 110, panel_w - 40, 12)
    fill = pygame.Rect(track.x, track.y, int(track.width * progress), track.height)
    pygame.draw.rect(screen, (44, 52, 72), track, border_radius=6)
    if fill.width > 0:
        pygame.draw.rect(screen, (60, 130, 210), fill, border_radius=6)

    percent = small_font.render(f"{int(progress * 100):d}%", True, (120, 132, 150))
    screen.blit(percent, (track.right - percent.get_width(), track.bottom + 10))
    pygame.display.flip()


def _pump_startup_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    return True


def _prewarm_creatures(screen: pygame.Surface) -> bool:
    stages = 2
    current_stage = 0
    cancelled = False

    def _progress(stage_label: str, index: int, total: int, _scale: float) -> bool:
        nonlocal cancelled
        if not _pump_startup_events():
            cancelled = True
            return False
        fraction = (current_stage + (index / max(1, total))) / stages
        _draw_startup_screen(
            screen,
            title="Loading Sandbox",
            message=f"Preparing {stage_label.lower()} sprites...",
            progress=fraction,
        )
        return True

    _draw_startup_screen(
        screen,
        title="Loading Sandbox",
        message="Preparing creature sprites...",
        progress=0.0,
    )
    if not _pump_startup_events():
        return False

    current_stage = 0
    if CreatureManager.prewarm_assets(progress=_progress) is False:
        return False

    current_stage = 1
    _draw_startup_screen(
        screen,
        title="Loading Sandbox",
        message="Creature sprites are ready.",
        progress=1.0,
    )
    return not cancelled


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


def _make_guide(config: Config) -> GuideEngine | None:
    if not config.ai_enabled:
        return None

    backend = config.llm_backend if config.llm_enabled else "template"
    provider_config = ProviderConfig(
        backend=backend,
        base_url=config.llm_base_url,
        model=config.llm_model,
        timeout_seconds=config.llm_timeout_seconds,
    )
    return GuideEngine(
        template_narrator=TemplateNarrator(),
        async_narrator=build_async_narrator(provider_config),
        event_cooldown=config.guide_update_seconds,
    )


def _make_calibration(config: Config) -> CalibrationData:
    points = config.vision_calibration_points or []
    if len(points) != 4:
        return CalibrationData()
    return CalibrationData(camera_points=tuple((float(x), float(y)) for x, y in points))


def main() -> int:
    pygame.init()
    pygame.display.set_caption("AR Sandbox")

    config  = Config.load()
    screen, screen_w, screen_h = _make_display(config)
    if not _prewarm_creatures(screen):
        pygame.quit()
        return 0
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
    creatures = CreatureManager(
        n_sharks=config.shark_count,
        n_dinos=config.dinosaur_count,
    )
    guide = _make_guide(config)
    vision = WebcamObserver(config.camera_index) if config.vision_enabled else None
    calibration = _make_calibration(config)
    interactions = InteractionEngine()
    camera_discovery = AsyncCameraDiscovery()
    camera_tester = AsyncCameraTester()
    llm_tester = AsyncConnectionTester()
    sidebar  = Sidebar()
    debug_font = pygame.font.SysFont(None, 18)
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

            if config.ai_changed:
                config.ai_changed = False
                if guide is not None:
                    guide.close()
                guide = _make_guide(config)

            if config.vision_changed:
                config.vision_changed = False
                if vision is not None:
                    vision.close()
                    vision = None
                calibration = _make_calibration(config)
                if config.vision_enabled:
                    vision = WebcamObserver(config.camera_index)
                    if vision.error_message:
                        config.camera_test_status = "error"
                        config.camera_test_message = vision.error_message
                    elif not config.camera_test_message:
                        config.camera_test_status = "ok"
                        config.camera_test_message = f"Camera {config.camera_index} ready."
                else:
                    config.available_cameras = []
                    config.camera_scan_status = "idle"
                    config.camera_scan_message = ""
                    config.camera_test_status = "idle"
                    config.camera_test_message = ""
                    config.calibration_message = ""

            if config.llm_test_requested:
                config.llm_test_requested = False
                backend = config.llm_backend if config.llm_enabled else "template"
                provider_config = ProviderConfig(
                    backend=backend,
                    base_url=config.llm_base_url,
                    model=config.llm_model,
                    timeout_seconds=config.llm_timeout_seconds,
                )
                started = llm_tester.start(provider_config)
                if not started and config.llm_test_status == "running":
                    config.llm_test_message = "A connection test is already running."

            if config.camera_test_requested:
                config.camera_test_requested = False
                started = camera_tester.start(config.camera_index)
                if not started and config.camera_test_status == "running":
                    config.camera_test_message = "A camera test is already running."

            if config.camera_scan_requested:
                config.camera_scan_requested = False
                started = camera_discovery.start()
                if not started and config.camera_scan_status == "scanning":
                    config.camera_scan_message = "A camera scan is already running."

            llm_test_result = llm_tester.poll()
            if llm_test_result is not None:
                config.llm_test_status = "ok" if llm_test_result.ok else "error"
                config.llm_test_message = llm_test_result.summary

            camera_test_result = camera_tester.poll()
            if camera_test_result is not None:
                config.camera_test_status = "ok" if camera_test_result.ok else "error"
                config.camera_test_message = camera_test_result.summary

            camera_scan_result = camera_discovery.poll()
            if camera_scan_result is not None:
                config.available_cameras = [
                    {
                        "index": camera.index,
                        "label": camera.summary.replace(" is working at ", "  "),
                    }
                    for camera in camera_scan_result
                ]
                if config.available_cameras:
                    config.camera_scan_status = "ok"
                    config.camera_scan_message = (
                        f"Found {len(config.available_cameras)} camera(s). Select one below."
                    )
                    available_indexes = {int(camera["index"]) for camera in config.available_cameras}
                    if config.camera_index not in available_indexes:
                        config.camera_index = int(config.available_cameras[0]["index"])
                        config.vision_changed = True
                else:
                    config.camera_scan_status = "error"
                    config.camera_scan_message = "No working cameras were detected."

            if config.save_requested:
                config.save_requested = False
                config.save()

            creatures.set_targets(
                sharks=config.shark_count,
                dinosaurs=config.dinosaur_count,
            )

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
                creatures.draw(scene, frame)

            vision_events = []
            vision_objects = []
            if config.vision_enabled and vision is not None:
                if config.vision_calibrating:
                    new_calibration = vision.auto_calibrate()
                    if new_calibration is not None:
                        calibration = new_calibration
                        config.vision_calibration_points = [
                            [float(x), float(y)] for x, y in new_calibration.camera_points
                        ]
                        config.vision_calibrating = False
                        config.calibration_message = "Calibration updated from corner markers."
                        config.request_save()
                    elif not config.calibration_message:
                        config.calibration_message = (
                            "Show corner tags 100, 101, 102, and 103 to calibrate the sandbox."
                        )
                vision_objects = vision.get_objects(calibration, (render_w, render_h), frame=frame)
                interactions.update(
                    vision_objects,
                    frame,
                    time.monotonic(),
                    reactions_enabled=config.object_reactions_enabled,
                )
                if config.vision_calibrating or config.vision_debug_enabled:
                    overlay_rgb = vision.get_warped_rgb(calibration, (render_w, render_h))
                    overlay_label = "Calibration View"
                    overlay_alpha = 92 if config.vision_calibrating else 64
                    if overlay_rgb is None:
                        overlay_rgb = vision.get_preview_rgb((render_w, render_h))
                        overlay_label = "Camera Preview"
                        overlay_alpha = 72 if config.vision_calibrating else 56
                    if overlay_rgb is not None:
                        overlay_surface = pygame.surfarray.make_surface(overlay_rgb.swapaxes(0, 1))
                        overlay_surface.set_alpha(overlay_alpha)
                        scene.blit(overlay_surface, (0, 0))
                        label = debug_font.render(overlay_label, True, (255, 230, 90))
                        shadow = debug_font.render(overlay_label, True, (0, 0, 0))
                        scene.blit(shadow, (11, 11))
                        scene.blit(label, (10, 10))
                interactions.draw(scene)
                if config.vision_debug_enabled:
                    for kind, _camera_pos, mapped_pos in vision.debug_points(calibration, (render_w, render_h)):
                        px, py = int(mapped_pos[0]), int(mapped_pos[1])
                        pygame.draw.circle(scene, (255, 230, 90), (px, py), 9, 2)
                        tag = debug_font.render(kind, True, (255, 230, 90))
                        scene.blit(tag, (px + 8, py - 8))
                vision_events = interactions.pop_events()

            guide_message = None
            active_challenge = None
            if guide is not None:
                counts = creatures.counts() if config.show_creatures else {"sharks": 0, "dinosaurs": 0}
                guide_message = guide.update(
                    frame,
                    time.monotonic(),
                    shark_count=counts["sharks"],
                    dinosaur_count=counts["dinosaurs"],
                    creatures_enabled=config.show_creatures,
                    guide_enabled=config.guide_enabled,
                    challenges_enabled=config.guide_challenges_enabled,
                    verbosity=config.guide_verbosity,
                )
                active_challenge = guide.active_challenge
                if config.guide_enabled:
                    for vision_event in vision_events:
                        pushed = guide.push_external_event(
                            event_key=vision_event.event_key,
                            title=vision_event.title,
                            body=vision_event.body,
                            now=time.monotonic(),
                            verbosity=config.guide_verbosity,
                        )
                        if pushed is not None:
                            guide_message = pushed
            else:
                vision_events.clear()

            if scene.get_size() == screen.get_size():
                screen.blit(scene, (0, 0))
            else:
                scaled = pygame.transform.scale(scene, screen.get_size())
                screen.blit(scaled, (0, 0))

            draw_guide_overlay(
                screen,
                config,
                title=guide_message.title if guide_message else None,
                body=guide_message.body if guide_message else None,
                challenge_text=(
                    active_challenge.prompt
                    if (active_challenge is not None and config.guide_challenges_enabled)
                    else None
                ),
                challenge_done=bool(guide_message.completed) if guide_message else False,
            )

            pygame.draw.circle(screen, (255, 255, 255), (mx, my), brush_radius, 1)

            creatures_state = "on" if config.show_creatures else "off"
            ai_state = "ai:on" if config.ai_enabled else "ai:off"
            hud = (f"Tab settings  C contours  G creatures:{creatures_state}  R reset  "
                   f"brush {brush_radius}px  |  {config.colour_scheme}  {ai_state}")
            label = hud_font.render(hud, True, (200, 200, 200))
            screen.blit(label, (10, screen.get_height() - 24))

            sidebar.update(dt)
            sidebar.draw(screen, config)

            pygame.display.flip()
    finally:
        try:
            config.save()
        except Exception:
            pass
        try:
            if guide is not None:
                guide.close()
        except Exception:
            pass
        try:
            camera_discovery.close()
        except Exception:
            pass
        try:
            camera_tester.close()
        except Exception:
            pass
        try:
            if vision is not None:
                vision.close()
        except Exception:
            pass
        try:
            llm_tester.close()
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
