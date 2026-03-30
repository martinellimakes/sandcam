"""
AR Sandbox POC

Controls
--------
Left-drag   raise terrain (simulator only)
Right-drag  lower terrain (simulator only)
Scroll      resize brush  (simulator only)
C           toggle contour lines
R           reset terrain (simulator only)
Tab         open / close settings sidebar
Esc / Q     quit
"""

import pygame

# from depth_source import MouseSimulator
from depth_source import KinectV1Source
from renderer import Renderer
from ui import Config, Sidebar

FPS           = 60
BRUSH_MIN     = 8
BRUSH_MAX     = 180
BRUSH_DEFAULT = 50
BRUSH_DELTA   = 0.012


def _make_display(config: Config) -> tuple[pygame.Surface, int, int]:
    """Create (or recreate) the pygame display surface from current config."""
    if config.fullscreen:
        desktops = pygame.display.get_desktop_sizes()
        idx = min(config.display_index, len(desktops) - 1)
        w, h = desktops[idx]
        screen = pygame.display.set_mode((w, h), pygame.FULLSCREEN, display=idx)
    else:
        w, h = config.windowed_w, config.windowed_h
        screen = pygame.display.set_mode((w, h))
    return screen, w, h


def main() -> None:
    pygame.init()
    pygame.display.set_caption("AR Sandbox")

    config  = Config()
    screen, W, H = _make_display(config)

    # source  = MouseSimulator(W, H)
    source   = KinectV1Source(W, H,
                              min_depth_mm=config.min_depth_mm,
                              max_depth_mm=config.max_depth_mm)
    renderer = Renderer(W, H)
    sidebar  = Sidebar()
    clock    = pygame.font.SysFont(None, 22)   # reused as font for HUD
    hud_font = pygame.font.SysFont(None, 22)
    clock    = pygame.time.Clock()

    brush_radius = BRUSH_DEFAULT

    running = True
    while running:
        # ── events ────────────────────────────────────────────────────────────
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
                    source.reset()
                elif event.key == pygame.K_c:
                    config.show_contours = not config.show_contours
            elif event.type == pygame.MOUSEWHEEL:
                brush_radius = max(BRUSH_MIN, min(BRUSH_MAX,
                                                  brush_radius + event.y * 4))

        # ── react to config changes ───────────────────────────────────────────
        if config.display_changed:
            config.display_changed = False
            screen, W, H = _make_display(config)
            renderer = Renderer(W, H)
            source.resize(W, H)

        if config.depth_range_changed:
            config.depth_range_changed = False
            if hasattr(source, "set_depth_range"):
                source.set_depth_range(config.min_depth_mm, config.max_depth_mm)

        # ── sculpt (simulator only) ───────────────────────────────────────────
        buttons = pygame.mouse.get_pressed()
        mx, my  = pygame.mouse.get_pos()
        if buttons[0]:
            source.sculpt(mx, my, brush_radius, +BRUSH_DELTA)
        elif buttons[2]:
            source.sculpt(mx, my, brush_radius, -BRUSH_DELTA)

        # ── render ────────────────────────────────────────────────────────────
        frame = source.get_frame()
        renderer.draw(screen, frame,
                      show_contours=config.show_contours,
                      colour_scheme=config.colour_scheme)

        # Brush cursor
        pygame.draw.circle(screen, (255, 255, 255), (mx, my), brush_radius, 1)

        # HUD
        hud = (f"Tab settings  C contours  R reset  "
               f"brush {brush_radius}px  |  {config.colour_scheme}")
        label = hud_font.render(hud, True, (200, 200, 200))
        screen.blit(label, (10, screen.get_height() - 24))

        # Sidebar (always on top)
        sidebar.update()
        sidebar.draw(screen, config)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    if hasattr(source, "close"):
        source.close()


if __name__ == "__main__":
    main()
