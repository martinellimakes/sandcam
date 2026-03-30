"""
AR Sandbox POC — mouse-sculpted terrain with elevation rendering.

Controls
--------
Left-drag   raise terrain
Right-drag  lower terrain
Scroll      resize brush
C           toggle contour lines
R           reset terrain
Esc / Q     quit
"""

import pygame

# from depth_source import MouseSimulator
from depth_source import KinectV1Source

from renderer import Renderer

WIDTH  = 900
HEIGHT = 650
FPS    = 60

BRUSH_MIN      = 8
BRUSH_MAX      = 180
BRUSH_DEFAULT  = 50
BRUSH_STRENGTH = 0.012   # height units per frame while button held


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AR Sandbox POC")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 22)

    # source   = MouseSimulator(WIDTH, HEIGHT)
    source = KinectV1Source(WIDTH, HEIGHT)   # tune min_depth_mm / max_depth_mm to your rig height
    renderer = Renderer(WIDTH, HEIGHT)

    brush_radius  = BRUSH_DEFAULT
    show_contours = True

    running = True
    while running:
        # ------------------------------------------------------------------ events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    source.reset()
                elif event.key == pygame.K_c:
                    show_contours = not show_contours
            elif event.type == pygame.MOUSEWHEEL:
                brush_radius = max(BRUSH_MIN, min(BRUSH_MAX, brush_radius + event.y * 4))

        # ------------------------------------------------------------------ sculpt
        buttons = pygame.mouse.get_pressed()
        mx, my  = pygame.mouse.get_pos()
        if buttons[0]:
            source.sculpt(mx, my, brush_radius, +BRUSH_STRENGTH)
        elif buttons[2]:
            source.sculpt(mx, my, brush_radius, -BRUSH_STRENGTH)

        # ------------------------------------------------------------------ render
        frame = source.get_frame()
        renderer.draw(screen, frame, show_contours=show_contours)

        # Brush cursor
        pygame.draw.circle(screen, (255, 255, 255), (mx, my), brush_radius, 1)

        # HUD
        hud_text = (
            f"LMB raise  RMB lower  Scroll resize brush ({brush_radius}px)"
            f"  C {'hide' if show_contours else 'show'} contours  R reset"
        )
        label = font.render(hud_text, True, (220, 220, 220))
        screen.blit(label, (10, HEIGHT - 26))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    if hasattr(source, "close"):
        source.close()


if __name__ == "__main__":
    main()
