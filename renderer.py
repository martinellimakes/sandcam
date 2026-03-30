"""
Renders a normalised height map as an AR-sandbox-style elevation image.

Colour schemes : terrain, heat, greyscale, desert
Contour lines  : drawn at fixed elevation intervals above the waterline
"""

from __future__ import annotations

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter


# ── colour maps  (normalised_height, (R, G, B)) ───────────────────────────────

_COLORMAPS: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "terrain": [
        (0.00, (5,   10,  80)),   # abyssal water
        (0.25, (15,  60, 160)),   # deep water
        (0.38, (25, 110, 200)),   # shallow water
        (0.43, (80, 170, 220)),   # surf
        (0.46, (220, 210, 140)),  # wet sand
        (0.50, (240, 230, 150)),  # beach
        (0.54, (120, 190,  70)),  # low grass
        (0.63, ( 70, 150,  40)),  # grass
        (0.72, (100, 110,  55)),  # highland
        (0.82, (130, 115,  85)),  # rock
        (0.91, (175, 165, 150)),  # high rock
        (1.00, (255, 255, 255)),  # snow
    ],
    "heat": [
        (0.00, (  0,   0,  50)),
        (0.20, (  0,   0, 180)),
        (0.40, (  0, 160, 200)),
        (0.60, ( 50, 210,  50)),
        (0.75, (240, 210,   0)),
        (0.90, (255,  80,   0)),
        (1.00, (255, 255, 255)),
    ],
    "greyscale": [
        (0.00, ( 10,  10,  10)),
        (1.00, (245, 245, 245)),
    ],
    "desert": [
        (0.00, ( 15,  10,   5)),
        (0.25, ( 70,  40,  15)),
        (0.45, (160, 110,  50)),
        (0.60, (210, 170,  90)),
        (0.72, (225, 190, 110)),
        (0.82, (180, 130,  70)),
        (0.92, (140,  95,  55)),
        (1.00, (230, 210, 180)),
    ],
}

WATER_LEVEL      = 0.46
CONTOUR_INTERVAL = 0.05


def _build_lut(colormap: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(colormap) - 1):
            t0, c0 = colormap[k]
            t1, c1 = colormap[k + 1]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0)
                lut[i] = [int(c0[j] + a * (c1[j] - c0[j])) for j in range(3)]
                break
    return lut


_LUTS = {name: _build_lut(cm) for name, cm in _COLORMAPS.items()}


class Renderer:
    def __init__(self, width: int, height: int) -> None:
        self._surf = pygame.Surface((width, height))

    def draw(
        self,
        target: pygame.Surface,
        height_data: np.ndarray,
        *,
        show_contours: bool = True,
        colour_scheme: str = "terrain",
    ) -> None:
        # Recreate internal surface if window was resized
        if self._surf.get_size() != target.get_size():
            self._surf = pygame.Surface(target.get_size())

        lut = _LUTS.get(colour_scheme, _LUTS["terrain"])

        # Mild display-only smoothing
        display = gaussian_filter(height_data, sigma=1.5).astype(np.float32)

        # Base colour from LUT
        indices = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
        rgb = lut[indices].copy()

        # Hillshading — NW light source
        dy, dx = np.gradient(display)
        shade = np.clip(1.0 - 0.6 * dy + 0.2 * dx, 0.5, 1.3).astype(np.float32)
        rgb = np.clip(rgb * shade[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # Contour lines (above waterline only)
        if show_contours:
            above_water = display >= WATER_LEVEL
            quantized   = (display / CONTOUR_INTERVAL).astype(np.int32)
            contour = (
                (quantized != np.roll(quantized, 1, axis=0)) |
                (quantized != np.roll(quantized, 1, axis=1))
            )
            contour &= above_water
            rgb[contour] = (rgb[contour] * 0.35).astype(np.uint8)

        # pygame surfarray is (W, H, 3) — transpose from numpy's (H, W, 3)
        pygame.surfarray.blit_array(self._surf, rgb.transpose(1, 0, 2))
        target.blit(self._surf, (0, 0))
