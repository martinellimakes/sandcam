"""
Renders a normalised height map as an AR-sandbox-style elevation image.

Colour map  : blue water → sand → grass → rock → snow
Contour lines: drawn at fixed elevation intervals above the waterline
"""

from __future__ import annotations

import numpy as np
import pygame
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Elevation colour map  –  (normalised_height, (R, G, B))
# ---------------------------------------------------------------------------
_COLORMAP: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (5,   10,  80)),   # abyssal water
    (0.25, (15,  60, 160)),   # deep water
    (0.38, (25, 110, 200)),   # shallow water
    (0.43, (80, 170, 220)),   # very shallow / surf
    (0.46, (220, 210, 140)),  # wet sand
    (0.50, (240, 230, 150)),  # dry sand / beach
    (0.54, (120, 190,  70)),  # low grass
    (0.63, ( 70, 150,  40)),  # grass
    (0.72, (100, 110,  55)),  # highland / scrub
    (0.82, (130, 115,  85)),  # rock
    (0.91, (175, 165, 150)),  # high rock
    (1.00, (255, 255, 255)),  # snow
]

WATER_LEVEL    = 0.46   # heights below this are rendered as water
CONTOUR_INTERVAL = 0.05  # spacing between contour lines (in normalised units)


def _build_lut() -> np.ndarray:
    """Pre-compute a 256-entry (index → RGB) lookup table from _COLORMAP."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(len(_COLORMAP) - 1):
            t0, c0 = _COLORMAP[k]
            t1, c1 = _COLORMAP[k + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0)
                lut[i] = [int(c0[j] + alpha * (c1[j] - c0[j])) for j in range(3)]
                break
    return lut


_LUT = _build_lut()


class Renderer:
    def __init__(self, width: int, height: int) -> None:
        self._surf = pygame.Surface((width, height))

    def draw(
        self,
        target: pygame.Surface,
        height_data: np.ndarray,
        *,
        show_contours: bool = True,
    ) -> None:
        # Mild display-only smoothing (keeps the sculpt map clean)
        display = gaussian_filter(height_data, sigma=1.5).astype(np.float32)

        # --- base colour from LUT ---
        indices = (np.clip(display, 0.0, 1.0) * 255).astype(np.uint8)
        rgb = _LUT[indices].copy()  # shape (H, W, 3)

        # --- hillshading: simple north-west light to give depth ---
        dy, dx = np.gradient(display)
        shade = np.clip(1.0 - 0.6 * dy + 0.2 * dx, 0.5, 1.3).astype(np.float32)
        rgb = np.clip(rgb * shade[:, :, np.newaxis], 0, 255).astype(np.uint8)

        # --- contour lines (terrain only, not water) ---
        if show_contours:
            above_water = display >= WATER_LEVEL
            quantized = (display / CONTOUR_INTERVAL).astype(np.int32)
            contour = (
                (quantized != np.roll(quantized, 1, axis=0)) |
                (quantized != np.roll(quantized, 1, axis=1))
            )
            contour &= above_water
            rgb[contour] = (rgb[contour] * 0.35).astype(np.uint8)

        # pygame surfarray is (W, H, 3) — transpose from numpy's (H, W, 3)
        pygame.surfarray.blit_array(self._surf, rgb.transpose(1, 0, 2))
        target.blit(self._surf, (0, 0))
