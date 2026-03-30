"""
Depth source abstraction.

Available sources
-----------------
MouseSimulator  – POC stand-in, no hardware required
KinectV1Source  – Microsoft Kinect v1 via libfreenect (ctypes, no PyPI package needed)

Setup for KinectV1Source (Windows)
-----------------------------------
1. Build libfreenect.dll (done — it lives at C:/Git/sandcam/freenect.dll).
   libusb-1.0.dll must also be next to it (also already copied there).

2. Install Zadig (https://zadig.akeo.ie/)
   Plug in the Kinect, open Zadig, select the Kinect device,
   replace its driver with "WinUSB".

3. In main.py swap MouseSimulator for KinectV1Source:
       from depth_source import KinectV1Source
       source = KinectV1Source()
   Remember to call source.close() on exit (or use it as a context manager).

No extra Python packages are needed beyond what is already in pyproject.toml.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import gaussian_filter


class DepthSource(ABC):
    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return a 2-D float32 array of shape (height, width), values in [0, 1]."""
        ...


class MouseSimulator(DepthSource):
    """
    Simulates a Kinect depth map via mouse sculpting.

    Left-drag  → raise terrain
    Right-drag → lower terrain
    Scroll     → resize brush
    R          → reset to flat
    """

    DEFAULT_LEVEL = 0.45  # resting height (sits just at the waterline)

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._map = np.full((height, width), self.DEFAULT_LEVEL, dtype=np.float32)

        # Pre-build coordinate grids once — sculpt() reuses them every frame
        self._ys, self._xs = np.mgrid[0:height, 0:width]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sculpt(self, cx: int, cy: int, radius: int, delta: float) -> None:
        """Apply a smooth Gaussian brush centred on (cx, cy)."""
        sigma = radius / 2.5
        dist_sq = (self._xs - cx) ** 2 + (self._ys - cy) ** 2
        brush = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)
        brush[dist_sq > radius ** 2] = 0.0
        self._map += brush * delta
        np.clip(self._map, 0.0, 1.0, out=self._map)
        # Light smoothing prevents ugly spikes at the brush edge
        self._map = gaussian_filter(self._map, sigma=1.0).astype(np.float32)

    def reset(self) -> None:
        self._map[:] = self.DEFAULT_LEVEL

    def get_frame(self) -> np.ndarray:
        return self._map.copy()


class KinectV1Source(DepthSource):
    """
    Kinect v1 depth source via libfreenect, using ctypes directly.

    Loads freenect.dll from the same directory as this file (no pip package
    required).  Runs the freenect event loop in a background daemon thread and
    exposes the latest depth frame through get_frame().

    Parameters
    ----------
    width, height : int
        Output frame dimensions (should match the display window).
        The Kinect's native 640×480 frame is resized to fit.
    min_depth_mm : int
        Distance (mm) → normalised 1.0 (closest expected point, i.e. tallest
        sand pile).  Tune after mounting the camera above your sandbox.
    max_depth_mm : int
        Distance (mm) → normalised 0.0 (bare sandbox floor).
    device_index : int
        Which Kinect to open (almost always 0).
    """

    # freenect constants
    _RESOLUTION_MEDIUM = 1   # 640 × 480
    _DEPTH_11BIT       = 0   # uint16, values 0–2047 mm
    _DEPTH_RAW_NO_DATA = 2047

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        min_depth_mm: int = 400,
        max_depth_mm: int = 1100,
        device_index: int = 0,
    ) -> None:
        self._out_width  = width
        self._out_height = height
        import ctypes
        import os
        import pathlib
        import threading

        # ------------------------------------------------------------------ load DLL
        dll_dir = pathlib.Path(__file__).parent
        import sys, ctypes.util

        if sys.platform == "win32":
            # Python 3.8+: add the project dir so libusb-1.0.dll is found too
            os.add_dll_directory(str(dll_dir))
            lib_path = str(dll_dir / "freenect.dll")
        else:
            # Linux / macOS: installed via apt / brew, resolvable by the linker
            found = ctypes.util.find_library("freenect")
            if not found:
                raise RuntimeError(
                    "libfreenect not found. On Debian/Ubuntu: apt install libfreenect-dev"
                )
            lib_path = found

        try:
            lib = ctypes.CDLL(lib_path)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot load libfreenect from {lib_path}.\n"
                "Windows: ensure freenect.dll + libusb-1.0.dll are next to this file.\n"
                "Linux:   apt install libfreenect-dev\n"
                f"Original error: {exc}"
            ) from exc

        # ------------------------------------------------------------------ struct
        # typedef struct {
        #   uint32_t reserved;      freenect_resolution resolution;
        #   union { int32_t dummy; freenect_video_format; freenect_depth_format; };
        #   int32_t bytes;
        #   int16_t width; int16_t height;
        #   int8_t data_bits; int8_t padding_bits; int8_t framerate; int8_t is_valid;
        # } freenect_frame_mode;   // 24 bytes total
        class _FrameMode(ctypes.Structure):
            _fields_ = [
                ("reserved",    ctypes.c_uint32),
                ("resolution",  ctypes.c_int32),
                ("format",      ctypes.c_int32),
                ("bytes",       ctypes.c_int32),
                ("width",       ctypes.c_int16),
                ("height",      ctypes.c_int16),
                ("data_bpp",    ctypes.c_int8),
                ("pad_bpp",     ctypes.c_int8),
                ("framerate",   ctypes.c_int8),
                ("is_valid",    ctypes.c_int8),
            ]

        # ------------------------------------------------------------------ API
        lib.freenect_init.restype  = ctypes.c_int
        lib.freenect_init.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        lib.freenect_shutdown.restype  = ctypes.c_int
        lib.freenect_shutdown.argtypes = [ctypes.c_void_p]

        lib.freenect_open_device.restype  = ctypes.c_int
        lib.freenect_open_device.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int
        ]
        lib.freenect_close_device.restype  = ctypes.c_int
        lib.freenect_close_device.argtypes = [ctypes.c_void_p]

        lib.freenect_find_depth_mode.restype  = _FrameMode
        lib.freenect_find_depth_mode.argtypes = [ctypes.c_int, ctypes.c_int]

        lib.freenect_set_depth_mode.restype  = ctypes.c_int
        lib.freenect_set_depth_mode.argtypes = [ctypes.c_void_p, _FrameMode]

        _DepthCB = ctypes.CFUNCTYPE(
            None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        )
        lib.freenect_set_depth_callback.restype  = None
        lib.freenect_set_depth_callback.argtypes = [ctypes.c_void_p, _DepthCB]

        lib.freenect_start_depth.restype  = ctypes.c_int
        lib.freenect_start_depth.argtypes = [ctypes.c_void_p]
        lib.freenect_stop_depth.restype   = ctypes.c_int
        lib.freenect_stop_depth.argtypes  = [ctypes.c_void_p]

        lib.freenect_process_events.restype  = ctypes.c_int
        lib.freenect_process_events.argtypes = [ctypes.c_void_p]

        # ------------------------------------------------------------------ init
        ctx = ctypes.c_void_p()
        if lib.freenect_init(ctypes.byref(ctx), None) != 0:
            raise RuntimeError("freenect_init failed")

        dev = ctypes.c_void_p()
        if lib.freenect_open_device(ctx, ctypes.byref(dev), device_index) != 0:
            lib.freenect_shutdown(ctx)
            raise RuntimeError(
                f"freenect_open_device({device_index}) failed — "
                "is the Kinect plugged in and using the WinUSB driver (Zadig)?"
            )

        mode = lib.freenect_find_depth_mode(self._RESOLUTION_MEDIUM, self._DEPTH_11BIT)
        if not mode.is_valid:
            raise RuntimeError("Could not find 640×480 11-bit depth mode")
        lib.freenect_set_depth_mode(dev, mode)

        # ------------------------------------------------------------------ state
        self._lib     = lib
        self._ctx     = ctx
        self._dev     = dev
        self._min     = float(min_depth_mm)
        self._max     = float(max_depth_mm)
        self._running = True
        self._lock    = threading.Lock()
        self._latest  = np.full((height, width), 0.5, dtype=np.float32)

        # Hold a Python reference to the callback so the GC doesn't collect it
        self._cb = _DepthCB(self._on_depth)
        lib.freenect_set_depth_callback(dev, self._cb)
        lib.freenect_start_depth(dev)

        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_depth(self, _dev, data_ptr: int, _timestamp: int) -> None:
        """Called from the freenect event thread on each new depth frame."""
        if not data_ptr:
            return
        # Interpret the raw buffer as a 480×640 uint16 array (no copy yet)
        import ctypes
        raw = np.frombuffer(
            (ctypes.c_uint16 * (480 * 640)).from_address(data_ptr),
            dtype=np.uint16,
        ).reshape(480, 640).copy().astype(np.float32)

        # Pixels with no valid reading → treat as maximum distance (floor)
        raw[(raw == 0) | (raw >= self._DEPTH_RAW_NO_DATA)] = self._max

        # Closer → higher normalised value (raised sand)
        frame = np.clip(
            1.0 - (raw - self._min) / (self._max - self._min),
            0.0, 1.0,
        ).astype(np.float32)

        # Resize to output dimensions if they differ from native 480×640
        if frame.shape != (self._out_height, self._out_width):
            from scipy.ndimage import zoom
            zy = self._out_height / frame.shape[0]
            zx = self._out_width  / frame.shape[1]
            frame = zoom(frame, (zy, zx), order=1).astype(np.float32)

        with self._lock:
            self._latest = frame

    def _event_loop(self) -> None:
        while self._running:
            self._lib.freenect_process_events(self._ctx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self._latest.copy()

    def close(self) -> None:
        """Stop the Kinect and release resources."""
        self._running = False
        self._thread.join(timeout=2.0)
        self._lib.freenect_stop_depth(self._dev)
        self._lib.freenect_close_device(self._dev)
        self._lib.freenect_shutdown(self._ctx)

    def __enter__(self) -> KinectV1Source:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
