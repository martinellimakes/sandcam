# sandcam — AR Sandbox POC
#
# Linux-based image.  On Linux, libfreenect is available via apt and the
# freenect Python C extension compiles without the pthreads workarounds
# required on Windows.
#
# Building
# --------
#   docker build -t sandcam .
#
# Running (Linux host with X11)
# --------------------------------
#   docker run --rm \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --device /dev/bus/usb \
#     --privileged \
#     sandcam
#
# Running (Windows host via WSL2 + VcXsrv or WSLg)
# -------------------------------------------------
# WSLg (Windows 11 22H2+) exposes a Wayland/X11 socket automatically.
# Inside WSL2:
#   export DISPLAY=:0
#   docker run --rm -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --device /dev/bus/usb --privileged sandcam
#
# USB / Kinect passthrough
# ------------------------
# Docker Desktop on Windows does NOT support USB device passthrough natively.
# Use usbipd-win (https://github.com/dorssel/usbipd-win) to attach the Kinect
# to WSL2 before running the container:
#
#   # PowerShell (admin), one-time:
#   winget install usbipd
#
#   # Attach the Kinect to WSL2:
#   usbipd list                  # find the Kinect's BUSID
#   usbipd attach --wsl --busid <BUSID>
#
# The Kinect will then appear as /dev/bus/usb inside WSL2 and the container.
#
# Simulator mode (no Kinect needed)
# ----------------------------------
# Comment out KinectV1Source and uncomment MouseSimulator in main.py before
# building the image.  Display still required (no headless mode in pygame).

FROM python:3.13-slim

# ── system dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # libfreenect + libusb (Kinect v1 driver)
    libfreenect-dev \
    libusb-1.0-0-dev \
    # SDL2 (required by pygame)
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    # build tools (for Python C extensions, e.g. scipy)
    build-essential \
    python3-dev \
    # udev rules for Kinect USB access without root
    udev \
    && rm -rf /var/lib/apt/lists/*

# Allow non-root access to Kinect USB device
RUN echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666"' \
    > /etc/udev/rules.d/51-kinect.rules

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install only the declared deps (pygame, numpy, scipy).
# libfreenect is loaded at runtime via ctypes from the system library.
RUN uv sync --frozen --no-dev

# ── application code ──────────────────────────────────────────────────────────
COPY depth_source.py renderer.py main.py ./

# SDL / pygame needs a display — set DISPLAY at runtime via -e DISPLAY=$DISPLAY
ENV SDL_VIDEODRIVER=x11

ENTRYPOINT ["uv", "run", "python", "main.py"]
