# sandcam — AR Sandbox POC

A real-time augmented-reality sandbox visualiser.  A depth camera (Kinect v1)
scans a physical sandbox; the app renders a topographic elevation map with
contour lines, hillshading, and colour-coded terrain that updates live.

A **mouse-sculpting simulator** is included so you can develop and test without
any hardware.

---

## How it works

```
Kinect v1 (depth camera)
        │  480×640 uint16 depth frame (mm)
        ▼
  KinectV1Source          ← ctypes wrapper around libfreenect, async event loop
        │  float32 [0,1] height map, resized to window
        ▼
    Renderer              ← elevation LUT + hillshading + contour lines
        │  RGB pixel array
        ▼
  pygame window           ← 60 fps display, optional projector output
```

---

## Running without hardware (mouse simulator)

```bash
# in main.py, ensure this line is active:
#   source = MouseSimulator(WIDTH, HEIGHT)
# and this is commented out:
#   source = KinectV1Source(WIDTH, HEIGHT)

uv run python main.py
```

Controls:

| Input | Action |
|---|---|
| Left-drag | Raise terrain |
| Right-drag | Lower terrain |
| Scroll wheel | Resize brush |
| `C` | Toggle contour lines |
| `R` | Reset terrain |
| `Esc` / `Q` | Quit |

---

## Windows setup (Kinect v1)

### Prerequisites

- Windows 10 / 11
- Python 3.13+ (via [python.org](https://www.python.org/))
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Visual Studio 2019 Build Tools (C++ workload) — needed to compile libfreenect
- CMake — `winget install Kitware.CMake`
- Git

### 1 — Automated setup (recommended)

```powershell
git clone https://github.com/yourname/sandcam
cd sandcam
.\setup-windows.ps1
```

The script clones libfreenect, builds it with the bundled libusb, copies the
DLLs next to `main.py`, and runs `uv sync`.  If it fails due to a permissions
error on `C:\libfreenect`, re-run from an elevated (Administrator) PowerShell.

Skip to [step 3 (Zadig)](#3--replace-the-kinect-usb-driver-zadig) when done.

---

### Manual setup (if the script fails)

#### 1a — Clone and install Python deps

```powershell
git clone https://github.com/yourname/sandcam
cd sandcam
uv sync
```

#### 2 — Build libfreenect from source

The `freenect` PyPI package cannot be built on Windows because its c_sync
wrapper requires pthreads.  Instead we build `libfreenect.dll` directly and
talk to it via ctypes — no pthreads required.

```powershell
# Clone libfreenect (includes bundled libusb-1.0 for Windows)
git clone https://github.com/OpenKinect/libfreenect C:\Git\libfreenect

# Configure — skip c_sync (pthreads) and examples; use bundled libusb
cd C:\Git\libfreenect
mkdir build; cd build

cmake .. `
  -DCMAKE_INSTALL_PREFIX="C:\libfreenect" `
  -DBUILD_EXAMPLES=OFF `
  -DBUILD_FAKENECT=OFF `
  -DBUILD_PYTHON3=OFF `
  -DBUILD_C_SYNC=OFF `
  -DLIBUSB_1_INCLUDE_DIR="C:\Git\libfreenect\libusb-1.0.29\include" `
  -DLIBUSB_1_LIBRARY="C:\Git\libfreenect\libusb-1.0.29\VS2019\MS64\dll\libusb-1.0.lib"

cmake --build . --config Release
cmake --install . --config Release
```

> **Note:** The install will fail if targeting `C:\Program Files\libfreenect`
> due to UAC.  Use `C:\libfreenect` (no spaces, no admin needed).

Copy the runtime DLLs next to `main.py`:

```powershell
copy "C:\libfreenect\lib\freenect.dll"                              C:\Git\sandcam\
copy "C:\Git\libfreenect\libusb-1.0.29\VS2019\MS64\dll\libusb-1.0.dll"  C:\Git\sandcam\
```

### 3 — Replace the Kinect USB driver (Zadig)

The Kinect ships with an Xbox HID driver.  libfreenect needs WinUSB instead.

1. Download [Zadig](https://zadig.akeo.ie/) and run it
2. Plug in the Kinect (and its power adapter if using an Xbox 360 unit)
3. In Zadig: **Options → List All Devices**, select the Kinect Camera device
4. Set the target driver to **WinUSB**, click **Replace Driver**

> ⚠️ This replaces the driver for this USB device only.  If you later install
> the official Kinect SDK 1.8, you would need to re-run Zadig.

### 4 — Run with hardware

In `main.py`, switch the source:

```python
# from depth_source import MouseSimulator
from depth_source import KinectV1Source
...
# source = MouseSimulator(WIDTH, HEIGHT)
source = KinectV1Source(WIDTH, HEIGHT)
```

Tune the depth range to your physical rig (camera height above sandbox):

```python
source = KinectV1Source(
    WIDTH, HEIGHT,
    min_depth_mm=400,   # camera-to-peak-of-sand distance
    max_depth_mm=1100,  # camera-to-bare-sandbox-floor distance
)
```

```powershell
uv run python main.py
```

---

## Linux setup (Kinect v1)

On Linux, libfreenect is in the system package manager and the full `freenect`
Python package compiles cleanly (no pthreads workaround needed).

```bash
# Ubuntu / Debian
sudo apt install libfreenect-dev libusb-1.0-0-dev

# Fedora / RHEL
sudo dnf install libfreenect-devel libusbx-devel

# udev rule so the Kinect is accessible without sudo
sudo cp /usr/share/doc/libfreenect/examples/51-kinect.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules

uv sync
uv run python main.py
```

No Zadig step needed — the Linux kernel's `gspca` or generic USB driver works
directly with libfreenect.

---

## Docker

See [Dockerfile](Dockerfile).  The container targets Linux and is useful for
reproducible dev environments and CI.

```bash
# Build
docker build -t sandcam .

# Run with X11 display forwarding (Linux host)
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/bus/usb \
  --privileged \
  sandcam

# Run simulator only (no Kinect, no display — headless is not supported by pygame)
# Use X11 or Xvfb:
docker run --rm -e DISPLAY=:99 sandcam
```

See the Dockerfile comments for Windows (WSL2) display notes.

---

## Project structure

```
sandcam/
├── main.py           Game loop, input handling, HUD
├── depth_source.py   DepthSource ABC, MouseSimulator, KinectV1Source
├── renderer.py       Elevation colourmap, hillshading, contour lines
├── pyproject.toml    uv-managed dependencies (pygame, numpy, scipy)
├── freenect.dll      libfreenect Windows runtime   ← built locally, not in git
├── libusb-1.0.dll    libusb Windows runtime        ← built locally, not in git
└── Dockerfile
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pygame` | Window, event loop, pixel blitting |
| `numpy` | Height map arithmetic |
| `scipy` | Gaussian smoothing, frame resize |
| libfreenect (native) | Kinect v1 USB driver + depth stream |
