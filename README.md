# sandcam — AR Sandbox POC

A real-time augmented-reality sandbox visualiser.  A depth camera (Kinect v1)
scans a physical sandbox; the app renders a topographic elevation map with
contour lines, hillshading, and colour-coded terrain that updates live.

A **mouse-sculpting simulator** is included so you can develop and test without
any hardware.

The app also now supports an **optional webcam vision layer** for tagged toy
interactions, plus an optional AI guide.  Both systems are independently
disableable, so the project can always run as a plain AR sandbox.

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

## Optional webcam vision

The webcam/object system is a second sensing layer on top of the depth-based
sandbox.  It is fully optional:

- `vision_enabled = false` keeps the app as a normal AR sandbox
- `ai_enabled = false` keeps the app free of guide / LLM features
- webcam object reactions work locally and do **not** require an LLM

### What it does

When enabled, a standard RGB webcam can detect tagged objects placed on the
sand and trigger local sandbox interactions such as:

- boat in water → ripple-style water reaction
- boat on land → stranded event
- dinosaur toy on land → habitat-themed reaction
- house or tree near coast → settlement / shoreline reaction
- volcano toy → hazard-style reaction

### Setup

1. Install dependencies:

```powershell
uv sync
```

This now includes `opencv-contrib-python`, which is used for ArUco marker
detection.

2. Open the settings sidebar in the app with `Tab`
3. In the `Vision` section:
   - turn `Vision ON`
   - click `Scan Cameras`
   - choose the correct webcam from `Detected Cameras`
   - `Camera Index` still exists, but `Detected Cameras` is the preferred way to pick the right device on multi-camera systems
   - click `Test Camera`
4. If the camera works, turn on `Calibration Mode`
5. Show the four calibration corner markers to the camera:
   - `100` = top-left
   - `101` = top-right
   - `102` = bottom-left
   - `103` = bottom-right
6. Once all four are visible, calibration is stored in `sandcam-settings.json`

Printable marker files are included in [calibration/](c:/Git/sandcam/calibration):

- [TL-100.svg](c:/Git/sandcam/calibration/TL-100.svg)
- [TR-101.svg](c:/Git/sandcam/calibration/TR-101.svg)
- [BL-102.svg](c:/Git/sandcam/calibration/BL-102.svg)
- [BR-103.svg](c:/Git/sandcam/calibration/BR-103.svg)

### What you should see during calibration

- before calibration is solved, the app shows a semi-transparent `Camera Preview`
- once the four corner markers are found, it switches to a homography-warped
  `Calibration View`
- if `Vision Debug` is enabled, mapped marker/object positions are drawn over
  the scene

The overlay is only shown during `Calibration Mode` or `Vision Debug`, so the
normal sandbox view stays clean during regular use.

### Calibration marker IDs

The calibration markers are ArUco tags with these fixed IDs:

| Marker ID | Corner |
|---|---|
| `100` | Top-left |
| `101` | Top-right |
| `102` | Bottom-left |
| `103` | Bottom-right |

These are separate from the object interaction markers below.

### Object marker IDs

The first version uses fixed marker IDs for object types:

| Marker ID | Object |
|---|---|
| `1` | Boat |
| `2` | Dinosaur toy |
| `3` | House |
| `4` | Tree |
| `5` | Volcano |

### Notes

- Vision imports are lazy at runtime, so the sandbox can still run with vision
  disabled.
- Calibration currently uses ArUco corner tags rather than manual point
  clicking.
- The webcam overlay only appears in `Calibration Mode` or `Vision Debug`.
- `Vision Debug` draws mapped object positions over the sandbox output.
- `Object Reactions` can be turned off while leaving the camera/debug path on.

---

## Windows setup (Kinect v1)

### Prerequisites

- Windows 10 / 11
- Python 3.13+ (via [python.org](https://www.python.org/))
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Git

`freenect.dll` and `libusb-1.0.dll` are included in the repo — no C compiler
or CMake needed.

### 1 — Clone and install Python deps

```powershell
git clone https://github.com/yourname/sandcam
cd sandcam
uv sync
```

### 2 — Replace the Kinect USB driver (Zadig)

The Kinect ships with an Xbox HID driver.  libfreenect needs WinUSB instead.

1. Download [Zadig](https://zadig.akeo.ie/) and run it
2. Plug in the Kinect (and its power adapter if using an Xbox 360 unit)
3. In Zadig: **Options → List All Devices**, select the Kinect Camera device
4. Set the target driver to **WinUSB**, click **Replace Driver**

> ⚠️ This replaces the driver for this USB device only.  If you later install
> the official Kinect SDK 1.8, you would need to re-run Zadig.

### 3 — Run with hardware

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

### Rebuilding the DLLs (optional)

Only needed if you want to recompile libfreenect (e.g. newer version or
different architecture).  Requires Visual Studio Build Tools and CMake:

```powershell
.\setup-windows.ps1
```

Or manually — see the comments inside [setup-windows.ps1](setup-windows.ps1).

---

## Linux setup (Kinect v1)

On Linux, libfreenect is available via the system package manager.

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

No Zadig step needed — the Linux kernel's generic USB driver works directly
with libfreenect.

---

## Project structure

```
sandcam/
├── main.py             Game loop, input handling, HUD
├── depth_source.py     DepthSource ABC, MouseSimulator, KinectV1Source
├── ai_guide.py         Optional guide logic and optional LLM narration
├── webcam_observer.py  Optional webcam capture, marker tracking, calibration
├── interaction_engine.py Local object-to-world interaction rules
├── renderer.py         Elevation colourmap, hillshading, contour lines
├── ui.py               Sidebar settings, overlays, persisted config
├── pyproject.toml      uv-managed dependencies
├── setup-windows.ps1   Automates building libfreenect from source (optional)
├── freenect.dll        libfreenect Windows runtime (pre-built, included)
└── libusb-1.0.dll      libusb Windows runtime (pre-built, included)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pygame` | Window, event loop, pixel blitting |
| `numpy` | Height map arithmetic |
| `scipy` | Gaussian smoothing, frame resize |
| `opencv-contrib-python` | Optional webcam capture + ArUco marker detection |
| libfreenect (native) | Kinect v1 USB driver + depth stream |
