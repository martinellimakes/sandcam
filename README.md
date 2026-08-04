# sandcam — AR Sandbox POC

A real-time augmented-reality sandbox visualiser.  A depth camera (Kinect v1)
scans a physical sandbox; the app renders a topographic elevation map with
contour lines, hillshading, and colour-coded terrain that updates live.

A **mouse-sculpting simulator** is included so you can develop and test without
any hardware.

The app also now supports an **optional webcam vision layer** for tagged toy
interactions, plus an optional AI guide.  Both systems are independently
disableable, so the project can always run as a plain AR sandbox.

For local AI/CV setups, the app now assumes Docker Model Runner's
OpenAI-compatible endpoint at `http://localhost:12434/engines/v1`.

New to the project: start with [FIRST-TIME-SETUP.md](c:/Git/sandcam/FIRST-TIME-SETUP.md).

If you only want to configure and test model settings without launching the
sandbox, run:

```powershell
uv run python llm_setup_app.py
```

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

The default depth source is the mouse simulator. Confirm it in the settings
sidebar (`Tab` → Terrain → **Simulator`), then run:

```bash
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

Calibration and object markers use OpenCV `DICT_4X4_250` so IDs `100`–`103`
are valid.

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

Open the settings sidebar with `Tab` and choose **Kinect** under Terrain.
If the device fails to open, the app falls back to the simulator and shows a
status message.

Tune the depth range to your physical rig (camera height above sandbox) under
Terrain Debug (enable Debug Mode first):

- Min ≈ camera-to-peak-of-sand distance (mm)
- Max ≈ camera-to-bare-sandbox-floor distance (mm)

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

---

## LLM and CV Requests

The app talks to local or remote models using an OpenAI-compatible
`chat/completions` request shape.

Important: the configured `base_url` should be the API root, not the final
endpoint path. The code appends `/chat/completions` itself.

Examples:

- Docker Model Runner: `http://localhost:12434/engines/v1`
- other local OpenAI-compatible servers: their equivalent API root

Do not set the base URL to something that already ends in
`/chat/completions` or `/completions`.

### Guide LLM request

When the optional guide narrator is enabled, the app sends a text-only chat
request containing the current guide message, active challenge text, and a
small world-state summary.

Example shape:

```json
{
  "model": "ai/qwen3-vl:2B-UD-Q4_K_XL",
  "temperature": 0.7,
  "messages": [
    {
      "role": "system",
      "content": "You are a kid-friendly narrator for an AR sandbox..."
    },
    {
      "role": "user",
      "content": "{\"message\":{\"kind\":\"observation\",\"title\":\"Large Lake\",\"body\":\"A wide lake has formed.\"},\"challenge_text\":\"\",\"world_state\":{\"water_ratio\":0.32,\"land_ratio\":0.68,\"coastline_ratio\":0.15,\"highest_peak\":0.81,\"islands\":0,\"lakes\":1,\"features\":[\"large_lake\",\"mountain_range\"],\"shark_count\":1,\"dinosaur_count\":0},\"verbosity\":\"normal\"}"
    }
  ]
}
```

### CV detection request

When `vision.detection.backend = openai_vision`, the app sends the current
camera frame as a base64 `image_url` plus a short detection instruction.

Example shape:

```json
{
  "model": "ai/qwen3-vl:2B-UD-Q4_K_XL",
  "max_tokens": 512,
  "messages": [
    {
      "role": "system",
      "content": "You are a vision detector for an AR sandbox..."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        },
        {
          "type": "text",
          "text": "Detect all objects."
        }
      ]
    }
  ]
}
```

The detector expects a JSON-array-style response describing objects with
`label`, `confidence`, and `bbox`.

### CV training request

When `Training Mode` is on and you click `Capture Object`, the app sends an
image request to the configured CV reasoner model asking it to identify one
object and return compact JSON.

Example shape:

```json
{
  "model": "ai/qwen3-vl:2B-UD-Q4_K_XL",
  "max_tokens": 200,
  "messages": [
    {
      "role": "system",
      "content": "You are identifying a specific physical toy or object placed in an augmented-reality sandbox so the user can track it by name..."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        },
        {
          "type": "text",
          "text": "Identify this object."
        }
      ]
    }
  ]
}
```

Expected response shape:

```json
{
  "label": "red toy car",
  "description": "small red plastic racing car with yellow wheels"
}
```

### Using YOLO with a multimodal reasoner

The most practical setup is still:

- `YOLO` for live detection and tracking
- a local or remote multimodal model for:
  - object naming during training
  - relabeling known custom objects
  - optional guide wording

This keeps the live path responsive while still letting a multimodal model
interpret images when needed.

---

## Asset attribution

The shark sprite sheets used by this project is based on:

- `Shark Sprites - animated 4-directional` by [Sevarihk](https://opengameart.org/users/sevarihk)
- Source: [OpenGameArt](https://opengameart.org/content/shark-sprites-animated-4-directional)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Per the attribution request on the asset page, please keep credit to the
original author and link back to the source page or the author's homepage when
redistributing the shark asset or edited versions of it.

---
