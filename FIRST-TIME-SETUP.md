# First-Time Setup

This guide is for someone opening `sandcam` for the first time and wanting the
fastest path to a working sandbox.

## Start Simple

Begin with the base sandbox only:

- no webcam vision
- no AI guide
- no CV training

Get the terrain running first. Add the optional systems later.

## What You Need

Minimum:

- Python 3.13+
- `uv`
- this repo checked out locally

Optional:

- Kinect v1 for live depth input
- a normal webcam for object sensing
- a local OpenAI-compatible endpoint for AI/CV features

Default local endpoint:

- `http://localhost:12434/engines/v1`

This is Docker Model Runner's OpenAI-compatible local endpoint.

## Install Dependencies

From the project root:

```powershell
uv sync
```

If you want to configure only the guide/CV model settings without launching the
sandbox itself, use:

```powershell
uv run python llm_setup_app.py
```

## First Run

The safest first run uses the mouse simulator (the default depth source).

Open the settings sidebar with `Tab` and confirm **Simulator** is selected under
Terrain. Switch to **Kinect** only after the base app works.

Then run:

```powershell
uv run python main.py
```

## Basic Controls

- `Tab` opens the settings sidebar
- `Esc` or `Q` quits
- `C` toggles contour lines
- `G` toggles creatures
- `R` resets terrain in simulator mode

Simulator-only terrain controls:

- left-drag raises terrain
- right-drag lowers terrain
- mouse wheel changes brush size

## Recommended First-Time Workflow

### 1. Confirm the base app works

- launch the app
- open the sidebar with `Tab`
- leave `Camera Sensing` off
- leave `AI Features` off
- verify the terrain renders and the window behaves correctly

### 2. Tune the base sandbox

- choose a colour scheme you like
- pick the correct display
- enable borderless if you are using a projector or secondary screen
- turn creatures on only after the terrain is working smoothly

### 3. Add Kinect depth

Once the simulator is working, open the sidebar with `Tab` and choose **Kinect**
under Terrain. If the Kinect fails to open, the app falls back to the simulator
and shows a status message.

Then:

- start the app
- open the sidebar
- if `Debug Mode` is needed, turn it on
- tune `Terrain Debug` depth and smoothing only if necessary

Interpretation:

- closer to the camera = higher terrain
- farther from the camera = lower terrain

## Optional Camera Object Sensing

Turn on `Camera Sensing` only after the base terrain is working.

Recommended setup:

1. Turn `Camera Sensing` on.
2. Click `Scan Cameras`.
3. Choose the correct webcam from the detected list.
4. Click `Test Camera`.
5. Turn on `Calibration Mode`.
6. Show the four calibration markers:
   - `100` top-left
   - `101` top-right
   - `102` bottom-left
   - `103` bottom-right

Calibration files are in [calibration](c:/Git/sandcam/calibration).

Normal users do not need `Debug Mode` for this unless troubleshooting.

## Optional AI Guide

The AI guide is separate from camera sensing.

You can keep the sandbox completely local and offline-looking by leaving it off.

If you want the guide:

1. Turn `AI Features` on.
2. Turn `LLM Wording` on only if you want model-generated wording.
3. Use a local OpenAI-compatible endpoint such as:
   - `http://localhost:12434/engines/v1`

Recommended local setup:

- `Model Provider` = `Local`
- `Endpoint URL` = `http://localhost:12434/engines/v1`
- `LLM Model` = your local multimodal or text model name

## Optional CV Object Detection

The best current setup is:

- `Object Detection` on
- backend `YOLO`
- local vision model only for relabeling/training

Recommended values:

- `Detection Backend` = `YOLO`
- `YOLO Model` = `yolo11n.pt`
- confidence = `0.5` to start

If you are using local Qwen with YOLO:

- YOLO handles live detection
- Qwen handles object naming, relabeling, and optional guide wording

## Training an Object

The training flow uses the configured CV reasoner model, even if YOLO is your
live detector.

Recommended flow:

1. Turn `Camera Sensing` on.
2. Turn `Object Detection` on.
3. Turn `Debug Mode` on.
4. In the CV section, set:
   - `Reasoning Provider`
   - `Reasoner URL`
   - `Reasoner Model`
5. Click `Test CV Reasoner`.
6. Turn `Training Mode` on.
7. Place one object in view by itself.
8. Click `Capture Object`.
9. Review the suggested label.
10. Save the object.

Mental model:

- `YOLO Model` = fast live detector
- `Reasoner Model` = multimodal model used for training, relabeling, and
  image understanding

## What the App Sends to the Model

The app uses an OpenAI-compatible `chat/completions` payload shape.

Important:

- set the `base_url` to the API root
- do not include `/chat/completions` in the saved URL
- do not include `/completions` either unless your provider explicitly expects
  that as part of its API root

Typical local Docker Model Runner URL:

- `http://localhost:12434/engines/v1`

### Training request shape

When you click `Capture Object`, the app sends an image plus a short prompt.

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
            "url": "data:image/jpeg;base64,..."
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

Expected response:

```json
{
  "label": "red toy car",
  "description": "small red plastic racing car with yellow wheels"
}
```

### OpenAI vision detection request shape

If you use the `openai_vision` backend instead of YOLO, the app sends the full
camera frame in a similar format:

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
            "url": "data:image/jpeg;base64,..."
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

Most setups should still prefer YOLO for live detection and use the multimodal
model for training and relabeling.

## Debug Mode

`Debug Mode` is a global switch for technical settings.

Leave it off for normal use.

Turn it on only when you need:

- depth calibration and smoothing controls
- model provider and endpoint controls
- CV backend and training controls
- camera debug overlay and manual camera index controls

Turning `Debug Mode` off hides those controls again.

## Settings File

Settings are saved in [sandcam-settings.json](c:/Git/sandcam/sandcam-settings.json).

The file is grouped by feature:

- `display`
- `terrain`
- `creatures`
- `guide`
- `vision`
- `debug`

You usually should not need to edit it by hand.

## If Something Feels Wrong

Start by reducing the system back to the simplest working state:

1. turn `AI Features` off
2. turn `Camera Sensing` off
3. turn creatures off
4. confirm the terrain alone still works

Then re-enable one system at a time.

## Best First Demo Configuration

For a reliable first demo:

- simulator or Kinect working
- no AI
- no camera sensing
- contour lines on
- creatures on
- debug mode off

Once that is stable, add camera sensing. Add AI last.
