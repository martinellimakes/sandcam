"""
Persistent store of user-trained custom objects for CV recognition.

Saved to cv-custom-objects.json alongside the other settings files.

Two API helpers are provided:
  identify_object  — training: ask the model what a captured object is
  match_object     — runtime:  ask the model which custom object a crop contains
"""
from __future__ import annotations

import base64
import json
import pathlib
from dataclasses import asdict, dataclass
from urllib import request as urllib_request

STORE_PATH = pathlib.Path(__file__).with_name("cv-custom-objects.json")

_IDENTIFY_SYSTEM = (
    "You are identifying a specific physical toy or object placed in an "
    "augmented-reality sandbox so the user can track it by name. "
    "Give it a short, memorable label (1-3 words) and a brief visual description "
    "that will help recognise it from other angles later. "
    'Return compact JSON only: {"label":"red toy car","description":"small red plastic racing car with yellow wheels"}'
)

_MATCH_SYSTEM_TMPL = (
    "You are identifying an object in an AR sandbox image against a list of known objects. "
    "Known objects:\n{known}\n"
    "If the image clearly shows one of these known objects, return its exact label and matched=true. "
    "Otherwise return the best generic label and matched=false. "
    'Return JSON only: {{"label":"...","matched":true}}'
)


@dataclass
class CustomObject:
    label: str
    description: str
    thumbnail_b64: str = ""  # base64 JPEG crop, may be empty


def load_objects() -> list[CustomObject]:
    if not STORE_PATH.exists():
        return []
    try:
        data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        return [CustomObject(**item) for item in data]
    except Exception:
        return []


def save_objects(objects: list[CustomObject]) -> None:
    STORE_PATH.write_text(
        json.dumps([asdict(o) for o in objects], indent=2),
        encoding="utf-8",
    )


def _vision_call(
    frame_b64: str,
    system: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> dict:
    payload = json.dumps({
        "model": model,
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                {"type": "text", "text": "Identify this object."},
            ]},
        ],
    }).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers=headers,
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"].strip()
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1 or e <= 0:
        return {}
    return json.loads(text[s:e])


def identify_object(
    frame_b64: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout: float = 12.0,
) -> dict[str, str]:
    """
    Training helper.  Ask the model to identify and label an object from a
    frame or crop.  Returns {label, description} on success, {error} on failure.
    """
    try:
        result = _vision_call(frame_b64, _IDENTIFY_SYSTEM, base_url, model, api_key, timeout)
        if not result.get("label"):
            return {"error": "Model did not return a label."}
        return result
    except Exception as exc:
        return {"error": str(exc)}


def match_object(
    frame_b64: str,
    objects: list[CustomObject],
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout: float = 8.0,
) -> str | None:
    """
    Runtime helper for YOLO tracks.  Ask the model which (if any) custom object
    is visible in a crop.  Returns the matched label, or None if no match.
    """
    if not objects:
        return None
    known = "\n".join(f'- "{o.label}": {o.description}' for o in objects)
    system = _MATCH_SYSTEM_TMPL.format(known=known)
    try:
        result = _vision_call(frame_b64, system, base_url, model, api_key, timeout)
        if result.get("matched") and result.get("label"):
            return str(result["label"])
    except Exception:
        pass
    return None
