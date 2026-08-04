"""
Standalone configuration utility for model setup and testing.

This tool edits the guide narrator and CV reasoner settings stored in
sandcam-settings.json without launching the full sandbox application.
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from ai_guide import ProviderConfig, test_provider_connection
from ui import Config, DEFAULT_LOCAL_BASE_URL


WINDOW_TITLE = "Sandcam Model Setup"


def _pretty_json(payload: dict) -> str:
    return json.dumps(payload, indent=2)


class ModelSetupApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1040x820")
        self.root.minsize(920, 720)

        self.config = Config.load()
        self._guide_test_thread: threading.Thread | None = None
        self._cv_test_thread: threading.Thread | None = None

        self._build_vars()
        self._build_ui()
        self._load_from_config()
        self._refresh_preview()

    def _build_vars(self) -> None:
        self.guide_enabled_var = tk.BooleanVar()
        self.guide_llm_enabled_var = tk.BooleanVar()
        self.guide_provider_var = tk.StringVar()
        self.guide_base_url_var = tk.StringVar()
        self.guide_model_var = tk.StringVar()
        self.guide_timeout_var = tk.StringVar()
        self.guide_temp_api_key_var = tk.StringVar()

        self.cv_detection_enabled_var = tk.BooleanVar()
        self.cv_provider_var = tk.StringVar()
        self.cv_base_url_var = tk.StringVar()
        self.cv_model_var = tk.StringVar()
        self.cv_api_key_var = tk.StringVar()

        for var in (
            self.guide_enabled_var,
            self.guide_llm_enabled_var,
            self.guide_provider_var,
            self.guide_base_url_var,
            self.guide_model_var,
            self.guide_timeout_var,
            self.cv_detection_enabled_var,
            self.cv_provider_var,
            self.cv_base_url_var,
            self.cv_model_var,
            self.cv_api_key_var,
        ):
            var.trace_add("write", lambda *_: self._refresh_preview())

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Guide and Vision Model Setup",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=(
                "Configure local or remote OpenAI-compatible models, test connections, "
                "and save the grouped sandcam settings file."
            ),
            foreground="#5a6475",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        left.columnconfigure(0, weight=1)
        body.add(left, weight=3)

        right = ttk.Frame(body)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        body.add(right, weight=2)

        self._build_guide_group(left).grid(row=0, column=0, sticky="ew")
        self._build_cv_group(left).grid(row=1, column=0, sticky="ew", pady=(12, 0))
        self._build_actions(left).grid(row=2, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(
            right,
            text="Request Preview",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.preview_text = ScrolledText(right, wrap="word", font=("Consolas", 10))
        self.preview_text.grid(row=1, column=0, sticky="nsew", pady=(8, 10))
        self.preview_text.configure(state="disabled")

        ttk.Label(
            right,
            text=(
                "Use the API root as the base URL. The app appends "
                "`/chat/completions` automatically."
            ),
            foreground="#5a6475",
        ).grid(row=2, column=0, sticky="w")

    def _build_guide_group(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Guide Narrator", padding=12)
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(frame, text="Guide features enabled", variable=self.guide_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Checkbutton(frame, text="LLM wording enabled", variable=self.guide_llm_enabled_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 8)
        )

        ttk.Label(frame, text="Provider").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.guide_provider_var,
            state="readonly",
            values=("local", "remote"),
        ).grid(row=2, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Base URL").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.guide_base_url_var).grid(row=3, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Model").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.guide_model_var).grid(row=4, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Timeout (seconds)").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.guide_timeout_var).grid(row=5, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Temp API key").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.guide_temp_api_key_var, show="*").grid(row=6, column=1, sticky="ew", pady=3)

        ttk.Label(
            frame,
            text="Guide API key is not saved to settings.json. It is used only for this tool's connection test.",
            foreground="#5a6475",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 8))

        button_row = ttk.Frame(frame)
        button_row.grid(row=8, column=0, columnspan=2, sticky="ew")
        button_row.columnconfigure(2, weight=1)
        ttk.Button(button_row, text="Use Docker Local Default", command=self._guide_defaults).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Test Guide Connection", command=self._start_guide_test).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.guide_status = tk.Text(frame, height=3, wrap="word", relief="flat", bg=self.root.cget("bg"))
        self.guide_status.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.guide_status.configure(state="disabled")
        return frame

    def _build_cv_group(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="CV Reasoner", padding=12)
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(frame, text="Vision detection enabled", variable=self.cv_detection_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        ttk.Label(frame, text="Provider").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.cv_provider_var,
            state="readonly",
            values=("local", "remote"),
        ).grid(row=1, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Base URL").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.cv_base_url_var).grid(row=2, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="Model").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.cv_model_var).grid(row=3, column=1, sticky="ew", pady=3)

        ttk.Label(frame, text="API key").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=self.cv_api_key_var, show="*").grid(row=4, column=1, sticky="ew", pady=3)

        ttk.Label(
            frame,
            text="This model is used for vision-style reasoning, object training, and relabeling.",
            foreground="#5a6475",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 8))

        button_row = ttk.Frame(frame)
        button_row.grid(row=6, column=0, columnspan=2, sticky="ew")
        button_row.columnconfigure(2, weight=1)
        ttk.Button(button_row, text="Use Docker Local Default", command=self._cv_defaults).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Test CV Reasoner", command=self._start_cv_test).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.cv_status = tk.Text(frame, height=3, wrap="word", relief="flat", bg=self.root.cget("bg"))
        self.cv_status.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.cv_status.configure(state="disabled")
        return frame

    def _build_actions(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Actions", padding=12)
        frame.columnconfigure(0, weight=1)

        row = ttk.Frame(frame)
        row.grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="Reload From File", command=self._reload).pack(side="left")
        ttk.Button(row, text="Save Settings", command=self._save).pack(side="left", padx=(8, 0))
        ttk.Button(row, text="Save and Close", command=self._save_and_close).pack(side="left", padx=(8, 0))

        ttk.Label(
            frame,
            text=(
                "This utility edits the saved guide and CV reasoner settings only. "
                "It does not launch the sandbox, camera, or Kinect."
            ),
            foreground="#5a6475",
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        return frame

    def _load_from_config(self) -> None:
        self.guide_enabled_var.set(self.config.ai_enabled)
        self.guide_llm_enabled_var.set(self.config.llm_enabled)
        self.guide_provider_var.set(self.config.llm_provider_location)
        self.guide_base_url_var.set(self.config.llm_base_url or DEFAULT_LOCAL_BASE_URL)
        self.guide_model_var.set(self.config.llm_model)
        self.guide_timeout_var.set(str(self.config.llm_timeout_seconds))
        self.guide_temp_api_key_var.set("")

        self.cv_detection_enabled_var.set(self.config.cv_detection_enabled)
        self.cv_provider_var.set(self.config.cv_reasoner_location)
        self.cv_base_url_var.set(self.config.cv_detection_api_url or self.config.llm_base_url or DEFAULT_LOCAL_BASE_URL)
        self.cv_model_var.set(self.config.cv_detection_api_model or self.config.llm_model)
        self.cv_api_key_var.set(self.config.cv_detection_api_key)

        self._set_status(self.guide_status, "")
        self._set_status(self.cv_status, "")

    def _apply_to_config(self) -> None:
        self.config.ai_enabled = bool(self.guide_enabled_var.get())
        self.config.guide_enabled = bool(self.guide_enabled_var.get())
        self.config.llm_enabled = bool(self.guide_llm_enabled_var.get())
        self.config.llm_provider_location = self.guide_provider_var.get().strip() or "local"
        self.config.llm_base_url = self.guide_base_url_var.get().strip() or DEFAULT_LOCAL_BASE_URL
        self.config.llm_model = self.guide_model_var.get().strip()
        try:
            self.config.llm_timeout_seconds = max(0.5, float(self.guide_timeout_var.get().strip() or "2.0"))
        except ValueError:
            self.config.llm_timeout_seconds = 2.0

        self.config.cv_detection_enabled = bool(self.cv_detection_enabled_var.get())
        self.config.cv_reasoner_location = self.cv_provider_var.get().strip() or "local"
        self.config.cv_detection_api_url = self.cv_base_url_var.get().strip()
        self.config.cv_detection_api_model = self.cv_model_var.get().strip()
        self.config.cv_detection_api_key = self.cv_api_key_var.get().strip()

    def _guide_defaults(self) -> None:
        self.guide_provider_var.set("local")
        self.guide_base_url_var.set(DEFAULT_LOCAL_BASE_URL)

    def _cv_defaults(self) -> None:
        self.cv_provider_var.set("local")
        self.cv_base_url_var.set(DEFAULT_LOCAL_BASE_URL)

    def _reload(self) -> None:
        self.config = Config.load()
        self._load_from_config()
        self._refresh_preview()

    def _save(self) -> None:
        self._apply_to_config()
        self.config.save()
        self._set_status(self.guide_status, "Settings saved.", "ok")
        self._set_status(self.cv_status, "Settings saved.", "ok")
        self._refresh_preview()

    def _save_and_close(self) -> None:
        self._save()
        self.root.after(120, self.root.destroy)

    def _start_guide_test(self) -> None:
        if self._guide_test_thread is not None and self._guide_test_thread.is_alive():
            self._set_status(self.guide_status, "A guide connection test is already running.", "warn")
            return

        self._set_status(self.guide_status, "Testing guide narrator connection...", "info")
        config = self._guide_provider_config()
        self._guide_test_thread = threading.Thread(
            target=self._run_test,
            args=(config, self.guide_status, "guide"),
            daemon=True,
        )
        self._guide_test_thread.start()

    def _start_cv_test(self) -> None:
        if self._cv_test_thread is not None and self._cv_test_thread.is_alive():
            self._set_status(self.cv_status, "A CV reasoner test is already running.", "warn")
            return

        self._set_status(self.cv_status, "Testing CV reasoner connection...", "info")
        config = self._cv_provider_config()
        self._cv_test_thread = threading.Thread(
            target=self._run_test,
            args=(config, self.cv_status, "cv"),
            daemon=True,
        )
        self._cv_test_thread.start()

    def _run_test(self, config: ProviderConfig, widget: tk.Text, _kind: str) -> None:
        try:
            result = test_provider_connection(config)
            level = "ok" if result.ok else "error"
            message = result.summary
        except Exception as exc:
            level = "error"
            message = f"Connection test failed: {exc}"
        self.root.after(0, lambda: self._set_status(widget, message, level))

    def _guide_provider_config(self) -> ProviderConfig:
        backend = (
            "local_openai_compatible"
            if (self.guide_provider_var.get().strip() or "local") == "local"
            else "cloud_openai_compatible"
        )
        try:
            timeout = max(0.5, float(self.guide_timeout_var.get().strip() or "2.0"))
        except ValueError:
            timeout = 2.0
        return ProviderConfig(
            backend=backend,
            base_url=self.guide_base_url_var.get().strip() or DEFAULT_LOCAL_BASE_URL,
            model=self.guide_model_var.get().strip(),
            timeout_seconds=timeout,
            api_key=self.guide_temp_api_key_var.get().strip() or None,
        )

    def _cv_provider_config(self) -> ProviderConfig:
        backend = (
            "local_openai_compatible"
            if (self.cv_provider_var.get().strip() or "local") == "local"
            else "cloud_openai_compatible"
        )
        try:
            timeout = max(0.5, float(self.guide_timeout_var.get().strip() or "2.0"))
        except ValueError:
            timeout = 2.0
        return ProviderConfig(
            backend=backend,
            base_url=self.cv_base_url_var.get().strip() or DEFAULT_LOCAL_BASE_URL,
            model=self.cv_model_var.get().strip(),
            timeout_seconds=timeout,
            api_key=self.cv_api_key_var.get().strip() or None,
        )

    def _refresh_preview(self, *_args: object) -> None:
        guide_url = (self.guide_base_url_var.get().strip() or DEFAULT_LOCAL_BASE_URL).rstrip("/") + "/chat/completions"
        cv_url = (self.cv_base_url_var.get().strip() or self.guide_base_url_var.get().strip() or DEFAULT_LOCAL_BASE_URL).rstrip("/") + "/chat/completions"

        guide_payload = {
            "model": self.guide_model_var.get().strip() or "default",
            "temperature": 0.7,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a kid-friendly narrator for an AR sandbox...",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "message": {
                                "kind": "observation",
                                "title": "Connection Check",
                                "body": "The sandbox is testing its narrator connection.",
                            },
                            "challenge_text": "",
                            "world_state": {
                                "water_ratio": 0.32,
                                "land_ratio": 0.68,
                                "coastline_ratio": 0.15,
                                "highest_peak": 0.81,
                                "islands": 0,
                                "lakes": 1,
                                "features": ["large_lake", "mountain_range"],
                                "shark_count": 1,
                                "dinosaur_count": 0,
                            },
                            "verbosity": "normal",
                        }
                    ),
                },
            ],
        }

        cv_payload = {
            "model": self.cv_model_var.get().strip() or "default",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "system",
                    "content": "You are identifying a specific physical toy or object placed in an augmented-reality sandbox...",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,...",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Identify this object.",
                        },
                    ],
                },
            ],
        }

        preview = (
            "Guide narrator request\n"
            f"POST {guide_url}\n\n"
            f"{_pretty_json(guide_payload)}\n\n"
            "CV reasoner training request\n"
            f"POST {cv_url}\n\n"
            f"{_pretty_json(cv_payload)}\n"
        )

        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", preview)
        self.preview_text.configure(state="disabled")

    def _set_status(self, widget: tk.Text, text: str, level: str = "info") -> None:
        colours = {
            "info": "#334155",
            "ok": "#166534",
            "warn": "#92400e",
            "error": "#991b1b",
        }
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.tag_configure("status", foreground=colours.get(level, "#334155"))
        widget.insert("1.0", text, ("status",))
        widget.configure(state="disabled")


def main() -> int:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("clam")
    except tk.TclError:
        pass
    app = ModelSetupApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
