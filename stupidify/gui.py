from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .config import DamageConfig
from .core import stupidify_model


class StupidifyGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Stupidify — model vandalism laboratory")
        self.geometry("760x620")
        self.minsize(680, 560)
        self.events: queue.Queue[str] = queue.Queue()
        self._build()
        self.after(100, self._drain_events)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(10, weight=1)

        ttk.Label(root, text="Stupidify", font=("Segoe UI", 22, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(root, text="Take a perfectly respectable safetensors checkpoint and make questionable decisions to it.").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        self.model = tk.StringVar(value="openai-community/gpt2-xl")
        self.output = tk.StringVar(value=str(Path.cwd() / "stupidified-model"))
        self.method = tk.StringVar(value="goblin")
        self.severity = tk.DoubleVar(value=0.10)
        self.bits = tk.IntVar(value=4)
        self.seed = tk.IntVar(value=42)
        self.include_embeddings = tk.BooleanVar(value=False)
        self.include_norms = tk.BooleanVar(value=False)
        self.include_biases = tk.BooleanVar(value=False)

        ttk.Label(root, text="Model repo or folder").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.model).grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse…", command=self._browse_model).grid(row=2, column=2)
        ttk.Label(root, text="Output folder").grid(row=3, column=0, sticky="w", pady=8)
        ttk.Entry(root, textvariable=self.output).grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse…", command=self._browse_output).grid(row=3, column=2)
        ttk.Label(root, text="Damage method").grid(row=4, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.method, state="readonly", values=("goblin", "zero", "noise", "signflip", "bitcrush", "shuffle")).grid(row=4, column=1, sticky="ew", padx=8)
        ttk.Label(root, text="Severity").grid(row=5, column=0, sticky="w", pady=8)
        severity_frame = ttk.Frame(root)
        severity_frame.grid(row=5, column=1, sticky="ew", padx=8)
        severity_frame.columnconfigure(0, weight=1)
        ttk.Scale(severity_frame, from_=0, to=0.75, variable=self.severity).grid(row=0, column=0, sticky="ew")
        self.severity_label = ttk.Label(severity_frame, width=6)
        self.severity_label.grid(row=0, column=1, padx=(8, 0))
        self.severity.trace_add("write", lambda *_: self._update_severity())
        self._update_severity()

        knobs = ttk.Frame(root)
        knobs.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Label(knobs, text="Bit depth").pack(side="left")
        ttk.Spinbox(knobs, from_=2, to=16, textvariable=self.bits, width=5).pack(side="left", padx=(6, 20))
        ttk.Label(knobs, text="Seed").pack(side="left")
        ttk.Spinbox(knobs, from_=0, to=2_147_483_647, textvariable=self.seed, width=10).pack(side="left", padx=6)

        advanced = ttk.LabelFrame(root, text="Things Stupidify normally protects", padding=10)
        advanced.grid(row=7, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(advanced, text="also damage embeddings", variable=self.include_embeddings).pack(side="left")
        ttk.Checkbutton(advanced, text="also damage normalization", variable=self.include_norms).pack(side="left", padx=16)
        ttk.Checkbutton(advanced, text="also damage biases", variable=self.include_biases).pack(side="left")

        action = ttk.Frame(root)
        action.grid(row=8, column=0, columnspan=3, sticky="ew", pady=14)
        self.run_button = ttk.Button(action, text="STUPIDIFY IT", command=self._run)
        self.run_button.pack(side="left")
        ttk.Label(action, text="The source model is never overwritten.").pack(side="left", padx=12)
        ttk.Separator(root).grid(row=9, column=0, columnspan=3, sticky="ew")
        self.log = tk.Text(root, height=14, wrap="word", state="disabled")
        self.log.grid(row=10, column=0, columnspan=3, sticky="nsew", pady=(12, 0))

    def _update_severity(self) -> None:
        self.severity_label.config(text=f"{self.severity.get():.2f}")

    def _browse_model(self) -> None:
        path = filedialog.askdirectory(title="Choose a local Hugging Face model folder")
        if path:
            self.model.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output.set(path)

    def _write(self, text: str) -> None:
        self.events.put(text)

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "__DONE__":
                self.run_button.config(state="normal")
                continue
            self.log.config(state="normal")
            self.log.insert("end", event + "\n")
            self.log.see("end")
            self.log.config(state="disabled")
        self.after(100, self._drain_events)

    def _run(self) -> None:
        self.run_button.config(state="disabled")
        cfg = DamageConfig(method=self.method.get(), severity=float(self.severity.get()), seed=int(self.seed.get()), bits=int(self.bits.get()), include_embeddings=self.include_embeddings.get(), include_norms=self.include_norms.get(), include_biases=self.include_biases.get())

        def worker() -> None:
            try:
                stupidify_model(self.model.get(), self.output.get(), cfg, progress=self._write)
                self._write("Finished. The model is now measurably less respectable.")
            except Exception as exc:
                self._write(f"ERROR: {exc}")
            finally:
                self._write("__DONE__")

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    app = StupidifyGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
