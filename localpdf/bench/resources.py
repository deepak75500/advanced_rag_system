"""Peak RAM / VRAM measurement. RAM is sampled from this process's RSS in a background thread."""
from __future__ import annotations

import threading
import time


class ResourceMonitor:
    """with ResourceMonitor() as m: ...  ->  m.peak_ram_mb, m.baseline_ram_mb, m.peak_vram_mb (None without CUDA)."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak_ram_mb = 0.0
        self.baseline_ram_mb = 0.0
        self.peak_vram_mb: float | None = None
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._proc = None
        self._torch = None

    def _rss(self) -> float:
        return self._proc.memory_info().rss / 1e6 if self._proc else 0.0

    def __enter__(self):
        try:
            import psutil

            self._proc = psutil.Process()
        except ImportError:
            self._proc = None
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._torch = None
        self.baseline_ram_mb = self.peak_ram_mb = self._rss()

        def loop():
            while not self._stop.wait(self.interval):
                self.peak_ram_mb = max(self.peak_ram_mb, self._rss())

        if self._proc:
            self._th = threading.Thread(target=loop, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1)
        self.peak_ram_mb = max(self.peak_ram_mb, self._rss())
        if self._torch is not None:
            self.peak_vram_mb = self._torch.cuda.max_memory_allocated() / 1e6
        return False

    def as_dict(self) -> dict:
        return {"peak_ram_mb": round(self.peak_ram_mb, 1), "baseline_ram_mb": round(self.baseline_ram_mb, 1),
                "peak_vram_mb": None if self.peak_vram_mb is None else round(self.peak_vram_mb, 1)}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False
