"""Hardware detection utilities.

Probes for CUDA availability and VRAM budget.
Provides a shared device string for all model-loading code.
Also provides ResourceGuard — a background thread that aborts the pipeline
if RAM or CPU usage exceeds configured thresholds.
"""
from __future__ import annotations

import logging
import platform
import threading

logger = logging.getLogger(__name__)


class ResourceLimitExceeded(RuntimeError):
    """Raised when RAM or CPU usage breaches the configured threshold."""


class ResourceGuard:
    """Background monitor that sets an abort event when resource limits are hit.

    Usage::

        guard = ResourceGuard(max_ram_pct=90.0, max_cpu_pct=95.0, interval_s=10)
        guard.start()
        # ... between pipeline stages ...
        guard.check()   # raises ResourceLimitExceeded if limits were breached
        guard.stop()
    """

    def __init__(
        self,
        max_ram_pct: float = 90.0,
        max_cpu_pct: float = 95.0,
        interval_s: float = 10.0,
    ) -> None:
        self._max_ram_pct = max_ram_pct
        self._max_cpu_pct = max_cpu_pct
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._breach_event = threading.Event()
        self._breach_reason: str = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, daemon=True, name="resource-guard")
        self._thread.start()
        logger.debug(
            "ResourceGuard started (RAM≤%.0f%% CPU≤%.0f%% check every %.0fs)",
            self._max_ram_pct, self._max_cpu_pct, self._interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def check(self) -> None:
        """Call between pipeline stages. Raises ResourceLimitExceeded if breached."""
        if self._breach_event.is_set():
            raise ResourceLimitExceeded(self._breach_reason)

    def _monitor(self) -> None:
        import psutil
        while not self._stop_event.wait(self._interval_s):
            try:
                ram_pct = psutil.virtual_memory().percent
                # cpu_percent(interval=1) blocks for 1 s — acceptable in a background thread
                cpu_pct = psutil.cpu_percent(interval=1)

                if ram_pct > self._max_ram_pct:
                    self._breach_reason = (
                        f"RAM usage {ram_pct:.1f}% exceeded limit {self._max_ram_pct:.0f}%"
                    )
                    logger.critical("ResourceGuard: %s — aborting pipeline", self._breach_reason)
                    self._breach_event.set()
                    return

                if cpu_pct > self._max_cpu_pct:
                    self._breach_reason = (
                        f"CPU usage {cpu_pct:.1f}% exceeded limit {self._max_cpu_pct:.0f}%"
                    )
                    logger.critical("ResourceGuard: %s — aborting pipeline", self._breach_reason)
                    self._breach_event.set()
                    return

                logger.debug("ResourceGuard: RAM=%.1f%% CPU=%.1f%%", ram_pct, cpu_pct)
            except Exception as exc:
                logger.warning("ResourceGuard monitor error: %s", exc)

_device: str | None = None


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available and FORCE_CPU is not set, else 'cpu'."""
    global _device
    if _device is not None:
        return _device

    from config import settings
    if settings.force_cpu:
        _device = "cpu"
        return _device

    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
            vram = get_vram_gb()
            logger.info(
                "GPU detected: %s (%.1f GB VRAM)",
                torch.cuda.get_device_name(0),
                vram,
            )
        else:
            _device = "cpu"
            logger.info("No CUDA GPU detected — using CPU")
    except ImportError:
        _device = "cpu"
        logger.info("PyTorch not installed — using CPU")

    return _device


def get_vram_gb() -> float:
    """Return available VRAM in GB for the first CUDA device, or 0.0."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def log_hardware_summary() -> None:
    """Log a one-time hardware summary at startup."""
    try:
        import torch
        cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda else "N/A"
        vram = get_vram_gb() if cuda else 0.0
    except ImportError:
        cuda = False
        device_name = "N/A"
        vram = 0.0

    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    logger.info(
        "Hardware: OS=%s | CPU=%d cores | RAM=%.0fGB | GPU=%s | VRAM=%.1fGB",
        platform.system(),
        psutil.cpu_count(logical=False) or 0,
        ram_gb,
        device_name,
        vram,
    )
