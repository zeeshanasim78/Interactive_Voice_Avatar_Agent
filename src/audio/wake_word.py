"""Wake trigger for starting a recording (module 5.1 / Phase 1).

Per NAB_AI_System_Design.md 5.1: push-to-talk is the recommended primary
trigger for a kiosk ("the safest, most robust option"), with `openWakeWord`
as an optional bonus layered on later. `PushToTalkTrigger` is the primary
implementation; `OpenWakeWordTrigger` is provided but only usable if the
optional `openwakeword` package is installed.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PushToTalkTrigger:
    """Simplest, most robust wake trigger: an explicit press event.

    A UI (or, for now, a test harness) calls `press()` when the user activates
    push-to-talk. `on_wake` fires synchronously so callers can flip an
    avatar/UI state within the ~300ms acceptance window (5.1 acceptance
    criteria); the avatar state change itself is wired up in a later phase.
    """

    def __init__(self, on_wake: Optional[Callable[[], None]] = None):
        self.on_wake = on_wake

    def press(self) -> float:
        """Activates the trigger. Returns the monotonic timestamp of the
        wake event, so callers can measure wake-to-listening latency."""
        wake_time = time.monotonic()
        logger.info("Push-to-talk activated")
        if self.on_wake:
            self.on_wake()
        return wake_time


class OpenWakeWordTrigger:
    """Optional continuous wake-phrase listener ("Hey NAB").

    Requires the `openwakeword` package, which is not part of the core
    install (see NAB_AI_System_Design.md Section 4) — push-to-talk remains
    the primary trigger regardless of whether this is enabled.
    """

    def __init__(
        self,
        on_wake: Callable[[], None],
        sample_rate: int = 16000,
        threshold: float = 0.5,
        inference_frames: int = 1280,
    ):
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is not installed; push-to-talk remains the "
                "primary trigger. Install the optional `openwakeword` "
                "package to enable wake-phrase detection."
            ) from exc

        self.on_wake = on_wake
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.inference_frames = inference_frames
        self._model = Model()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        import sounddevice as sd

        self._stop_event.clear()

        def _run():
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.inference_frames,
            ) as stream:
                while not self._stop_event.is_set():
                    frame, _ = stream.read(self.inference_frames)
                    scores = self._model.predict(frame[:, 0])
                    if any(score >= self.threshold for score in scores.values()):
                        logger.info("Wake word detected")
                        self.on_wake()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
