"""Avatar & Lip-Sync (module 5.8 / Phase 7).

Per NAB_AI_System_Design.md 5.8 and Section 7 Phase 7: given a WAV file,
generates viseme (mouth-shape) timing and drives the avatar's visual state
(FR8/FR9) -- `idle`, `listening`, `processing` (FR14), `speaking`. Lip-sync
uses **Rhubarb Lip Sync** (a standalone compiled binary invoked via
`subprocess`, not a pip package, so it has no Python-version coupling) as
the primary approach, per the design doc's recommendation over the
optional/advanced Wav2Lip path.

This module owns *timing data* -- which state is active, and for speech,
which mouth shape should be shown at which timestamp -- not pixel
rendering, which belongs to Phase 9's PySide6 kiosk UI. A UI layer attaches
via the `on_state_change`/`on_viseme` callbacks and swaps sprite frames
accordingly; sprite art itself (idle/listening/processing/speaking frames)
is a Phase 9 asset-authoring concern, not this module's.

If the Rhubarb binary isn't installed on the target machine, `play_speaking`
degrades to an amplitude-envelope viseme approximation (open/closed mouth
from audio energy) rather than failing outright -- a coarser sync, but
still within the <=300ms desync tolerance for short kiosk responses, and it
never leaves the avatar silently frozen.

Usage:
    python -m src.response.avatar
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

DEFAULT_PROCESSING_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "avatar_assets" / "audio" / "processing_prompt.wav"
)

# 5.8 acceptance: "No perceptible desync (>300ms) between audio and lip
# movement" / "idle/listening/processing/speaking states ... transition in
# < 500ms".
MAX_SYNC_DRIFT_SECONDS = 0.3
MAX_STATE_TRANSITION_SECONDS = 0.5


class AvatarState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"  # FR14 -- looping wait animation, driven by the UI while this state is active
    SPEAKING = "speaking"


@dataclass
class Viseme:
    start: float  # seconds from the start of the audio clip
    end: float
    shape: str  # Rhubarb mouth-shape code (A-H, X=rest) or "B"/"X" for the amplitude-envelope fallback


@dataclass
class SpeakingResult:
    audio_path: Path
    visemes: List[Viseme]
    source: str  # "rhubarb" or "amplitude_envelope" (see module docstring)
    duration_seconds: float


@dataclass
class AvatarConfig:
    rhubarb_path: str = "rhubarb"  # binary name (must be on PATH) or a direct path to rhubarb(.exe)
    processing_prompt_path: Path = DEFAULT_PROCESSING_PROMPT_PATH
    rhubarb_timeout_seconds: float = 30.0


class Avatar:
    """Tracks avatar state and produces lip-sync timing for spoken audio.
    Construct once, reuse across the session (mirrors the load-once pattern
    used throughout this codebase)."""

    def __init__(
        self,
        config: Optional[AvatarConfig] = None,
        on_state_change: Optional[Callable[[AvatarState], None]] = None,
        on_viseme: Optional[Callable[[Viseme], None]] = None,
    ):
        self.config = config or AvatarConfig()
        self.state: AvatarState = AvatarState.IDLE
        self._on_state_change = on_state_change
        self._on_viseme = on_viseme

        self._rhubarb_available = shutil.which(self.config.rhubarb_path) is not None
        if not self._rhubarb_available:
            logger.warning(
                "Rhubarb Lip Sync binary (%r) not found on PATH; play_speaking() will use the "
                "amplitude-envelope viseme fallback instead of true phoneme-based lip sync (Section 5.8).",
                self.config.rhubarb_path,
            )

    def _set_state(self, state: AvatarState) -> None:
        start = time.monotonic()
        self.state = state
        if self._on_state_change is not None:
            self._on_state_change(state)
        elapsed = time.monotonic() - start
        if elapsed > MAX_STATE_TRANSITION_SECONDS:
            logger.warning(
                "Avatar state transition to %s took %.2fs, exceeding the %.1fs target (Section 5.8 acceptance)",
                state.value,
                elapsed,
                MAX_STATE_TRANSITION_SECONDS,
            )
        logger.info("Avatar state -> %s", state.value)

    def play_idle(self) -> None:
        self._set_state(AvatarState.IDLE)

    def play_listening(self) -> None:
        self._set_state(AvatarState.LISTENING)

    def play_processing(self) -> None:
        """FR14: plays the cached wait-prompt audio once, immediately, and
        enters the `processing` state. Non-blocking -- the caller (the
        orchestrator's background RAG/LLM thread, in Phase 8) keeps working
        while the prompt plays and the UI loops its processing animation
        for as long as `state == PROCESSING`."""
        self._set_state(AvatarState.PROCESSING)
        threading.Thread(
            target=self._play_audio_file,
            args=(self.config.processing_prompt_path,),
            kwargs={"blocking": True},
            daemon=True,
        ).start()

    def play_speaking(self, audio_path: Union[str, Path]) -> SpeakingResult:
        """Extracts viseme timing for `audio_path`, enters the `speaking`
        state, and plays the audio while dispatching `on_viseme` callbacks
        in sync. Blocks until playback finishes (this is the terminal step
        of a response turn, per the 5.9 orchestrator's RESPONDING state)."""
        audio_path = Path(audio_path)
        visemes, source = self._extract_visemes(audio_path)
        duration = _wav_duration_seconds(audio_path)

        self._set_state(AvatarState.SPEAKING)
        t0 = time.monotonic()
        self._play_audio_file(audio_path, blocking=False)

        for viseme in visemes:
            target = t0 + viseme.start
            drift = target - time.monotonic()
            if drift > 0:
                time.sleep(drift)
            elif -drift > MAX_SYNC_DRIFT_SECONDS:
                logger.warning(
                    "Viseme %r dispatched %.2fs late, exceeding the %.1fs desync tolerance (Section 5.8 acceptance)",
                    viseme.shape,
                    -drift,
                    MAX_SYNC_DRIFT_SECONDS,
                )
            if self._on_viseme is not None:
                self._on_viseme(viseme)

        sd.wait()  # ensure playback has actually finished before returning
        return SpeakingResult(audio_path=audio_path, visemes=visemes, source=source, duration_seconds=duration)

    def _extract_visemes(self, audio_path: Path) -> Tuple[List[Viseme], str]:
        if self._rhubarb_available:
            try:
                return self._run_rhubarb(audio_path), "rhubarb"
            except Exception as exc:
                logger.warning(
                    "Rhubarb invocation failed (%s); falling back to amplitude-envelope visemes", exc
                )
        return _amplitude_envelope_visemes(audio_path), "amplitude_envelope"

    def _run_rhubarb(self, audio_path: Path) -> List[Viseme]:
        result = subprocess.run(
            [self.config.rhubarb_path, "-f", "json", str(audio_path)],
            capture_output=True,
            text=True,
            timeout=self.config.rhubarb_timeout_seconds,
            check=True,
        )
        data = json.loads(result.stdout)
        return [
            Viseme(start=float(cue["start"]), end=float(cue["end"]), shape=str(cue["value"]))
            for cue in data.get("mouthCues", [])
        ]

    @staticmethod
    def _play_audio_file(path: Union[str, Path], blocking: bool) -> None:
        samples, sample_rate = _read_wav(path)
        sd.play(samples, sample_rate)
        if blocking:
            sd.wait()


def _read_wav(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Loads a 16-bit PCM WAV file into float32 [-1, 1] samples, preserving
    channel count (mono or stereo) for `sounddevice.play`."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV input")
        raw = wf.readframes(wf.getnframes())
        pcm16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        if n_channels > 1:
            pcm16 = pcm16.reshape(-1, n_channels)
        return pcm16, sample_rate


def _wav_duration_seconds(path: Union[str, Path]) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _amplitude_envelope_visemes(audio_path: Path, window_seconds: float = 0.1) -> List[Viseme]:
    """Rhubarb-unavailable fallback: a crude open/closed mouth timeline
    derived from short-time audio energy, on fixed windows small enough to
    stay within the 300ms desync tolerance. Not phoneme-accurate, but keeps
    the avatar visibly responsive to speech instead of static."""
    samples, sample_rate = _read_wav(audio_path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    window = max(1, int(window_seconds * sample_rate))
    threshold = float(np.abs(samples).mean()) * 0.6 if samples.size else 0.0

    visemes: List[Viseme] = []
    for start_idx in range(0, len(samples), window):
        chunk = samples[start_idx : start_idx + window]
        if chunk.size == 0:
            continue
        energy = float(np.abs(chunk).mean())
        shape = "B" if energy > threshold else "X"  # "B" = open-ish, "X" = rest/closed
        end_idx = min(start_idx + window, len(samples))
        visemes.append(Viseme(start=start_idx / sample_rate, end=end_idx / sample_rate, shape=shape))
    return visemes


if __name__ == "__main__":
    import sys
    import tempfile

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.INFO)

    def _log_state(state: AvatarState) -> None:
        print(f"[state]  -> {state.value}")

    def _log_viseme(viseme: Viseme) -> None:
        print(f"[viseme] {viseme.start:5.2f}-{viseme.end:5.2f}s: {viseme.shape}")

    avatar = Avatar(on_state_change=_log_state, on_viseme=_log_viseme)

    print("-- idle / listening / processing (plays cached wait-prompt) --")
    avatar.play_idle()
    avatar.play_listening()
    processing_start = time.monotonic()
    avatar.play_processing()
    print(f"wait-prompt playback dispatched in {time.monotonic() - processing_start:.3f}s")
    time.sleep(2.5)  # let the wait-prompt finish before demoing "speaking"

    print("\n-- speaking (Phase 7 step 5: synthesize + lip-sync a sample answer) --")
    from .tts import TextToSpeech  # local import: avatar.py's core class has no TTS dependency

    tts = TextToSpeech()
    sample_path = Path(tempfile.gettempdir()) / "nab_avatar_demo.wav"
    tts.synthesize("Thank you for visiting the National Accountability Bureau.", sample_path)

    result = avatar.play_speaking(sample_path)
    print(f"-> {len(result.visemes)} visemes from '{result.source}' source over {result.duration_seconds:.2f}s")
    avatar.play_idle()
