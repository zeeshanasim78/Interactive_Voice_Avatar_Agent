"""Microphone capture with Silero VAD-based silence trimming (module 5.1 / Phase 1).

Records one utterance at a time via `sounddevice`, using `silero-vad`'s
streaming `VADIterator` to detect speech and auto-stop after a period of
trailing silence (~1.5s), then trims leading/trailing silence from the
returned buffer with a batch VAD pass.
"""

from __future__ import annotations

import logging
import queue
import time
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from silero_vad import VADIterator, get_speech_timestamps, load_silero_vad

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _vad_window_samples(sample_rate: int) -> int:
    """Silero VAD requires exactly 512 samples per chunk at 16kHz (256 at 8kHz)."""
    if sample_rate not in (8000, 16000):
        raise ValueError("silero-vad only supports 8000 or 16000 Hz sample rates")
    return 512 if sample_rate == 16000 else 256


@dataclass
class CaptureConfig:
    sample_rate: int = SAMPLE_RATE
    vad_threshold: float = 0.5
    min_silence_duration_ms: int = 100  # VADIterator's own end-of-speech smoothing
    speech_pad_ms: int = 30  # padding kept around detected speech
    trailing_silence_s: float = 1.5  # auto-stop after this much silence once speech has started
    max_utterance_s: float = 30.0  # hard safety cap so we never record forever
    pre_speech_timeout_s: float = 8.0  # give up if no speech is detected at all


class AudioCapture:
    """Records a single utterance from the microphone, auto-stopping on silence."""

    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        # Loaded once and reused across recordings — model loading is the expensive part.
        self._vad_model = load_silero_vad()

    def record_utterance(self) -> np.ndarray:
        """Blocks until an utterance is captured. Returns mono float32 samples
        in [-1, 1] at `config.sample_rate`, trimmed of leading/trailing silence.
        """
        cfg = self.config
        window_samples = _vad_window_samples(cfg.sample_rate)
        vad_iterator = VADIterator(
            self._vad_model,
            threshold=cfg.vad_threshold,
            sampling_rate=cfg.sample_rate,
            min_silence_duration_ms=cfg.min_silence_duration_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _on_audio(indata, frames, time_info, status):
            if status:
                logger.warning("sounddevice input status: %s", status)
            audio_q.put(indata[:, 0].copy())

        recorded_chunks: list[np.ndarray] = []
        speech_started = False
        in_silence_since: Optional[float] = None
        start_time = time.monotonic()

        logger.info("Listening for speech...")

        try:
            with sd.InputStream(
                samplerate=cfg.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=window_samples,
                callback=_on_audio,
            ):
                while True:
                    try:
                        chunk = audio_q.get(timeout=1.0)
                    except queue.Empty:
                        chunk = None

                    now = time.monotonic()
                    if not speech_started and (now - start_time) > cfg.pre_speech_timeout_s:
                        raise TimeoutError("No speech detected within pre_speech_timeout_s")

                    if chunk is None:
                        continue

                    recorded_chunks.append(chunk)
                    event = vad_iterator(torch.from_numpy(chunk))

                    if event and "start" in event:
                        speech_started = True
                        in_silence_since = None
                    elif event and "end" in event:
                        in_silence_since = now
                    # else: mid-speech or mid-silence, no state change

                    if (
                        speech_started
                        and in_silence_since is not None
                        and (now - in_silence_since) >= cfg.trailing_silence_s
                    ):
                        break
                    if (now - start_time) >= cfg.max_utterance_s:
                        logger.warning("Max utterance duration reached, stopping recording")
                        break
        finally:
            vad_iterator.reset_states()

        audio = np.concatenate(recorded_chunks) if recorded_chunks else np.array([], dtype=np.float32)
        return self._trim_silence(audio)

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trims leading/trailing silence using a batch VAD pass for a clean buffer."""
        if audio.size == 0:
            return audio
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio),
            self._vad_model,
            sampling_rate=self.config.sample_rate,
            threshold=self.config.vad_threshold,
        )
        if not timestamps:
            return audio
        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
        return audio[start:end]


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Saves float32 [-1, 1] mono samples to a 16-bit PCM WAV file."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    # Manual test harness for Phase 1 acceptance criteria: press Enter to
    # simulate push-to-talk, speak a sentence, confirm a clean WAV buffer is
    # produced (play back nab_test_utterance.wav afterwards).
    logging.basicConfig(level=logging.INFO)

    from wake_word import PushToTalkTrigger

    capture = AudioCapture()
    trigger = PushToTalkTrigger(on_wake=lambda: print("Listening..."))

    input("Press Enter to start talking...")
    trigger.press()
    audio = capture.record_utterance()
    out_path = "nab_test_utterance.wav"
    save_wav(out_path, audio)
    print(f"Recorded {len(audio) / SAMPLE_RATE:.2f}s of trimmed audio -> {out_path}")
