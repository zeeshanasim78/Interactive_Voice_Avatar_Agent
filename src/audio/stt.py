"""Speech-to-text via faster-whisper (module 5.2 / Phase 2).

Loads a `faster-whisper` model once at startup and transcribes recorded
utterances (the WAV/np.ndarray buffers produced by `capture.py`) into text,
with auto language detection covering English and Urdu (FR3) and a
confidence score consumers can use to trigger a "could you repeat that?"
re-prompt (NFR10) rather than guessing on unclear audio.
"""

from __future__ import annotations

import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class STTConfig:
    model_size: str = "small"  # "small" or "medium" depending on hardware, per Section 5.2/10
    device: str = "cpu"
    compute_type: str = "int8"  # fast, low-memory default for CPU-only kiosks
    beam_size: int = 5
    language: Optional[str] = None  # None = auto-detect (English/Urdu, FR3)
    low_confidence_threshold: float = 0.5  # below this, caller should ask user to repeat (NFR10)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    is_low_confidence: bool


class SpeechToText:
    """Transcribes recorded utterances. Load once, reuse across requests —
    model loading (not transcription) is the expensive part."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self._model = WhisperModel(
            self.config.model_size,
            device=self.config.device,
            compute_type=self.config.compute_type,
        )

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: int = SAMPLE_RATE,
    ) -> TranscriptionResult:
        """Transcribes a mono utterance.

        `audio` is either the float32 [-1, 1] ndarray returned by
        `AudioCapture.record_utterance()`, or a path to a WAV file.
        """
        if isinstance(audio, (str, Path)):
            audio = _load_wav(audio, expected_sample_rate=sample_rate)
        elif sample_rate != SAMPLE_RATE:
            raise ValueError(f"faster-whisper expects {SAMPLE_RATE}Hz audio, got {sample_rate}")

        segments, info = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,  # silence already trimmed upstream by capture.py's silero-vad pass
        )
        segments = list(segments)

        text = " ".join(segment.text.strip() for segment in segments).strip()
        confidence = _estimate_confidence(segments, info.language_probability)
        is_low_confidence = not text or confidence < self.config.low_confidence_threshold

        if is_low_confidence:
            logger.info("Low-confidence transcription (%.2f): %r", confidence, text)

        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=confidence,
            is_low_confidence=is_low_confidence,
        )


def _estimate_confidence(segments: list, language_probability: float) -> float:
    """faster-whisper exposes no single transcript-level confidence score, so
    this approximates one from each segment's average token log-probability
    (`avg_logprob`, exponentiated into a ~[0, 1] likelihood) and its
    `no_speech_prob`, duration-weighted across segments, then scaled by the
    language-detection confidence since a wrong language guess also means an
    unreliable transcript.
    """
    if not segments:
        return 0.0

    total_duration = sum(max(segment.end - segment.start, 1e-6) for segment in segments)
    weighted_sum = sum(
        math.exp(segment.avg_logprob) * (1.0 - segment.no_speech_prob) * max(segment.end - segment.start, 1e-6)
        for segment in segments
    )
    segment_confidence = weighted_sum / total_duration
    confidence = segment_confidence * language_probability
    return max(0.0, min(1.0, confidence))


def _load_wav(path: Union[str, Path], expected_sample_rate: int) -> np.ndarray:
    """Loads a 16-bit PCM mono WAV file into float32 [-1, 1] samples."""
    with wave.open(str(path), "rb") as wf:
        if wf.getframerate() != expected_sample_rate:
            raise ValueError(
                f"WAV sample rate {wf.getframerate()} != expected {expected_sample_rate}"
            )
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV input")
        raw = wf.readframes(wf.getnframes())
        pcm16 = np.frombuffer(raw, dtype=np.int16)
        if wf.getnchannels() > 1:
            pcm16 = pcm16.reshape(-1, wf.getnchannels()).mean(axis=1)
        return (pcm16.astype(np.float32) / 32767.0)


if __name__ == "__main__":
    # Manual test harness for Phase 2 acceptance criteria: speak 10 English
    # and 10 Urdu sample sentences, one at a time, and check the printed
    # transcripts/confidence manually against what was actually said.
    logging.basicConfig(level=logging.INFO)

    from capture import AudioCapture, save_wav
    from wake_word import PushToTalkTrigger

    capture = AudioCapture()
    stt = SpeechToText()
    trigger = PushToTalkTrigger(on_wake=lambda: print("Listening..."))

    while True:
        input("Press Enter to speak (Ctrl+C to quit)...")
        trigger.press()
        audio = capture.record_utterance()
        save_wav("nab_test_utterance.wav", audio)

        result = stt.transcribe(audio)
        print(f"Text:       {result.text!r}")
        print(f"Language:   {result.language}")
        print(f"Confidence: {result.confidence:.2f}")
        if result.is_low_confidence:
            print("-> Low confidence: would prompt 'Could you repeat that?' (NFR10)")
