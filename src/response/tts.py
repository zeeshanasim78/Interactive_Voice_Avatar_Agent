"""Text-to-Speech (module 5.7 / Phase 7).

Per NAB_AI_System_Design.md 5.7 and Section 7 Phase 7: converts a response
string (RAG answer, fallback message, or the fixed processing-wait prompt)
into a WAV file, using **`coqui-tts`** (the actively-maintained
`idiap/coqui-ai-TTS` fork, `pip install coqui-tts` -- the import path stays
`from TTS.api import TTS`, unchanged from the original package) as the
primary engine. `pyttsx3` is the lightweight offline fallback the design doc
calls for "if the neural TTS is too heavy for the target hardware" -- this
module also falls back to it automatically if the coqui model can't be
loaded at all (e.g. no internet for the one-time model download), so the
rest of the pipeline never breaks on a machine that hasn't fetched a coqui
model yet.

This module also owns generating `avatar_assets/audio/processing_prompt.wav`
(FR14, 5.8): per Phase 7 step 2, that file is synthesized **once**, offline,
at setup time from the fixed `processing_prompt` string in
`config/settings.yaml` (already populated by Phase 6) and cached to disk --
never regenerated per-request, since that would add latency to the very
message meant to hide latency.

Usage:
    python -m src.response.tts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
DEFAULT_PROCESSING_PROMPT_WAV = (
    Path(__file__).resolve().parents[2] / "avatar_assets" / "audio" / "processing_prompt.wav"
)

# Single-speaker, no-ToS-prompt model -- safe to auto-download unattended,
# unlike the multi-speaker/XTTS models that require interactively accepting
# a license. Urdu-capable coqui models are not consistently available, so a
# strong English model is the pragmatic default (5.7 notes Urdu "where
# available"); swap `model_name` per-deployment if a suitable one is found.
DEFAULT_COQUI_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

# 5.7 acceptance criterion: "Audio generated within ~1-2 seconds for a
# typical response length" -- logged as a warning, not enforced as a hard
# error, since it's inherently hardware-dependent.
TARGET_SYNTHESIS_SECONDS = 2.0


@dataclass
class TTSConfig:
    engine: str = "coqui"  # "coqui" (primary, 5.7) or "pyttsx3" (explicit fallback)
    model_name: str = DEFAULT_COQUI_MODEL
    device: str = "cpu"  # kiosk hardware is CPU-only per NFR2; GPU optional
    pyttsx3_rate: int = 175  # words/minute, pyttsx3 default is ~200


@dataclass
class SynthesisResult:
    audio_path: Path
    engine_used: str  # "coqui" or "pyttsx3" -- which one actually ran
    elapsed_seconds: float


class TextToSpeech:
    """Loads a TTS engine once at construction (mirrors the load-once
    pattern used throughout this codebase -- `SpeechToText`, `IntentRouter`,
    `QueryEngine`) and reuses it across requests."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._coqui_tts = None
        self._pyttsx3_engine = None
        self.engine_name: str

        if self.config.engine == "coqui":
            self._coqui_tts = self._try_load_coqui()

        if self._coqui_tts is not None:
            self.engine_name = "coqui"
        else:
            self._pyttsx3_engine = self._load_pyttsx3()
            self.engine_name = "pyttsx3"

        logger.info("TextToSpeech ready, engine=%s", self.engine_name)

    def _try_load_coqui(self):
        try:
            from TTS.api import TTS

            tts = TTS(model_name=self.config.model_name, progress_bar=False)
            tts.to(self.config.device)
            return tts
        except Exception as exc:  # model download/import/runtime failure
            logger.warning(
                "coqui-tts unavailable (%s); falling back to pyttsx3 per Section 5.7's "
                "documented fallback path.",
                exc,
            )
            return None

    def _load_pyttsx3(self):
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self.config.pyttsx3_rate)
        return engine

    def synthesize(self, text: str, output_path: Union[str, Path]) -> SynthesisResult:
        """Synthesizes `text` to a WAV file at `output_path`, overwriting it
        if present. Returns which engine actually ran and how long it took."""
        text = text.strip()
        if not text:
            raise ValueError("synthesize() requires non-empty text")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        if self.engine_name == "coqui":
            self._coqui_tts.tts_to_file(text=text, file_path=str(output_path))
        else:
            self._pyttsx3_engine.save_to_file(text, str(output_path))
            self._pyttsx3_engine.runAndWait()
        elapsed = time.monotonic() - start

        logger.info(
            "Synthesized %d chars via %s in %.2fs -> %s", len(text), self.engine_name, elapsed, output_path
        )
        if elapsed > TARGET_SYNTHESIS_SECONDS:
            logger.warning(
                "TTS synthesis took %.2fs, exceeding the ~%.0fs target (Section 5.7 acceptance)",
                elapsed,
                TARGET_SYNTHESIS_SECONDS,
            )

        return SynthesisResult(audio_path=output_path, engine_used=self.engine_name, elapsed_seconds=elapsed)


def generate_processing_prompt(
    tts: Optional[TextToSpeech] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_PROCESSING_PROMPT_WAV,
) -> SynthesisResult:
    """Phase 7 step 2: synthesizes the fixed FR14 wait-prompt text (read
    verbatim from `config/settings.yaml`'s `processing_prompt` entry, the
    same file Phase 6 populated) into a cached WAV file, once. Callers
    should run this at setup time only -- `avatar.py`'s `play_processing()`
    plays this cached file directly and never calls this at request time."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompt_text = (data.get("processing_prompt") or "").strip()
    if not prompt_text:
        raise ValueError(f"{config_path}: 'processing_prompt' is required and must be non-empty")

    tts = tts or TextToSpeech()
    result = tts.synthesize(prompt_text, output_path)
    logger.info("Cached processing-prompt audio at %s (engine=%s)", result.audio_path, result.engine_used)
    return result


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    logging.basicConfig(level=logging.INFO)

    tts = TextToSpeech()

    # Phase 7 step 2 (added): generate the cached wait-prompt WAV once.
    prompt_result = generate_processing_prompt(tts)
    print(f"-> processing_prompt.wav: engine={prompt_result.engine_used}, {prompt_result.elapsed_seconds:.2f}s")

    # Phase 7 step 5: synthesize a few sample answers and report timing, for
    # a manual clarity/latency check.
    samples = [
        "Thank you for visiting the National Accountability Bureau.",
        "NAB's mandate is to eliminate corruption through awareness, prevention, and enforcement.",
        "I have boundaries in accessing that information; it is not available with me at this time.",
    ]
    for i, text in enumerate(samples, start=1):
        out = Path(__file__).resolve().parents[2] / "avatar_assets" / "audio" / f"sample_{i}.wav"
        result = tts.synthesize(text, out)
        print(f"-> sample_{i}.wav: engine={result.engine_used}, {result.elapsed_seconds:.2f}s, {out}")
