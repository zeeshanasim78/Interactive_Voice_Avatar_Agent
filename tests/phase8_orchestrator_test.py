"""Phase 8 acceptance test: Orchestrator / Dialogue Manager (module 5.9).

Per NAB_AI_System_Design.md Section 7, Phase 8, Acceptance:
    "No state is skipped; a full log row exists for every single user
    interaction, with no raw audio retained beyond transcription unless
    explicitly enabled; PROCESSING triggers on 100% of interactions and
    never blocks or slows down the subsequent real response."

This drives the *real* `Orchestrator` state machine and SQLite logging code
end-to-end, but swaps in lightweight stubs for the hardware/model-backed
Phase 1-7 leaf modules (mic, faster-whisper, VLC, Ollama/Chroma, coqui-tts,
Rhubarb) so the orchestration logic itself can be verified deterministically
without a kiosk PC. Manual, real-hardware testing (10 live interactions
across all 4 response types, per Section 7 Phase 8 step 4) remains a
separate, non-automatable check.

Verifies:
1. Running interactions covering all 4 response types (video, RAG answer,
   restricted, fallback) plus the CLARIFY branch (both trigger paths --
   low STT confidence and borderline routing confidence) each produce the
   full, unskipped state sequence in `state_transitions`.
2. Every interaction gets exactly one complete row in `interactions`, with
   `processing_started_at`/`processing_ended_at` populated on 100% of runs.
3. A video-playback failure is caught, logged as response_type="error" with
   the error recorded, and answered with the apology -- not a crash
   (Section 9 review finding #6).
4. No raw microphone audio is written to disk by the orchestrator itself
   (NFR5).

Usage:
    python -m tests.phase8_orchestrator_test
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio.stt import TranscriptionResult  # noqa: E402
from src.orchestrator import (  # noqa: E402
    Orchestrator,
    OrchestratorConfig,
)
from src.rag.query_engine import QueryResult  # noqa: E402
from src.response.fallback_handler import FallbackResponse  # noqa: E402
from src.routing.intent_router import IntentCategory, RoutingResult  # noqa: E402
from src.video.player import VideoPlaybackError  # noqa: E402

SCRATCH_DIR = Path(__file__).resolve().parent / "phase8_scratch"
SCRATCH_DB = SCRATCH_DIR / "test_interactions.db"

EXPECTED_SEQUENCE = {
    "video": ["listening", "transcribing", "processing", "routing", "video", "responding", "idle"],
    "rag_answer": ["listening", "transcribing", "processing", "routing", "rag_answer", "responding", "idle"],
    "restricted": ["listening", "transcribing", "processing", "routing", "restricted", "responding", "idle"],
    "fallback": ["listening", "transcribing", "processing", "routing", "fallback", "responding", "idle"],
    "clarify": ["listening", "transcribing", "processing", "routing", "clarify", "responding", "idle"],
}


class StubCapture:
    def record_utterance(self) -> np.ndarray:
        return np.zeros(1600, dtype=np.float32)


class StubSTT:
    def __init__(self, result: TranscriptionResult):
        self._result = result

    def transcribe(self, audio) -> TranscriptionResult:
        return self._result


class StubRouter:
    """Mimics IntentRouter's public surface (`route()` + `.config.semantic_threshold`)."""

    def __init__(self, result: RoutingResult, semantic_threshold: float = 0.45):
        self._result = result
        self.config = SimpleNamespace(semantic_threshold=semantic_threshold)

    def route(self, text: str) -> RoutingResult:
        return self._result


class StubVideoLibrary:
    def entry_for_topic(self, topic):
        return SimpleNamespace(topic=topic, file=Path("videos/nab_achievements.mp4"))

    def resolved_path(self, entry):
        return SCRATCH_DIR / "fake_video.mp4"


class StubVideoPlayer:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.played: List[Path] = []

    def play(self, path):
        if self.fail:
            raise VideoPlaybackError("simulated VLC failure")
        self.played.append(Path(path))

    def release(self):
        pass


class StubQueryEngine:
    def __init__(self, result: QueryResult):
        self._result = result

    def answer(self, query: str) -> QueryResult:
        return self._result


class StubFallbackHandler:
    def get_response(self) -> FallbackResponse:
        return FallbackResponse(message="I have boundaries in accessing that information.")


class StubTTS:
    def __init__(self):
        self.calls: List[str] = []

    def synthesize(self, text: str, output_path):
        self.calls.append(text)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"stub-audio")
        return SimpleNamespace(audio_path=output_path, engine_used="stub", elapsed_seconds=0.0)


class StubAvatar:
    def __init__(self):
        self.states: List[str] = []
        self.spoken_paths: List[Path] = []

    def play_idle(self):
        self.states.append("idle")

    def play_listening(self):
        self.states.append("listening")

    def play_processing(self):
        self.states.append("processing")

    def play_speaking(self, path):
        self.states.append("speaking")
        self.spoken_paths.append(Path(path))


def _build_orchestrator(
    *,
    stt_result: TranscriptionResult,
    routing_result: RoutingResult = None,
    query_result: QueryResult = None,
    video_fails: bool = False,
) -> Orchestrator:
    routing_result = routing_result or RoutingResult(category=IntentCategory.FALLBACK, confidence=0.0, method="fallback")
    return Orchestrator(
        config=OrchestratorConfig(db_path=SCRATCH_DB, response_audio_path=SCRATCH_DIR / "response_output.wav"),
        capture=StubCapture(),
        stt=StubSTT(stt_result),
        router=StubRouter(routing_result),
        video_library=StubVideoLibrary(),
        video_player=StubVideoPlayer(fail=video_fails),
        query_engine=StubQueryEngine(query_result) if query_result is not None else None,
        fallback_handler=StubFallbackHandler(),
        tts=StubTTS(),
        avatar=StubAvatar(),
    )


def _sequence_for_session(conn: sqlite3.Connection, session_id: str) -> List[str]:
    rows = conn.execute(
        "SELECT state FROM state_transitions WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [r[0] for r in rows]


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
    SCRATCH_DIR.mkdir(parents=True)

    checks: List[tuple[str, object, str]] = []
    session_ids: dict[str, str] = {}
    orchestrators: List[Orchestrator] = []

    scenarios = [
        (
            "video",
            dict(
                stt_result=TranscriptionResult(text="Tell me about NAB achievements", language="en", confidence=0.9, is_low_confidence=False),
                routing_result=RoutingResult(
                    category=IntentCategory.VIDEO, confidence=0.9, video_topic="NAB Achievements",
                    video_file="videos/nab_achievements.mp4", method="keyword",
                ),
            ),
        ),
        (
            "rag_answer",
            dict(
                stt_result=TranscriptionResult(text="How can I file a complaint with NAB?", language="en", confidence=0.9, is_low_confidence=False),
                routing_result=RoutingResult(category=IntentCategory.PDF_QA, confidence=0.7, method="semantic"),
                query_result=QueryResult(query="How can I file a complaint with NAB?", has_answer=True, answer="File it via NAB's official complaint portal.", best_similarity=0.72, reason="answered"),
            ),
        ),
        (
            "restricted",
            dict(
                stt_result=TranscriptionResult(text="What is the status of case number 4521?", language="en", confidence=0.9, is_low_confidence=False),
                routing_result=RoutingResult(
                    category=IntentCategory.RESTRICTED, confidence=1.0, restricted_category="case_specific",
                    restricted_response="I have boundaries in accessing that information.", method="restricted",
                ),
            ),
        ),
        (
            "fallback",
            dict(
                stt_result=TranscriptionResult(text="What is the weather today?", language="en", confidence=0.9, is_low_confidence=False),
                routing_result=RoutingResult(category=IntentCategory.FALLBACK, confidence=0.05, method="fallback"),
            ),
        ),
        (
            "clarify",  # trigger 1: low STT confidence
            dict(
                stt_result=TranscriptionResult(text="mm mumble", language="en", confidence=0.2, is_low_confidence=True),
            ),
        ),
        (
            "clarify",  # trigger 2: borderline routing confidence (0.40, within margin of 0.45 threshold)
            dict(
                stt_result=TranscriptionResult(text="tell me something about the process maybe", language="en", confidence=0.9, is_low_confidence=False),
                routing_result=RoutingResult(category=IntentCategory.FALLBACK, confidence=0.40, method="fallback"),
            ),
        ),
    ]

    for label, kwargs in scenarios:
        orch = _build_orchestrator(**kwargs)
        orchestrators.append(orch)
        result = orch.run_once()
        session_ids.setdefault(label, result.session_id)
        checks.append((f"{label}: response_type == '{label}'", result.response_type == label, result.response_type))
        checks.append((f"{label}: processing_started_at/ended_at populated", bool(result.processing_started_at and result.processing_ended_at), "ok"))

    # Error-handling path: video playback raises -> caught, logged as "error", apology spoken, no crash.
    error_orch = _build_orchestrator(
        stt_result=TranscriptionResult(text="Tell me about NAB achievements", language="en", confidence=0.9, is_low_confidence=False),
        routing_result=RoutingResult(category=IntentCategory.VIDEO, confidence=0.9, video_topic="NAB Achievements", video_file="videos/nab_achievements.mp4", method="keyword"),
        video_fails=True,
    )
    orchestrators.append(error_orch)
    try:
        error_result = error_orch.run_once()
        checks.append(("video failure: no crash, response_type == 'error'", error_result.response_type == "error", error_result.response_type))
        checks.append(("video failure: error message recorded", bool(error_result.error), str(error_result.error)))
    except Exception as exc:
        checks.append(("video failure: no crash, response_type == 'error'", False, f"raised {exc!r}"))

    for orch in orchestrators:
        orch.close()

    # Inspect the SQLite log directly (mirrors README 5.3's "DB Browser for SQLite" manual check).
    conn = sqlite3.connect(SCRATCH_DB)

    for label, session_id in session_ids.items():
        actual_sequence = _sequence_for_session(conn, session_id)
        expected = EXPECTED_SEQUENCE[label]
        checks.append((f"{label}: full state sequence, no state skipped", actual_sequence == expected, ",".join(actual_sequence)))

    interaction_count = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    expected_count = len(scenarios) + 1  # + the error-path run
    checks.append((
        "one complete 'interactions' row exists per run",
        interaction_count == expected_count,
        f"{interaction_count}/{expected_count}",
    ))

    complete_rows = conn.execute(
        "SELECT COUNT(*) FROM interactions WHERE processing_started_at IS NOT NULL AND processing_ended_at IS NOT NULL "
        "AND transcript IS NOT NULL AND response_type IS NOT NULL"
    ).fetchone()[0]
    checks.append((
        "PROCESSING timestamps present on 100% of interaction rows",
        complete_rows == expected_count,
        f"{complete_rows}/{expected_count}",
    ))

    conn.close()

    # NFR5: the orchestrator itself must never persist raw mic audio -- only
    # the synthesized response_output.wav (TTS output, not mic input) may exist.
    wav_files = [p for p in SCRATCH_DIR.rglob("*.wav") if p.name != "response_output.wav"]
    checks.append(("no raw microphone audio persisted by the orchestrator (NFR5)", len(wav_files) == 0, str(wav_files)))

    passed = True
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} ({detail!r})")
        passed = passed and bool(ok)

    print(f"\nPhase 8 acceptance: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
