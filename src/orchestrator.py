"""Orchestrator / Dialogue Manager (module 5.9 / Phase 8).

Ties Phases 1-7 together as the explicit, auditable state machine specified
in Section 5.9:

    IDLE -> LISTENING -> TRANSCRIBING -> PROCESSING -> ROUTING ->
    (VIDEO | RAG_ANSWER | RESTRICTED | FALLBACK | CLARIFY) -> RESPONDING -> IDLE

Every transition is written to `logs/interactions.db` (FR11, NFR4), and one
complete summary row is written per finished interaction. `PROCESSING` is
entered the instant `TRANSCRIBING` finishes: it calls `avatar.play_processing()`
immediately (cached wait-prompt audio + looping animation, FR14) while the
real routing/RAG/LLM decision runs on a background thread, per Phase 8 step 2 --
this is a pure UX/feedback layer and never changes what `ROUTING` decides or
delays the real answer (Section 5.9 / Section 13).

`CLARIFY` (Section 9 review finding #2, NFR10) is entered instead of guessing
whenever the STT transcript itself was low-confidence, or the router's own
fallback confidence was borderline rather than clearly "no match" -- the user
is asked to repeat instead of the system inventing an answer.

Per Section 9 review finding #6, the VIDEO and RESPONDING stages wrap their
file I/O / playback calls in try/except and degrade to a spoken apology
(`technical_difficulty_prompt`) plus a logged error, rather than crashing the
kiosk app.

This module only wires the pipeline together; it does not change any of the
Phase 1-7 modules' own decision logic (restricted-topic filtering, RAG
grounding/hallucination prevention, fallback wording) -- see Section 14.2.

Usage:
    python -m src.orchestrator
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .audio.capture import AudioCapture
from .audio.stt import SpeechToText, TranscriptionResult
from .audio.wake_word import PushToTalkTrigger
from .rag.query_engine import QueryEngine, QueryResult
from .response.avatar import Avatar
from .response.fallback_handler import FallbackHandler
from .response.tts import TextToSpeech
from .routing.intent_router import IntentCategory, IntentRouter, RoutingResult
from .video.player import VideoLibrary, VideoLibraryError, VideoPlaybackError, VideoPlayer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "logs" / "interactions.db"
DEFAULT_RESPONSE_AUDIO_PATH = PROJECT_ROOT / "avatar_assets" / "audio" / "response_output.wav"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrchestratorState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    ROUTING = "routing"
    VIDEO = "video"
    RAG_ANSWER = "rag_answer"
    RESTRICTED = "restricted"
    FALLBACK = "fallback"
    CLARIFY = "clarify"
    RESPONDING = "responding"


class Branch(str, Enum):
    """The five outcomes `ROUTING` can resolve to (Section 5.9 diagram)."""

    VIDEO = "video"
    RAG_ANSWER = "rag_answer"
    RESTRICTED = "restricted"
    FALLBACK = "fallback"
    CLARIFY = "clarify"


_BRANCH_STATE = {
    Branch.VIDEO: OrchestratorState.VIDEO,
    Branch.RAG_ANSWER: OrchestratorState.RAG_ANSWER,
    Branch.RESTRICTED: OrchestratorState.RESTRICTED,
    Branch.FALLBACK: OrchestratorState.FALLBACK,
    Branch.CLARIFY: OrchestratorState.CLARIFY,
}


@dataclass
class BranchResult:
    branch: Branch
    routing: Optional[RoutingResult] = None
    query_result: Optional[QueryResult] = None
    clarify_reason: Optional[str] = None  # "low_stt_confidence" | "borderline_routing_confidence"


@dataclass
class InteractionResult:
    session_id: str
    started_at: str
    ended_at: str
    transcript: str
    language: Optional[str]
    stt_confidence: float
    matched_route: str
    response_type: str
    video_topic: Optional[str]
    restricted_category: Optional[str]
    pdf_similarity: Optional[float]
    processing_started_at: str
    processing_ended_at: str
    error: Optional[str]


@dataclass
class OrchestratorConfig:
    db_path: Path = DEFAULT_DB_PATH
    response_audio_path: Path = DEFAULT_RESPONSE_AUDIO_PATH
    # Section 9 review finding #2 / NFR10: how far below the router's own
    # semantic_threshold a FALLBACK-routed confidence can be while still
    # counting as "borderline" (-> CLARIFY) rather than a clean miss.
    clarify_confidence_margin: float = 0.10
    clarify_prompt: str = "I didn't quite catch that - could you repeat your question?"
    technical_difficulty_prompt: str = "I'm sorry, something went wrong on my end. Please try again."


class InteractionLogger:
    """Append-only SQLite audit log at `logs/interactions.db` (FR11, NFR4).

    Two tables:
      - `state_transitions`: one row per state transition, ever -- satisfies
        5.9's "every state transition is logged", never updated.
      - `interactions`: one complete summary row per finished interaction,
        inserted once at the end -- satisfies Phase 8's "log has one
        complete row per interaction", also never updated.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                state TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                transcript TEXT,
                matched_intent TEXT,
                response_type TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                transcript TEXT,
                language TEXT,
                stt_confidence REAL,
                matched_route TEXT,
                response_type TEXT NOT NULL,
                video_topic TEXT,
                restricted_category TEXT,
                pdf_similarity REAL,
                processing_started_at TEXT,
                processing_ended_at TEXT,
                error TEXT
            )
            """
        )
        self._conn.commit()

    def log_transition(
        self,
        session_id: str,
        state: str,
        timestamp: str,
        transcript: Optional[str] = None,
        matched_intent: Optional[str] = None,
        response_type: Optional[str] = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO state_transitions (session_id, state, timestamp, transcript, matched_intent, response_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, state, timestamp, transcript, matched_intent, response_type),
        )
        self._conn.commit()

    def log_interaction(self, result: InteractionResult) -> None:
        self._conn.execute(
            "INSERT INTO interactions (session_id, started_at, ended_at, transcript, language, stt_confidence, "
            "matched_route, response_type, video_topic, restricted_category, pdf_similarity, "
            "processing_started_at, processing_ended_at, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                result.session_id,
                result.started_at,
                result.ended_at,
                result.transcript,
                result.language,
                result.stt_confidence,
                result.matched_route,
                result.response_type,
                result.video_topic,
                result.restricted_category,
                result.pdf_similarity,
                result.processing_started_at,
                result.processing_ended_at,
                result.error,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class Orchestrator:
    """Explicit state machine tying the Phase 1-7 modules together.

    All collaborators are injected explicitly (no hidden default
    construction of heavy models/hardware here) so this class stays testable
    without a microphone, VLC, Ollama, or a GPU -- see `build_orchestrator()`
    below for wiring the real Phase 1-7 components together for an actual
    kiosk run.
    """

    def __init__(
        self,
        *,
        capture: AudioCapture,
        stt: SpeechToText,
        router: IntentRouter,
        video_library: VideoLibrary,
        video_player: Optional[VideoPlayer],
        query_engine: Optional[QueryEngine],
        fallback_handler: FallbackHandler,
        tts: TextToSpeech,
        avatar: Avatar,
        config: Optional[OrchestratorConfig] = None,
        wake_trigger: Optional[PushToTalkTrigger] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.capture = capture
        self.stt = stt
        self.router = router
        self.video_library = video_library
        self.video_player = video_player
        self.query_engine = query_engine
        self.fallback_handler = fallback_handler
        self.tts = tts
        self.avatar = avatar
        self.wake_trigger = wake_trigger or PushToTalkTrigger(on_wake=self.avatar.play_listening)

        self.db = InteractionLogger(self.config.db_path)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nab-orchestrator-routing")

    def run_once(self) -> InteractionResult:
        """Runs exactly one full IDLE->...->IDLE interaction, blocking until
        the response has finished playing. Never raises -- any failure
        during the RESPONDING stage is caught, logged, and answered with a
        spoken apology (Section 9 review finding #6)."""
        session_id = str(uuid.uuid4())
        started_at = _now_iso()

        # LISTENING -- wake_trigger.press() flips the avatar synchronously (5.1 acceptance: within 300ms).
        self.wake_trigger.press()
        self._log_transition(session_id, OrchestratorState.LISTENING)
        audio = self.capture.record_utterance()

        # TRANSCRIBING
        stt_result = self.stt.transcribe(audio)
        self._log_transition(session_id, OrchestratorState.TRANSCRIBING, transcript=stt_result.text)

        # PROCESSING -- wait-prompt fires immediately (FR14); routing/RAG/LLM
        # work runs on a background thread (Phase 8 step 2) while it plays.
        processing_started_at = _now_iso()
        self.avatar.play_processing()
        self._log_transition(session_id, OrchestratorState.PROCESSING, transcript=stt_result.text)
        branch_result = self._executor.submit(self._decide_branch, stt_result).result()
        processing_ended_at = _now_iso()

        # ROUTING (+ the resolved branch state)
        self._log_transition(
            session_id, OrchestratorState.ROUTING, transcript=stt_result.text, matched_intent=branch_result.branch.value
        )
        self._log_transition(
            session_id,
            _BRANCH_STATE[branch_result.branch],
            transcript=stt_result.text,
            matched_intent=branch_result.branch.value,
        )

        # RESPONDING
        response_type, error = self._respond(branch_result)
        self._log_transition(
            session_id,
            OrchestratorState.RESPONDING,
            transcript=stt_result.text,
            matched_intent=branch_result.branch.value,
            response_type=response_type,
        )

        # back to IDLE
        self.avatar.play_idle()
        self._log_transition(session_id, OrchestratorState.IDLE)
        ended_at = _now_iso()

        result = InteractionResult(
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            transcript=stt_result.text,
            language=stt_result.language,
            stt_confidence=stt_result.confidence,
            matched_route=branch_result.branch.value,
            response_type=response_type,
            video_topic=branch_result.routing.video_topic if branch_result.routing else None,
            restricted_category=branch_result.routing.restricted_category if branch_result.routing else None,
            pdf_similarity=branch_result.query_result.best_similarity if branch_result.query_result else None,
            processing_started_at=processing_started_at,
            processing_ended_at=processing_ended_at,
            error=error,
        )
        self.db.log_interaction(result)
        return result

    def run_forever(self) -> None:
        """Kiosk main loop: repeatedly waits for a wake/press and runs one
        interaction. A failure inside one interaction never crashes the
        loop -- the kiosk logs it and keeps listening (Section 9 finding #6)."""
        logger.info("Orchestrator ready, entering main loop (Ctrl+C to stop)")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Unhandled error during interaction; returning to idle and continuing")
                try:
                    self.avatar.play_idle()
                except Exception:
                    logger.exception("Also failed to reset avatar to idle")

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        if self.video_player is not None:
            try:
                self.video_player.release()
            except Exception:
                logger.exception("Error releasing video player")
        self.db.close()

    def __enter__(self) -> "Orchestrator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _log_transition(
        self,
        session_id: str,
        state: OrchestratorState,
        *,
        transcript: Optional[str] = None,
        matched_intent: Optional[str] = None,
        response_type: Optional[str] = None,
    ) -> None:
        self.db.log_transition(session_id, state.value, _now_iso(), transcript, matched_intent, response_type)

    def _decide_branch(self, stt_result: TranscriptionResult) -> BranchResult:
        """The ROUTING decision. Runs on the background thread submitted from
        `run_once()`; does not touch the avatar/TTS (that's RESPONDING)."""
        if stt_result.is_low_confidence:
            logger.info("CLARIFY: low STT confidence (%.2f)", stt_result.confidence)
            return BranchResult(branch=Branch.CLARIFY, clarify_reason="low_stt_confidence")

        routing = self.router.route(stt_result.text)

        if routing.category == IntentCategory.RESTRICTED:
            return BranchResult(branch=Branch.RESTRICTED, routing=routing)

        if routing.category == IntentCategory.VIDEO:
            return BranchResult(branch=Branch.VIDEO, routing=routing)

        if routing.category == IntentCategory.PDF_QA:
            if self.query_engine is None:
                logger.warning("PDF_QA routed but no QueryEngine is available; falling back")
                return BranchResult(branch=Branch.FALLBACK, routing=routing)
            query_result = self.query_engine.answer(stt_result.text)
            if query_result.has_answer:
                return BranchResult(branch=Branch.RAG_ANSWER, routing=routing, query_result=query_result)
            return BranchResult(branch=Branch.FALLBACK, routing=routing, query_result=query_result)

        # routing.category == IntentCategory.FALLBACK -- Section 9 review
        # finding #2 (NFR10): a borderline confidence (not clearly a miss)
        # asks the user to repeat instead of guessing.
        semantic_threshold = getattr(getattr(self.router, "config", None), "semantic_threshold", 0.45)
        borderline = routing.confidence >= (semantic_threshold - self.config.clarify_confidence_margin)
        if borderline:
            logger.info("CLARIFY: borderline routing confidence (%.2f)", routing.confidence)
            return BranchResult(branch=Branch.CLARIFY, routing=routing, clarify_reason="borderline_routing_confidence")
        return BranchResult(branch=Branch.FALLBACK, routing=routing)

    def _respond(self, branch_result: BranchResult) -> Tuple[str, Optional[str]]:
        """RESPONDING. Returns (response_type, error). Never raises -- any
        failure here is caught, logged, and answered with a spoken apology
        (Section 9 review finding #6)."""
        try:
            if branch_result.branch == Branch.VIDEO:
                return self._respond_video(branch_result.routing)
            if branch_result.branch == Branch.RAG_ANSWER:
                self._speak(branch_result.query_result.answer)
                return "rag_answer", None
            if branch_result.branch == Branch.RESTRICTED:
                self._speak(branch_result.routing.restricted_response)
                return "restricted", None
            if branch_result.branch == Branch.CLARIFY:
                self._speak(self.config.clarify_prompt)
                return "clarify", None
            # Branch.FALLBACK
            self._speak(self.fallback_handler.get_response().message)
            return "fallback", None
        except Exception as exc:
            logger.exception("Unhandled error while responding (branch=%s)", branch_result.branch.value)
            try:
                self._speak(self.config.technical_difficulty_prompt)
            except Exception:
                logger.exception("Also failed to speak the technical-difficulty apology")
            return "error", str(exc)

    def _respond_video(self, routing: RoutingResult) -> Tuple[str, Optional[str]]:
        if self.video_player is None:
            msg = "Video playback unavailable (VLC not installed/initialized)"
            logger.error(msg)
            self._speak(self.config.technical_difficulty_prompt)
            return "error", msg
        try:
            entry = self.video_library.entry_for_topic(routing.video_topic)
            path = self.video_library.resolved_path(entry)
            self.video_player.play(path)
            return "video", None
        except (FileNotFoundError, VideoPlaybackError, VideoLibraryError) as exc:
            logger.error("Video playback failed for topic %r: %s", routing.video_topic, exc)
            self._speak(self.config.technical_difficulty_prompt)
            return "error", str(exc)

    def _speak(self, text: str) -> None:
        self.tts.synthesize(text, self.config.response_audio_path)
        self.avatar.play_speaking(self.config.response_audio_path)


def build_orchestrator(config: Optional[OrchestratorConfig] = None) -> Orchestrator:
    """Constructs an `Orchestrator` wired to the real Phase 1-7 components.

    Video playback and the RAG knowledge base degrade gracefully rather than
    raising if VLC isn't installed or `python -m src.rag.ingest` hasn't been
    run yet (Section 9 review findings #6/#7) -- VIDEO/PDF_QA routes then
    resolve to a logged "error"/"fallback" response instead of crashing at
    startup.
    """
    config = config or OrchestratorConfig()

    capture = AudioCapture()
    stt = SpeechToText()
    router = IntentRouter()
    video_library = VideoLibrary()

    try:
        video_player: Optional[VideoPlayer] = VideoPlayer()
    except VideoPlaybackError as exc:
        logger.warning("VLC unavailable (%s); video responses will report an error instead of playing", exc)
        video_player = None

    try:
        query_engine: Optional[QueryEngine] = QueryEngine()
    except RuntimeError as exc:
        logger.warning("Knowledge base not ready (%s); PDF-QA routes will fall back", exc)
        query_engine = None

    fallback_handler = FallbackHandler()
    tts = TextToSpeech()
    avatar = Avatar()

    return Orchestrator(
        config=config,
        capture=capture,
        stt=stt,
        router=router,
        video_library=video_library,
        video_player=video_player,
        query_engine=query_engine,
        fallback_handler=fallback_handler,
        tts=tts,
        avatar=avatar,
    )


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print Urdu script

    logging.basicConfig(level=logging.INFO)

    with build_orchestrator() as orchestrator:
        print("Press Enter to speak (Ctrl+C to quit)...")
        try:
            while True:
                input()
                result = orchestrator.run_once()
                print(
                    f"-> transcript={result.transcript!r} route={result.matched_route} "
                    f"response={result.response_type} (logged to {orchestrator.config.db_path})"
                )
        except KeyboardInterrupt:
            print("\nStopping.")
