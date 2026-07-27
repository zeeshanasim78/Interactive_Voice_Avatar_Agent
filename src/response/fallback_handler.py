"""Fallback Handler (module 5.6 / Phase 6, FR7).

Per NAB_AI_System_Design.md 5.6 and Section 7 Phase 6: when a query is out
of scope of both the video library (5.4) and the PDF knowledge base (5.5) --
i.e. `IntentRouter` finds no video match and `QueryEngine.answer()` returns
`has_answer=False` -- the system must never fabricate an answer. It responds
with a fixed, non-fabricated boundary statement pulled verbatim from
`config/settings.yaml`, delivered via TTS + avatar (Phase 7, later) plus an
on-screen subtitle (Phase 9, later), same as any other answer.

This is distinct from `RestrictedTopicFilter.restricted_response`
(config/restricted_topics.yaml): that's the dedicated boundary statement for
case/individual/investigation queries (FR12, 5.3 Step 1), which runs as a
hard pre-filter *before* routing/RAG ever see the query. This handler is the
generic "not available" fallback for anything else that falls through
routing and RAG with no match (FR7).

Per Phase 6 acceptance criterion ("Message text matches config exactly,
every time"), `get_response()` takes no query/context input at all -- the
whole point of the Fallback Handler is that it is NOT a function of
anything; it always returns the identical config string, never an
LLM-generated one. Wiring this into the orchestrator's fallback branch is
Phase 8 (5.9); this module is what gets wired in.

Usage:
    python -m src.response.fallback_handler
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


@dataclass
class FallbackResponse:
    message: str
    source: str = "config"  # always "config" -- never "llm" (Phase 6 acceptance criterion)


class FallbackHandler:
    """Loads the fixed boundary message from `config/settings.yaml` and
    hands it back verbatim.

    Loaded once at construction (mirrors the load-once pattern used
    throughout this codebase -- `RestrictedTopicFilter`, `IntentRouter`,
    `QueryEngine`).
    """

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.message: str = ""
        self._load()

    def _load(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        message = (data.get("fallback_response") or "").strip()
        if not message:
            raise ValueError(f"{self.config_path}: 'fallback_response' is required and must be non-empty")
        self.message = message
        logger.info("Loaded fallback handler: %d-char message from %s", len(self.message), self.config_path)

    def get_response(self) -> FallbackResponse:
        """Returns the fixed boundary message verbatim, unconditionally."""
        return FallbackResponse(message=self.message)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print Urdu script

    logging.basicConfig(level=logging.INFO)

    handler = FallbackHandler()
    for _ in range(3):
        print(f"-> {handler.get_response().message!r}")
