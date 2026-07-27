"""Phase 6 acceptance test: Fallback Handler (module 5.6).

Per NAB_AI_System_Design.md Section 7, Phase 6, Acceptance:
    "Message text matches config exactly, every time."

Verifies:
1. `FallbackHandler` loads `config/settings.yaml` and returns a message
   that is byte-for-byte the (whitespace-normalized) `fallback_response`
   entry in that file -- not something re-derived or paraphrased.
2. Calling `get_response()` repeatedly, across repeated instantiations,
   always returns the identical string -- i.e. it is a fixed lookup, never
   an LLM-generated or otherwise varying response.
3. `config/settings.yaml` also carries the Section 5.8/FR14
   `processing_prompt` entry with the exact wording specified in the design
   doc, since Phase 6 is responsible for creating that entry (consumed by
   Phase 7, not exercised by this handler).

Usage:
    python -m tests.phase6_fallback_test
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.response.fallback_handler import DEFAULT_CONFIG_PATH, FallbackHandler  # noqa: E402

EXPECTED_PROCESSING_PROMPT = "Please wait, I am processing your request."


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    expected_message = (raw_config.get("fallback_response") or "").strip()

    checks: list[tuple[str, bool, str]] = []

    # Check 1: message matches config exactly.
    handler = FallbackHandler()
    response = handler.get_response()
    matches_config = response.message == expected_message
    checks.append(("message matches config verbatim", matches_config, response.message))

    # Check 2: identical across repeated calls and fresh instantiations (no
    # per-call variation -- rules out anything LLM-generated or randomized).
    repeats = [FallbackHandler().get_response().message for _ in range(5)]
    all_identical = all(m == expected_message for m in repeats)
    checks.append(("identical across 5 repeated instantiations/calls", all_identical, str(len(set(repeats))) + " distinct value(s)"))

    # Check 3: source is always "config", never "llm".
    source_is_config = response.source == "config"
    checks.append(("response.source == 'config'", source_is_config, response.source))

    # Check 4: settings.yaml also carries the Phase 6-owned processing_prompt
    # entry with the exact FR14 wording.
    processing_prompt = (raw_config.get("processing_prompt") or "").strip()
    processing_prompt_ok = processing_prompt == EXPECTED_PROCESSING_PROMPT
    checks.append(("processing_prompt matches FR14 wording exactly", processing_prompt_ok, processing_prompt))

    passed = True
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {name} ({detail!r})")
        passed = passed and ok

    print(f"\nPhase 6 acceptance: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
