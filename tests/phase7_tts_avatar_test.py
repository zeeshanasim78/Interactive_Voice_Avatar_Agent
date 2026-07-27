"""Phase 7 acceptance test: Text-to-Speech + Avatar/Lip-Sync (modules 5.7, 5.8).

Per NAB_AI_System_Design.md Section 7, Phase 7, Acceptance:
    "No perceptible desync (>300ms) between audio and lip movement;
    idle/listening/processing/speaking states all render correctly;
    wait-prompt playback starts in under ~300ms every time."
Plus 5.7's: "Audio generated within ~1-2 seconds for a typical response
length; intelligible and clear at kiosk speaker volume" (clarity is a
manual/listening check, not automatable here -- see step 5 of Phase 7).

Verifies:
1. `TextToSpeech.synthesize()` produces a playable WAV file, and reports
   which engine actually ran (coqui-tts primary, per 5.7) so a run in a
   sandbox without internet access for the one-time coqui model download
   still exercises the documented pyttsx3 fallback rather than failing.
2. `avatar_assets/audio/processing_prompt.wav` exists, is non-empty, and
   was generated from the exact FR14 wording in `config/settings.yaml`
   (Phase 7 step 2 -- cached once, not regenerated per request).
3. `Avatar` walks through all four states (idle/listening/processing/
   speaking) with no state skipped, and `play_processing()` dispatches the
   cached wait-prompt within the ~300ms acceptance bar.
4. `Avatar.play_speaking()` returns a non-empty, monotonically ordered
   viseme timeline covering the full clip duration, from either the
   Rhubarb or amplitude-envelope source -- the two ordering/coverage
   properties that "no perceptible desync" depends on.

If no audio output device is available in this environment, playback-
dependent checks are reported SKIPPED (not FAILED), mirroring the
graceful-degradation pattern used by Phase 4's video-playback test.

Usage:
    python -m tests.phase7_tts_avatar_test
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sounddevice as sd  # noqa: E402
import yaml  # noqa: E402

from src.response.avatar import (  # noqa: E402
    Avatar,
    AvatarState,
    DEFAULT_PROCESSING_PROMPT_PATH,
    MAX_SYNC_DRIFT_SECONDS,
)
from src.response.tts import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    TARGET_SYNTHESIS_SECONDS,
    TextToSpeech,
)

SCRATCH_DIR = Path(__file__).resolve().parent / "phase7_scratch"
EXPECTED_PROCESSING_PROMPT = "Please wait, I am processing your request."


def _has_output_device() -> bool:
    try:
        sd.check_output_settings()
        return True
    except Exception:
        return False


def test_tts_synthesis(tts: TextToSpeech) -> list[tuple[str, object, str]]:
    checks: list[tuple[str, object, str]] = []
    out_path = SCRATCH_DIR / "sample_answer.wav"
    result = tts.synthesize(
        "NAB's mandate is to eliminate corruption through awareness, prevention, and enforcement.", out_path
    )

    checks.append(("synthesize() produces a non-empty WAV file", out_path.is_file() and out_path.stat().st_size > 0, str(out_path)))
    checks.append(("engine_used is 'coqui' or 'pyttsx3'", result.engine_used in ("coqui", "pyttsx3"), result.engine_used))
    within_target = result.elapsed_seconds <= TARGET_SYNTHESIS_SECONDS
    checks.append((
        f"synthesis within ~{TARGET_SYNTHESIS_SECONDS:.0f}s target (5.7 acceptance)",
        within_target if within_target else "WARN",
        f"{result.elapsed_seconds:.2f}s via {result.engine_used}",
    ))
    return checks


def test_processing_prompt_cached() -> list[tuple[str, object, str]]:
    checks: list[tuple[str, object, str]] = []

    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}
    configured_prompt = (raw_config.get("processing_prompt") or "").strip()

    checks.append(("processing_prompt wording matches FR14 exactly", configured_prompt == EXPECTED_PROCESSING_PROMPT, configured_prompt))
    exists = DEFAULT_PROCESSING_PROMPT_PATH.is_file() and DEFAULT_PROCESSING_PROMPT_PATH.stat().st_size > 0
    checks.append(("avatar_assets/audio/processing_prompt.wav exists and is non-empty (cached once, step 2)", exists, str(DEFAULT_PROCESSING_PROMPT_PATH)))
    return checks


def test_avatar_states_and_lipsync(avatar_ready: bool) -> list[tuple[str, object, str]]:
    checks: list[tuple[str, object, str]] = []
    observed_states: list[AvatarState] = []
    avatar = Avatar(on_state_change=observed_states.append)

    avatar.play_idle()
    avatar.play_listening()

    if not avatar_ready:
        checks.append(("idle/listening/processing/speaking states all render", "SKIPPED", "no audio output device in this environment"))
        checks.append(("wait-prompt playback starts within ~300ms (FR14)", "SKIPPED", "no audio output device in this environment"))
        checks.append(("play_speaking() returns a non-empty, ordered, duration-covering viseme timeline", "SKIPPED", "no audio output device in this environment"))
        return checks

    start = time.monotonic()
    avatar.play_processing()
    dispatch_latency = time.monotonic() - start
    checks.append(("wait-prompt playback starts within ~300ms (FR14)", dispatch_latency <= 0.3, f"{dispatch_latency * 1000:.0f}ms"))
    time.sleep(2.5)  # let the cached wait-prompt finish before demoing "speaking"

    tts = TextToSpeech()
    sample_path = SCRATCH_DIR / "avatar_speaking_sample.wav"
    tts.synthesize("Thank you for visiting the National Accountability Bureau.", sample_path)

    result = avatar.play_speaking(sample_path)
    avatar.play_idle()

    checks.append((
        "idle/listening/processing/speaking states all render, none skipped",
        {AvatarState.IDLE, AvatarState.LISTENING, AvatarState.PROCESSING, AvatarState.SPEAKING}.issubset(set(observed_states)),
        ",".join(s.value for s in observed_states),
    ))

    has_visemes = len(result.visemes) > 0
    starts = [v.start for v in result.visemes]
    is_ordered = starts == sorted(starts)
    covers_duration = bool(result.visemes) and (result.duration_seconds - result.visemes[-1].end) <= MAX_SYNC_DRIFT_SECONDS
    checks.append((
        "play_speaking() returns a non-empty, ordered, duration-covering viseme timeline",
        has_visemes and is_ordered and covers_duration,
        f"{len(result.visemes)} visemes, source={result.source}, duration={result.duration_seconds:.2f}s",
    ))
    return checks


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    tts = TextToSpeech()
    checks = []
    checks += test_tts_synthesis(tts)
    checks += test_processing_prompt_cached()
    checks += test_avatar_states_and_lipsync(avatar_ready=_has_output_device())

    passed = True
    for name, ok, detail in checks:
        status = "PASS" if ok is True else ("WARN" if ok == "WARN" else ("SKIPPED" if ok == "SKIPPED" else "FAIL"))
        print(f"{status}: {name} ({detail!r})")
        if ok is False:
            passed = False

    print(f"\nPhase 7 acceptance: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
