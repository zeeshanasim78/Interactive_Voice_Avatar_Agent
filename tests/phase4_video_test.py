"""Phase 4 acceptance test: video-to-keyword mapping + interrupt latency.

Per NAB_AI_System_Design.md Section 7, Phase 4, step 3/Acceptance:
    "Trigger each of the 4+ videos by voice and by keyword text directly;
    confirm correct file plays and can be interrupted."
    "Acceptance: 100% correct video-to-keyword mapping; interrupt works
    within 1 second."

Two parts, tested independently:

1. Mapping accuracy — for a set of keyword/voice-style phrases (English +
   Urdu) per configured topic, `IntentRouter.route()` (the same routing
   used for live voice input, per Section 5.3 Step 2) must resolve to the
   VIDEO category with the correct `video_file`. This needs no VLC install.

2. Playback + interrupt latency — for each configured video whose file
   actually exists on disk, `VideoPlayer.play()`/`stop()` is exercised and
   the stop latency is measured against the < 1s acceptance bar. Per the
   design doc's Assumptions (2.3), NAB's real video files aren't available
   in this repo yet, and per Section 9 review finding #6 a missing asset
   must degrade gracefully rather than crash — so entries with no file on
   disk (or no working local VLC install) are reported as SKIPPED, not
   FAILED, and don't count against the mapping-accuracy score.

Usage:
    python -m tests.phase4_video_test
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.routing.intent_router import IntentCategory, IntentRouter  # noqa: E402
from src.video.player import VideoLibrary, VideoPlaybackError, VideoPlayer  # noqa: E402

OUTPUT_CSV = Path(__file__).resolve().parent / "phase4_video_results.csv"

# Keyword/voice-style phrases per configured topic (English + Urdu), mirroring
# the keywords in config/video_map.yaml plus natural full-sentence phrasing —
# i.e. both "by keyword text directly" and "by voice" (voice input reaches the
# router as text after STT, so a transcript-shaped phrase exercises the same
# path). At least 2 phrases per topic, covering all 4+ configured videos.
VIDEO_TEST_SET = [
    ("achievements", "NAB Achievements"),
    ("Tell me about NAB's success stories", "NAB Achievements"),
    ("نیب کی کامیابیاں بتائیں", "NAB Achievements"),
    ("AI investigation", "NAB AI Investigation System"),
    ("Tell me about the artificial intelligence system", "NAB AI Investigation System"),
    ("اے آئی تحقیقاتی نظام کے بارے میں بتائیں", "NAB AI Investigation System"),
    ("what is nab", "About NAB"),
    ("Can you tell me about NAB?", "About NAB"),
    ("نیب کیا ہے", "About NAB"),
    ("how nab works", "Working of NAB"),
    ("Explain the working of NAB", "Working of NAB"),
    ("نیب کیسے کام کرتا ہے", "Working of NAB"),
]


def test_mapping_accuracy(router: IntentRouter, library: VideoLibrary) -> tuple[int, list[dict]]:
    rows = []
    correct = 0
    for text, expected_topic in VIDEO_TEST_SET:
        result = router.route(text)
        expected_file = library.entry_for_topic(expected_topic).file
        actual_file = Path(result.video_file) if result.video_file else None
        is_correct = result.category == IntentCategory.VIDEO and actual_file == expected_file
        correct += is_correct
        rows.append(
            {
                "check": "mapping",
                "text": text,
                "expected_topic": expected_topic,
                "actual_category": result.category.value,
                "actual_topic": result.video_topic or "",
                "correct": is_correct,
                "method": result.method,
                "detail": "",
            }
        )
    return correct, rows


def test_playback_and_interrupt(library: VideoLibrary) -> list[dict]:
    rows = []
    for entry in library.entries:
        path = library.resolved_path(entry)
        if not path.is_file():
            rows.append(
                {
                    "check": "playback",
                    "text": "",
                    "expected_topic": entry.topic,
                    "actual_category": "",
                    "actual_topic": "",
                    "correct": "SKIPPED",
                    "method": "",
                    "detail": f"video file not present on disk: {path}",
                }
            )
            continue

        try:
            player = VideoPlayer()
        except VideoPlaybackError as exc:
            rows.append(
                {
                    "check": "playback",
                    "text": "",
                    "expected_topic": entry.topic,
                    "actual_category": "",
                    "actual_topic": "",
                    "correct": "SKIPPED",
                    "method": "",
                    "detail": f"VLC unavailable: {exc}",
                }
            )
            continue

        try:
            player.play(path)
            playing_ok = player.is_playing()
            time.sleep(0.2)
            stop_start = time.monotonic()
            player.stop()
            stop_elapsed = time.monotonic() - stop_start
            interrupt_ok = stop_elapsed < 1.0 and not player.is_playing()
            rows.append(
                {
                    "check": "playback",
                    "text": "",
                    "expected_topic": entry.topic,
                    "actual_category": "",
                    "actual_topic": "",
                    "correct": bool(playing_ok and interrupt_ok),
                    "method": "",
                    "detail": f"played={playing_ok} stop_latency={stop_elapsed:.3f}s",
                }
            )
        except (FileNotFoundError, VideoPlaybackError) as exc:
            rows.append(
                {
                    "check": "playback",
                    "text": "",
                    "expected_topic": entry.topic,
                    "actual_category": "",
                    "actual_topic": "",
                    "correct": False,
                    "method": "",
                    "detail": f"playback error: {exc}",
                }
            )
        finally:
            player.release()
    return rows


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    library = VideoLibrary()
    router = IntentRouter()

    mapping_correct, mapping_rows = test_mapping_accuracy(router, library)
    playback_rows = test_playback_and_interrupt(library)
    rows = mapping_rows + playback_rows

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["check", "text", "expected_topic", "actual_category", "actual_topic", "correct", "method", "detail"],
        )
        writer.writeheader()
        writer.writerows(rows)

    mapping_accuracy = mapping_correct / len(VIDEO_TEST_SET)
    print(f"Video mapping accuracy:  {mapping_correct}/{len(VIDEO_TEST_SET)} = {mapping_accuracy:.1%} (need 100%)")

    print("\nMisrouted phrases:")
    for row in mapping_rows:
        if not row["correct"]:
            print(f"  {row['text']!r}: expected={row['expected_topic']} actual_topic={row['actual_topic']!r} category={row['actual_category']}")

    print("\nPlayback/interrupt checks:")
    for row in playback_rows:
        print(f"  {row['expected_topic']}: correct={row['correct']} {row['detail']}")

    playback_failed = any(row["correct"] is False for row in playback_rows)
    playback_all_skipped = all(row["correct"] == "SKIPPED" for row in playback_rows)

    print(f"\nResults written to: {OUTPUT_CSV}")

    passed = mapping_accuracy == 1.0 and not playback_failed
    print(f"\nPhase 4 mapping acceptance: {'PASS' if mapping_accuracy == 1.0 else 'FAIL'} (100% required)")
    if playback_all_skipped:
        print(
            "Phase 4 playback/interrupt acceptance: UNVERIFIED — no configured video file + working local "
            "VLC install was available in this environment (see docstring). Re-run on the target kiosk "
            "once real videos and a matching-architecture VLC install are in place."
        )
    else:
        print(f"Phase 4 playback/interrupt acceptance: {'PASS' if not playback_failed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
