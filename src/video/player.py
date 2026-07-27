"""Video Library Manager — playback (module 5.4 / Phase 4).

Config-driven full-screen video playback via `python-vlc`. `VideoLibrary`
loads the same `config/video_map.yaml` used by `IntentRouter`
(src/routing/intent_router.py, Step 2) to resolve a topic name to its
video file; `VideoPlayer` plays that file full-screen through the local
VLC installation and exposes `stop()` for the "interrupt and return to
listening" UX required by FR5.

This module owns *playback only*. Topic/keyword -> file matching during a
live conversation happens in `IntentRouter.route()` (Section 5.3 Step 2);
`VideoLibrary` re-loads the same YAML independently so a video can also be
played directly by topic name (e.g. a Phase 9 topic-button click, or this
module's own test harness) without needing a full IntentRouter /
sentence-transformers model just to resolve a filename.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_MAP_PATH = PROJECT_ROOT / "config" / "video_map.yaml"


class VideoLibraryError(RuntimeError):
    """Raised for video_map.yaml problems: missing/empty config, unknown topic."""


class VideoPlaybackError(RuntimeError):
    """Raised when libVLC can't be initialized or playback fails to start.

    Per Section 9 review finding #6, callers (ultimately the Orchestrator)
    should catch this and `FileNotFoundError` and fall back to a spoken
    apology + log entry rather than crashing the kiosk app.
    """


@dataclass
class VideoEntry:
    topic: str
    keywords: List[str]
    file: Path  # as given in config; relative paths are resolved against project root


class VideoLibrary:
    """Loads `config/video_map.yaml` (module 5.4) and resolves topics to files."""

    def __init__(self, config_path: Path = DEFAULT_VIDEO_MAP_PATH, project_root: Path = PROJECT_ROOT):
        self.config_path = Path(config_path)
        self.project_root = Path(project_root)
        self.entries: List[VideoEntry] = self._load()

    def _load(self) -> List[VideoEntry]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        entries = [
            VideoEntry(topic=v["topic"], keywords=list(v.get("keywords", [])), file=Path(v["file"]))
            for v in data.get("videos", [])
        ]
        if not entries:
            raise VideoLibraryError(f"{self.config_path}: no videos configured under 'videos:'")
        logger.info("Loaded video library: %d topics from %s", len(entries), self.config_path)
        return entries

    def topics(self) -> List[str]:
        return [e.topic for e in self.entries]

    def entry_for_topic(self, topic: str) -> VideoEntry:
        for entry in self.entries:
            if entry.topic == topic:
                return entry
        raise VideoLibraryError(f"Unknown video topic: {topic!r}")

    def resolved_path(self, entry_or_topic: Union[VideoEntry, str]) -> Path:
        """Resolves a `VideoEntry` (or topic name) to an absolute file path."""
        entry = entry_or_topic if isinstance(entry_or_topic, VideoEntry) else self.entry_for_topic(entry_or_topic)
        return entry.file if entry.file.is_absolute() else self.project_root / entry.file

    def missing_files(self) -> List[VideoEntry]:
        """Entries whose configured file does not exist on disk — for a
        startup/admin sanity check so a missing video is caught proactively
        rather than discovered mid-interaction (Section 9 finding #6)."""
        return [e for e in self.entries if not self.resolved_path(e).is_file()]


@dataclass
class VideoPlayerConfig:
    fullscreen: bool = True
    vlc_args: List[str] = field(default_factory=lambda: ["--no-video-title-show", "--quiet"])


class VideoPlayer:
    """Full-screen playback via `python-vlc` (Section 5.4).

    `stop()` is meant to be bound to an interrupt key/button by the kiosk UI
    (Phase 9) so the user can stop mid-video and return to listening (FR5).
    """

    def __init__(self, config: Optional[VideoPlayerConfig] = None):
        self.config = config or VideoPlayerConfig()
        self._current_file: Optional[Path] = None

        try:
            import vlc  # imported lazily so importing this module never requires a working VLC install
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise VideoPlaybackError(
                f"python-vlc is not installed or failed to load libVLC ({exc}). Install the VLC media "
                "player matching your Python interpreter's architecture (e.g. 32-bit VLC + 64-bit Python "
                "will fail to load) and ensure it is on PATH, then `pip install python-vlc`."
            ) from exc

        try:
            self._instance = vlc.Instance(self.config.vlc_args)
            self._player = self._instance.media_player_new()
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise VideoPlaybackError(f"Could not initialize libVLC: {exc}") from exc

    def play(self, file: Union[Path, str], *, fullscreen: Optional[bool] = None) -> None:
        """Loads and plays `file` full-screen.

        Raises `FileNotFoundError` if the file doesn't exist and
        `VideoPlaybackError` if VLC fails to start it.
        """
        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"Video file not found: {path}")

        media = self._instance.media_new(str(path))
        self._player.set_media(media)
        self._player.set_fullscreen(self.config.fullscreen if fullscreen is None else fullscreen)

        if self._player.play() == -1:
            raise VideoPlaybackError(f"VLC failed to start playback for {path}")

        self._current_file = path
        logger.info("Playing video: %s", path)

    def stop(self) -> None:
        """Stops playback immediately (the interrupt action, FR5). No-op if nothing is playing."""
        self._player.stop()
        if self._current_file is not None:
            logger.info("Playback stopped: %s", self._current_file)
        self._current_file = None

    def is_playing(self) -> bool:
        return bool(self._player.is_playing())

    def wait_until_finished(self, poll_interval_s: float = 0.1, timeout_s: Optional[float] = None) -> bool:
        """Blocks until playback ends on its own (True) or `timeout_s` elapses
        (False). For test harnesses / non-UI callers — the real kiosk UI
        (Phase 9) will drive this from its own event loop instead."""
        start = time.monotonic()
        while self.is_playing():
            if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
                return False
            time.sleep(poll_interval_s)
        return True

    def release(self) -> None:
        """Releases the underlying VLC player/instance. Call on app shutdown."""
        self._player.release()
        self._instance.release()

    def __enter__(self) -> "VideoPlayer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


if __name__ == "__main__":
    # Manual test harness for Phase 4 acceptance criteria: lists configured
    # topics (flagging any missing files), then, given a topic name as
    # argv[1], plays it full-screen and lets you press Enter to interrupt —
    # confirm the correct file plays and that stop() takes effect immediately.
    import sys

    logging.basicConfig(level=logging.INFO)

    library = VideoLibrary()
    print("Configured topics:")
    for entry in library.entries:
        exists = library.resolved_path(entry).is_file()
        print(f"  - {entry.topic!r} -> {entry.file} ({'found' if exists else 'MISSING on disk'})")

    if len(sys.argv) > 1:
        topic = sys.argv[1]
        entry = library.entry_for_topic(topic)
        player = VideoPlayer()
        try:
            player.play(library.resolved_path(entry))
            input("Playing... press Enter to interrupt and stop.\n")
            stop_start = time.monotonic()
            player.stop()
            print(f"Stopped in {time.monotonic() - stop_start:.3f}s (need < 1s)")
        finally:
            player.release()
    else:
        print('\nRun with a topic name as argv[1] to play it, e.g.:')
        print('  python -m src.video.player "NAB Achievements"')
