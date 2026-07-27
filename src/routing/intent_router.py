"""Intent Recognition & Routing Engine (module 5.3 / Phase 3).

Decides, for a transcript, whether to: (a) refuse under the restricted-topic
policy, (b) play a known video, (c) answer from the PDF knowledge base
(RAG — implemented in a later phase), or (d) fall back with the "not
available" boundary message. Implements the four steps of Section 5.3
exactly, in order:

    Step 1 — restricted_filter.RestrictedTopicFilter runs first, always,
             independent of everything else (FR12).
    Step 2 — keyword/fuzzy match against `video_map.yaml` via `rapidfuzz`
             (`fuzz.ratio` over sliding word-windows — see
             `_keyword_match_score` for why plain `partial_ratio` isn't
             used directly), threshold 75.
    Step 3 — if no strong keyword hit, semantic similarity
             (`sentence-transformers`, multilingual model per Section 9
             review finding #1) between the transcript and both (a) video
             topic descriptions and (b) a "general knowledge question"
             prototype, to decide video vs. PDF-QA.
    Step 4 — if similarity to everything is below threshold, fallback.

This module only *decides the route* — it does not play videos, run RAG, or
speak the fallback message; those are later phases. It also does not
implement the `CLARIFY` state (that is an Orchestrator/Phase 8
responsibility per Section 9 review finding #2, decided from this module's
confidence scores).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
from rapidfuzz import fuzz

from .restricted_filter import DEFAULT_CONFIG_PATH as DEFAULT_RESTRICTED_CONFIG_PATH
from .restricted_filter import RestrictedTopicFilter

logger = logging.getLogger(__name__)

DEFAULT_VIDEO_MAP_PATH = Path(__file__).resolve().parents[2] / "config" / "video_map.yaml"

# Multilingual model per Section 9 review finding #1 / Section 5.5 — English-only
# models like all-MiniLM-L6-v2 retrieve poorly across English/Urdu, so routing
# uses the same multilingual family recommended there.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Prototype phrases (English + Urdu) representing "a general knowledge
# question answerable from NAB's public PDFs" (FR4b) — deliberately spans
# several distinct question shapes (mandate, law, complaints, definitions,
# vision) rather than one narrow theme, since Step 3 scores a query against
# the *nearest* prototype (see _best_pdf_qa_match), not a single averaged
# centroid that would blur these apart.
PDF_QA_PROTOTYPES = [
    "A general knowledge question about NAB's mandate, history, laws, or public information.",
    "Please explain NAB's role, functions, or anti-corruption process.",
    "What are the laws and procedures NAB follows?",
    "How can I file a complaint with NAB?",
    "What counts as corruption under NAB's law?",
    "What is NAB's vision and mission?",
    "NAB کے قانون اور طریقہ کار کے بارے میں ایک عام سوال۔",
    "NAB کا کردار اور ذمہ داریاں کیا ہیں؟",
    "میں نیب میں شکایت کیسے درج کروا سکتا ہوں؟",
    "بدعنوانی کی تعریف کیا ہے؟",
]


def _word_windows(text: str, size: int) -> List[str]:
    """Splits `text` into space-joined sliding windows of `size` words —
    e.g. `_word_windows("what laws does nab operate", 2)` yields "what
    laws", "laws does", "does nab", "nab operate"."""
    words = re.findall(r"[\w']+", text.lower())
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def _keyword_match_score(text: str, keyword: str) -> float:
    """Best `fuzz.ratio` between `keyword` and any same-length word-window of
    `text`, 0-100.

    A plain `fuzz.partial_ratio(text, keyword)` over the raw strings looks
    tempting for "is this keyword phrase present in this longer utterance,"
    but it scores whichever substring of `text` best aligns with `keyword`
    without penalizing how much of `keyword` itself went unmatched — so a
    short, filler-word-heavy keyword like "what is nab" scores ~85 against
    a wholly unrelated query like "what is the weather today?" purely
    because both share "what is". Comparing `keyword` against equal-length
    windows of `text` with the (non-partial) `fuzz.ratio` instead requires
    the *whole* keyword to resemble the *whole* window, so shared filler
    words alone can't inflate the score — the window still has to be a good
    overall match to the keyword's content words, not just its first two.
    """
    kw_words = re.findall(r"[\w']+", keyword.lower())
    if not kw_words:
        return 0.0
    best = 0.0
    for window in _word_windows(text, len(kw_words)):
        score = fuzz.ratio(window, keyword.lower())
        if score > best:
            best = score
    return best


class IntentCategory(str, Enum):
    RESTRICTED = "restricted"
    VIDEO = "video"
    PDF_QA = "pdf_qa"
    FALLBACK = "fallback"


@dataclass
class VideoTopic:
    topic: str
    keywords: List[str]
    file: str


@dataclass
class RoutingResult:
    category: IntentCategory
    confidence: float  # the score that decided the route, in [0, 1] or [0, 100] for keyword hits — see field below
    # Step 1 (restricted) fields
    restricted_category: Optional[str] = None
    restricted_response: Optional[str] = None
    # Step 2/3 (video) fields
    video_topic: Optional[str] = None
    video_file: Optional[str] = None
    matched_keyword: Optional[str] = None
    keyword_score: Optional[float] = None  # rapidfuzz fuzz.ratio (best word-window), 0-100
    # Step 3 (semantic) fields
    semantic_video_score: Optional[float] = None  # cosine similarity, -1..1
    semantic_pdf_score: Optional[float] = None
    method: str = ""  # "restricted" | "keyword" | "semantic" | "fallback" — how the decision was made


@dataclass
class IntentRouterConfig:
    video_map_path: Path = DEFAULT_VIDEO_MAP_PATH
    restricted_topics_path: Path = DEFAULT_RESTRICTED_CONFIG_PATH
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    keyword_threshold: float = 75.0  # rapidfuzz partial_ratio threshold, per Section 5.3 Step 2 ("e.g., 75")
    semantic_threshold: float = 0.45  # cosine similarity floor for Step 3/4; below this on both classes -> fallback


class IntentRouter:
    """Routes a transcript to RESTRICTED / VIDEO / PDF_QA / FALLBACK.

    Loads the restricted-topic filter and the sentence-transformers model
    once at construction (model loading, not inference, is the expensive
    part — mirrors `SpeechToText`'s load-once pattern in stt.py).
    """

    def __init__(self, config: Optional[IntentRouterConfig] = None):
        self.config = config or IntentRouterConfig()

        self.restricted_filter = RestrictedTopicFilter(self.config.restricted_topics_path)
        self.videos: List[VideoTopic] = _load_video_map(self.config.video_map_path)

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.config.embedding_model_name)

        video_descriptions = [f"{v.topic}. {', '.join(v.keywords)}" for v in self.videos]
        self._video_embeddings = (
            self._model.encode(video_descriptions, normalize_embeddings=True)
            if video_descriptions
            else np.zeros((0, self._model.get_sentence_embedding_dimension()))
        )
        # Kept as individual prototype embeddings (not mean-pooled into one
        # centroid) — the prototypes deliberately span several distinct
        # question shapes, and a query only needs to resemble the *nearest*
        # one to count as PDF-QA (see _best_pdf_qa_match).
        self._pdf_qa_embeddings = self._model.encode(PDF_QA_PROTOTYPES, normalize_embeddings=True)

        logger.info(
            "IntentRouter ready: %d video topics, keyword_threshold=%.0f, semantic_threshold=%.2f",
            len(self.videos),
            self.config.keyword_threshold,
            self.config.semantic_threshold,
        )

    def route(self, text: str) -> RoutingResult:
        text = (text or "").strip()

        # Step 1 — restricted-topic filter, always first, independent of everything else (FR12).
        restricted = self.restricted_filter.check(text)
        if restricted.is_restricted:
            return RoutingResult(
                category=IntentCategory.RESTRICTED,
                confidence=1.0,
                restricted_category=restricted.category,
                restricted_response=restricted.response,
                matched_keyword=restricted.matched_term,
                method="restricted",
            )

        if not text:
            return RoutingResult(category=IntentCategory.FALLBACK, confidence=0.0, method="fallback")

        # Step 2 — keyword/fuzzy match against video_map.yaml.
        best_video_idx, best_keyword, keyword_score = self._best_keyword_match(text)
        if keyword_score >= self.config.keyword_threshold:
            video = self.videos[best_video_idx]
            logger.info("Routed VIDEO via keyword: topic=%s score=%.1f keyword=%r", video.topic, keyword_score, best_keyword)
            return RoutingResult(
                category=IntentCategory.VIDEO,
                confidence=keyword_score / 100.0,
                video_topic=video.topic,
                video_file=video.file,
                matched_keyword=best_keyword,
                keyword_score=keyword_score,
                method="keyword",
            )

        # Step 3 — semantic similarity: transcript vs. video topic descriptions, and vs. the PDF-QA prototype.
        query_embedding = self._model.encode(text, normalize_embeddings=True)
        semantic_video_score, semantic_video_idx = self._best_semantic_video_match(query_embedding)
        semantic_pdf_score = self._best_pdf_qa_match(query_embedding)

        if semantic_video_score >= self.config.semantic_threshold and semantic_video_score >= semantic_pdf_score:
            video = self.videos[semantic_video_idx]
            logger.info(
                "Routed VIDEO via semantic match: topic=%s score=%.3f (pdf_score=%.3f)",
                video.topic,
                semantic_video_score,
                semantic_pdf_score,
            )
            return RoutingResult(
                category=IntentCategory.VIDEO,
                confidence=semantic_video_score,
                video_topic=video.topic,
                video_file=video.file,
                keyword_score=keyword_score,
                semantic_video_score=semantic_video_score,
                semantic_pdf_score=semantic_pdf_score,
                method="semantic",
            )

        if semantic_pdf_score >= self.config.semantic_threshold:
            logger.info("Routed PDF_QA via semantic match: score=%.3f (video_score=%.3f)", semantic_pdf_score, semantic_video_score)
            return RoutingResult(
                category=IntentCategory.PDF_QA,
                confidence=semantic_pdf_score,
                keyword_score=keyword_score,
                semantic_video_score=semantic_video_score,
                semantic_pdf_score=semantic_pdf_score,
                method="semantic",
            )

        # Step 4 — below threshold on everything -> fallback.
        logger.info("Routed FALLBACK: keyword_score=%.1f video_score=%.3f pdf_score=%.3f", keyword_score, semantic_video_score, semantic_pdf_score)
        return RoutingResult(
            category=IntentCategory.FALLBACK,
            confidence=max(semantic_video_score, semantic_pdf_score),
            keyword_score=keyword_score,
            semantic_video_score=semantic_video_score,
            semantic_pdf_score=semantic_pdf_score,
            method="fallback",
        )

    def _best_keyword_match(self, text: str) -> tuple[int, Optional[str], float]:
        best_idx, best_keyword, best_score = -1, None, 0.0
        for idx, video in enumerate(self.videos):
            for keyword in video.keywords:
                score = _keyword_match_score(text, keyword)
                if score > best_score:
                    best_idx, best_keyword, best_score = idx, keyword, score
        return best_idx, best_keyword, best_score

    def _best_semantic_video_match(self, query_embedding: np.ndarray) -> tuple[float, int]:
        if len(self.videos) == 0:
            return -1.0, -1
        scores = self._video_embeddings @ query_embedding
        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), best_idx

    def _best_pdf_qa_match(self, query_embedding: np.ndarray) -> float:
        scores = self._pdf_qa_embeddings @ query_embedding
        return float(np.max(scores))


def _load_video_map(path: Path) -> List[VideoTopic]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [
        VideoTopic(topic=v["topic"], keywords=list(v.get("keywords", [])), file=v["file"])
        for v in data.get("videos", [])
    ]


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print Urdu script

    logging.basicConfig(level=logging.INFO)

    router = IntentRouter()
    samples = [
        "Tell me about NAB's achievements",
        "kamyabi ke baare mein bataen",
        "NAB kya hai",
        "کیا آپ مجھے کیس نمبر 123 کے بارے میں بتا سکتے ہیں؟",
        "What is the weather today?",
        "How does NAB fight corruption?",
    ]
    for s in samples:
        result = router.route(s)
        print(f"{s!r} -> {result.category.value} (method={result.method}, confidence={result.confidence:.2f})")
