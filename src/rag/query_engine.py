"""Knowledge Base (PDF RAG) query engine (module 5.5 / Phase 5).

Query-time half of Section 5.5's spec:

    1. Embed the user's transcript.
    2. Retrieve top-k (e.g., 4) most similar chunks.
    3. Feed transcript + retrieved chunks into local LLM (via `Ollama`) with
       a strict system prompt: "Answer only using the provided context. If
       the answer is not in the context, say you don't have that
       information. Do not speculate."
    4. If the retrieval similarity score is too low, skip generation and go
       straight to fallback -- this avoids hallucinated answers.

This module only decides whether a grounded answer exists and, if so, what
it is -- it does not speak the answer (`response/tts.py`, later phase) or
choose the fixed "not available" boundary wording (`5.6` Fallback Handler,
later phase). Per Phase 5 step 2, on a miss it signals "no answer
available" (`QueryResult.has_answer = False`) up to the orchestrator/caller,
which is responsible for invoking the actual Fallback Handler.

Usage:
    python -m src.rag.query_engine
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import chromadb
import ollama

from .ingest import COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, DEFAULT_VECTOR_STORE_DIR

logger = logging.getLogger(__name__)

# Section 5.5 step 3, verbatim, plus a citation clause for FR6 ("citing that
# it is based on official NAB documents") -- the citation instruction is
# additive to, not a replacement of, the spec's exact anti-hallucination
# wording.
SYSTEM_PROMPT = (
    "Answer only using the provided context. If the answer is not in the "
    "context, say you don't have that information. Do not speculate. "
    "The context is drawn from NAB's official public documents -- when you "
    "do answer, make clear the information comes from official NAB "
    "documents."
)

# Local LLM per Section 4: "Ollama running Llama 3.1 8B or Mistral 7B
# (quantized GGUF), called via the ollama Python client."
DEFAULT_LLM_MODEL = "llama3.1:8b"

# Phrases indicating the LLM itself declined to answer from context, even
# though retrieval similarity cleared the threshold below (e.g. the
# retrieved chunks were topically close but didn't actually contain the
# answer). Treating these as "no answer" -- not a spoken RAG answer -- is
# what makes the 0%-hallucination acceptance bar (Phase 5 step 4) achievable
# in practice, since a small local LLM can ignore the system prompt's
# strictness and answer vaguely instead of cleanly refusing upstream.
_NO_ANSWER_PHRASES = [
    "don't have that information",
    "do not have that information",
    "not mentioned in the context",
    "not in the context",
    "not in the provided context",
    "no information",
    "cannot find",
    "can't find",
    "not available in the",
    "context does not",
    "context doesn't",
    "i don't know",
    "unable to find",
    "not provided in the context",
    "not specified in the context",
]


@dataclass
class QueryEngineConfig:
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL  # must match ingest.py's -- see ingest.py's comment
    collection_name: str = COLLECTION_NAME
    top_k: int = 4  # "top-k (e.g., 4)" per Section 5.5
    similarity_threshold: float = 0.40  # cosine similarity floor; calibrated empirically (see Phase 5 test) -- clear separation observed between grounded (~0.5-0.8) and unrelated (~0.1-0.3) queries against the sample knowledge base
    llm_model: str = DEFAULT_LLM_MODEL
    llm_temperature: float = 0.0  # deterministic, factual -- matches the "do not speculate" instruction


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    page: int
    similarity: float  # cosine similarity, -1..1 (typically 0..1 for these embeddings)


@dataclass
class QueryResult:
    query: str
    has_answer: bool
    answer: Optional[str]
    chunks: List[RetrievedChunk] = field(default_factory=list)
    best_similarity: float = 0.0
    # "answered" | "empty_query" | "below_similarity_threshold" | "llm_declined" | "llm_unavailable"
    reason: str = ""


class QueryEngine:
    """Retrieves grounded context from the Chroma knowledge base built by
    `ingest.py` and generates an answer via a local Ollama LLM, or signals
    that no grounded answer is available.

    Loads the embedding model and opens the Chroma collection once at
    construction (mirrors the load-once pattern used throughout this
    codebase -- `SpeechToText`, `IntentRouter`, `KnowledgeBaseIngestor`).
    """

    def __init__(self, config: Optional[QueryEngineConfig] = None):
        self.config = config or QueryEngineConfig()

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.config.embedding_model_name)

        client = chromadb.PersistentClient(path=str(self.config.vector_store_dir))
        try:
            self._collection = client.get_collection(self.config.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Knowledge base collection {self.config.collection_name!r} not found under "
                f"{self.config.vector_store_dir}. Run `python -m src.rag.ingest` first."
            ) from exc

        logger.info(
            "QueryEngine ready: %d chunk(s) in collection, top_k=%d, similarity_threshold=%.2f, llm_model=%s",
            self._collection.count(),
            self.config.top_k,
            self.config.similarity_threshold,
            self.config.llm_model,
        )

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Embeds `query` and returns its top-k most similar chunks, best
        (highest similarity) first."""
        query_embedding = self._model.encode(query, normalize_embeddings=True)
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.config.top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        return [
            RetrievedChunk(
                text=doc,
                source_file=meta["source_file"],
                page=meta["page"],
                similarity=1.0 - distance,  # collection uses cosine space (see ingest.py) -> distance = 1 - similarity
            )
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def answer(self, query: str) -> QueryResult:
        """Full query-time pipeline: retrieve -> similarity gate -> grounded
        LLM generation, or a structured "no answer available" result."""
        query = (query or "").strip()
        if not query:
            return QueryResult(query=query, has_answer=False, answer=None, reason="empty_query")

        chunks = self.retrieve(query)
        best_similarity = chunks[0].similarity if chunks else 0.0

        if not chunks or best_similarity < self.config.similarity_threshold:
            logger.info(
                "No grounded answer: best_similarity=%.3f < threshold=%.2f for %r",
                best_similarity,
                self.config.similarity_threshold,
                query,
            )
            return QueryResult(
                query=query,
                has_answer=False,
                answer=None,
                chunks=chunks,
                best_similarity=best_similarity,
                reason="below_similarity_threshold",
            )

        context = "\n\n".join(f"[Source: {c.source_file}, page {c.page}]\n{c.text}" for c in chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        try:
            response = ollama.chat(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": self.config.llm_temperature},
            )
            answer_text = response["message"]["content"].strip()
        except Exception:
            logger.exception("Ollama call failed (model=%s) -- treating as no answer available", self.config.llm_model)
            return QueryResult(
                query=query,
                has_answer=False,
                answer=None,
                chunks=chunks,
                best_similarity=best_similarity,
                reason="llm_unavailable",
            )

        if not answer_text or _looks_like_no_answer(answer_text):
            logger.info("LLM declined to answer despite similarity=%.3f: %r", best_similarity, answer_text)
            return QueryResult(
                query=query,
                has_answer=False,
                answer=None,
                chunks=chunks,
                best_similarity=best_similarity,
                reason="llm_declined",
            )

        return QueryResult(
            query=query,
            has_answer=True,
            answer=answer_text,
            chunks=chunks,
            best_similarity=best_similarity,
            reason="answered",
        )


def _looks_like_no_answer(answer_text: str) -> bool:
    normalized = answer_text.lower()
    return any(phrase in normalized for phrase in _NO_ANSWER_PHRASES)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252, which can't print Urdu script

    logging.basicConfig(level=logging.INFO)

    # DEFAULT_LLM_MODEL ("llama3.1:8b") is Section 4's recommended model;
    # override here to whatever's actually pulled on this dev machine
    # (`ollama list`) -- NFR7 extensibility means swapping the model name is
    # a config change, not a code change.
    engine = QueryEngine(QueryEngineConfig(llm_model="llama3.2:1b"))

    samples = [
        "How can I file a complaint with NAB?",
        "What is NAB's vision statement?",
        "What is the capital of Japan?",
    ]
    for q in samples:
        result = engine.answer(q)
        print(f"\nQ: {q}")
        print(f"has_answer={result.has_answer} reason={result.reason} best_similarity={result.best_similarity:.3f}")
        if result.has_answer:
            print(f"A: {result.answer}")
