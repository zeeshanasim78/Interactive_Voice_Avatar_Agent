"""Knowledge Base (PDF RAG) ingestion pipeline (module 5.5 / Phase 5).

Run once, whenever PDFs are added/updated under `knowledge_base/pdfs/`
(Section 7, Phase 5, step 1):

    1. Load PDFs (`pdfplumber`).
    2. Extract text, split into ~300-500 token chunks with overlap using
       `langchain-text-splitters` (the small, focused package -- not the
       full `langchain` meta-package, per Section 4/15's Python 3.12
       dependency-risk rationale).
    3. Embed chunks (`sentence-transformers`, multilingual model to cover
       Urdu+English per Section 9 review finding #1) and store in `Chroma`
       (local persistent DB).

Chunking is done per PDF page rather than over the whole-document text.
This keeps each chunk's `page` metadata exact (needed for citing "based on
official NAB documents," FR6) at the cost of overlap never spanning a page
boundary -- an acceptable simplification since NAB brochures/reports are
organized in self-contained pages/sections, not paragraphs that
deliberately straddle a page break.

Usage:
    python -m src.rag.ingest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_PDF_DIR = Path(__file__).resolve().parents[2] / "knowledge_base" / "pdfs"
DEFAULT_VECTOR_STORE_DIR = Path(__file__).resolve().parents[2] / "knowledge_base" / "vector_store"
COLLECTION_NAME = "nab_knowledge_base"

# Must match query_engine.py's embedding model -- a query embedded with a
# different model than the one used to build the index would not be
# comparable to the stored vectors. Same multilingual family already used by
# src/routing/intent_router.py (Section 9 review finding #1: English-only
# models retrieve poorly across English/Urdu).
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# "~300-500 token chunks with overlap" (Section 5.5). Measured in the actual
# embedding model's subword tokens (via from_huggingface_tokenizer) rather
# than raw characters or a separate tokenizer library like `tiktoken`, so
# the chunk boundary matches what the embedding model itself will see.
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60


@dataclass
class IngestConfig:
    pdf_dir: Path = DEFAULT_PDF_DIR
    vector_store_dir: Path = DEFAULT_VECTOR_STORE_DIR
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS


@dataclass
class IngestSummary:
    num_pdfs: int
    num_pages: int
    num_chunks: int
    skipped_files: List[str]


class KnowledgeBaseIngestor:
    """Loads PDFs from `knowledge_base/pdfs/`, chunks and embeds them, and
    (re)builds the Chroma collection queried by `query_engine.py`.

    Loads the embedding model once at construction -- model loading, not
    embedding, is the expensive part (mirrors `SpeechToText`'s load-once
    pattern in stt.py and `IntentRouter`'s in intent_router.py).
    """

    def __init__(self, config: IngestConfig | None = None):
        self.config = config or IngestConfig()

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.config.embedding_model_name)
        self._splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self._model.tokenizer,
            chunk_size=self.config.chunk_size_tokens,
            chunk_overlap=self.config.chunk_overlap_tokens,
        )

        self._client = chromadb.PersistentClient(path=str(self.config.vector_store_dir))

    def ingest_all(self) -> IngestSummary:
        """(Re)builds the knowledge-base collection from every PDF currently
        under `pdf_dir`. Full rebuild each run keeps this idempotent and
        simple to reason about at NAB's document-library scale, consistent
        with "run this once whenever docs are added/updated" (Phase 5 step
        1) rather than diffing against the previous ingestion.
        """
        if _collection_exists(self._client, COLLECTION_NAME):
            self._client.delete_collection(COLLECTION_NAME)
        collection = self._client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we always supply our own embeddings, never chroma's default
        )

        pdf_paths = sorted(self.config.pdf_dir.glob("*.pdf"))
        skipped_files: List[str] = []
        num_pages = 0
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for pdf_path in pdf_paths:
            try:
                pages_text = _extract_pages(pdf_path)
            except Exception:
                logger.exception("Skipping unreadable/corrupt PDF: %s", pdf_path)
                skipped_files.append(pdf_path.name)
                continue

            num_pages += len(pages_text)
            for page_number, page_text in enumerate(pages_text, start=1):
                if not page_text.strip():
                    continue
                for chunk_index, chunk_text in enumerate(self._splitter.split_text(page_text)):
                    ids.append(f"{pdf_path.name}::p{page_number}::c{chunk_index}")
                    documents.append(chunk_text)
                    metadatas.append(
                        {
                            "source_file": pdf_path.name,
                            "page": page_number,
                            "chunk_index": chunk_index,
                        }
                    )

        if documents:
            embeddings = self._model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        summary = IngestSummary(
            num_pdfs=len(pdf_paths) - len(skipped_files),
            num_pages=num_pages,
            num_chunks=len(documents),
            skipped_files=skipped_files,
        )
        logger.info(
            "Ingested %d PDF(s), %d page(s) -> %d chunk(s) into %r (skipped: %s)",
            summary.num_pdfs,
            summary.num_pages,
            summary.num_chunks,
            COLLECTION_NAME,
            summary.skipped_files or "none",
        )
        return summary


def _extract_pages(pdf_path: Path) -> List[str]:
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _collection_exists(client: chromadb.ClientAPI, name: str) -> bool:
    return any(c.name == name for c in client.list_collections())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ingestor = KnowledgeBaseIngestor()
    result = ingestor.ingest_all()

    print(f"PDFs ingested:   {result.num_pdfs}")
    print(f"Pages processed: {result.num_pages}")
    print(f"Chunks stored:   {result.num_chunks}")
    if result.skipped_files:
        print(f"Skipped files:   {', '.join(result.skipped_files)}")
