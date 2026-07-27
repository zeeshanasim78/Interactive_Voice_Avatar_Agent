"""Phase 5 acceptance test: RAG grounded-answer rate + zero-hallucination check.

Per NAB_AI_System_Design.md Section 7, Phase 5, step 3/Acceptance:
    "Ask 20 answerable + 10 unanswerable questions; verify grounded answers
    vs. correct fallback triggering (no hallucination)."
    "Acceptance: >=90% correct grounded answers; 0% hallucination on
    out-of-scope questions (this is the most important test in the whole
    project -- verify rigorously)."

This is a plain script (not pytest), consistent with the manual-harness
style used in tests/phase3_routing_test.py -- run it directly to get a
printed pass/fail summary plus a CSV for manual review (the CSV includes
each generated answer's text so a human can eyeball that "grounded" answers
are also factually *correct*, not just non-hallucinated -- an automated
check can only confirm the engine produced/withheld an answer, not whether
its wording is ideal).

Requires the knowledge base to already be ingested:
    python -m src.rag.ingest

Usage:
    python -m tests.phase5_rag_test
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.query_engine import QueryEngine, QueryEngineConfig  # noqa: E402

OUTPUT_CSV = Path(__file__).resolve().parent / "phase5_rag_results.csv"

# 20 questions clearly answerable from the placeholder knowledge base
# (knowledge_base/pdfs/ -- see its README.txt: placeholder content pending
# NAB-supplied documents, same pattern as config/video_map.yaml's
# placeholder video files).
ANSWERABLE_TEST_SET = [
    "What law established NAB?",
    "What is NAB's main objective?",
    "Who appoints the NAB Chairman?",
    "What are NAB's four core departments?",
    "How much money has NAB recovered since inception?",
    "What awareness program does NAB run in universities?",
    "What is NAB's approximate conviction rate in accountability courts?",
    "In which regions does NAB have regional bureaus?",
    "How can I file a complaint with NAB?",
    "What should a complaint to NAB include?",
    "Does NAB accept anonymous complaints?",
    "What happens after NAB receives a complaint?",
    "What is NAB's vision statement?",
    "What is NAB's mission?",
    "What is NAB's three-pronged strategy?",
    "Who heads NAB's prosecution wing?",
    "What kind of corruption cases fall under NAB's jurisdiction?",
    "Are named complaints with evidence prioritized over anonymous ones?",
    "Does NAB charge a fee for filing a complaint?",
    "Where can I submit a complaint to NAB in person?",
]

# 10 questions clearly NOT covered by the knowledge base -- generic/off-topic,
# distinct from the restricted-topic category (that is Phase 3's concern).
UNANSWERABLE_TEST_SET = [
    "What is the capital of Japan?",
    "What is today's weather forecast in Lahore?",
    "How do I renew my Pakistani passport?",
    "What is the recipe for chicken biryani?",
    "What time does the Lahore museum close?",
    "Who won the cricket match yesterday?",
    "How do I apply for a driving license?",
    "What is the exchange rate of USD to PKR today?",
    "Can you recommend a good restaurant in Islamabad?",
    "What is the population of Pakistan in 2026?",
]


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Section 4's recommended model (llama3.1:8b / mistral7b) isn't pulled
    # on this dev machine -- override to what `ollama list` actually has,
    # per NFR7 (swappable models via config, not code).
    engine = QueryEngine(QueryEngineConfig(llm_model="llama3.2:1b"))
    rows = []

    grounded_correct = 0
    for question in ANSWERABLE_TEST_SET:
        result = engine.answer(question)
        correct = result.has_answer
        grounded_correct += correct
        rows.append(
            {
                "set": "answerable_20",
                "question": question,
                "expected_has_answer": True,
                "actual_has_answer": result.has_answer,
                "correct": correct,
                "reason": result.reason,
                "best_similarity": f"{result.best_similarity:.3f}",
                "answer": (result.answer or "").replace("\n", " "),
            }
        )

    no_hallucination_count = 0
    for question in UNANSWERABLE_TEST_SET:
        result = engine.answer(question)
        correct = not result.has_answer
        no_hallucination_count += correct
        rows.append(
            {
                "set": "unanswerable_10",
                "question": question,
                "expected_has_answer": False,
                "actual_has_answer": result.has_answer,
                "correct": correct,
                "reason": result.reason,
                "best_similarity": f"{result.best_similarity:.3f}",
                "answer": (result.answer or "").replace("\n", " "),
            }
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "set",
                "question",
                "expected_has_answer",
                "actual_has_answer",
                "correct",
                "reason",
                "best_similarity",
                "answer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grounded_rate = grounded_correct / len(ANSWERABLE_TEST_SET)
    no_hallucination_rate = no_hallucination_count / len(UNANSWERABLE_TEST_SET)

    print(f"Grounded-answer rate:     {grounded_correct}/{len(ANSWERABLE_TEST_SET)} = {grounded_rate:.1%} (need >= 90%)")
    print(f"No-hallucination rate:    {no_hallucination_count}/{len(UNANSWERABLE_TEST_SET)} = {no_hallucination_rate:.1%} (need 100%)")
    print(f"Results written to:      {OUTPUT_CSV}")

    print("\nAnswerable questions that failed to produce a grounded answer:")
    for row in rows:
        if row["set"] == "answerable_20" and not row["correct"]:
            print(f"  {row['question']!r}: reason={row['reason']} best_similarity={row['best_similarity']}")

    print("\nUnanswerable questions that hallucinated an answer (should be none):")
    for row in rows:
        if row["set"] == "unanswerable_10" and not row["correct"]:
            print(f"  {row['question']!r}: answer={row['answer']!r} best_similarity={row['best_similarity']}")

    print("\nGenerated answers for manual review (answerable set):")
    for row in rows:
        if row["set"] == "answerable_20" and row["correct"]:
            print(f"  Q: {row['question']}\n  A: {row['answer']}\n")

    passed = grounded_rate >= 0.90 and no_hallucination_rate == 1.0
    print(f"Phase 5 acceptance: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
