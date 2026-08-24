"""Phase 6: measure retrieval and answer quality against a known-good set.

    python -m scripts.evaluate                      # reranked, with generation
    python -m scripts.evaluate --retrieval-only     # fast: skip the LLM
    python -m scripts.evaluate --compare            # all three modes, side by side
    python -m scripts.evaluate --verify-anchors     # check ground truth still resolves
    python -m scripts.evaluate --json runs/x.json   # dump raw results

This is a script rather than a test suite: it measures, it does not
pass/fail, and the project has no test infrastructure to slot into.

Ground truth lives in eval/eval_set.json, anchored on source_id plus a
distinctive text snippet -- never chunk_id, which is DB-assigned and
would shift on re-ingestion.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select

from app import rag
from app.db import SessionLocal
from app.embeddings import embed_texts
from app.llm import generate
from app.models import Chunk, Document

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"
MODES = ("vector", "hybrid", "reranked")

# The extracted PDF text carries two opposite artifact classes: words split
# by line breaks and hyphenation ("land- ing", "extravehicul ar") and words
# run together with no space at all ("Thepurposeof", "missionwasto"). Merely
# collapsing whitespace fixes the first and not the second, so both sides of
# every comparison drop whitespace and hyphens entirely. Applied identically
# to needle and haystack, this cannot create a mismatch, and snippets are
# long enough that spurious cross-word matches are not a practical risk.
_STRIP = re.compile(r"[\s‐-―\-]+")


def normalize(text: str) -> str:
    return _STRIP.sub("", text.lower())


def load_eval_set(path: Path) -> list[dict]:
    with path.open() as fh:
        return json.load(fh)["questions"]


def source_id_by_chunk(db) -> dict[int, str]:
    """chunk_id -> owning document's source_id.

    Built here rather than added to rag.RetrievedChunk so the /ask response
    shape stays untouched by the eval code.
    """
    rows = db.execute(
        select(Chunk.id, Document.source_id).join(
            Document, Chunk.document_id == Document.id
        )
    ).all()
    return {chunk_id: source_id for chunk_id, source_id in rows}


def matched_anchors(chunk, question: dict, source_by_chunk: dict[int, str]) -> set[int]:
    """Indices of the question's ground-truth anchors this chunk satisfies."""
    text = normalize(chunk.text)
    source_id = source_by_chunk.get(chunk.chunk_id)
    return {
        i
        for i, anchor in enumerate(question["relevant"])
        if anchor["source_id"] == source_id and normalize(anchor["must_contain"]) in text
    }


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def score_answer(answer: str, question: dict) -> dict:
    """Lightweight, local, deterministic answer scoring -- no LLM judge."""
    facts = question["expected_facts"]
    normalized = normalize(answer)
    found = [f for f in facts if normalize(f) in normalized]
    vectors = embed_texts([answer, question["reference_answer"]])
    return {
        "facts_present": len(found) / len(facts) if facts else None,
        "facts_missing": [f for f in facts if f not in found],
        "ref_similarity": cosine(vectors[0], vectors[1]),
    }


def evaluate_question(
    question: dict,
    db,
    mode: str,
    top_k: int,
    source_by_chunk: dict[int, str],
    with_generation: bool,
) -> dict:
    chunks = rag.retrieve(question["question"], db, top_k=top_k, mode=mode)

    first_hit_rank = None
    found: set[int] = set()
    for rank, chunk in enumerate(chunks, start=1):
        hits = matched_anchors(chunk, question, source_by_chunk)
        if hits and first_hit_rank is None:
            first_hit_rank = rank
        found |= hits

    total_anchors = len(question["relevant"])
    result = {
        "id": question["id"],
        "question": question["question"],
        "mode": mode,
        "hit": first_hit_rank is not None,
        "first_hit_rank": first_hit_rank,
        "reciprocal_rank": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "recall": len(found) / total_anchors if total_anchors else 0.0,
        "retrieved_chunk_ids": [c.chunk_id for c in chunks],
    }

    if with_generation:
        # Pinned so repeat runs are comparable -- see app.llm.generate.
        answer = generate(rag.build_prompt(question["question"], chunks), temperature=0)
        result["answer"] = answer
        result["citations"] = sorted(rag.parse_citations(answer, len(chunks)))
        result.update(score_answer(answer, question))
    return result


def aggregate(results: list[dict], top_k: int) -> dict:
    n = len(results)
    summary = {
        "n": n,
        f"hit@{top_k}": sum(r["hit"] for r in results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n,
        f"recall@{top_k}": sum(r["recall"] for r in results) / n,
    }
    scored = [r for r in results if "facts_present" in r]
    if scored:
        summary["facts_present"] = sum(r["facts_present"] for r in scored) / len(scored)
        summary["ref_similarity"] = sum(r["ref_similarity"] for r in scored) / len(scored)
        summary["answered_with_citation"] = sum(
            1 for r in scored if r["citations"]
        ) / len(scored)
    return summary


def verify_anchors(db, questions: list[dict]) -> int:
    """Confirm every ground-truth anchor still resolves to a real chunk.

    Ground truth that has silently drifted is worse than no ground truth,
    so this is runnable on demand rather than trusted once at authoring time.
    """
    rows = db.execute(
        select(Chunk.text, Document.source_id).join(
            Document, Chunk.document_id == Document.id
        )
    ).all()
    corpus = [(source_id, normalize(text)) for text, source_id in rows]

    problems = 0
    for question in questions:
        for anchor in question["relevant"]:
            needle = normalize(anchor["must_contain"])
            in_doc = sum(1 for sid, t in corpus if sid == anchor["source_id"] and needle in t)
            corpus_wide = sum(1 for _, t in corpus if needle in t)
            if in_doc == 0:
                problems += 1
                status = "MISSING"
            elif corpus_wide > 8:
                problems += 1
                status = "TOO VAGUE"
            else:
                status = "ok"
            print(
                f"  [{status:9s}] {question['id']} {anchor['source_id']} "
                f"chunks={in_doc} corpus-wide={corpus_wide}  "
                f"{anchor['must_contain'][:48]}"
            )
    print(
        f"\n{problems} problem(s)" if problems else "\nAll anchors resolve to real chunks."
    )
    return problems


def print_summary(summaries: dict[str, dict], top_k: int) -> None:
    keys = [f"hit@{top_k}", "mrr", f"recall@{top_k}"]
    if any("facts_present" in s for s in summaries.values()):
        keys += ["facts_present", "ref_similarity", "answered_with_citation"]

    width = max(len(k) for k in keys) + 2
    print(f"\n{'metric'.ljust(width)}" + "".join(m.rjust(12) for m in summaries))
    print("-" * (width + 12 * len(summaries)))
    for key in keys:
        row = key.ljust(width)
        for summary in summaries.values():
            value = summary.get(key)
            row += ("-" if value is None else f"{value:.3f}").rjust(12)
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="reranked")
    parser.add_argument("--compare", action="store_true", help="run all three modes")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="skip LLM generation (much faster; retrieval metrics only)",
    )
    parser.add_argument("--top-k", type=int, default=rag.TOP_K)
    parser.add_argument("--json", type=Path, help="write raw per-question results here")
    parser.add_argument(
        "--verify-anchors",
        action="store_true",
        help="check ground truth still resolves against the corpus, then exit",
    )
    args = parser.parse_args()

    questions = load_eval_set(EVAL_SET_PATH)
    db = SessionLocal()
    try:
        if args.verify_anchors:
            print(f"Verifying {sum(len(q['relevant']) for q in questions)} anchors\n")
            sys.exit(1 if verify_anchors(db, questions) else 0)

        source_by_chunk = source_id_by_chunk(db)
        modes = MODES if args.compare else (args.mode,)
        with_generation = not args.retrieval_only

        all_results: dict[str, list[dict]] = {}
        summaries: dict[str, dict] = {}
        for mode in modes:
            print(f"\n=== {mode} (top_k={args.top_k}"
                  f"{', retrieval only' if args.retrieval_only else ''}) ===")
            results = []
            for question in questions:
                result = evaluate_question(
                    question, db, mode, args.top_k, source_by_chunk, with_generation
                )
                results.append(result)
                rank = result["first_hit_rank"]
                print(
                    f"  {result['id']}  {'HIT ' if result['hit'] else 'miss'}"
                    f"  rank={rank if rank else '-':<3}"
                    f" recall={result['recall']:.2f}"
                    + (
                        f" facts={result['facts_present']:.2f}"
                        f" sim={result['ref_similarity']:.2f}"
                        if "facts_present" in result
                        else ""
                    )
                )
            all_results[mode] = results
            summaries[mode] = aggregate(results, args.top_k)

        print_summary(summaries, args.top_k)

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("w") as fh:
                json.dump(
                    {"top_k": args.top_k, "summaries": summaries, "results": all_results},
                    fh,
                    indent=2,
                )
            print(f"\nWrote {args.json}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
