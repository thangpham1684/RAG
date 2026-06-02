import argparse
import json
import os
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from generators.llm_generator import ResponseGenerator
from parsers.router import DocumentRouter
from retrievers.hybrid_search import AdvancedHybridRetriever
from logging_config import get_logger

logger = get_logger(__name__)


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are", "was", "were",
    "la", "va", "cua", "cho", "trong", "duoc", "mot", "nhung", "cac", "tai", "tu", "vao", "phan", "de",
}


@dataclass
class CaseResult:
    case_id: str
    language: str
    query: str
    rank: int | None
    top_files: list[str]
    raw_nodes: int
    best_nodes: int
    faithfulness: float | None
    has_citation: bool


def normalize_ascii(text: str) -> str:
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )


def tokenize(text: str) -> list[str]:
    text = normalize_ascii(text)
    return [tok for tok in re.findall(r"[a-z0-9]+", text) if tok and tok not in STOPWORDS]


def load_cases(path: str, max_cases: int | None = None) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if max_cases is not None:
        return cases[:max_cases]
    return cases


def find_rank_by_expected_file(best_nodes, expected_file_tokens: list[str]) -> int | None:
    expected_tokens_norm = [normalize_ascii(t) for t in expected_file_tokens]
    for idx, node in enumerate(best_nodes, 1):
        file_name = normalize_ascii(node.node.metadata.get("file_name", ""))
        if any(tok in file_name for tok in expected_tokens_norm):
            return idx
    return None


def split_claim_sentences(answer: str) -> list[str]:
    lines = [ln.strip() for ln in re.split(r"[\n\r]+", answer) if ln.strip()]
    candidates = []
    for ln in lines:
        ln = re.sub(r"^[-*#>\d\.\)\s]+", "", ln).strip()
        if len(ln) >= 20:
            candidates.append(ln)
    return candidates


def heuristic_faithfulness(answer: str, context_nodes) -> float:
    claims = split_claim_sentences(answer)
    if not claims:
        return 1.0
    context_text = " ".join(n.node.text for n in context_nodes)
    context_tokens = set(tokenize(context_text))
    if not context_tokens:
        return 0.0

    supported = 0
    for claim in claims:
        claim_tokens = tokenize(claim)
        if not claim_tokens:
            continue
        overlap = len(set(claim_tokens) & context_tokens) / len(set(claim_tokens))
        if overlap >= 0.50:
            supported += 1
    return supported / max(1, len(claims))


def build_or_load_retriever(rebuild_index: bool) -> AdvancedHybridRetriever:
    db_manager = QdrantDBManager()
    index = None
    nodes = None

    if not rebuild_index:
        index = db_manager.get_existing_index()
        if index is not None:
            nodes = list(index.docstore.docs.values())

    if index is None:
        router = DocumentRouter()
        docs = router.process_directory("data")
        if not docs:
            raise RuntimeError("No documents found in data/ for benchmark.")
        chunker = AdvancedChunker()
        nodes = chunker.split_into_chunks(docs)
        index = db_manager.save_and_index(nodes)

    return AdvancedHybridRetriever(index=index, nodes=nodes, llm=None, db_manager=db_manager)


def compute_summary(results: list[CaseResult], hit_ks: list[int]) -> dict[str, Any]:
    ranks = [r.rank for r in results]
    summary: dict[str, Any] = {
        "total_cases": len(results),
        "mrr": statistics.mean(0.0 if r is None else 1.0 / r for r in ranks),
    }

    for k in hit_ks:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        summary[f"hit@{k}"] = hit / len(results)

    faith_scores = [r.faithfulness for r in results if r.faithfulness is not None]
    summary["faithfulness_mean"] = statistics.mean(faith_scores) if faith_scores else None
    summary["citation_rate"] = sum(1 for r in results if r.has_citation) / len(results)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run standardized RAG benchmark (VN/EN).")
    parser.add_argument(
        "--benchmark-file",
        default=os.path.join("tests", "rag_benchmark_vn_en.json"),
        help="Path to fixed benchmark JSON file.",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Evaluate only first N cases.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index from data/ before evaluating.")
    parser.add_argument("--no-generation", action="store_true", help="Skip answer generation and faithfulness.")
    parser.add_argument(
        "--hit-ks",
        default="1,3,5,8",
        help="Comma-separated list of K values for Hit@K metrics.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save full benchmark report JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv(".env")

    hit_ks = [int(x.strip()) for x in args.hit_ks.split(",") if x.strip()]
    cases = load_cases(args.benchmark_file, args.max_cases)
    retriever = build_or_load_retriever(rebuild_index=args.rebuild_index)

    generator = None
    if not args.no_generation:
        try:
            generator = ResponseGenerator()
        except Exception as exc:
            logger.warning("⚠️ Generation unavailable, skip faithfulness: %s", exc)

    results: list[CaseResult] = []
    for case in cases:
        query = case["query"]
        expected_file_tokens = case.get("expected_file_tokens", [])
        selected_files = case.get("selected_files")

        best_nodes, raw_nodes, _ = retriever.retrieve_and_rerank(query, selected_files)
        rank = find_rank_by_expected_file(best_nodes, expected_file_tokens)
        top_files = [n.node.metadata.get("file_name", "Unknown") for n in best_nodes[:5]]

        answer = ""
        faith = None
        has_citation = False
        if generator is not None:
            chunks = []
            for ch in generator.generate_answer_stream(query, best_nodes):
                chunks.append(ch)
            answer = "".join(chunks)
            faith = heuristic_faithfulness(answer, best_nodes)
            has_citation = "[File:" in answer

        item = CaseResult(
            case_id=case["id"],
            language=case.get("language", "unknown"),
            query=query,
            rank=rank,
            top_files=top_files,
            raw_nodes=len(raw_nodes),
            best_nodes=len(best_nodes),
            faithfulness=faith,
            has_citation=has_citation,
        )
        results.append(item)
        logger.info("[%s] rank=%s raw=%d best=%d faith=%s",
            item.case_id, item.rank, item.raw_nodes, item.best_nodes, item.faithfulness)

    summary = compute_summary(results, hit_ks)
    logger.info("\n=== BENCHMARK SUMMARY ===")
    for k, v in summary.items():
        logger.info("%s: %s", k, v)

    if args.output_json:
        report = {
            "summary": summary,
            "cases": [r.__dict__ for r in results],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Saved report: %s", args.output_json)


if __name__ == "__main__":
    main()
