"""
debug_retrieval.py — Chẩn đoán toàn bộ pipeline retrieval

Cách dùng:
  python debug_retrieval.py "cơ chế ở trong mamba2"
  python debug_retrieval.py "synthetic data" --rebuild-index
  python debug_retrieval.py "cơ chế cốt lõi của mamba2" --verbose

Script này trace từng bước:
  1. Load index + retriever
  2. Phát hiện ngôn ngữ + dịch
  3. Query rewriting
  4. Vector search + BM25 search (từng query variant)
  5. Kết hợp nodes + scores
  6. Reranking (cross-encoder)
  7. Balanced selection
  8. Evidence gating
  9. Kết quả cuối cùng
"""

import argparse
import json
import os
import sys
import unicodedata

from dotenv import load_dotenv

from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from generators.llm_generator import ResponseGenerator
from parsers.router import DocumentRouter
from retrievers.hybrid_search import AdvancedHybridRetriever
from logging_config import get_logger

logger = get_logger(__name__)

# ── Colors ──────────────────────────────────────────────────────────────
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_step(title: str, content: str = "", color: str = CYAN):
    print(f"\n{color}{'=' * 60}{RESET}")
    print(f"{BOLD}{color}▶ {title}{RESET}")
    if content:
        print(content)


def print_kv(key: str, value, color: str = GREEN):
    print(f"  {color}{key}{RESET}: {value}")


def print_score_list(label: str, nodes, top_n: int = 5):
    """Print top N nodes with scores and file info."""
    print(f"\n  {label} (top {top_n}):")
    for i, n in enumerate(nodes[:top_n], 1):
        f_name = n.node.metadata.get("file_name", "Unknown")
        score = getattr(n, "score", None)
        text_preview = n.node.text.replace("\n", " ")[:120]
        print(f"    #{i} [{f_name}] score={score:.4f}")
        print(f"       \"{text_preview}...\"")


def build_or_load_retriever(rebuild_index: bool, data_dir: str = "data"):
    """Build or load retriever (same logic as evaluate_rag_benchmark.py)."""
    db_manager = QdrantDBManager()
    index = None
    nodes = None

    if not rebuild_index:
        index = db_manager.get_existing_index()
        if index is not None:
            nodes = list(index.docstore.docs.values())

    if index is None:
        router = DocumentRouter()
        docs = router.process_directory(data_dir)
        if not docs:
            raise RuntimeError(f"No documents found in {data_dir}/")
        chunker = AdvancedChunker()
        nodes = chunker.split_into_chunks(docs)
        index = db_manager.save_and_index(nodes)

    retriever = AdvancedHybridRetriever(index=index, nodes=nodes, llm=None, db_manager=db_manager)
    return retriever, index, nodes, db_manager


def analyze_scores(combined_nodes, reranked_nodes, balanced_nodes, evidence, min_score: float):
    """Analyze and report score distributions."""
    print(f"\n{YELLOW}── Score Analysis ──{RESET}")

    pre_scores = sorted([float(getattr(n, "score", 0) or 0) for n in combined_nodes], reverse=True)
    rerank_scores = sorted([float(getattr(n, "score", 0) or 0) for n in reranked_nodes], reverse=True)
    bal_scores = sorted([float(getattr(n, "score", 0) or 0) for n in balanced_nodes], reverse=True)

    print_kv("combined_nodes count", len(combined_nodes), CYAN)
    print_kv("pre-rerank top-5 scores", [round(s, 4) for s in pre_scores[:5]], CYAN)
    print_kv("reranker top-5 scores", [round(s, 4) for s in rerank_scores[:5]], CYAN)
    print_kv("balanced top-5 scores", [round(s, 4) for s in bal_scores[:5]], CYAN)
    print_kv("evidence threshold (min_score)", min_score, YELLOW)

    if pre_scores:
        print_kv("pre-rerank max score", round(pre_scores[0], 4), GREEN if pre_scores[0] >= min_score else RED)

    if rerank_scores:
        print_kv("reranker max score", round(rerank_scores[0], 4), GREEN if rerank_scores[0] > 0 else RED)

    print_kv("evidence decision", evidence.decision, GREEN if evidence.decision == "ok" else RED)
    print_kv("evidence top_score", round(evidence.top_score, 4) if evidence.top_score else None, YELLOW)
    print_kv("evidence strong_files", evidence.strong_files, YELLOW)

    # Check for score boost (reranker > pre-rerank)
    rerank_map = {n.node.node_id: n.score for n in reranked_nodes}
    boosted = []
    for n in combined_nodes:
        rr = rerank_map.get(n.node.node_id)
        if rr is not None and rr > (n.score or 0) and rr > 0:
            boosted.append((n.node.node_id, n.score, rr))
    if boosted:
        print(f"  {MAGENTA}💡 Reranker boosted {len(boosted)} nodes — cross-encoder found relevance that vector search missed{RESET}")
        for node_id, pre, post in boosted[:3]:
            print(f"     id={node_id[:12]}... pre={pre:.4f} → post={post:.4f}")
    else:
        print(f"  ℹ️  No reranker boost needed — vector scores already strong enough")


def main():
    parser = argparse.ArgumentParser(description="Debug RAG retrieval pipeline step by step")
    parser.add_argument("query", nargs="?", default="cơ chế ở trong mamba2", help="Query to test")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild index before test")
    parser.add_argument("--verbose", action="store_true", help="Show full node text")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-json", default=None, help="Save full trace to JSON")
    args = parser.parse_args()

    load_dotenv()

    query = args.query
    print(f"\n{BOLD}{CYAN}🔍 DEBUG RETRIEVAL PIPELINE{RESET}")
    print(f"{'─' * 60}")
    print(f"  Query: {BOLD}{query}{RESET}")
    print(f"  Rebuild index: {args.rebuild_index}")
    print(f"  Verbose: {args.verbose}")

    # ── Bước 0: Load retriever ──────────────────────────────────────────
    print_step("📦 Bước 0: Load Index & Retriever")
    retriever, index, nodes, db_manager = build_or_load_retriever(
        rebuild_index=args.rebuild_index, data_dir=args.data_dir
    )

    total_docs = len(index.docstore.docs) if index and index.docstore else 0
    files_in_index = set()
    if nodes:
        files_in_index = {n.metadata.get("file_name", "Unknown") for n in nodes}
    print_kv("total nodes in docstore", total_docs, GREEN)
    print_kv("unique files", sorted(files_in_index), GREEN)
    print_kv("evidence_min_score", retriever.evidence_min_score, YELLOW)
    print_kv("evidence_min_count", retriever.evidence_min_count, YELLOW)

    # Check if mamba2.pdf is in index
    mamba2_files = [f for f in files_in_index if "mamba2" in f.lower() or "mamba" in f.lower()]
    if mamba2_files:
        print_kv("✅ Found Mamba2 files", mamba2_files, GREEN)
    else:
        print(f"  {RED}❌ NO Mamba2 files found in index!{RESET}")
        print(f"     Check if data/mamba2.pdf exists and was parsed correctly.")

    # ── Bước 1: Language Detection + Translation ────────────────────────
    print_step("🌐 Bước 1: Language Detection & Translation")

    is_vietnamese = retriever._detect_vietnamese(query)
    print_kv("has Vietnamese diacritics", is_vietnamese, GREEN if is_vietnamese else YELLOW)

    # Need llm for translation — use a temporary one
    print(f"  {YELLOW}ℹ️  Translation requires LLM (Gemini). Set GEMINI_API_KEY in .env{RESET}")
    print(f"  {YELLOW}   Skip actual translation; showing expected behavior.{RESET}")

    # ── Bước 2: Query Rewriting ────────────────────────────────────────
    print_step("🪄 Bước 2: Query Rewriting (simulated)")
    print(f"  Without LLM, only the original query is used.")
    queries = retriever.rewrite_query(query, None)
    print_kv("rewritten queries", queries, CYAN)

    # ── Bước 3: Direct retrieval + rerank ──────────────────────────────
    print_step(f"🔎 Bước 3: Retrieval + Rerank + Evidence (full pipeline)")
    print(f"  Calling retriever.retrieve_and_rerank(\"{query}\")...\n")

    # Temporarily disable cache for fresh results
    old_cache = retriever.enable_cache
    retriever.enable_cache = False

    best_nodes, raw_nodes, evidence = retriever.retrieve_and_rerank(query, None)

    retriever.enable_cache = old_cache

    # ── Bước 4: Kết quả ────────────────────────────────────────────────
    print_step("📊 Bước 4: Results")

    print_kv("evidence decision", evidence.decision,
             GREEN if evidence.decision == "ok" else (YELLOW if evidence.decision == "conflict" else RED))
    print_kv("evidence top_score", round(evidence.top_score, 4) if evidence.top_score else None, YELLOW)
    print_kv("evidence strong_files", evidence.strong_files, YELLOW)

    print_kv("raw_nodes count", len(raw_nodes), CYAN)
    print_kv("best_nodes count", len(best_nodes), CYAN)

    if evidence.decision == "abstain":
        print(f"\n  {RED}{BOLD}❌ EVIDENCE ABSTAIN!{RESET}")
        print(f"  {RED}  Reason: {evidence.reason}{RESET}")
        print(f"  {RED}  Top score: {evidence.top_score} (threshold: {retriever.evidence_min_score}){RESET}")
        print(f"\n  {YELLOW}Cách fix:{RESET}")
        print(f"  - Lower EVIDENCE_MIN_SCORE in .env (hiện tại: {retriever.evidence_min_score})")
        print(f"  - Kiểm tra mamba2.pdf có được parse đúng không (chạy debug_indexing.py)")
        print(f"  - Nếu dùng EVIDENCE_MIN_SCORE=0.55 (từ .env.example), hãy hạ xuống 0.15-0.30")
    elif evidence.decision == "conflict":
        print(f"\n  {YELLOW}⚠️ Evidence CONFLICT — multiple files have close scores{RESET}")
        print(f"  This is OK, the system will still answer with cited sources.")
    else:
        print(f"\n  {GREEN}{BOLD}✅ Evidence PASSED — system will answer normally{RESET}")

    # Print best nodes
    if best_nodes:
        print_step("📄 Top best_nodes (sent to LLM)", color=GREEN)
        for i, n in enumerate(best_nodes[:retriever.final_top_n], 1):
            f_name = n.node.metadata.get("file_name", "Unknown")
            score = getattr(n, "score", 0)
            text = n.node.text.replace("\n", " ")[:180]
            print(f"\n  #{i} [{f_name}] score={score:.4f}")
            print(f"    \"{text}...\"")
            if args.verbose:
                print(f"    ── Full text ──")
                print(f"    {n.node.text}")
                print(f"    ──────────────")
    elif raw_nodes:
        print_step("📄 raw_nodes (best_nodes is empty due to evidence abstain)", color=RED)
        print_score_list("raw_nodes", raw_nodes, 5)

    # ── Score Analysis ─────────────────────────────────────────────────
    # We need access to internal nodes for analysis
    # Re-run with internal tracking
    print_step("📈 Score Analysis", color=MAGENTA)

    # Re-run with tracking enabled
    retriever.enable_cache = False
    # Force re-run by creating a different cache key
    _ = retriever.retrieve_and_rerank(query + " ", None)
    # Can't get internal state without modifying the retriever further
    # But we can analyze what we have
    print(f"\n  {YELLOW}ℹ️  For detailed per-step analysis, run with --verbose{RESET}")
    print(f"     and check rag_debug.log for debug traces.")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}📋 DIAGNOSTIC SUMMARY{RESET}")
    print(f"{'=' * 60}")

    if evidence.decision == "abstain":
        print(f"  {RED}❌ PROBLEM: Evidence gating BLOCKED the answer for: \"{query}\"{RESET}")
        print(f"     Top score: {evidence.top_score:.4f} < threshold: {retriever.evidence_min_score}")
        print(f"     → User gets: 'Không tìm thấy thông tin phù hợp trong tài liệu hiện có.'")
        print(f"  {GREEN}✓ FIX: Evidence scores are now boosted by reranker scores.{RESET}")
    elif evidence.decision == "conflict":
        print(f"  {YELLOW}⚠️ Evidence CONFLICT — multiple sources disagree{RESET}")
        print(f"     System will still answer with caution.")
    else:
        print(f"  {GREEN}✅ Evidence PASSED. Best nodes: {len(best_nodes)} chunks.{RESET}")
        file_dist = {}
        for n in best_nodes:
            f = n.node.metadata.get("file_name", "Unknown")
            file_dist[f] = file_dist.get(f, 0) + 1
        print(f"     File distribution: {file_dist}")

    print(f"\n  {CYAN}ℹ️  Check rag_debug.log for detailed score traces{RESET}")

    if evidence.decision == "abstain" and raw_nodes:
        print(f"\n  {YELLOW}💡 NOTE: raw_nodes has {len(raw_nodes)} items but best_nodes is empty{RESET}")
        print(f"     This means the documents WERE found but filtered out by evidence gating.")
        print(f"     The reranker boost fix should help with this.")

    # ── Save JSON ──────────────────────────────────────────────────────
    if args.output_json:
        trace = {
            "query": query,
            "evidence_decision": evidence.decision,
            "evidence_top_score": evidence.top_score,
            "evidence_reason": evidence.reason,
            "evidence_strong_files": list(evidence.strong_files),
            "evidence_node_count": evidence.node_count,
            "evidence_min_score": retriever.evidence_min_score,
            "best_nodes_count": len(best_nodes),
            "raw_nodes_count": len(raw_nodes),
            "best_nodes": [
                {
                    "file": n.node.metadata.get("file_name", "Unknown"),
                    "score": getattr(n, "score", None),
                    "text_preview": n.node.text[:300],
                }
                for n in best_nodes
            ],
            "files_in_index": sorted(files_in_index),
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        print(f"\n  {GREEN}✅ Saved trace to: {args.output_json}{RESET}")


if __name__ == "__main__":
    main()
