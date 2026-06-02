import os
import statistics
import unicodedata
from dotenv import load_dotenv

from parsers.router import DocumentRouter
from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from retrievers.hybrid_search import AdvancedHybridRetriever
from logging_config import get_logger

logger = get_logger(__name__)


def normalize_ascii(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )


def pick_file(files, token_ascii: str):
    token_ascii = token_ascii.lower()
    for f in files:
        if token_ascii in normalize_ascii(f):
            return f
    return None


def print_stats(label, items):
    if not items:
        logger.info(f"{label}: no items")
        return
    avg_len = statistics.mean(items)
    min_len = min(items)
    max_len = max(items)
    logger.info(f"{label}: count={len(items)} avg={avg_len:.1f} min={min_len} max={max_len}")


def main():
    load_dotenv(".env")

    data_dir = "data"
    router = DocumentRouter()
    docs = router.process_directory(data_dir)
    if not docs:
        logger.warning("No documents loaded.")
        return

    # Doc length stats per file
    doc_lengths = {}
    for d in docs:
        f_name = d.metadata.get("file_name", "Unknown")
        doc_lengths.setdefault(f_name, []).append(len(d.text))

    logger.info("\n[DOC STATS]")
    for f_name in sorted(doc_lengths.keys()):
        print_stats(f"- {f_name}", doc_lengths[f_name])

    # Chunking
    chunker = AdvancedChunker()
    nodes = chunker.split_into_chunks(docs)

    node_lengths = {}
    for n in nodes:
        f_name = n.metadata.get("file_name", "Unknown")
        node_lengths.setdefault(f_name, []).append(len(n.text))

    logger.info("\n[NODE STATS]")
    for f_name in sorted(node_lengths.keys()):
        print_stats(f"- {f_name}", node_lengths[f_name])

    # Build index + retriever
    db_manager = QdrantDBManager()
    index = db_manager.save_and_index(nodes)
    retriever = AdvancedHybridRetriever(index=index, nodes=nodes)

    files = sorted({n.metadata.get("file_name", "Unknown") for n in nodes})
    datn_file = pick_file(files, "datn")
    algo_file = pick_file(files, "algoritmos")

    logger.info("\n[FILES]")
    logger.info(f"DATN file: {datn_file}")
    logger.info(f"ALGO file: {algo_file}")

    test_queries = [
        "datn",
        "do an tot nghiep",
        "synthetic data",
        "synthetic data trong datn",
        "du lieu tong hop",
        "algoritmos geneticos",
        "muc tieu nghien cuu",
    ]

    def run_case(case_name, query, selected_files):
        best_nodes, raw_nodes, _ = retriever.retrieve_and_rerank(query, selected_files)
        logger.info("\n" + "=" * 70)
        logger.info(f"CASE: {case_name}")
        logger.info(f"Query: {query}")
        logger.info(f"Selected files: {selected_files}")
        logger.info(f"Raw nodes: {len(raw_nodes)} | Best nodes: {len(best_nodes)}")

        by_file = {}
        for n in best_nodes:
            f_name = n.node.metadata.get("file_name", "Unknown")
            by_file[f_name] = by_file.get(f_name, 0) + 1

        logger.info("Top file distribution:")
        for f_name, cnt in sorted(by_file.items(), key=lambda x: (-x[1], x[0])):
            logger.info("- %s: %d", f_name, cnt)

        logger.info("Top 3 snippets:")
        for i, n in enumerate(best_nodes[:3], 1):
            f_name = n.node.metadata.get("file_name", "Unknown")
            score = getattr(n, "score", None)
            text = n.node.text.replace("\n", " ")[:180]
            logger.info("#%d file=%s score=%s text=%s...", i, f_name, score, text)

    # Run cases
    for q in test_queries:
        run_case("no_filter", q, None)
        if datn_file:
            run_case("filter_datn", q, [datn_file])
        if algo_file:
            run_case("filter_algo", q, [algo_file])


if __name__ == "__main__":
    main()
