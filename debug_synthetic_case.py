import os
from dotenv import load_dotenv

from parsers.router import DocumentRouter
from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from retrievers.hybrid_search import AdvancedHybridRetriever
from logging_config import get_logger

logger = get_logger(__name__)


def main():
    load_dotenv(".env")

    data_dir = "data"
    router = DocumentRouter()
    docs = router.process_directory(data_dir)
    if not docs:
        logger.warning("No documents loaded.")
        return

    chunker = AdvancedChunker()
    nodes = chunker.split_into_chunks(docs)

    db_manager = QdrantDBManager()
    index = db_manager.save_and_index(nodes)
    retriever = AdvancedHybridRetriever(index=index, nodes=nodes)

    queries = [
        "synthetic data",
        "synthetic data trong datn",
    ]

    for q in queries:
        best_nodes, raw_nodes, _ = retriever.retrieve_and_rerank(q, None)
        logger.info("\n" + "=" * 70)
        logger.info(f"Query: {q}")
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


if __name__ == "__main__":
    main()
