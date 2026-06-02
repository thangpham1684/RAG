from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from logging_config import get_logger

logger = get_logger(__name__)

class AdvancedChunker:
    def __init__(self):
        logger.info("✂️ MODULE 2: Khởi tạo Advanced Chunking (Semantic & Hierarchical)...")
        # Nới lỏng cấu trúc 3 tầng để tận dụng context lớn (8k) của model bge-m3: 2048 (Cha), 1024 (Trung gian), 512 (Lá)
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1024, 512],
            chunk_overlap=64
        )

    def split_into_chunks(self, documents):
        """
        Cắt tài liệu và tạo quan hệ Parent-Child
        """
        logger.info("⏳ Đang phân tích cấu trúc Cha-Con cho tài liệu...")
        all_nodes = self.node_parser.get_nodes_from_documents(documents)
        logger.info(f"✅ Đã tạo thành {len(all_nodes)} node với quan hệ phân cấp.")
        return all_nodes