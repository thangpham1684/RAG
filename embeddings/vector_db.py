import os
import json
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import get_leaf_nodes

from retrievers.sparse_bm25 import SparseBM25Encoder
from logging_config import get_logger

logger = get_logger(__name__)

class QdrantDBManager:
    def __init__(self, client=None):
        logger.info("💽 MODULE 3: Embedding & Storage (Qdrant)...")
        # Sử dụng mô hình nhúng mạnh mẽ BGE-M3
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
        self.dense_dim = int(os.getenv("DENSE_VECTOR_SIZE", "1024"))
        self.use_sparse = os.getenv("USE_SPARSE_BM25", "true").lower() in {"1", "true", "yes"}
        self.sparse_vector_name = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25")
        self.sparse_dim = int(os.getenv("SPARSE_VECTOR_DIM", "1048576"))
        self.sparse_encoder = None

        if client:
            self.client = client
        elif qdrant_url:
            self.client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Local Qdrant (không phù hợp multi-process). Dùng cho dev/test.
            self.client = qdrant_client.QdrantClient(path=qdrant_path)

        # Docstore vẫn cần lưu local để BM25/parent-child dùng được.
        docstore_dir_env = os.getenv("DOCSTORE_DIR")
        default_docstore_dir = "./docstore"
        legacy_docstore_dir = "./qdrant_data"
        if docstore_dir_env:
            self.persist_dir = docstore_dir_env
        elif os.path.exists(os.path.join(legacy_docstore_dir, "docstore.json")):
            # Tương thích dữ liệu cũ
            self.persist_dir = legacy_docstore_dir
        elif os.path.exists(os.path.join(default_docstore_dir, "docstore.json")):
            self.persist_dir = default_docstore_dir
        else:
            self.persist_dir = default_docstore_dir if qdrant_url else qdrant_path

        os.makedirs(self.persist_dir, exist_ok=True)

        self._ensure_collection()

        if self.use_sparse:
            encoder_path = os.path.join(self.persist_dir, "bm25_sparse.json")
            if os.path.exists(encoder_path):
                self.sparse_encoder = SparseBM25Encoder.load(encoder_path)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )

        # Khởi tạo StorageContext. Nếu đã có dữ liệu persist, load lên. Nếu không, khởi tạo mới.
        docstore_path = os.path.join(self.persist_dir, "docstore.json")
        if os.path.exists(docstore_path):
            logger.info(f"📦 Tìm thấy dữ liệu docstore tại {self.persist_dir}, đang tải...")
            try:
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store,
                    persist_dir=self.persist_dir
                )
            except (MemoryError, Exception) as exc:
                logger.warning(
                    "⚠️ Không thể tải docstore: %s. "
                    "Khởi tạo storage context mới. Vector trong Qdrant vẫn còn, "
                    "nhưng cần chạy 'Xây dựng index' để rebuild docstore local.",
                    exc
                )
                self.storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )
        else:
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def save_and_index(self, all_nodes):
        """
        Lưu toàn bộ node vào DocStore và chỉ index node lá vào Vector DB
        """
        logger.info("🧠 Đang xây dựng Index và DocStore (Parent-Child)...")
        
        # 1. Lưu toàn bộ (bao gồm node cha) vào DocStore
        self.storage_context.docstore.add_documents(all_nodes)
        
        # 2. Tách node lá để đánh chỉ mục Vector
        leaf_nodes = get_leaf_nodes(all_nodes)
        
        # 3. Tạo Index
        index = VectorStoreIndex(
            leaf_nodes, 
            storage_context=self.storage_context,
            show_progress=True
        )
        
        # 4. Ghi DocStore xuống ổ cứng 
        self.storage_context.persist(persist_dir=self.persist_dir)
        
        if self.use_sparse:
            self._build_sparse_vectors(leaf_nodes)

        logger.info(f"✅ Đã index {len(leaf_nodes)} node lá. Toàn bộ {len(all_nodes)} node đã được lưu vào DocStore tại {self.persist_dir}.")
        return index

    def get_existing_index(self):
        try:
            return load_index_from_storage(self.storage_context)
        except Exception as e:
            # Trường hợp có nhiều index_id trong cùng persist_dir.
            idx = self._load_latest_index_by_id()
            if idx is not None:
                return idx
            logger.warning("Không thể load index cũ: %s", e)
            return None

    def _ensure_collection(self):
        sparse_config = None
        if self.use_sparse:
            sparse_config = {
                self.sparse_vector_name: models.SparseVectorParams()
            }

        if self.client.collection_exists(self.collection_name):
            self._ensure_payload_indexes()
            if self.use_sparse:
                info = self.client.get_collection(self.collection_name)
                existing_sparse = getattr(info.config.params, "sparse_vectors", None) or {}
                if self.sparse_vector_name not in existing_sparse:
                    try:
                        self.client.update_collection(
                            collection_name=self.collection_name,
                            sparse_vectors_config=sparse_config,
                        )
                    except UnexpectedResponse as exc:
                        self._fallback_disable_sparse(exc)
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dense_dim, distance=models.Distance.COSINE),
            sparse_vectors_config=sparse_config,
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            msg = str(exc).lower()
            if "already exists" not in msg and "already created" not in msg:
                raise

    def _fallback_disable_sparse(self, exc: UnexpectedResponse):
        logger.warning("⚠️ Không thể migrate sparse vector cho collection hiện tại. Fallback sang BM25 in-memory. Chi tiết: %s", exc)
        self.use_sparse = False
        self.sparse_encoder = None

    def _load_latest_index_by_id(self):
        index_store_path = os.path.join(self.persist_dir, "index_store.json")
        if not os.path.exists(index_store_path):
            return None
        try:
            with open(index_store_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            ids = list(payload.get("index_store/data", {}).keys())
            if not ids:
                return None
            latest_id = ids[-1]
            return load_index_from_storage(self.storage_context, index_id=latest_id)
        except Exception:
            return None

    def _build_sparse_vectors(self, leaf_nodes):
        if not leaf_nodes:
            return

        encoder_path = os.path.join(self.persist_dir, "bm25_sparse.json")
        self.sparse_encoder = SparseBM25Encoder(sparse_dim=self.sparse_dim)
        self.sparse_encoder.fit([n.get_content() for n in leaf_nodes])
        self.sparse_encoder.save(encoder_path)

        batch = []
        batch_size = int(os.getenv("SPARSE_UPSERT_BATCH", "128"))
        for node in leaf_nodes:
            sparse_vec = self.sparse_encoder.encode_document(node.get_content())
            if not sparse_vec.indices:
                continue
            batch.append(
                models.PointVectors(
                    id=str(node.node_id),
                    vector={self.sparse_vector_name: sparse_vec},
                )
            )
            if len(batch) >= batch_size:
                self.client.update_vectors(self.collection_name, points=batch)
                batch = []

        if batch:
            self.client.update_vectors(self.collection_name, points=batch)

    def delete_file_nodes(self, file_name: str) -> int:
        """Remove all docstore/vector entries that belong to a file_name."""
        file_name_norm = (file_name or "").strip().lower()
        if not file_name_norm:
            return 0

        docs = getattr(self.storage_context.docstore, "docs", {})
        node_ids = []
        for node_id, node in list(docs.items()):
            node_file = str(node.metadata.get("file_name", "")).strip().lower()
            if node_file == file_name_norm:
                node_ids.append(str(node_id))

        if not node_ids:
            return 0

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=node_ids),
            wait=True,
        )

        for node_id in node_ids:
            docs.pop(node_id, None)

        self.storage_context.persist(persist_dir=self.persist_dir)
        return len(node_ids)

    def reset_index_storage(self):
        """
        Reset vector collection + local index artifacts.
        Used when a full rebuild is required (e.g. deleted source files).
        """
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self._ensure_collection()
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        for name in (
            "docstore.json",
            "index_store.json",
            "graph_store.json",
            "image__vector_store.json",
            "bm25_sparse.json",
        ):
            path = os.path.join(self.persist_dir, name)
            if os.path.exists(path):
                os.remove(path)

        if self.use_sparse:
            self.sparse_encoder = None