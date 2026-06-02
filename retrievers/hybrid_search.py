import hashlib
import math
import os
import re
import threading
from collections import OrderedDict
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition

from retrievers.sparse_bm25 import QdrantSparseRetriever
from retrievers.evidence_policy import evaluate_evidence
from logging_config import get_logger

logger = get_logger(__name__)

class AdvancedHybridRetriever:
    def __init__(self, index, nodes, llm=None, db_manager=None):
        logger.info("🔍 MODULE 4: Khởi tạo Hybrid Retrieval (Vector + BM25)...")
        self.index = index
        self.llm = llm
        self.vector_top_k = int(os.getenv("RETRIEVAL_TOP_K", "20"))
        self.sparse_top_k = int(os.getenv("SPARSE_TOP_K", str(self.vector_top_k)))
        self.final_top_n = int(os.getenv("FINAL_TOP_N", "8"))
        self.rerank_pool_size = int(os.getenv("RERANK_POOL_SIZE", "24"))
        self.max_query_rewrites = max(1, int(os.getenv("MAX_QUERY_REWRITES", "3")))
        self.enable_cache = os.getenv("ENABLE_RETRIEVAL_CACHE", "true").lower() in {"1", "true", "yes"}
        self.cache_max_size = int(os.getenv("RETRIEVAL_CACHE_SIZE", "64"))
        self._result_cache = OrderedDict()
        self._cache_lock = threading.Lock()
        auto_merge_verbose = os.getenv("AUTO_MERGE_VERBOSE", "false").lower() in {"1", "true", "yes"}

        # 1. Vector Retriever với cơ chế AutoMerging (Parent-Child)
        self.base_vector_retriever = index.as_retriever(similarity_top_k=self.vector_top_k)
        self.vector_retriever = AutoMergingRetriever(
            self.base_vector_retriever,
            index.storage_context,
            verbose=auto_merge_verbose
        )
        
        # 2. Sparse Retriever (Qdrant) hoặc BM25 in-memory fallback
        self.sparse_retriever = None
        self.total_nodes_count = len(nodes) if nodes else 0
        self.bm25_retriever = None
        if db_manager and getattr(db_manager, "use_sparse", False) and db_manager.sparse_encoder:
            self.sparse_retriever = QdrantSparseRetriever(
                client=db_manager.client,
                collection_name=db_manager.collection_name,
                docstore=index.docstore,
                encoder=db_manager.sparse_encoder,
                vector_name=db_manager.sparse_vector_name,
                top_k=self.sparse_top_k,
            )
        else:
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.sparse_top_k)
        
        # 3. MODULE 5: Reranking (Cross-Encoder)
        logger.info("⚖️ MODULE 5: Khởi tạo Reranker (BGE-Reranker-v2-m3)...")
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3", top_n=self.rerank_pool_size
        )
        # Evidence gating thresholds (configurable via environment variables)
        self.evidence_min_score = float(os.getenv("EVIDENCE_MIN_SCORE", "0.15"))
        self.evidence_min_count = int(os.getenv("EVIDENCE_MIN_COUNT", "1"))
        self.evidence_conflict_margin = float(os.getenv("EVIDENCE_CONFLICT_MARGIN", "0.05"))

    def _cache_key(self, query_str: str, selected_files: list | None):
        files = ()
        if selected_files:
            files = tuple(sorted(f.strip().lower() for f in selected_files))
        return query_str.strip().lower(), files

    def _cache_get(self, key):
        if not self.enable_cache:
            return None
        with self._cache_lock:
            value = self._result_cache.get(key)
            if value is None:
                return None
            self._result_cache.move_to_end(key)
            return value

    def _cache_set(self, key, value):
        if not self.enable_cache:
            return
        with self._cache_lock:
            self._result_cache[key] = value
            self._result_cache.move_to_end(key)
            while len(self._result_cache) > self.cache_max_size:
                self._result_cache.popitem(last=False)

    # ── Vietnamese detection + translation ────────────────────────────────
    _VIETNAMESE_RE = re.compile(
        r'[àáảãạăắằẵặâấầẫậđèéẻẽẹêếềễệìíỉĩịòóỏõọôốồỗộơớờỡợùúủũụưứừữựỳýỷỹỵ]',
        re.IGNORECASE
    )

    def _detect_vietnamese(self, text: str) -> bool:
        """Check if text contains Vietnamese diacritics."""
        return bool(self._VIETNAMESE_RE.search(text))

    def _translate_to_english(self, query: str) -> str | None:
        """Translate Vietnamese query to English using LLM.

        Returns:
            Translated English query, or None if translation is not needed/possible.
        """
        if not self._detect_vietnamese(query) or not self.llm:
            return None

        logger.info("🌐 Đang dịch câu hỏi từ Tiếng Việt sang Tiếng Anh...")
        prompt = (
            f"Dịch câu hỏi sau từ Tiếng Việt sang Tiếng Anh.\n"
            f"Giữ nguyên các từ viết tắt, tên riêng, tên kỹ thuật (ví dụ: Mamba2, SSM, BM25, RAG).\n"
            f"Chỉ trả lời DUY NHẤT bản dịch, không giải thích, không thêm gì khác.\n\n"
            f"Câu hỏi: {query}\n"
            f"Dịch:"
        )
        try:
            response = self.llm.complete(prompt).text.strip()
            # Validate: response should be different from original and reasonable length
            if response and response.lower() != query.strip().lower() and len(response) > len(query) * 0.3:
                logger.info(f"✅ Đã dịch: '{query}' → '{response}'")
                return response
            else:
                logger.info(f"ℹ️ Bỏ qua dịch thuật: response quá giống hoặc quá ngắn")
        except Exception as e:
            logger.warning("⚠️ Lỗi translate: %s", e)

        return None

    def _select_balanced_nodes(self, nodes, top_n):
        if not nodes:
            return []

        # ── Bỏ focus heuristic (chỉ giữ 1 file) vì lý do sau:
        #    Cross-encoder BGE-Reranker-v2-m3 cho score thấp hơn cosine similarity
        #    nhiều, và việc focus chỉ 1 file có thể drop mất chunks từ file khác
        #    vẫn có liên quan (vd: query "mamba2" match cả mamba2.pdf và ĐATN.pdf).
        #    Balanced selection thuần (max_per_file) đủ tốt.

        distinct_files = {
            n.node.metadata.get("file_name", "Unknown") for n in nodes
        }
        max_per_file = max(1, math.ceil(top_n / max(1, len(distinct_files))))

        selected = []
        selected_ids = set()
        file_counts = {}

        for n in nodes:
            node_id = n.node.node_id
            if node_id in selected_ids:
                continue
            f_name = n.node.metadata.get("file_name", "Unknown")
            if file_counts.get(f_name, 0) >= max_per_file:
                continue
            selected.append(n)
            selected_ids.add(node_id)
            file_counts[f_name] = file_counts.get(f_name, 0) + 1
            if len(selected) >= top_n:
                return selected

        # Fallback: nếu bị giới hạn quá chặt, lấy tiếp theo thứ hạng
        for n in nodes:
            node_id = n.node.node_id
            if node_id in selected_ids:
                continue
            selected.append(n)
            selected_ids.add(node_id)
            if len(selected) >= top_n:
                break

        return selected

    def rewrite_query(self, query: str, conversation_history: list | None = None):
        """
        MODULE 4: Query Routing & Rewriting

        Nếu có conversation_history, dùng LLM để contextualize câu hỏi follow-up
        thành một câu hỏi độc lập trước khi sinh keyword variants.
        """
        # Bước 1: Contextualize — biến follow-up thành câu hỏi độc lập
        #   Luôn chạy khi có history (≥ 2 turns), không dùng heuristic vì query
        #   như "cơ chế cốt lõi của mamba2" dài nhưng vẫn là follow-up ngữ cảnh.
        #   LLM prompt được thiết kế để giữ nguyên câu hỏi độc lập, guard
        #   len(response) > len(query)*0.5 bảo vệ khỏi garbage output.
        contextualized = query
        if conversation_history and len(conversation_history) >= 2 and self.llm:
            logger.info("🔄 Contextualizing query with conversation history...")
            # Lấy 2-4 turn gần nhất để có ngữ cảnh (không cần toàn bộ)
            recent = conversation_history[-4:]
            history_lines = []
            for msg in recent:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"Người dùng: {content}")
                elif role == "assistant":
                    # Chỉ lấy 200 ký tự đầu của câu trả lời để tránh trôi context
                    history_lines.append(f"Bạn: {content[:200]}")
            history_str = "\n".join(history_lines)

            ctx_prompt = (
                f"Dựa vào lịch sử hội thoại dưới đây, hãy viết lại câu hỏi hiện tại "
                f"thành một câu hỏi hoàn chỉnh, độc lập, có đầy đủ ngữ cảnh (Tiếng Việt).\n"
                f"Nếu câu hỏi hiện tại ĐÃ là một câu hỏi độc lập, hãy giữ nguyên.\n"
                f"KHÔNG thêm thông tin mới, CHỈ dùng thông tin từ lịch sử để điền vào chỗ trống.\n\n"
                f"--- LỊCH SỬ HỘI THOẠI ---\n"
                f"{history_str}\n"
                f"-------------------------\n"
                f"Câu hỏi hiện tại: {query}\n"
                f"Trả lời DUY NHẤT câu hỏi đã viết lại, không giải thích."
            )
            try:
                response = self.llm.complete(ctx_prompt).text.strip()
                if response and len(response) > len(query) * 0.5:
                    # Giới hạn độ dài để tránh embedding cost cao và noise
                    contextualized = response[:200]
                    logger.info(f"✅ Contextualized: '{query}' → '{contextualized}'")
                else:
                    logger.info(f"ℹ️ Contextualize trả về quá ngắn, giữ nguyên query")
            except Exception as e:
                logger.warning("⚠️ Lỗi contextualize: %s", e)

        # Bước 2: Query Rewriting — sinh keyword variants từ câu đã contextualize
        if not self.llm or len(contextualized.split()) < 3:
            return [contextualized]

        logger.info("🪄 Đang tối ưu hóa câu hỏi (Query Rewriting)...")
        prompt = (
            f"Hãy viết lại câu hỏi sau thành 2 phiên bản tìm kiếm:\n"
            f"1. Rút gọn thành các từ khóa quan trọng (Tiếng Việt).\n"
            f"2. Dịch sang từ khóa tìm kiếm bằng Tiếng Anh.\n"
            f"Câu hỏi: {contextualized}\n"
            f"Trả lời DUY NHẤT 2 dòng, không đánh số, không giải thích."
        )
        try:
            response = self.llm.complete(prompt).text
            rewritten = [contextualized]
            for line in response.split('\n'):
                line = line.strip()
                if not line: continue
                clean_line = re.sub(r'^(\d+\.|\-|\*|•)\s*', '', line).strip()
                if clean_line and clean_line != contextualized:
                    rewritten.append(clean_line)
            return rewritten[:self.max_query_rewrites]
        except Exception as e:
            logger.warning("⚠️ Lỗi rewrite: %s", e)
            return [contextualized]

    def _debug_log(self, label: str, data):
        """Write debug info to a file visible on Windows."""
        import json, datetime
        with open(os.path.join(os.path.dirname(__file__), '..', 'rag_debug.log'), 'a', encoding='utf-8') as f:
            f.write(f'[{datetime.datetime.now().isoformat()}] {label}: {json.dumps(data, ensure_ascii=False)[:500]}\n')

    def retrieve_and_rerank(self, query_str: str, selected_files: list = None, conversation_history: list | None = None):
        # Cache key dựa trên query_str + history (không bao gồm translation để
        #   cache key ổn định). Translation chỉ chạy sau cache check.
        cache_key = self._cache_key(query_str, selected_files)
        if conversation_history:
            recent = conversation_history[-4:]
            ctx_digest = hashlib.md5(str(recent).encode()).hexdigest()[:8]
            cache_key = (cache_key[0], cache_key[1], ctx_digest)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Debug: ghi nhận query gốc
        self._debug_log('query_str', query_str)
        self._debug_log('selected_files', selected_files)

        # Bước 0: Language detection + translation (Vietnamese → English)
        #   Dịch query tiếng Việt sang tiếng Anh để reranker (cross-encoder) khớp
        #   với nội dung tài liệu tiếng Anh, và thêm vào danh sách retrieval queries.
        translated = self._translate_to_english(query_str)
        reranker_query = translated or query_str
        self._debug_log('translated', translated)
        self._debug_log('reranker_query', reranker_query)

        # Bước 1: Contextualize + Viết lại câu hỏi
        queries = self.rewrite_query(query_str, conversation_history)
        # Thêm bản dịch tiếng Anh vào danh sách retrieval để tìm đúng tài liệu tiếng Anh
        if translated and translated.lower() not in [q.lower() for q in queries]:
            queries.append(translated)
            logger.info(f"🔎 Added translated query: '{translated}'")
        logger.info(f"🔎 Queries for retrieval: {queries}")
        
        # Bước 2: Thiết lập bộ lọc (Metadata Filtering)
        current_vector_retriever = self.vector_retriever
        if selected_files:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="file_name", value=f) for f in selected_files],
                condition=FilterCondition.OR
            )
            filtered_base = self.index.as_retriever(similarity_top_k=self.vector_top_k, filters=filters)
            current_vector_retriever = AutoMergingRetriever(
                filtered_base,
                self.index.storage_context,
                verbose=False
            )

        # Bước 3: Hybrid Retrieval (vector search with ALL query variants including translation)
        all_nodes_dict = {}
        
        for q in queries:
            # Vector Search
            v_nodes = current_vector_retriever.retrieve(q)

            # Sparse Search (Qdrant) hoặc BM25 fallback
            if self.sparse_retriever:
                b_nodes = self.sparse_retriever.retrieve(q, selected_files)
            else:
                b_nodes = self.bm25_retriever.retrieve(q)

                if selected_files:
                    sel_files_clean = [f.strip().lower() for f in selected_files]
                    filtered_b_nodes = []
                    for n in b_nodes:
                        if n.node.metadata.get('file_name', '').strip().lower() in sel_files_clean:
                            filtered_b_nodes.append(n)
                            if len(filtered_b_nodes) >= self.sparse_top_k:
                                break
                    b_nodes = filtered_b_nodes

            for node in v_nodes + b_nodes:
                node_id = node.node.node_id
                if node_id not in all_nodes_dict:
                    all_nodes_dict[node_id] = node
                else:
                    current_score = getattr(all_nodes_dict[node_id], "score", 0) or 0
                    new_score = getattr(node, "score", 0) or 0
                    if new_score > current_score:
                        all_nodes_dict[node_id].score = new_score
        
        combined_nodes = list(all_nodes_dict.values())
        if not combined_nodes:
            logger.warning("⚠️ Cảnh báo: Không tìm thấy kết quả nào sau khi lọc!")
            evidence = evaluate_evidence(
                [],
                min_score=self.evidence_min_score,
                min_count=self.evidence_min_count,
                conflict_margin=self.evidence_conflict_margin,
            )
            result = ([], [], evidence)
            self._cache_set(cache_key, result)
            return result

        # Bước 4: Reranking (Cross-Encoder) — dùng reranker_query (tiếng Anh nếu đã dịch)
        #   để cross-encoder có thể matching chính xác với nội dung tài liệu tiếng Anh.
        query_bundle = QueryBundle(reranker_query)
        reranked_nodes = self.reranker.postprocess_nodes(combined_nodes, query_bundle=query_bundle)
        balanced_nodes = self._select_balanced_nodes(reranked_nodes, self.final_top_n)

        # ── Hybrid Evidence: kết hợp pre-rerank (vector/BM25) + reranker (cross-encoder) scores
        #    Vấn đề: Với query tiếng Việt, vector search cho cosine similarity thấp (0.3-0.5)
        #    trên tài liệu tiếng Anh. BM25/Qdrant sparse cho score thấp nếu keyword ít trùng.
        #    Nếu chỉ dùng pre-rerank scores, evidence gate dễ ABSTAIN dù doc có liên quan.
        #    Cross-encoder dùng query tiếng Anh (đã dịch) match chính xác hơn với nội dung.
        #    Giải pháp: boost combined_nodes scores bằng reranker scores khi reranker > pre-rerank.
        reranker_score_map = {n.node.node_id: n.score for n in reranked_nodes}
        boosted_count = 0
        for n in combined_nodes:
            pre_score = n.score or 0
            rerank_score = reranker_score_map.get(n.node.node_id)
            if rerank_score is not None and rerank_score > pre_score:
                n.score = rerank_score
                boosted_count += 1
        if boosted_count > 0:
            logger.info("🚀 Boosted %d/%d evidence scores with reranker assessment", boosted_count, len(combined_nodes))

        top_combined = sorted([float(getattr(n, 'score', 0) or 0) for n in combined_nodes], reverse=True)[:5]
        self._debug_log('boosted_combined_scores', top_combined)
        self._debug_log('combined_nodes_count', len(combined_nodes))
        self._debug_log('evidence_min_score', self.evidence_min_score)

        evidence = evaluate_evidence(
            combined_nodes,
            min_score=self.evidence_min_score,
            min_count=self.evidence_min_count,
            conflict_margin=self.evidence_conflict_margin,
        )
        self._debug_log('evidence_decision', getattr(evidence, 'decision', None))
        self._debug_log('evidence_top_score', getattr(evidence, 'top_score', None))

        logger.info("📊 [Retrieval] Nodes: %d -> Pool %d -> Top %d balanced.", len(combined_nodes), len(reranked_nodes), len(balanced_nodes))
        logger.info("🔐 Evidence gating decision: %s; top_score=%s (min=%.2f); strong_count=%s; strong_files=%s",
            getattr(evidence, 'decision', None),
            getattr(evidence, 'top_score', None),
            self.evidence_min_score,
            getattr(evidence, 'node_count', None),
            getattr(evidence, 'strong_files', ()))

        if getattr(evidence, 'decision', None) == "abstain":
            # Nếu evidence vẫn ABSTAIN sau khi boost, log warning chi tiết
            logger.warning("⚠️ Evidence ABSTAIN even after reranker boost! Scores: %s", top_combined)
            result = ([], combined_nodes, evidence)
        else:
            result = (balanced_nodes, combined_nodes, evidence)

        self._cache_set(cache_key, result)
        return result