"""
Tests for conversation history contextualization in the hybrid retriever.

Verifies that follow-up queries (e.g. "cơ chế cốt lõi của mamba2") are
contextualized against prior conversation history before retrieval,
preventing false abstain from evidence gating.
"""

import unittest
from unittest.mock import MagicMock, patch
from llama_index.core.schema import TextNode, NodeWithScore
from retrievers.hybrid_search import AdvancedHybridRetriever


class MockLLM:
    """LLM mock that returns controlled responses for contextualize + rewrite calls."""

    def __init__(self):
        self.call_count = 0
        self.calls = []

    def complete(self, prompt):
        self.call_count += 1
        self.calls.append(prompt)

        class MockResponse:
            text = ""

        resp = MockResponse()
        if self.call_count == 1:
            resp.text = "cơ chế cốt lõi của Mamba2 (selective state space model)"
        elif self.call_count == 2:
            resp.text = "cơ chế cốt lõi Mamba2\nMamba2 core mechanisms"
        else:
            resp.text = "mamba2 core mechanisms"
        return resp


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_mamba2_nodes():
    """Create realistic Mamba2 document nodes."""
    return [
        TextNode(
            text=(
                "Mamba2 là một kiến trúc mô hình state space (SSM) chọn lọc, "
                "được phát triển dựa trên nền tảng của Mamba1. Cơ chế cốt lõi "
                "của Mamba2 bao gồm selective scan mechanism cho phép mô hình "
                "chọn lọc thông tin cần nhớ và quên."
            ),
            metadata={"file_name": "mamba2.pdf"},
        ),
        TextNode(
            text=(
                "Cơ chế cốt lõi của Mamba2: (1) Selective State Space Model — "
                "cho phép mô hình điều chỉnh hành vi dựa trên nội dung đầu vào, "
                "(2) Hardware-aware parallel scan — tối ưu hóa tính toán trên GPU, "
                "(3) Simplified architecture — loại bỏ MLP block."
            ),
            metadata={"file_name": "mamba2.pdf"},
        ),
        TextNode(
            text=(
                "Mamba2 sử dụng selective scan thay vì attention mechanism truyền "
                "thống. Selective scan cho phép mô hình quyết định thông tin nào "
                "được giữ lại ở mỗi bước thời gian."
            ),
            metadata={"file_name": "mamba2.pdf"},
        ),
    ]


def _make_mock_index():
    """Create a mock index suitable for AdvancedHybridRetriever.__init__."""
    mock_idx = MagicMock()
    mock_idx.as_retriever.return_value = MagicMock()
    mock_idx.storage_context = MagicMock()
    return mock_idx


MAMBA2_HISTORY = [
    {"role": "user", "content": "viết mamba2"},
    {"role": "assistant", "content": "Mamba2 là kiến trúc SSM chọn lọc... [File: mamba2.pdf]"},
]


# Patch SentenceTransformerRerank to avoid actual model loading (torch segfault)
@patch("retrievers.hybrid_search.SentenceTransformerRerank")
class TestContextualization(unittest.TestCase):
    """Test suite for conversation history contextualization."""

    def setUp(self):
        self.nodes = _make_mamba2_nodes()

    # ── rewrite_query tests ─────────────────────────────────────────────

    def test_rewrite_without_history_skips_contextualization(self, _mock_reranker):
        """Without history, rewrite_query should skip contextualize step."""
        llm = MockLLM()
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = False

        queries = retriever.rewrite_query("cơ chế cốt lõi của mamba2")

        # Should have original query + 2 keyword variants
        self.assertGreaterEqual(len(queries), 1)
        self.assertIn("cơ chế cốt lõi của mamba2", queries)
        # Only keyword rewrite step (1 call), no contextualize step
        self.assertEqual(llm.call_count, 1)

    def test_rewrite_with_history_contextualizes_follow_up(self, _mock_reranker):
        """With history, rewrite_query should contextualize then keyword rewrite."""
        llm = MockLLM()
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = False

        queries = retriever.rewrite_query("cơ chế cốt lõi của mamba2", MAMBA2_HISTORY)

        # First query should be the contextualized form (references Mamba2 SSM)
        self.assertGreaterEqual(len(queries), 1)
        self.assertIn("mamba2", queries[0].lower())
        # LLM called twice: contextualize + keyword rewrite
        self.assertEqual(llm.call_count, 2)

    def test_rewrite_with_history_includes_history_in_prompt(self, _mock_reranker):
        """The contextualize prompt should contain the conversation history."""
        llm = MockLLM()
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = False

        retriever.rewrite_query("cơ chế cốt lõi của mamba2", MAMBA2_HISTORY)

        first_prompt = llm.calls[0] if llm.calls else ""
        self.assertIn("LỊCH SỬ HỘI THOẠI", first_prompt)
        self.assertIn("viết mamba2", first_prompt)
        self.assertIn("cơ chế cốt lõi của mamba2", first_prompt)

    def test_rewrite_without_llm_skips_all_rewriting(self, _mock_reranker):
        """Without an LLM, rewrite_query returns only the original query."""
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=None)
        retriever.enable_cache = False

        queries = retriever.rewrite_query("cơ chế cốt lõi của mamba2")
        self.assertEqual(queries, ["cơ chế cốt lõi của mamba2"])

        # With history but no LLM — still returns original
        queries = retriever.rewrite_query("cơ chế cốt lõi của mamba2", MAMBA2_HISTORY)
        self.assertEqual(queries, ["cơ chế cốt lõi của mamba2"])

    def test_rewrite_with_short_query_skips_rewrite(self, _mock_reranker):
        """Queries < 3 words should skip keyword rewriting entirely."""
        llm = MockLLM()
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = False

        queries = retriever.rewrite_query("mamba2")
        # Only original query returned, no LLM calls
        self.assertEqual(queries, ["mamba2"])
        self.assertEqual(llm.call_count, 0)

    def test_rewrite_with_empty_history_list(self, _mock_reranker):
        """Empty list [] should be treated like no history (falsy)."""
        llm = MockLLM()
        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = False

        queries = retriever.rewrite_query("cơ chế cốt lõi của mamba2", [])
        # Should skip contextualize (empty history is falsy) but keyword rewrite runs
        self.assertIn("cơ chế cốt lõi của mamba2", queries)
        self.assertEqual(llm.call_count, 1)

    # ── retrieve_and_rerank tests ──────────────────────────────────────

    def _make_mocked_retriever(self, vector_score=0.8, llm=None, enable_cache=False):
        """Create retriever with all internals mocked for controlled testing.

        Args:
            vector_score: Score assigned to each node by vector/BM25 retrievers
                           (used for evidence gating — cosine similarity [0,1]).
            llm: Optional MockLLM instance.
            enable_cache: Whether to enable result caching.
        """
        if llm is None:
            llm = MockLLM()

        retriever = AdvancedHybridRetriever(_make_mock_index(), self.nodes, llm=llm)
        retriever.enable_cache = enable_cache
        retriever._result_cache.clear()

        # Mock vector retriever — scores used for evidence gating (cosine similarity)
        mock_vector = MagicMock()
        mock_vector.retrieve.return_value = [
            NodeWithScore(node=n, score=vector_score) for n in self.nodes
        ]
        retriever.vector_retriever = mock_vector

        # Mock BM25 retriever — scores used for evidence gating
        mock_bm25 = MagicMock()
        mock_bm25.retrieve.return_value = [
            NodeWithScore(node=n, score=vector_score) for n in self.nodes
        ]
        retriever.bm25_retriever = mock_bm25

        # Mock reranker — still used for ranking but NOT for evidence gate anymore
        def mock_postprocess(nodes, query_bundle=None):
            # Keep vector scores so balanced selection + LLM still works
            return [NodeWithScore(node=n.node, score=getattr(n, 'score', 0.0)) for n in nodes]
        retriever.reranker.postprocess_nodes = mock_postprocess

        return retriever

    def test_retrieve_with_history_does_not_abstain(self, _mock_reranker):
        """Follow-up query with history must pass evidence gating (vector scores ≥ 0.4)."""
        retriever = self._make_mocked_retriever(vector_score=0.75)

        best_nodes, raw_nodes, evidence = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi của mamba2",
            conversation_history=MAMBA2_HISTORY,
        )

        self.assertNotEqual(
            evidence.decision, "abstain",
            f"Follow-up with history should NOT abstain. "
            f"decision={evidence.decision}, top_score={evidence.top_score:.4f}",
        )
        self.assertEqual(evidence.decision, "ok")

    def test_retrieve_without_history_abstains_on_low_vector_scores(self, _mock_reranker):
        """Without history, vector scores < 0.4 (cosine similarity) should abstain."""
        retriever = self._make_mocked_retriever(vector_score=0.15)

        best_nodes, raw_nodes, evidence = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi của mamba2",
            # No conversation_history
        )

        self.assertEqual(evidence.decision, "abstain")

    def test_retrieve_with_history_passes_vector_scores(self, _mock_reranker):
        """History + vector scores ≥ 0.4 should pass evidence gate."""
        retriever = self._make_mocked_retriever(vector_score=0.45)

        best_nodes, raw_nodes, evidence = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi của mamba2",
            conversation_history=MAMBA2_HISTORY,
        )

        self.assertEqual(
            evidence.decision, "ok",
            f"Vector score 0.45 >= 0.4 threshold should pass. "
            f"decision={evidence.decision}, top_score={evidence.top_score:.4f}",
        )

    def test_retrieve_caches_with_history_context(self, _mock_reranker):
        """Different conversation history → different cache key → no cache hit."""
        retriever = self._make_mocked_retriever(enable_cache=True)

        history_a = MAMBA2_HISTORY
        history_b = [
            {"role": "user", "content": "viết transformer"},
            {"role": "assistant", "content": "Transformer là attention..."},
        ]

        # First call with history_a
        _, _, ev_a = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi", conversation_history=history_a,
        )

        # Second call with history_b — should miss cache
        calls_before = retriever.llm.call_count
        _, _, ev_b = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi", conversation_history=history_b,
        )

        self.assertGreater(
            retriever.llm.call_count, calls_before,
            "Different history should produce different cache key",
        )

    def test_retrieve_caches_same_history_hits_cache(self, _mock_reranker):
        """Same conversation history → same cache key → cache hit."""
        retriever = self._make_mocked_retriever(enable_cache=True)

        # First call populates cache
        retriever.retrieve_and_rerank(
            "cơ chế cốt lõi", conversation_history=MAMBA2_HISTORY,
        )

        # Second call with identical history and query — should hit cache
        calls_before = retriever.llm.call_count
        _, _, evidence = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi", conversation_history=MAMBA2_HISTORY,
        )

        self.assertEqual(
            retriever.llm.call_count, calls_before,
            "Same history should hit cache (no additional LLM calls)",
        )
        self.assertEqual(evidence.decision, "ok")

    def test_retrieve_with_none_vs_empty_history(self, _mock_reranker):
        """None history should be equivalent to no history."""
        retriever = self._make_mocked_retriever()

        # None → should skip contextualize
        _, _, ev_none = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi của mamba2", conversation_history=None,
        )
        calls_after_none = retriever.llm.call_count

        # Empty list → also skip contextualize (falsy)
        retriever.llm.calls.clear()
        retriever.llm.call_count = 0
        _, _, ev_empty = retriever.retrieve_and_rerank(
            "cơ chế cốt lõi của mamba2", conversation_history=[],
        )
        calls_after_empty = retriever.llm.call_count

        # Both should behave the same: translate (1) + keyword rewrite (1) = 2 calls
        self.assertEqual(calls_after_none, 2)
        self.assertEqual(calls_after_empty, 2)


if __name__ == "__main__":
    unittest.main()
