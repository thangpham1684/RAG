import unittest
import os
import shutil
from llama_index.core import Document
from llama_index.core.schema import TextNode

# Import các components để test
from retrievers.file_routing import normalize_for_match, build_file_aliases, detect_files_from_query, FILE_HINT_STOPWORDS
from embeddings.chunker import AdvancedChunker
from embeddings.vector_db import QdrantDBManager
from retrievers.hybrid_search import AdvancedHybridRetriever
from parsers.router import DocumentRouter

class TestRAGComponents(unittest.TestCase):

    def test_01_heuristic_matching(self):
        """Test cơ chế nhận diện file định tuyến dựa trên từ khóa"""
        files = ["Bao cao tai chinh 2023.pdf", "HDSD_PhanMem.docx", "Quy_trinh_nhan_su.pdf"]
        
        # 1. Test hàm build_file_aliases
        aliases = build_file_aliases(files)
        self.assertIn("Bao cao tai chinh 2023.pdf", aliases)
        self.assertTrue(any("taichinh" in a or "tai" in a for a in aliases["Bao cao tai chinh 2023.pdf"]))
        
        # 2. Test detect_files_from_query
        # Truy vấn có chứa cụm từ liên quan
        query_1 = "Hãy tóm tắt bao cao tai chinh"
        hits_1 = detect_files_from_query(query_1, aliases)
        self.assertIn("Bao cao tai chinh 2023.pdf", hits_1)
        
        query_2 = "quy trinh nhan su the nao"
        hits_2 = detect_files_from_query(query_2, aliases)
        self.assertIn("Quy_trinh_nhan_su.pdf", hits_2)

    def test_02_advanced_chunker(self):
        """Test module phân tách văn bản Chunker"""
        chunker = AdvancedChunker()
        # Tạo một Document giả lập dài để test cắt
        long_text = "Nội dung kiểm thử " * 500  # Khoảng 1500 từ
        doc = Document(text=long_text, metadata={"file_name": "test_doc.txt"})
        
        nodes = chunker.split_into_chunks([doc])
        # Đảm bảo chia ra được các node (có Node cha, Node con)
        self.assertGreater(len(nodes), 0)
        # Chắc chắn node kế thừa metadata
        self.assertEqual(nodes[0].metadata["file_name"], "test_doc.txt")

    def test_03_vector_db_manager(self):
        """Test chức năng khởi tạo DB và Index"""
        # Test vào một temp dir độc lập để không kẹt chung với Qdrant chính
        os.makedirs("./temp_test_db", exist_ok=True)
        import qdrant_client
        temp_client = qdrant_client.QdrantClient(path="./temp_test_db")
        db_manager = QdrantDBManager(client=temp_client)
        # Bắt buộc docstore lưu vào temp
        db_manager.storage_context.persist_dir = "./temp_test_db"
        
        test_nodes = [
            TextNode(text="Fake chunk 1", metadata={"file_name": "a.txt"}),
            TextNode(text="Fake chunk 2", metadata={"file_name": "a.txt"})
        ]
        
        try:
            index = db_manager.save_and_index(test_nodes)
            self.assertIsNotNone(index)
            # Kiểm tra xem Docstore có chứa node không
            docstore_nodes = list(db_manager.storage_context.docstore.docs.values())
            self.assertGreaterEqual(len(docstore_nodes), 2)
        finally:
            temp_client.close()
            import shutil
            shutil.rmtree("./temp_test_db", ignore_errors=True)

    def test_04_hybrid_retriever_query_rewrite(self):
        """Test Cắt ghép Query Rewriting (Mock LLM)"""
        
        # Tạo một class Mock LLM để giả lập kết quả trả về từ Gemini
        class MockLLM:
            def complete(self, prompt):
                class MockResponse:
                    text = "1. AI ứng dụng\n2. AI application"
                return MockResponse()

        mock_index = type('MockIndex', (), {'as_retriever': lambda *a, **kw: None, 'storage_context': None})()
        mock_nodes = [TextNode(text="fake")]
        
        retriever = AdvancedHybridRetriever(mock_index, mock_nodes, llm=MockLLM())
        
        # Tránh LLM route nếu dưới 3 từ
        self.assertEqual(retriever.rewrite_query("ngắn"), ["ngắn"])
        
        # Sẽ chạy qua LLM mock
        rewrites = retriever.rewrite_query("Phân tích ứng dụng AI vào đời sống")
        self.assertEqual(len(rewrites), 3) # Câu gốc + 2 câu sinh thêm
        self.assertEqual(rewrites[1], "AI ứng dụng")
        self.assertEqual(rewrites[2], "AI application")

if __name__ == "__main__":
    unittest.main()
