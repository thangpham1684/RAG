# Bài giảng: thư mục `retrievers`

Đây là nơi quyết định "tìm đúng cái gì để đưa cho LLM".  
Trong thực tế, chất lượng RAG thường chết ở retrieval trước khi đến generation.

---

## 1. Lý thuyết retrieval quan trọng

### 1.1 Dense retrieval
Dựa vào embedding vector:
- tốt cho hiểu nghĩa, paraphrase, đa ngôn ngữ.
- yếu khi cần match từ khóa cực cụ thể.

### 1.2 Sparse/BM25 retrieval
Dựa vào từ khóa và tần suất token:
- tốt cho match exact term (tên dataset, viết tắt, mã số).
- yếu hơn dense ở ngữ nghĩa trừu tượng.

### 1.3 Hybrid retrieval
Kết hợp dense + sparse để lấy ưu điểm cả hai.

### 1.4 Rerank
Sau khi gom candidate, dùng cross-encoder để chấm lại mức liên quan với query.  
Đây là bước tăng precision top-K rất đáng kể.

---

## 2. Các module chính

### `hybrid_search.py`
Pipeline retrieval đầy đủ:
1. query rewrite (nếu có LLM),
2. dense retrieval,
3. sparse retrieval hoặc BM25 fallback,
4. merge/dedup candidate,
5. rerank,
6. chọn top node cân bằng theo file.

Đây là thành phần ảnh hưởng trực tiếp đến `hit@1`, `mrr`.

### `sparse_bm25.py`
Chứa:
- encoder BM25 dạng sparse vector,
- retriever sparse trên Qdrant.

Nếu sparse vector của collection không có, hệ thống fallback sang BM25 in-memory.

---

## 3. Công nghệ dùng trong thư mục này

- `AutoMergingRetriever` (LlamaIndex): gộp child node về parent context.
- `SentenceTransformerRerank` (`bge-reranker-v2-m3`): cross-encoder rerank.
- `BM25Retriever`: lexical retrieval fallback.
- `QdrantSparseRetriever`: sparse retrieval trên DB.

---

## 4. Vì sao thư mục này thường là "điểm yếu"?

Trong hệ RAG thật, lỗi phổ biến:
- query chung chung -> top-1 bị nhiễu,
- sparse không hoạt động -> mất lợi thế keyword,
- rerank pool chưa đủ rộng -> bỏ sót evidence tốt.

Do đó việc tuning `retrievers` thường cải thiện mạnh hơn tuning prompt.

---

## 5. Liên kết hệ thống

- Nhận index/docstore từ `embeddings\vector_db.py`.
- Trả `best_nodes` cho `generators\llm_generator.py`.
- Được gọi từ `api.py`, benchmark/eval scripts, test scripts.

Tóm lại: `retrievers` là "bộ não tìm chứng cứ" của toàn hệ RAG.
