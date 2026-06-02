# Bài giảng: thư mục `embeddings`

Đây là trái tim của phần "biến tri thức thành không gian tìm kiếm" trong RAG.

Nếu parser là khâu "đọc tài liệu", thì `embeddings` là khâu:
1. cắt nội dung thành đơn vị truy xuất,
2. mã hóa thành vector,
3. lưu vào hệ index để truy vấn nhanh.

---

## 1. Lý thuyết cần nắm

### 1.1 Chunking
LLM và retriever không làm việc tốt với văn bản quá dài, nên cần cắt thành **chunk**.

Chunk tốt phải cân bằng:
- đủ ngữ cảnh (tránh mất nghĩa),
- đủ ngắn để retrieval chính xác,
- có liên kết với nguồn để truy vết.

### 1.2 Embedding
Embedding là vector số biểu diễn ngữ nghĩa của text.  
Text gần nghĩa -> vector gần nhau trong không gian nhiều chiều.

### 1.3 Vector Store
Vector store (Qdrant) giúp tìm nhanh các vector gần query vector (nearest neighbors).

---

## 2. Các file chính

### `chunker.py`
Sử dụng chia chunk phân cấp (hierarchical):
- parent chunk lớn,
- child chunk nhỏ hơn để truy xuất chính xác hơn.

Thiết kế phân cấp giúp vừa giữ bối cảnh rộng, vừa có điểm rơi retrieval chi tiết.

### `vector_db.py`
Quản lý toàn bộ vòng đời vector index:
- khởi tạo Qdrant local/cloud,
- tạo collection dense/sparse,
- persist docstore/index,
- load index cũ khi restart,
- xóa dữ liệu theo file, reset index khi cần.

Nó cũng xử lý fallback khi sparse vector không khả dụng.

### `graph_db.py`
Tầng mở rộng cho graph-based knowledge indexing (dùng Ollama local).  
Hiện chưa phải tuyến chính khi chat, nhưng quan trọng cho hướng nâng cấp reasoning bằng đồ thị tri thức.

---

## 3. Công nghệ chính

- `BAAI/bge-m3`: model embedding cho dense retrieval.
- `Qdrant`: vector database (local path hoặc cloud URL).
- `SparseBM25Encoder`: sparse retrieval kiểu lexical.
- `LlamaIndex StorageContext`: persist/load index + docstore.
- `Ollama` (trong `graph_db.py`): local LLM cho trích xuất graph path.

---

## 4. Luồng xử lý điển hình

1. Nhận `Document` từ `parsers`.
2. `chunker.py` tạo node phân cấp.
3. `vector_db.py`:
   - lấy node lá để index vector,
   - lưu toàn bộ node vào docstore,
   - persist index artifacts.
4. Trả index cho `retrievers`.

---

## 5. Liên kết hệ thống

- Input: `parsers\`.
- Output runtime: `retrievers\`.
- Output artifact: `docstore\`, `qdrant_data\`.
- Được gọi bởi: `ingest_worker.py`, `api.py`, benchmark scripts.

Nói ngắn: `embeddings` là tầng chuyển "text" thành "khả năng tìm kiếm ngữ nghĩa".
