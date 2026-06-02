# Bài giảng: thư mục `tests`

Thư mục này là "phòng thí nghiệm" để đo chất lượng kỹ thuật của hệ RAG.

Trong dự án RAG, test không chỉ để tránh bug code, mà còn để đo:
- retrieval có đúng không,
- answer có bám context không,
- thay đổi parser/index có làm tụt chất lượng không.

---

## 1. Lý thuyết test trong hệ RAG

Khác với CRUD app, RAG có thêm các loại kiểm thử đặc thù:

1. **Component test**: parser/chunker/retriever hoạt động đúng hành vi kỹ thuật.
2. **Retrieval benchmark**: đo hit@k, mrr.
3. **Answer-level evaluation**: đo faithfulness, relevancy (RAGAS hoặc heuristic).

Nếu chỉ có unit test truyền thống, bạn vẫn có thể "pass test nhưng trả lời dở".

---

## 2. Các file hiện có

- `test_unit_components.py`  
  Test các khối lõi: heuristic match, chunker, vector index, query rewrite.

- `test_ragas_evaluation.py`  
  Test helper của script RAGAS (chuẩn hóa record, chọn metric, parse config, tổng hợp điểm).

- `test_pdf_parser_kreuzberg.py`  
  Test parser PDF mới: ưu tiên Kreuzberg và fallback khi lỗi.

- `rag_benchmark_vn_en.json`  
  Bộ câu hỏi benchmark đa ngôn ngữ cho retrieval.

---

## 3. Chiến lược test nên áp dụng cho người mới

1. Chạy unit test nhanh sau mỗi thay đổi logic.
2. Chạy retrieval benchmark sau thay đổi parser/index.
3. Chạy answer-eval theo batch nhỏ để tránh quota LLM evaluator.
4. So sánh trước/sau bằng cùng benchmark file.

---

## 4. Liên kết với các thư mục khác

- Test đọc và gọi trực tiếp module từ:
  - `parsers\`
  - `embeddings\`
  - `retrievers\`
  - script ở root (`evaluate_*`, `app.py`)

- Benchmark phụ thuộc dữ liệu đã ingest từ `data\`.

Tóm lại: `tests` giúp bạn biến nhận xét cảm tính "hình như hệ thống tốt hơn" thành số đo có kiểm chứng.
