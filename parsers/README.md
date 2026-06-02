# Bài giảng: thư mục `parsers`

Đây là tầng **chuyển đổi tài liệu thô -> tài liệu có cấu trúc** cho RAG.

Nếu `data` là "sách gốc", thì `parsers` là "bộ phận số hóa và chuẩn hóa nội dung" trước khi đưa vào hệ thống tìm kiếm.

---

## 1. Lý thuyết nền: vì sao parser rất quan trọng?

RAG không đọc trực tiếp PDF/Excel như con người.  
Nó cần text sạch + metadata nhất quán.

Parser tốt sẽ:
- giữ đúng nội dung ngữ nghĩa,
- giữ đúng liên hệ nguồn (`file_name`, category...),
- giảm ký tự rác (noise),
- và tránh mất cấu trúc quan trọng (bảng, heading, đoạn).

Parser kém sẽ gây lỗi dây chuyền:
- chunk xấu,
- embedding không đại diện,
- retrieval lệch ngữ cảnh,
- answer hallucination.

---

## 2. Các module chính

### `router.py`
Là "bộ điều phối":
- quét thư mục dữ liệu,
- gọi parser phù hợp theo định dạng,
- gom kết quả về cùng chuẩn `Document`,
- chạy bước dọn rác văn bản.

Đây là nơi giúp toàn hệ thống có một **điểm vào thống nhất** cho ingestion.

### `pdf_parser.py`
Tập trung cho PDF:
- ưu tiên **Kreuzberg** (local parser hiện đại, hỗ trợ extraction tốt),
- fallback sang `SimpleDirectoryReader` khi parser chính lỗi,
- luôn gắn `file_name` để trích dẫn nguồn về sau.

Điểm mạnh của thiết kế này: vừa tận dụng parser tốt hơn, vừa có đường lui để tránh gãy pipeline.

### `docx_parser.py`
Xử lý file Word (`.doc`, `.docx`) bằng reader chuyên dụng.

### `universal_parser.py`
Xử lý dữ liệu bảng/trình chiếu:
- `.xlsx`, `.csv`, `.pptx`
- chuẩn hóa metadata như `file_name`, `category`.

---

## 3. Công nghệ và thư viện liên quan

- `llama_index.core.SimpleDirectoryReader`: lớp reader nền.
- Reader theo định dạng: `DocxReader`, `PandasExcelReader`, `PandasCSVReader`, `PptxReader`.
- `kreuzberg`: parser đa định dạng, hiện dùng làm parser PDF chính.
- Regex cleaning trong `router.py`: chuẩn hóa whitespace, loại pattern rác phổ biến.

---

## 4. Luồng dữ liệu trong chính thư mục này

1. `router.py` quét thư mục.
2. Gọi parser theo định dạng file.
3. Nhận list `Document`.
4. Chuẩn hóa metadata + dọn nội dung.
5. Trả list `Document` sạch sang tầng chunking.

---

## 5. Liên kết với các tầng khác

- Đầu vào từ `data\`.
- Đầu ra chuyển sang `embeddings\chunker.py`.
- Được gọi bởi `ingest_worker.py` (ingestion chính), `evaluate_*` (benchmark/eval), và các script test.

Hiểu ngắn gọn: `parsers` là tầng "ETL text" của hệ RAG.
