# Bài giảng: thư mục `data` trong hệ thống RAG

Thư mục `data` là **nơi chứa tri thức gốc** của toàn bộ hệ thống.  
Hiểu đơn giản: nếu RAG là một "thư viện thông minh", thì `data` chính là kệ sách.

---

## 1. Về mặt lý thuyết, `data` có vai trò gì trong RAG?

RAG (Retrieval-Augmented Generation) gồm 2 bước cốt lõi:
1. **Retrieval**: tìm các đoạn liên quan trong kho tri thức.
2. **Generation**: dùng các đoạn đó để sinh câu trả lời.

Vì vậy chất lượng của `data` quyết định trực tiếp:
- độ đúng của retrieval,
- mức độ "bám nguồn" (faithfulness),
- và mức độ nhiễu của ngữ cảnh.

Nếu dữ liệu đầu vào lộn xộn, trùng lặp, sai ngôn ngữ, thiếu ngữ cảnh... thì dù mô hình mạnh đến đâu, câu trả lời vẫn dễ sai.

---

## 2. Các loại tài liệu trong thư mục này

- `ĐATN (3).pdf`: tài liệu chính tiếng Việt, xuất hiện nhiều trong benchmark.
- `ALGORITMOS_GENETICOS_APLICADOS_AO_PROBLEMA_DE_LOCA.pdf`: tài liệu học thuật về GA/facility location.
- `drones-08-00724-v2.pdf`, `hub location.pdf`, `mamba2.pdf`, `ts-mixer.pdf`: tài liệu tham khảo bổ sung.
- `test.xlsx`: dữ liệu dạng bảng để kiểm thử parser đa định dạng.

Mỗi định dạng sẽ được parser tương ứng xử lý khác nhau:
- PDF -> `pdf_parser.py` (Kreuzberg + fallback),
- DOC/DOCX -> `docx_parser.py`,
- XLSX/CSV/PPTX -> `universal_parser.py`.

---

## 3. Kiến thức thực hành cho người mới

Khi thêm tài liệu mới vào `data`, nên chú ý:

1. **Tên file rõ nghĩa**: giúp metadata `file_name` dễ hiểu khi trích dẫn nguồn.
2. **Tránh quá nhiều file gần như trùng nội dung**: làm retrieval nhiễu.
3. **Đảm bảo mã hóa và ngôn ngữ ổn định**: tránh lỗi ký tự gây giảm chất lượng chunk/retrieval.
4. **Chạy lại ingestion** sau khi thay dữ liệu, nếu không index không cập nhật.

---

## 4. Liên kết với các thư mục khác

- `parsers\`: đọc file từ `data` và chuyển thành `Document`.
- `embeddings\`: cắt chunk + tạo vector index.
- `docstore\` và `qdrant_data\`: lưu chỉ mục đã xây từ dữ liệu.
- `retrievers\`: truy xuất các chunk đã được index.
- `tests\`: benchmark truy xuất dựa trên dữ liệu trong `data`.

Tóm lại: `data` là **nguồn sự thật** (source of truth). Mọi tầng còn lại chỉ là cách tổ chức và khai thác nguồn này.
