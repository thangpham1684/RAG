# Bài giảng: thư mục `ui`

Đây là giao diện web tĩnh để người dùng tương tác với hệ RAG.  
Nó không làm retrieval hay embedding; nó làm đúng vai trò frontend: gửi câu hỏi, nhận stream, hiển thị kết quả.

---

## 1. Lý thuyết kiến trúc frontend trong dự án này

Hệ thống dùng mô hình tách lớp rõ:
- **Frontend (`ui`)**: phần hiển thị và tương tác người dùng.
- **Backend (`api.py`)**: xử lý nghiệp vụ RAG.

Lợi ích của tách lớp:
1. UI thay đổi không ảnh hưởng lõi RAG.
2. Dễ thay frontend khác (Streamlit, web app khác, mobile).
3. Debug thuận tiện: phân biệt lỗi hiển thị và lỗi thuật toán.

---

## 2. Các file và vai trò

### `index.html`
Định nghĩa cấu trúc giao diện:
- sidebar trạng thái,
- vùng chat,
- ô nhập câu hỏi,
- các nút thao tác.

### `styles.css`
Thiết kế UX:
- màu sắc, spacing, typography,
- style message bubble,
- animation cursor stream.

### `app.js`
Logic chính của frontend:
- health-check backend định kỳ,
- gọi API chat,
- đọc stream text,
- render Markdown an toàn (DOMPurify),
- highlight code block.

---

## 3. Công nghệ frontend dùng ở đây

- Vanilla JS (không framework nặng).
- `marked` để parse Markdown.
- `DOMPurify` để sanitize HTML, giảm XSS.
- `highlight.js` để tô màu code block.

Thiết kế này đơn giản, phù hợp môi trường nội bộ và dễ bảo trì cho người mới.

---

## 4. Luồng tương tác từ góc nhìn người dùng

1. Mở UI.
2. UI kiểm tra `/health`.
3. Người dùng nhập câu hỏi.
4. UI gọi `/api/v1/chat`.
5. Backend stream từng chunk.
6. UI render theo thời gian thực.

---

## 5. Liên kết với các phần khác

- Chỉ giao tiếp với `api.py` qua HTTP.
- Không truy cập trực tiếp `parsers\`, `embeddings\`, `retrievers\`.

Đây là nguyên tắc rất quan trọng: frontend không phụ thuộc nội bộ thuật toán.
