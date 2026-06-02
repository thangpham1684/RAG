# Bài giảng: thư mục `generators`

Đây là tầng "viết câu trả lời" của RAG:  
nhận câu hỏi + evidence rồi sinh output cuối cùng cho người dùng.

---

## 1. Lý thuyết generation trong RAG

Generation trong RAG không nên là "trả lời theo trí nhớ model", mà là:

1. đọc context đã retrieval,
2. tổng hợp theo yêu cầu câu hỏi,
3. giữ nguồn trích dẫn rõ ràng,
4. tránh bịa (hallucination).

Vì vậy prompt của generator đóng vai trò như "hợp đồng hành vi":
- chỉ dùng dữ liệu đã cung cấp,
- nếu thiếu dữ liệu thì nói rõ thiếu,
- trả về định dạng dễ đọc (Markdown),
- kèm citation.

---

## 2. Module chính

### `llm_generator.py`
Module này làm 3 việc chính:
1. Khởi tạo LLM (Gemini).
2. Dựng prompt QA có ràng buộc trích dẫn nguồn.
3. Stream output theo chunk để UI hiển thị realtime.

Điểm hay của stream:
- UX tốt hơn (nhìn thấy model đang trả lời),
- backend không cần chờ full answer mới trả.

---

## 3. Công nghệ chính

- `llama_index.llms.gemini.Gemini`: wrapper LLM.
- `PromptTemplate`: template hóa prompt.
- Streaming API (`stream_complete`): trả kết quả từng phần.

---

## 4. Những điểm người mới thường nhầm

1. **Generator không tự biết sự thật**  
   Nó chỉ giỏi tổng hợp từ context đã đưa vào.
2. **Prompt không sửa được retrieval kém**  
   Nếu retriever đưa nhầm context, generator vẫn có nguy cơ trả lời lệch.
3. **Citation không chỉ để đẹp**  
   Nó là cơ chế audit: giúp kiểm chứng đáp án.

---

## 5. Liên kết hệ thống

- Nhận `best_nodes` từ `retrievers\`.
- Được gọi từ `api.py` trong endpoint chat.
- Output stream lên frontend (`app.py` hoặc `ui\app.js`).

Tóm lại: `generators` là tầng "diễn đạt tri thức", không phải tầng "tìm tri thức".
