import os
from llama_index.llms.gemini import Gemini
from llama_index.core.prompts import PromptTemplate
from logging_config import get_logger

logger = get_logger(__name__)

class ResponseGenerator:
    def __init__(self, model_name="models/gemini-flash-latest"):
        logger.info(f"🧠 MODULE 6: Khởi tạo Generator ({model_name})...")
        self.llm = Gemini(api_key=os.getenv("GEMINI_API_KEY"), model_name=model_name)
        
        # Prompt mẫu chuyên nghiệp theo yêu cầu sơ đồ
        self.qa_prompt_tmpl = PromptTemplate(
            "Bạn là một trợ lý AI phân tích tài liệu chuyên nghiệp.\n"
            "Dưới đây là các thông tin được trích xuất từ tài liệu của người dùng:\n"
            "--- NGỮ CẢNH TÀI LIỆU ---\n"
            "{context_str}\n"
            "-------------------------\n"
            "QUY TẮC NGẮT NGỮ CẢNH (GROUNDING):\n"
            "1. Bạn chỉ được sử dụng thông tin trực tiếp có trong các tài liệu được cung cấp. Bạn không được suy luận, không được thêm thông tin ngoài nội dung tài liệu.\n"
            "2. Nếu thông tin không rõ ràng hoặc không có trong tài liệu, hãy trả lời rằng không tìm thấy trong kho dữ liệu hiện tại và không bịa đặt.\n"
            "3. Luôn trích dẫn nguồn cho mỗi ý quan trọng bằng ký hiệu: [File: Tên_File] và chỉ sử dụng các trích dẫn từ {evidence_status}.\n"
            "4. Trình bày chuyên nghiệp, dễ đọc; nếu cần, dùng Markdown.\n"
            "5. Dựa vào lịch sử hội thoại bên dưới để hiểu ngữ cảnh của câu hỏi hiện tại. Nếu câu hỏi là follow-up (ví dụ: 'còn phần này thì sao?', 'giải thích thêm'), hãy dùng thông tin từ lịch sử để trả lời mạch lạc.\n\n"
            "{history_str}"
            "Câu hỏi: {query_str}\n"
            "Trả lời chi tiết bằng Tiếng Việt:"
        )

    def generate_answer_stream(self, query_str, retrieved_nodes, evidence_status="OK", conversation_history=None):
        # Tạo chuỗi ngữ cảnh với tên file rõ ràng để trích dẫn
        context_list = []
        for i, node in enumerate(retrieved_nodes, 1):
            file_name = node.node.metadata.get('file_name', 'Tài liệu')
            text = node.node.text.replace("\n", " ")
            # Use the exact citation token required by the prompt: [File: {file_name}] followed by the text.
            context_list.append(f"[File: {file_name}] {text}")
            
        context_str = "\n\n".join(context_list)
        
        # Format conversation history
        history_str = ""
        if conversation_history:
            formatted = []
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    formatted.append(f"Người dùng: {content}")
                elif role == "assistant":
                    formatted.append(f"Bạn: {content}")
            if formatted:
                history_str = "--- LỊCH SỬ HỘI THOẠI ---\n" + "\n".join(formatted) + "\n\n"
        
        fmt_prompt = self.qa_prompt_tmpl.format(
            context_str=context_str,
            query_str=query_str,
            evidence_status=evidence_status,
            history_str=history_str,
        )
        
        response = self.llm.stream_complete(fmt_prompt)
        for chunk in response:
            yield chunk.delta