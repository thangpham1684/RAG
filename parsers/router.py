import os
import re
from .pdf_parser import AdvancedPDFParser
from .docx_parser import AdvancedDocxParser
from .universal_parser import UniversalDataParser
from logging_config import get_logger

logger = get_logger(__name__)

class DocumentRouter:
    def __init__(self):
        logger.info("⚙️ Khởi tạo hệ thống Phân luồng dữ liệu (Document Router)...")
        self.pdf_parser = AdvancedPDFParser()
        self.docx_parser = AdvancedDocxParser()
        self.universal_parser = UniversalDataParser()

    def clean_garbage_text(self, text):
        if not text:
            return ""
        # Xóa nhiều khoảng trắng liên tiếp
        text = re.sub(r'\s+', ' ', text)
        # Xóa các mẫu header/footer lặp lại, vd: Page 1 of 10
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Confidential', '', text, flags=re.IGNORECASE)
        # Đã loại bỏ dòng Regex sai lệch gây mất dấu Tiếng Việt ở đây
        return text.strip()

    def process_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {directory_path}")
            
        logger.info("-" * 50)
        logger.info(f"🔍 Đang quét thư mục: '{directory_path}'")
        
        all_documents = []

        try:
            pdf_docs = self.pdf_parser.load_documents(directory_path)
            all_documents.extend(pdf_docs)
        except Exception as e:
            logger.warning(f"⏭️ Bỏ qua PDF (hoặc lỗi): {e}")

        try:
            docx_docs = self.docx_parser.load_documents(directory_path)
            all_documents.extend(docx_docs)
        except Exception as e:
            logger.warning(f"⏭️ Bỏ qua Word (hoặc lỗi): {e}")

        try:
            uni_docs = self.universal_parser.load_documents(directory_path)
            all_documents.extend(uni_docs)
        except Exception as e:
            logger.warning(f"⏭️ Bỏ qua Excel/CSV/PPTX (hoặc lỗi): {e}")

        logger.info("-" * 50)
        if len(all_documents) > 0:
            logger.info(f"🧹 Đang dọn dẹp rác văn bản (Data Cleaning)...")
            for doc in all_documents:
                cleaned_text = self.clean_garbage_text(doc.get_content())
                doc.set_content(cleaned_text)
                
            logger.info(f"🎉 Hoàn tất Phase 1! Trích xuất {len(all_documents)} khối dữ liệu.")
            
        return all_documents