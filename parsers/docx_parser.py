import os
from llama_index.readers.file import DocxReader
from llama_index.core import SimpleDirectoryReader
from logging_config import get_logger

logger = get_logger(__name__)

class AdvancedDocxParser:
    def __init__(self):
        # Khởi tạo công cụ đọc Word
        self.parser = DocxReader()
        self.file_extractor = {".docx": self.parser}

    def load_documents(self, directory_path: str):
        """
        Quét và trích xuất toàn bộ file Word trong một thư mục
        """
        logger.info(f"📝 Đang trích xuất các file Docx từ '{directory_path}'...")
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".docx", ".doc"],
            file_extractor=self.file_extractor
        )
        
        documents = reader.load_data()
        
        # Gắn file_name vào metadata để trích dẫn nguồn (giống pdf_parser và universal_parser)
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if file_path and not doc.metadata.get("file_name"):
                file_name = os.path.basename(file_path)
                doc.metadata["file_name"] = file_name
        
        logger.info(f"✅ Đã trích xuất thành công {len(documents)} khối dữ liệu từ Word.")
        return documents