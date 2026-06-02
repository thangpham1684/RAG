import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PandasCSVReader, PandasExcelReader, PptxReader
from logging_config import get_logger

logger = get_logger(__name__)

class UniversalDataParser:
    def __init__(self):
        logger.info("📂 Khởi tạo bộ đọc Đa định dạng (Excel, CSV, PPTX)...")
        self.file_extractor = {
            ".csv": PandasCSVReader(pandas_config={"header": 0}),
            ".xlsx": PandasExcelReader(),
            ".pptx": PptxReader()
        }

    def load_documents(self, directory_path: str):
        logger.info(f"📊 Đang trích xuất Excel/CSV/PPTX từ '{directory_path}'...")
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            required_exts=[".csv", ".xlsx", ".pptx"],
            file_extractor=self.file_extractor
        )
        
        try:
            documents = reader.load_data()
            logger.info(f"✅ Đã trích xuất thành công {len(documents)} khối dữ liệu từ Excel/CSV/PPTX.")
            
            for doc in documents:
                file_path = doc.metadata.get("file_path", "")
                if file_path:
                    file_name = os.path.basename(file_path)
                    doc.metadata["file_name"] = file_name
                    category = os.path.basename(os.path.dirname(file_path))
                    doc.metadata["category"] = category
            
            return documents
        except Exception as e:
            logger.warning(f"⚠️ Không tìm thấy hoặc lỗi đọc file Excel/CSV/PPTX: {e}")
            return []
