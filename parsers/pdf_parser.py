import asyncio
import os

from llama_index.core import Document, SimpleDirectoryReader

from logging_config import get_logger

logger = get_logger(__name__)

try:
    from kreuzberg import ExtractionConfig, extract_file

    KREUZBERG_AVAILABLE = True
except Exception:
    ExtractionConfig = None
    extract_file = None
    KREUZBERG_AVAILABLE = False

class AdvancedPDFParser:
    def __init__(self):
        self.use_kreuzberg = KREUZBERG_AVAILABLE

    def _extract_with_kreuzberg(self, file_path: str, file_name: str):
        if not self.use_kreuzberg or extract_file is None:
            return []

        config = ExtractionConfig(use_cache=True, enable_quality_processing=True)
        # Sử dụng asyncio.run() an toàn hơn bằng cách tạo event loop mới nếu cần
        try:
            result = asyncio.run(extract_file(file_path, config=config))
        except RuntimeError as exc:
            exc_msg = str(exc).lower()
            if "cannot be called from a running event loop" in exc_msg or "event loop" in exc_msg:
                # Fallback khi đã có event loop đang chạy (ví dụ trong FastAPI)
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(extract_file(file_path, config=config))
                finally:
                    loop.close()
            else:
                raise  # Re-throw các RuntimeError khác (lỗi thực sự của coroutine)

        content = (getattr(result, "content", "") or "").strip()
        if not content:
            return []

        metadata_obj = getattr(result, "metadata", None)
        format_type = getattr(metadata_obj, "format_type", None) if metadata_obj else None
        page_count = getattr(metadata_obj, "page_count", None) if metadata_obj else None

        doc = Document(
            text=content,
            metadata={
                "file_name": file_name,
                "format_type": format_type,
                "page_count": page_count,
                "parser": "kreuzberg",
            },
        )
        return [doc]

    def load_documents(self, directory_path: str):
        logger.info(f"📄 Đang trích xuất file PDF từ '{directory_path}'...")
        
        all_pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith(".pdf")]
        documents = []
        
        for file_name in all_pdf_files:
            file_path = os.path.join(directory_path, file_name)
            doc_added = False

            # Thử dùng Kreuzberg trước
            if self.use_kreuzberg:
                try:
                    logger.info(f"🔮 Đang dùng Kreuzberg cho: {file_name}...")
                    file_docs = self._extract_with_kreuzberg(file_path, file_name)
                    if file_docs:
                        documents.extend(file_docs)
                        doc_added = True
                        logger.info(f"✅ Kreuzberg thành công: {file_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Kreuzberg lỗi cho {file_name}: {e}. Sẽ dùng fallback.")

            # Fallback nếu Kreuzberg không được dùng hoặc lỗi
            if not doc_added:
                try:
                    logger.info(f"🔄 Đang dùng bộ đọc tiêu chuẩn cho: {file_name}...")
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    file_docs = reader.load_data()
                    for d in file_docs:
                        d.metadata["file_name"] = file_name
                    documents.extend(file_docs)
                    logger.info(f"✅ Fallback thành công: {file_name}")
                except Exception as e:
                    logger.error(f"❌ Lỗi nghiêm trọng khi đọc {file_name}: {e}")

        logger.info(f"🎉 Hoàn tất trích xuất PDF. Tổng cộng {len(documents)} khối dữ liệu.")
        return documents