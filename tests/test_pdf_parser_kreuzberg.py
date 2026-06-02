import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core import Document

from parsers.pdf_parser import AdvancedPDFParser


class TestPdfParserKreuzberg(unittest.TestCase):
    def _make_temp_pdf(self) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp(prefix="pdf_parser_test_")
        file_name = "sample.pdf"
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(b"%PDF-1.4 test content")
        return temp_dir, file_name

    def test_prefers_kreuzberg_when_available(self):
        temp_dir, file_name = self._make_temp_pdf()
        try:
            k_result = SimpleNamespace(
                content="Kreuzberg extracted text",
                metadata=SimpleNamespace(format_type="pdf", page_count=1),
            )
            with patch("parsers.pdf_parser.KREUZBERG_AVAILABLE", True, create=True), patch(
                "parsers.pdf_parser.ExtractionConfig",
                return_value=SimpleNamespace(),
                create=True,
            ), patch(
                "parsers.pdf_parser.extract_file",
                new=AsyncMock(return_value=k_result),
                create=True,
            ) as mocked_extract, patch("parsers.pdf_parser.SimpleDirectoryReader") as mocked_reader:
                parser = AdvancedPDFParser()
                docs = parser.load_documents(temp_dir)

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].get_content(), "Kreuzberg extracted text")
            self.assertEqual(docs[0].metadata.get("file_name"), file_name)
            mocked_extract.assert_awaited_once()
            mocked_reader.assert_not_called()
        finally:
            for name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, name))
            os.rmdir(temp_dir)

    def test_falls_back_to_simple_reader_when_kreuzberg_fails(self):
        temp_dir, file_name = self._make_temp_pdf()
        try:
            fallback_doc = Document(text="Fallback text", metadata={})
            mocked_reader_instance = MagicMock()
            mocked_reader_instance.load_data.return_value = [fallback_doc]

            with patch("parsers.pdf_parser.KREUZBERG_AVAILABLE", True, create=True), patch(
                "parsers.pdf_parser.ExtractionConfig",
                return_value=SimpleNamespace(),
                create=True,
            ), patch(
                "parsers.pdf_parser.extract_file",
                new=AsyncMock(side_effect=RuntimeError("kreuzberg fail")),
                create=True,
            ), patch("parsers.pdf_parser.SimpleDirectoryReader", return_value=mocked_reader_instance):
                parser = AdvancedPDFParser()
                docs = parser.load_documents(temp_dir)

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].get_content(), "Fallback text")
            self.assertEqual(docs[0].metadata.get("file_name"), file_name)
        finally:
            for name in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, name))
            os.rmdir(temp_dir)


if __name__ == "__main__":
    unittest.main()
