import os
import sys
import io

# Fix for windows terminal encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
from logging_config import get_logger
from parsers.router import DocumentRouter
from embeddings.chunker import AdvancedChunker

logger = get_logger(__name__)

load_dotenv()

def debug_indexing():
    router = DocumentRouter()
    chunker = AdvancedChunker()
    
    data_dir = "data"
    documents = router.process_directory(data_dir)
    logger.info(f"Total documents loaded: {len(documents)}")
    
    file_counts = {}
    for doc in documents:
        f_name = doc.metadata.get('file_name', 'Unknown')
        file_counts[f_name] = file_counts.get(f_name, 0) + 1
    
    logger.info("\nDocument counts per file:")
    for f, count in file_counts.items():
        logger.info("- %s: %d blocks", f, count)

    nodes = chunker.split_into_chunks(documents)
    logger.info(f"\nTotal nodes created: {len(nodes)}")
    
    node_file_counts = {}
    for node in nodes:
        f_name = node.metadata.get('file_name', 'Unknown')
        node_file_counts[f_name] = node_file_counts.get(f_name, 0) + 1
        
    logger.info("\nNode counts per file:")
    for f, count in node_file_counts.items():
        logger.info("- %s: %d nodes", f, count)

if __name__ == "__main__":
    debug_indexing()
