import os
import sys
import glob
import chromadb
from typing import List, Generator, Iterator, Dict, Tuple
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import gc
from pypdf import PdfReader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_FOLDER, VECTOR_DB_PATH, EMBEDDING_MODEL

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection("knowledge_base")

# Constants for batch processing
BATCH_SIZE = 16  # Reduced batch size for large documents
MAX_CHUNK_SIZE = 500  # Maximum characters per chunk
CHUNK_OVERLAP = 50   # Overlap between chunks

def stream_pdf_pages(pdf_path: str) -> Generator[Tuple[str, int], None, None]:
    """
    Stream PDF pages one at a time to conserve memory.
    Returns a generator of (text, page_number) tuples.
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    for page_num in tqdm(range(total_pages), desc="Streaming PDF pages"):
        try:
            page = reader.pages[page_num]
            text = page.extract_text()
            if text.strip():  # Only yield non-empty pages
                # Add 1 to page_num because PDF reader is 0-indexed
                yield (text, page_num + 1)
            
            # Force garbage collection after each page
            del page
            if page_num % 100 == 0:  # GC every 100 pages
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            continue

def chunk_generator(text: str, text_splitter: RecursiveCharacterTextSplitter) -> Generator[str, None, None]:
    """Generate chunks one at a time instead of all at once."""
    try:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            if chunk.strip():  # Only yield non-empty chunks
                yield chunk
    except Exception as e:
        logger.error(f"Error generating chunks: {e}")

def process_pdf_page(page_text: str, page_number: int, file_path: str, 
                     embedding_model: HuggingFaceEmbeddings, 
                     text_splitter: RecursiveCharacterTextSplitter, 
                     start_chunk_number: int = 0):
    """Process a single PDF page with correct page number tracking."""
    chunks = list(chunk_generator(page_text, text_splitter))
    
    batch = []
    batch_ids = []
    batch_metadata = []
    chunk_count = start_chunk_number
    
    for i, chunk in enumerate(chunks):
        chunk_count += 1
        file_basename = os.path.splitext(os.path.basename(file_path))[0]
        # Include page number in the chunk ID
        chunk_id = f"{file_basename}-page{page_number}-chunk{i+1}"
        
        batch.append(chunk)
        batch_ids.append(chunk_id)
        batch_metadata.append({
            "source": file_path,
            "chunk_number": chunk_count,
            "page_number": page_number,  # Use actual page number
            "chunk_index_in_page": i+1   # Track position within page
        })
        
        if len(batch) >= BATCH_SIZE:
            try:
                embeddings = embedding_model.embed_documents(batch)
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch,
                    metadatas=batch_metadata
                )
                logger.info(f"Processed {len(batch)} chunks from page {page_number}")
            except Exception as e:
                logger.error(f"Error processing batch from page {page_number}: {e}")
            
            batch = []
            batch_ids = []
            batch_metadata = []
            gc.collect()
    
    # Process any remaining chunks
    if batch:
        try:
            embeddings = embedding_model.embed_documents(batch)
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch,
                metadatas=batch_metadata
            )
        except Exception as e:
            logger.error(f"Error processing final batch from page {page_number}: {e}")
    
    return chunk_count

def process_chunks_in_batches(chunks: Iterator[str], file_path: str, embedding_model: HuggingFaceEmbeddings, start_chunk_number: int = 0):
    """Process chunks in batches to manage memory usage."""
    batch = []
    batch_ids = []
    batch_metadata = []
    chunk_count = start_chunk_number
    
    try:
        for chunk in chunks:
            chunk_count += 1
            # Create a unique ID using file name and a global counter
            chunk_id = f"{os.path.splitext(os.path.basename(file_path))[0]}-chunk-{chunk_count}"
            batch.append(chunk)
            batch_ids.append(chunk_id)
            batch_metadata.append({
                "source": file_path,
                "chunk_number": chunk_count,
                "page_number": 1  # For text files, we use page 1
            })
            
            if len(batch) >= BATCH_SIZE:
                try:
                    embeddings = embedding_model.embed_documents(batch)
                    collection.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        documents=batch,
                        metadatas=batch_metadata
                    )
                    logger.info(f"Processed batch of {len(batch)} chunks")
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                
                batch = []
                batch_ids = []
                batch_metadata = []
                gc.collect()
        
        if batch:
            try:
                embeddings = embedding_model.embed_documents(batch)
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch,
                    metadatas=batch_metadata
                )
            except Exception as e:
                logger.error(f"Error processing final batch: {e}")
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
    
    return chunk_count

def process_documents():
    """Process documents with memory-efficient streaming and batching."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Process both PDF and TXT files
    file_patterns = [f"{DATA_FOLDER}/*.pdf", f"{DATA_FOLDER}/*.txt"]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            total_chunks = 0  # Track total chunks for this file
            
            if file_extension == '.pdf':
                logger.info(f"Processing PDF: {file_path}")
                # Stream PDF pages and process each page with its actual page number
                for page_text, page_number in stream_pdf_pages(file_path):
                    total_chunks = process_pdf_page(
                        page_text, 
                        page_number,
                        file_path, 
                        embedding_model, 
                        text_splitter, 
                        total_chunks
                    )
            
            elif file_extension == '.txt':
                logger.info(f"Processing TXT: {file_path}")
                loader = TextLoader(file_path)
                documents = loader.load()
                text = "\n".join([doc.page_content for doc in documents])
                chunks = chunk_generator(text, text_splitter)
                total_chunks = process_chunks_in_batches(chunks, file_path, embedding_model, 0)
            
            logger.info(f"✓ Successfully processed {file_path} with {total_chunks} total chunks")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
        
        gc.collect()

if __name__ == "__main__":
    try:
        process_documents()
        logger.info("All documents processed successfully!")
    except Exception as e:
        logger.error(f"Fatal error in document processing: {e}")

