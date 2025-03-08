import chromadb
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, List, Tuple
from pathlib import Path

# Add the root project directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import VECTOR_DB_PATH, EMBEDDING_MODEL

# Initialize ChromaDB ================================================================================================================

chroma_client = chromadb.PersistentClient(VECTOR_DB_PATH)

collection = chroma_client.get_collection("knowledge_base")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Constants
N_RESULTS = 5  # Increased from 3 to 5 results

# Use the user prompt to query the vector DB for relevant chunks =================================================================================================================

def retrieve_relevant_chunks(query: str) -> Tuple[List[str], List[Dict]]:
    """
    Retrieve relevant chunks and their metadata for a given query.
    Returns a tuple of (documents, metadata).
    """
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,  # Using the constant
        include=["documents", "metadatas"]
    )

    if not results or not results["metadatas"] or not results["documents"]:
        return (["No relevant data found."], [{"source": "None", "page_number": "N/A"}])

    # Filter out None values
    documents = [doc for doc in results["documents"][0] if doc is not None]
    
    # Process metadata to ensure page numbers are preserved
    metadata = []
    for meta in results["metadatas"][0]:
        if meta is not None:
            source_file = meta["source"]
            
            # Extract page number directly from metadata
            # Use actual page number if available
            if "page_number" in meta:
                page_number = meta["page_number"]
            else:
                page_number = "Unknown"
                
            metadata.append({
                "source": source_file,
                "page_number": page_number
            })

    return documents, metadata

# Calling this file will use relevant chunks to =================================================================================================================

#This will never be accessed, only for debugging purposes.
if __name__ == "__main__":
    query = input("Enter your question: ")
    docs, meta = retrieve_relevant_chunks(query)
    print("\nRetrieved {} chunks and their sources:".format(len(docs)))
    for doc, m in zip(docs, meta):
        print("\nSource: {}, Page: {}".format(os.path.basename(m['source']), m['page_number']))
        print("Preview: \"{}...\"".format(doc[:100]))
