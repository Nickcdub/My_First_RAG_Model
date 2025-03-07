import chromadb
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, List, Tuple

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
        return (["No relevant data found."], [{"source": "None"}])

    # Filter out None values and pair documents with their metadata
    documents = [doc for doc in results["documents"][0] if doc is not None]
    metadata = [{"source": meta["source"]} for meta in results["metadatas"][0] if meta is not None]

    return documents, metadata

# Calling this file will use relevant chunks to =================================================================================================================

#This will never be accessed, only for debugging purposes.
if __name__ == "__main__":
    query = input("Enter your question: ")
    docs, meta = retrieve_relevant_chunks(query)
    print(f"\nRetrieved {len(docs)} chunks and their sources:")
    for doc, m in zip(docs, meta):
        print(f"\nSource: {os.path.basename(m['source'])}")
        print(f"Context: \"{doc}\"")
