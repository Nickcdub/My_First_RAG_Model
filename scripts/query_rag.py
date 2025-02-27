import chromadb
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import VECTOR_DB_PATH, EMBEDDING_MODEL

# Initialize ChromaDB ================================================================================================================

chroma_client = chromadb.PersistentClient(VECTOR_DB_PATH)

collection = chroma_client.get_collection("knowledge_base")

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Use the user prompt to query the vector DB for relevant chunks =================================================================================================================

def retrieve_relevant_chunks(query):
    query_embedding = embedding_model.embed_query(query)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results or not results["metadatas"] or not results["documents"]:
        print(" No relevant data found in the database!")
        return [" No relevant data found."]

    #print(f" Retrieved Chunks: {results}") #For debugging, check if retriebed chunks returns None

    return [doc for doc in results["documents"][0] if doc is not None]

# Calling this file will use relevant chunks to =================================================================================================================

#This will never be accessed, only for debugging purposes.
if __name__ == "__main__":
    query = input("Enter your question: ")
    relevant_docs = retrieve_relevant_chunks(query)
    print(" Retrieved Documents:", relevant_docs)
