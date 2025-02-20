import os
import sys
import glob
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  

# Add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_FOLDER, VECTOR_DB_PATH, EMBEDDING_MODEL

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(VECTOR_DB_PATH)
collection = chroma_client.get_or_create_collection("knowledge_base")

# Load documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_documents():
    for file_path in glob.glob(f"{DATA_FOLDER}/*.txt"):
        print(f"Processing file: {file_path}")  # Debug output
        encoding = "utf-8"
        
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as file:
                text = file.read()
            print(f" Successfully read {file_path}")
        except Exception as e:
            print(f" Error reading {file_path}: {e}")
        
        # Debugging text chunking
        chunks = text_splitter.split_text(text)
        print(f"🔹 {len(chunks)} text chunks created for {file_path}")

        # Generate embeddings
        vectors = embedding_model.embed_documents(chunks)
        print(f" Generated embeddings for {file_path}")

        # Store embeddings in ChromaDB
    for i, chunk in enumerate(chunks):
        chunk_id = f"{file_path}-{i}"

        collection.add(
            ids=[chunk_id], 
            embeddings=[vectors[i]], 
            documents=[chunk],  #  Ensure document text is stored
            metadatas=[{"source": file_path}]
        )

        print(f" Stored Chunk {i}: {chunk[:100]}")  # Print first 100 chars

if __name__ == "__main__":
    process_documents()
    print(" All documents processed successfully!")

