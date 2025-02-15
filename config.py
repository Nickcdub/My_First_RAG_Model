import os

# Paths
DATA_FOLDER = "data/"
VECTOR_DB_PATH = "vector_store/chroma_db/"
#VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vector_store", "chroma_db"))
#MODEL_PATH = "models/mathstral-7B-v0.1-Q4_K_M.gguf"

# Model Parameters
MODEL_NAME = "mistral-7b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Hugging Face SentenceTransformer