import os

# Paths
DATA_FOLDER = "data/"
VECTOR_DB_PATH = "vector_store/chroma_db/"

# Model Parameters
MODEL_NAME = "mistral-7b"

# LM Studio Configuration
LM_STUDIO_URL = "http://localhost:1234/v1/completions"  # LM Studio API endpoint

#Embedding Models===================================================================================

#EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Hugging Face SentenceTransformer 384 Dimensions
#MiniLM Better for sentence identifiers, i.e. Tell me about the XG-250 Quantum Processor

EMBEDDING_MODEL = "BAAI/bge-m3"


