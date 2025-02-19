# My_First_RAG_Model'

# Retrieval-Augmented Generation (RAG) Model

## Overview
This project explores the implementation of a **Retrieval-Augmented Generation (RAG) Model** by integrating **AI models with vector-based retrieval**. The goal is to enhance language model responses with relevant retrieved information from a structured knowledge base.

By working with test data first, this experiment will provide insight into:
- Loading and embedding textual data for AI processing.
- Storing and retrieving information using **vector databases (ChromaDB)**.
- Using **retrieved information** to enhance AI-generated responses.

---

## Objective
This project aims to:
- Build a functional **RAG system** to augment LLM responses with retrieved data.
- Load and preprocess documents by **splitting, embedding, and storing** them in **ChromaDB**.
- Query the stored knowledge to **retrieve relevant information** before responding.

---

## How To Use

### 1. Load Documents
Before use, **delete any existing ChromaDB vector storage** to prevent conflicts:

'$ rm -rf vector_store/chroma_db'

This ensures:

- No dimension mismatch errors (if a different embedding model is used).
- No duplicate embeddings (if the same data is loaded multiple times).
- Once cleared, reload the documents:

'$ python3 scripts/load_documents.py'

This will:

- Process text files from the data/ directory.
- Split documents into chunks.
- Generate embeddings using the configured model.
- Store the embeddings in ChromaDB for retrieval.

### 2. Activate LM Studio
Once the embedding model is loaded with data, start LM Studio to serve the LLM:

1. Open LM Studio.
2. Select the local model to use (e.g., mistral-7b).
3. Click "Run API Server" to enable API mode.
4. Ensure the server is running on port 1234 (or update the script if needed).
3. Start the Chatbot

### 3. Run the chatbot application:

'$ python3 app.py'

This initializes the system, allowing it to query ChromaDB for relevant information before generating responses.

### 4. Query the Chatbot
Once running, ask questions that relate to the stored dataset:

You: Can you tell me about ID-1002?
Chatbot: The XG-250 Quantum Processor was released by QuantumCorp in 2023. It features a 5nm architecture and offers a 25% increase in processing speed compared to its predecessor.
If the chatbot responds with "I don't know", it may indicate:

The embedding model is not retrieving relevant data.
The query does not closely match stored embeddings.
The data was not loaded correctly.