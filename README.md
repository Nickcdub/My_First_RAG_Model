# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on your documents.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your documents:
- Put your PDF or text files in the `data/` folder

3. Process documents:
```bash
python scripts/load_documents.py
```

> **Note:** If you've previously processed documents and want accurate page numbers in citations, 
> you should clear your vector store and reprocess your documents:
> ```bash
> rm -rf vector_store/*
> python scripts/load_documents.py
> ```

## Running the Application

1. Start LM Studio:
- Open LM Studio
- Load the Mistral-7B model
- Click "Start Server"
- Ensure it's running on http://localhost:1234

2. Launch the application:
```bash
python app.py
```

The application will:
1. Check LM Studio connection
2. Start the web interface
3. Open the chat interface at http://localhost:8501

## Usage

1. Type your questions in the chat input at the bottom
2. The bot will search through your documents and provide relevant answers
3. Each answer includes citations from the source documents with page numbers
4. Use the "Clear Chat" button to start a fresh conversation

## Troubleshooting

If you encounter issues:

1. Check LM Studio is running and accessible at http://localhost:1234
2. Ensure your documents are properly loaded in the `data/` folder
3. Check the vector store was created successfully in `vector_store/`
4. Make sure all dependencies are installed correctly
5. If page numbers are incorrect, clear the vector store and reprocess documents

## Directory Structure

```
.
├── app.py               # Main application entry point
├── data/               # Your source documents
├── vector_store/      # Embedded document storage
├── frontend/         # Web interface
│   ├── server.py    # Flask server implementation
│   └── templates/   # HTML templates
│       └── index.html
├── scripts/          # Core RAG implementation
│   ├── load_documents.py
│   ├── query_rag.py
│   └── chatbot.py
└── config.py        # Configuration settings
```