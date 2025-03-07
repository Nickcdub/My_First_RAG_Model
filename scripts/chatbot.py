import requests
import sys
import os
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from query_rag import retrieve_relevant_chunks
from config import LM_STUDIO_URL

# Constants
MAX_TOKENS = 800  # Increased from 300 to allow for longer responses
TEMPERATURE = 0.2  # Keep temperature low for factual responses

def format_sources(chunks: List[str], metadata: List[Dict]) -> str:
    """Format source references with their context chunks."""
    sources = []
    for chunk, meta in zip(chunks, metadata):
        source = os.path.basename(meta['source'])
        sources.append(f"\n[Source: {source}]\nContext used: \"{chunk}\"\n")
    return "\n".join(sources)

def generate_chat_response(user_input: str) -> str:
    # Retrieve relevant knowledge and metadata from vector DB
    relevant_chunks, chunk_metadata = retrieve_relevant_chunks(user_input)

    if "No relevant data found." in relevant_chunks:
        return "⚠️ No relevant data found in the database."

    # Format the context with source references
    context_with_sources = []
    for chunk, meta in zip(relevant_chunks, chunk_metadata):
        source = os.path.basename(meta['source'])
        context_with_sources.append(f"[From {source}]: {chunk}")
    
    context = "\n\n".join(context_with_sources)

    # Properly format the prompt to include retrieved context
    prompt = f"""You are an AI assistant that strictly answers based on the provided information. 
If the information is not relevant to the question, respond with "I don't know."

Context with sources:
{context}

Question: {user_input}

Instructions:
1. Answer the question based on the provided context
2. Synthesize information from multiple chunks when relevant
3. After your answer, list the sources you used
4. If different sources provide conflicting information, mention this explicitly
5. If the context contains technical details, include them in your response

Answer:"""

    payload = {
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload).json()

        if "error" in response:
            return f"API Error: {response['error']}"

        if "choices" not in response:
            return "Error: No 'choices' key found in API response."

        answer = response["choices"][0]["text"].strip()
        
        # Add source reference footer if not already included by the model
        if not any(f"Source:" in line for line in answer.split('\n')):
            sources_footer = "\n\nSources and Context Used:\n" + format_sources(relevant_chunks, chunk_metadata)
            answer += sources_footer

        return answer

    except Exception as e:
        return f"Error generating response: {str(e)}"

#This will never be accessed, only for future debugging purposes.
if __name__ == "__main__":
    print("RAG Chatbot Running! Type 'exit' to stop.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        response = generate_chat_response(user_input)
        print("\nChatbot:", response)
