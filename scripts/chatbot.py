import requests
import sys
import os
from typing import List, Dict

# Import from the same directory
from scripts.query_rag import retrieve_relevant_chunks
from config import LM_STUDIO_URL

# Constants
MAX_TOKENS = 800  # Increased from 300 to allow for longer responses
TEMPERATURE = 0.2  # Keep temperature low for factual responses

def format_sources(metadata: List[Dict]) -> str:
    """Format source references with page numbers only."""
    sources = []
    for meta in metadata:
        source = os.path.basename(meta['source'])
        page = meta.get('page_number', 'Unknown')
        sources.append(f"[Source: {source}, Page: {page}]")
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
        page = meta.get('page_number', 'Unknown')
        context_with_sources.append(f"[From {source}, Page {page}]: {chunk}")
    
    context = "\n\n".join(context_with_sources)

    # Modify the prompt to discourage the model from adding its own sources
    prompt = f"""You are an AI assistant that strictly answers based on the provided information. 
If the information is not relevant to the question, respond with "I don't know."

Context with sources:
{context}

Question: {user_input}

Instructions:
1. Answer the question based on the provided context
2. Organize your response into logical paragraphs for readability
3. DO NOT list the sources at the end of your response - this will be handled automatically
4. If different sources provide conflicting information, mention this explicitly
5. Keep your response concise and well-structured

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
        
        # Check if the response already contains a sources section
        sources_keywords = ["Sources:", "Source:", "References:", "Reference:"]
        has_sources_section = False
        
        for keyword in sources_keywords:
            if keyword in answer:
                # Remove any existing sources section to avoid duplication
                answer_parts = answer.split(keyword, 1)
                answer = answer_parts[0].strip()
                has_sources_section = True
                break
        
        # Always add our standardized sources section
        sources_footer = "\n\nSources:\n" + format_sources(chunk_metadata)
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
