import requests
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from query_rag import retrieve_relevant_chunks
from config import MODEL_NAME

LM_STUDIO_URL = "http://localhost:1234/v1/completions"  # LM Studio API

def generate_chat_response(user_input):
    # Retrieve relevant knowledge from vector DB
    relevant_context = retrieve_relevant_chunks(user_input)

    if not relevant_context or "⚠️ No relevant data found." in relevant_context:
        return "⚠️ No relevant data found in the database."

    # Properly format the prompt to include retrieved context
    prompt = f"""You are an AI assistant that strictly answers based on the provided information. 
If the information is not relevant to the question, respond with "I don't know."

Context:
{relevant_context}

Question: {user_input}

Answer:
"""

    payload = {
        "prompt": prompt,  # Includes retrieved context
        "max_tokens": 200,
        "temperature": 0.2  # Reduce hallucination
    }

    response = requests.post(LM_STUDIO_URL, json=payload).json()

    # Debugging output
    #print("Debugging API Response:", response)

    if "error" in response:
        return f"API Error: {response['error']}"

    if "choices" not in response:
        return "Error: No 'choices' key found in API response."

    return response["choices"][0]["text"]

if __name__ == "__main__":
    while True:
        user_input = input(" You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_chat_response(user_input)
        print(" Chatbot:", response)
