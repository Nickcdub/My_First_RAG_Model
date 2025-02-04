import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))

from chatbot import generate_chat_response

print("RAG Chatbot Running! Type 'exit' to stop.")
while True:
    user_input = input(" You: ")
    if user_input.lower() == "exit":
        break
    response = generate_chat_response(user_input)
    print(" Chatbot:", response)
