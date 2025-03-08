import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the Python path and verify critical components"""
    # Add the current directory to Python path
    ROOT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(ROOT_DIR))
    
    # Verify required directories exist
    required_dirs = ['data', 'vector_store', 'frontend/templates']
    for dir_path in required_dirs:
        full_path = ROOT_DIR / dir_path
        if not full_path.exists():
            print(f"Creating directory: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)

def main():
    """Main entry point for the RAG Chatbot application"""
    # Setup environment
    setup_environment()
    
    print("Starting RAG Chatbot...")
    print("1. Checking LM Studio connection...")
    
    # Import after environment setup
    from scripts.chatbot import generate_chat_response
    from frontend.server import start_server
    
    try:
        # Test LM Studio connection with a simple query
        test_response = generate_chat_response("test")
        print("✓ LM Studio connection successful")
    except Exception as e:
        print("⚠️  Warning: Could not connect to LM Studio.")
        print("   Please make sure LM Studio is running and the server is started.")
        print("   Error: {}".format(str(e)))
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            sys.exit(1)
    
    print("\n2. Starting web interface...")
    print("   Access the chatbot at http://localhost:8501")
    
    # Start the Flask server
    start_server(port=8501, debug=True)

if __name__ == "__main__":
    main()
