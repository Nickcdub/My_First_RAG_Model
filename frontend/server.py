from flask import Flask, render_template, request, jsonify
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import our chatbot
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.chatbot import generate_chat_response

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/chat', methods=['POST'])
    def chat():
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        try:
            response = generate_chat_response(user_message)
            return jsonify({
                'response': response,
                'success': True
            })
        except Exception as e:
            return jsonify({
                'error': str(e),
                'success': False
            }), 500

    return app

def start_server(port=8501, debug=True):
    """Start the Flask server with the specified configuration"""
    app = create_app()
    print("Server starting...")
    print("Project root: {}".format(ROOT_DIR))
    app.run(debug=debug, port=port) 