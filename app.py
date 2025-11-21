from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Print environment variables for debugging
print("=== Azure OpenAI Configuration ===")
print(f"Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print(f"Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
print(f"API Key present: {'Yes' if os.getenv('AZURE_OPENAI_API_KEY') else 'No'}")
print("==================================")

# Enhanced RAG system prompt for better formatting
SYSTEM_PROMPT = """You are an AI assistant that helps people find movies and show recommendations on Netflix.

IMPORTANT FORMATTING RULES:
- Use bullet points with emojis for lists
- Use clear section headings
- Keep responses well-structured and easy to read
- Use line breaks between sections
- Bold important titles or key points
- Use a friendly, engaging tone

RESPONSE STRUCTURE:
🎬 **Movie/TV Show Recommendations:**
• Title 1 (Year) - Brief description
• Title 2 (Year) - Brief description

📝 **Why you might like these:**
• Reason 1
• Reason 2

💡 **Pro Tip:** Helpful additional information

Use only the tv shows and movies that are given in the prompt. Give the user a concise and helpful response.
If you don't know about a movie or it's not available on Netflix, be honest about it.

Keep responses friendly and helpful! Use Netflix-style recommendations with engaging formatting."""

def call_azure_openai(user_message):
    """Call Azure OpenAI using direct HTTP requests"""
    try:
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION')}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY")
        }
        
        data = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        print(f"🌐 Calling Azure OpenAI endpoint...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP Request error: {e}")
        raise Exception(f"Network error: {e}")
    except KeyError as e:
        print(f"❌ Response parsing error: {e}")
        raise Exception("Invalid response format from Azure OpenAI")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise e

# Test the connection on startup
try:
    test_response = call_azure_openai("What are 2 popular movies on Netflix?")
    print(f"✅ Azure OpenAI connection test successful! Response: {test_response}")
    azure_connected = True
except Exception as e:
    print(f"❌ Azure OpenAI connection test failed: {e}")
    azure_connected = False

@app.route('/')
def home():
    return jsonify({
        'message': 'Movie Recommendation Backend is running!',
        'status': 'success',
        'azure_openai': 'Connected' if azure_connected else 'Not connected'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not azure_connected:
            return jsonify({
                'reply': "Azure OpenAI is not connected. Please check your configuration.",
                'status': 'error'
            }), 500

        user_message = request.json.get('message', '')
        
        print(f"📨 User message: {user_message}")
        
        # Call Azure OpenAI using direct HTTP
        bot_reply = call_azure_openai(user_message)
        
        print(f"🤖 Bot response: {bot_reply}")
        
        return jsonify({
            'reply': bot_reply,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error calling Azure OpenAI: {e}")
        return jsonify({
            'reply': f"I'm having trouble connecting to the movie database: {str(e)}",
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'Backend is running!',
        'azure_openai': 'Connected' if azure_connected else 'Not connected',
        'deployment': os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
    })

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test endpoint to check Azure OpenAI connection"""
    try:
        test_response = call_azure_openai("What are 2 popular movies on Netflix?")
        return jsonify({
            'success': True,
            'response': test_response,
            'message': 'Azure OpenAI is working!'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server with Azure OpenAI (HTTP Direct)...")
    print("📊 Check health at: http://127.0.0.1:5000/api/health")
    print("🧪 Test connection at: http://127.0.0.1:5000/api/test")
    app.run(debug=True, port=5000)