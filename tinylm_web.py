"""
tinylm_web.py - Simple web interface for TinyLM
"""

from flask import Flask, render_template_string, request, jsonify
from tinylm import TinyLMManager

app = Flask(__name__)
manager = TinyLMManager()
inference = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>TinyLM Web Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .chat-box {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            height: 300px;
            overflow-y: auto;
            margin-bottom: 20px;
            background: #f9f9f9;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background: #e3f2fd;
            text-align: right;
        }
        .bot-message {
            background: #f3e5f5;
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #5a67d8;
        }
        .status {
            text-align: center;
            color: #666;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤏 TinyLM Chat</h1>
        <div class="chat-box" id="chatBox"></div>
        <div class="input-group">
            <input type="text" id="userInput" placeholder="Type your message..." 
                   onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
        <div class="status" id="status">Ready</div>
    </div>
    
    <script>
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Update status
            document.getElementById('status').textContent = 'Thinking...';
            
            // Send to server
            fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: message})
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.response, 'bot');
                document.getElementById('status').textContent = 'Ready';
            })
            .catch(error => {
                addMessage('Error: ' + error, 'bot');
                document.getElementById('status').textContent = 'Error';
            });
        }
        
        function addMessage(text, sender) {
            const chatBox = document.getElementById('chatBox');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + 
                (sender === 'user' ? 'user-message' : 'bot-message');
            messageDiv.textContent = (sender === 'user' ? 'You: ' : 'TinyLM: ') + text;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        // Initial message
        addMessage('Hello! I am TinyLM. How can I help you today?', 'bot');
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    global inference
    
    if inference is None:
        return jsonify({'response': 'Model not loaded. Please train first.'}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    
    try:
        response = inference.generate_text(prompt, max_length=30)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

def start_web_server():
    global inference
    
    print("Loading model...")
    if manager.load_model('tinylm_trained'):
        inference = manager.get_inference_engine()
        print("✅ Model loaded!")
    else:
        print("⚠️ No model found. Training new model...")
        manager.create_model()
        manager.train_model(1000)
        inference = manager.get_inference_engine()
        manager.save_model('tinylm_trained')
    
    print("Starting web server at http://localhost:5000")
    app.run(debug=False, port=5000)

if __name__ == '__main__':
    start_web_server()
