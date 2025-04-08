from transformers import AutoTokenizer, AutoModel
import torch
from flask import Flask, request, jsonify

# Load the tokenizer and model (example: MiniLM for faster inference)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

app = Flask(__name__)

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    try:
        data = request.get_json()
        text = data.get("text", "")

        # Tokenize and encode the text with truncation
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512,  # Ensure the input is no longer than 512 tokens
            truncation=True
        )
        
        # Generate the embedding
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)

        # Convert to list and return the response
        embedding_vector = embeddings[0].tolist()
        return jsonify({"embedding": embedding_vector})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

