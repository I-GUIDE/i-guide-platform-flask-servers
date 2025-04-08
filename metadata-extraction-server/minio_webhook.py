from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    event = request.json
    print("Received event:", event)
    
    # Trigger metadata extraction script
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    subprocess.run(["python3", "extract_metadata_code_notebooks.py", bucket, key])
    
    return jsonify({"message": "Event received"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)