from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
import tool_app  # Import your main script
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Configuration
UPLOAD_FOLDER = './Documents'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
script_result = None
uploaded_file_path = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_file_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_file_path)
        return jsonify({'message': 'File uploaded successfully', 'file_path': uploaded_file_path}), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/run-script', methods=['GET'])
async def run_script():
    global uploaded_file_path, script_result
    if not uploaded_file_path:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        script_result = await asyncio.create_task(tool_app.main(uploaded_file_path))
        return jsonify({"message": "Script started"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-result', methods=['GET'])
async def get_result():
    global script_result
    if script_result is None:
        return jsonify({"message": "Script still running or hasn't been started"}), 202
    elif "error" in script_result:
        return jsonify(script_result), 400
    else:
        return jsonify(script_result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
