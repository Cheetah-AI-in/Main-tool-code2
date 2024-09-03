from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
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
uploaded_file_paths = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_script_thread(uploaded_file_paths):
    global script_result
    if uploaded_file_paths:
        script_result = tool_app.main(uploaded_file_paths)
    else:
        script_result = {"error": "No file uploaded"}

@app.route('/upload', methods=['POST'])
def upload_files():
    global uploaded_file_paths
    uploaded_file_paths = []
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('files')
    for file in files:
        if file.filename == '':
            continue
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_file_paths.append(file_path)
    if not uploaded_file_paths:
        return jsonify({'error': 'No valid files uploaded'}), 400
    return jsonify({'message': 'Files uploaded successfully', 'file_paths': uploaded_file_paths}), 200

@app.route('/run-script', methods=['GET'])
def run_script():
    global uploaded_file_paths
    if not uploaded_file_paths:
        return jsonify({"error": "No file uploaded"}), 400

    thread = threading.Thread(target=run_script_thread, args=(uploaded_file_paths,))
    thread.start()
    return jsonify({"message": "Script started"}), 202

@app.route('/get-result', methods=['GET'])
def get_result():
    global script_result
    if script_result is None:
        return jsonify({"message": "Script still running or hasn't been started"}), 202
    return jsonify(script_result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
