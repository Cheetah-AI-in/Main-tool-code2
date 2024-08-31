from flask import Flask, jsonify, request
import threading
import web_app  # Import your main script
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

def run_script_thread():
    global script_result, uploaded_file_path
    if uploaded_file_path:
        script_result = web_app.main(uploaded_file_path)
    else:
        script_result = {"error": "No file uploaded"}

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
def run_script():
    global script_result
    script_result = None  # Reset the result
    thread = threading.Thread(target=run_script_thread)
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
