import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import main

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'opus', 'mp3', 'wav', 'm4a', 'aac', 'flac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file[]' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('file[]')
        if not files:
            return redirect(request.url)
        
        processed_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                file.save(upload_path)

                results_path = os.path.join(app.config['RESULTS_FOLDER'])
                if not os.path.exists(results_path):
                    os.makedirs(results_path)

                result_zip = main.process_audio_file(upload_path, results_path)
                if result_zip:
                    processed_files.append({"name": filename, "result_zip": os.path.basename(result_zip)})

        return render_template('results.html', files=processed_files)

    return render_template('index.html')

@app.route('/results/<filename>')
def download_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

