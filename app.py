import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import main

UPLOAD_FOLDER = 'uploads'
# RESULTS_FOLDER is now managed by main.py's TTS_DATASET_DIR

ALLOWED_EXTENSIONS = {'opus', 'mp3', 'wav', 'm4a', 'aac', 'flac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = main.TTS_DATASET_DIR # Use the TTS_DATASET_DIR from main.py

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file[]' not in request.files:
            return redirect(request.url)
        
        files = request.files.getlist('file[]')
        processing_mode = request.form.get('processing_mode', 'transcription')
        segmentation_mode = request.form.get('segmentation_mode', 'silence')

        if not files or files[0].filename == '':
            return redirect(request.url)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        processed_files_info = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                result = main.process_audio_file(
                    upload_path, 
                    app.config['RESULTS_FOLDER'], 
                    processing_mode, 
                    segmentation_mode
                )
                
                if result["status"] == "success":
                    processed_files_info.append({"name": filename, "status": "Processed"})
                else:
                    processed_files_info.append({"name": filename, "status": f"Failed: {result['message']}"})

        final_zip_path = main.create_zip_archive_of_tts_dataset(app.config['RESULTS_FOLDER'])
        
        return render_template('results.html', files=processed_files_info, final_zip=os.path.basename(final_zip_path) if final_zip_path else None)

    return render_template('index.html')

@app.route('/results/<filename>')
def download_file(filename):
    # This route now serves the single TTS dataset zip file
    return send_from_directory(os.path.dirname(app.config['RESULTS_FOLDER']), filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)