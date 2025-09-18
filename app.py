import os
import uuid
import shutil
from flask import Flask, request, render_template, send_from_directory, url_for, session
from werkzeug.utils import secure_filename
import whisper

# Wir importieren Ihre Verarbeitungslogik als Modul
import main as audio_processor

# --- Flask App Konfiguration ---
class Config:
    SECRET_KEY = os.urandom(24)
    UPLOAD_FOLDER = 'user_uploads'
    ALLOWED_EXTENSIONS = audio_processor.SUPPORTED_FORMATS

app = Flask(__name__)
app.config.from_object(Config)

# Laden des Whisper-Modells (nur einmal beim Start)
# WICHTIG: Das Herunterladen kann beim ersten Start dauern!
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper model loaded.")

def allowed_file(filename):
    return '.' in filename and \
           os.path.splitext(filename).lower() in app.config['ALLOWED_EXTENSIONS']
audio_processor.py
# ==============================================================================
# ROUTE 1: Die Startseite mit dem Upload-Formular
# ==============================================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Eindeutige Session-ID erstellen, um Uploads zu gruppieren
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id # Speichern in der User-Session
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or uploaded_files.filename == '':
            return render_template('index.html', error="No files selected.")

        processing_results = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                original_filename = secure_filename(file.filename)
                saved_path = os.path.join(session_folder, original_filename)
                file.save(saved_path)

                # Aufruf Ihrer Verarbeitungsfunktion aus 
                output_dir_for_session = os.path.join('results', session_id)
                
                result = audio_processor.process_audio_file(
                    saved_path, 
                    output_dir_for_session, 
                    model
                )
                result['original_filename'] = original_filename
                processing_results.append(result)

        # ZIP-Archiv für diese spezifische Session erstellen
        final_zip_path = audio_processor.create_zip_archive_of_tts_dataset(output_dir_for_session)
        final_zip_filename = os.path.basename(final_zip_path) if final_zip_path else None
        
        return render_template('results.html', 
                               files=processing_results, 
                               final_zip=final_zip_filename, 
                               session_id=session_id)

    return render_template('index.html')

# ==============================================================================
# ROUTE 2: Die Download-Route, jetzt korrekt implementiert
# ==============================================================================
@app.route('/downloads/<session_id>/<filename>')
def download_file(session_id, filename):
    # Der Pfad zum Download ist jetzt sicher und sessions-spezifisch
    directory = os.path.join(os.getcwd(), 'results')
    download_directory = os.path.join(directory, session_id)
    
    # Sicherheitscheck
    safe_path = os.path.abspath(download_directory)
    if not safe_path.startswith(os.path.abspath(directory)):
        return "Access Denied", 403

    return send_from_directory(directory=safe_path, path=filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)