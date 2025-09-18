import os
from flask import Flask, request, send_from_directory
from redis import Redis, ConnectionPool, exceptions

# Import der neuen Audio-Bibliotheken
import pydub
import librosa
import noisereduce as nr
import scipy.signal
import soundfile as sf
import numpy as np

# Konfiguration (bleibt gleich)
class Config:
    REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    REDIS_CONNECTION_POOL = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'

# ==============================================================================
# Die Audio-Verarbeitungspipeline
# ==============================================================================
def enhance_audio_for_whisper(input_path, output_path):
    """
    Wendet eine Kette von Verbesserungen auf eine Audiodatei an, 
    um sie für ein KI-Modell wie Whisper zu optimieren.
    """
    # Schritt 1 & 2: Laden und in Mono umwandeln mit pydub
    audio_segment = pydub.AudioSegment.from_file(input_path).set_channels(1)
    
    # Schritt 3: Resampling auf 16kHz
    audio_segment = audio_segment.set_frame_rate(16000)
    
    # Konvertierung zu numpy-Array für wissenschaftliche Verarbeitung
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    sample_rate = audio_segment.frame_rate

    # Schritt 4: Hochpassfilter, um Rumpeln unter 100Hz zu entfernen
    b, a = scipy.signal.butter(5, 100, btype='high', fs=sample_rate)
    filtered_samples = scipy.signal.lfilter(b, a, samples)

    # Schritt 5: Rauschunterdrückung
    reduced_noise_samples = nr.reduce_noise(y=filtered_samples, sr=sample_rate)
    
    # Schritt 6: Stille am Anfang und Ende entfernen
    trimmed_samples, _ = librosa.effects.trim(reduced_noise_samples, top_db=20)
    
    # Schritt 7: Normalisierung auf -1.0 dBFS Peak-Amplitude
    # Wir wandeln zurück zu pydub, da es eine einfache Normalisierung bietet
    normalized_segment = pydub.AudioSegment(
        trimmed_samples.astype(np.int16).tobytes(), 
        frame_rate=sample_rate,
        sample_width=2, # 16-bit
        channels=1
    ).normalize()
    
    # Speichern der finalen, optimierten Datei
    normalized_segment.export(output_path, format="wav")

# ==============================================================================
# Application Factory
# ==============================================================================
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.redis_client = Redis(connection_pool=app.config['REDIS_CONNECTION_POOL'])

    # Sicherstellen, dass die Ordner existieren
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

    @app.route('/')
    def hello():
        count = app.redis_client.incr('hits')
        return f'Hello World! Counter: {count}. Use the /process-audio endpoint to enhance an audio file.\n'

    @app.route('/process-audio', methods=['POST'])
    def process_audio_endpoint():
        if 'file' not in request.files:
            return "Fehler: Keine Datei im Request gefunden.", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Fehler: Keine Datei ausgewählt.", 400

        if file:
            input_filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_path)

            output_filename = f"processed_{os.path.splitext(input_filename)[0]}.wav"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

            try:
                enhance_audio_for_whisper(input_path, output_path)
                return send_from_directory(app.config['PROCESSED_FOLDER'], output_filename, as_attachment=True)
            except Exception as e:
                app.logger.error(f"Fehler bei der Audioverarbeitung: {e}")
                return "Fehler bei der Audioverarbeitung.", 500

    return app

# App-Start
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
```

### Schritt 3: Visualisierung der neuen Pipeline

Um diese professionelle Pipeline zu verdeutlichen, hier eine Visualisierung des Ablaufs:

```mermaid
graph TD
    A[Client sendet Audiodatei (MP3, WAV, etc.)] --> B{Flask Endpoint `/process-audio`}
    
    subgraph "Audio-Verbesserungs-Pipeline"
        direction LR
        C[1. Laden & Umwandeln in Mono] -->
        D[2. Resampling auf 16kHz] -->
        E[3. Hochpassfilter] -->
        F[4. Rauschunterdrückung] -->
        G[5. Stille entfernen] -->
        H[6. Normalisieren]
    end

    B --> C
    H --> I{Speichern als `processed_... .wav`}
    I --> J[An Client als Download zurücksenden]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#c8e6c9,stroke:#333,stroke-width:2px
```

### Zusammenfassung und nächste Schritte

Sie haben jetzt ein komplettes, professionelles Setup:

1.  **Infrastruktur:** Eine robuste Docker-Umgebung mit Flask und Redis.
2.  **State-of-the-Art-Pipeline:** Eine Kette von Audio-Verbesserungen, die Ihre Audiodateien optimal für KI-Modelle wie Whisper vorbereitet.
3.  **API-Endpunkt:** Einen funktionierenden Endpunkt, an den Sie Audiodateien per `POST`-Request senden und die optimierte Version als Antwort erhalten.