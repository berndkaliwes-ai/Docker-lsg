
import os
import shutil
import csv
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.generators import Sine

SUPPORTED_FORMATS = [".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"]
RESULTS_DIR = "/app/results"

def convert_to_wav(file_path):
    """
    Konvertiert eine Audiodatei in das .wav-Format, wenn sie in den unterstützten Formaten vorliegt.
    Löscht die Originaldatei nach erfolgreicher Konvertierung.
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext.lower() not in SUPPORTED_FORMATS:
        print(f"Fehler: Dateiformat '{file_ext}' wird nicht unterstützt.")
        return None
    if file_ext.lower() == ".wav":
        return file_path
    try:
        audio = AudioSegment.from_file(file_path, format=file_ext[1:])
        wav_path = f"{file_name}.wav"
        audio.export(wav_path, format="wav")
        # In a docker env, the original file is the one we want to process
        if not os.getenv('DOCKER_ENV') == 'true':
            os.remove(file_path)
        return wav_path
    except Exception as e:
        print(f"Fehler bei der Konvertierung von '{file_path}': {e}")
        return None

def segment_audio(file_path, base_output_dir):
    """
    Segmentiert eine .wav-Audiodatei anhand von Stillen und speichert die Segmente.
    """
    if not file_path or not file_path.lower().endswith(".wav"):
        return [], None

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(base_output_dir, f"{file_name}_segments")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    audio = AudioSegment.from_wav(file_path)
    segments = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=250)

    segment_data = []
    total_duration = 0
    for i, segment in enumerate(segments):
        segment_number = i + 1
        segment_file_name = f"segment_{segment_number}.wav"
        segment_path = os.path.join(output_dir, segment_file_name)
        segment.export(segment_path, format="wav")

        start_time = total_duration / 1000.0
        end_time = (total_duration + len(segment)) / 1000.0
        total_duration += len(segment)

        segment_data.append({
            "original_filename": os.path.basename(file_path),
            "segment_number": segment_number,
            "audio_file": segment_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": len(segment) / 1000.0,
            "transcript": "",
            "error": ""
        })
    return segment_data, output_dir

def transcribe_audio(segment_data):
    """
    Transkribiert die audio_file-Einträge in den Segmentdaten.
    """
    model = whisper.load_model("base")
    for segment in segment_data:
        result = model.transcribe(segment["audio_file"])
        segment["transcript"] = result["text"]
    return segment_data

def save_to_csv(segment_data, output_dir):
    """
    Speichert die Segmentdaten in einer CSV-Datei.
    """
    if not segment_data:
        return None
    
    csv_file_path = os.path.join(output_dir, "transcript.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["original_filename", "segment_number", "audio_file", "transcript", "start_time", "end_time", "duration", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segment_data)
    return csv_file_path

def create_zip_archive(output_dir):
    """
    Erstellt ein ZIP-Archiv aus dem angegebenen Verzeichnis.
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    zip_file_path = f"{output_dir}.zip"
    shutil.make_archive(os.path.splitext(zip_file_path)[0], 'zip', output_dir)
    return zip_file_path

if __name__ == '__main__':
    is_docker = os.getenv('DOCKER_ENV') == 'true'
    base_output_dir = "."
    if is_docker and os.path.exists(RESULTS_DIR) and os.path.isdir(RESULTS_DIR):
        base_output_dir = RESULTS_DIR

    # In docker, we expect the files to be in the /app directory
    test_file = "test.mp3"
    if is_docker:
        # We could look for any audio file in the /app directory
        # For now, we just create the test file for demonstration
        if not os.path.exists(test_file):
            tone1 = Sine(440).to_audio_segment(duration=500)
            silence1 = AudioSegment.silent(duration=1000)
            tone2 = Sine(660).to_audio_segment(duration=500)
            test_audio = tone1 + silence1 + tone2
            test_audio.export(test_file, format="mp3")
    else:
        if not os.path.exists(test_file):
            tone1 = Sine(440).to_audio_segment(duration=500)
            silence1 = AudioSegment.silent(duration=1000)
            tone2 = Sine(660).to_audio_segment(duration=500)
            test_audio = tone1 + silence1 + tone2
            test_audio.export(test_file, format="mp3")

    converted_file = convert_to_wav(test_file)

    zip_file = None
    output_dir = None
    if converted_file:
        segments, output_dir = segment_audio(converted_file, base_output_dir)
        if segments:
            transcribed_segments = transcribe_audio(segments)
            csv_file = save_to_csv(transcribed_segments, output_dir)
            if csv_file:
                print(f"\n--- Inhalt der CSV-Datei: {csv_file} ---")
                with open(csv_file, 'r', encoding='utf-8') as f:
                    print(f.read())
                print("---------------------------------")
            
            zip_file = create_zip_archive(output_dir)
            if zip_file:
                print(f"\nZIP-Archiv erstellt: {zip_file}")

    if not is_docker:
        print("\n--- Bereinigung ---")
        if os.path.exists(converted_file):
            os.remove(converted_file)
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        print("--------------------")

