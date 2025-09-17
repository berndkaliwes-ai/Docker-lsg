import os
import shutil
import csv
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence

SUPPORTED_FORMATS = [".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"]

def convert_to_wav(file_path):
    """
    Konvertiert eine Audiodatei in das .wav-Format, wenn sie in den unterstützten Formaten vorliegt.
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
        return wav_path
    except Exception as e:
        print(f"Fehler bei der Konvertierung von '{file_path}': {e}")
        return None

def analyze_quality(file_path):
    """
    Analysiert die Qualität einer Audiodatei.
    """
    quality_metrics = {}
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # Clipping detection
        clipped_samples = np.sum(np.abs(y) >= 0.99)
        quality_metrics['clipping_percentage'] = (clipped_samples / len(y)) * 100 if len(y) > 0 else 0

        # SNR estimation
        non_silent_intervals = librosa.effects.split(y, top_db=30)
        if len(non_silent_intervals) > 0:
            signal_parts = np.concatenate([y[start:end] for start, end in non_silent_intervals])
            
            silent_mask = np.ones(len(y), dtype=bool)
            for start, end in non_silent_intervals:
                silent_mask[start:end] = False
            noise_parts = y[silent_mask]

            if len(signal_parts) > 0 and len(noise_parts) > 0:
                power_signal = np.mean(signal_parts**2)
                power_noise = np.mean(noise_parts**2)
                if power_noise > 0:
                    quality_metrics['snr_db'] = 10 * np.log10(power_signal / power_noise)
        
        # Dynamic range
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        quality_metrics['dynamic_range_db'] = np.max(S_db) - np.min(S_db)
        
        # Spectral centroid
        quality_metrics['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    except Exception as e:
        print(f"Fehler bei der Qualitätsanalyse von '{file_path}': {e}")
    return quality_metrics

def segment_audio(file_path, original_filename, base_output_dir):
    """
    Segmentiert eine .wav-Audiodatei anhand von Stillen und speichert die Segmente.
    """
    if not file_path or not file_path.lower().endswith(".wav"):
        return [], None

    file_name_no_ext = os.path.splitext(original_filename)[0]
    output_dir = os.path.join(base_output_dir, file_name_no_ext)
    
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
            "original_filename": original_filename,
            "segment_number": segment_number,
            "audio_file": segment_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration": len(segment) / 1000.0,
            "transcript": "",
            "error": "",
        })
    return segment_data, output_dir

def transcribe_audio(segment_data):
    """
    Transkribiert die audio_file-Einträge in den Segmentdaten.
    """
    model = whisper.load_model("base")
    for segment in segment_data:
        if os.path.exists(segment["audio_file"]):
            try:
                result = model.transcribe(segment["audio_file"])
                segment["transcript"] = result["text"]
            except Exception as e:
                segment["error"] += f"Transcription failed: {e}; "
        else:
            segment["error"] += "Audio file not found; "
    return segment_data

def save_to_csv(segment_data, output_dir):
    """
    Speichert die Segmentdaten in einer CSV-Datei.
    """
    if not segment_data:
        return None
    
    csv_file_path = os.path.join(output_dir, "transcript.csv")
    
    fieldnames = [
        "original_filename", "segment_number", "audio_file", "transcript", 
        "start_time", "end_time", "duration", "error", 
        "clipping_percentage", "snr_db", "dynamic_range_db", "spectral_centroid"
    ]
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for segment in segment_data:
            # Ensure all keys are present before writing
            row = {key: segment.get(key, "") for key in fieldnames}
            writer.writerow(row)
            
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

def process_audio_file(file_path, base_output_dir):
    """
    Führt den gesamten Verarbeitungsprozess für eine einzelne Audiodatei aus.
    """
    original_filename = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(original_filename)[0]
    output_dir = os.path.join(base_output_dir, file_name_no_ext)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    converted_file = convert_to_wav(file_path)
    if not converted_file:
        error_data = [{
            "original_filename": original_filename, "error": "File conversion failed.",
        }]
        save_to_csv(error_data, output_dir)
        return create_zip_archive(output_dir)

    quality_metrics = analyze_quality(converted_file)
    
    error_messages = []
    if quality_metrics.get('snr_db', 100) < 20:
        error_messages.append("Low SNR value")
    if quality_metrics.get('clipping_percentage', 0) > 1:
        error_messages.append("Clipping detected")

    if error_messages:
        error_data = [{
            "original_filename": original_filename,
            "error": "; ".join(error_messages),
            **quality_metrics
        }]
        save_to_csv(error_data, output_dir)
        return create_zip_archive(output_dir)

    segments, segment_output_dir = segment_audio(converted_file, original_filename, base_output_dir)
    
    if not segments:
        error_data = [{
            "original_filename": original_filename,
            "error": "No segments found.",
            **quality_metrics
        }]
        save_to_csv(error_data, segment_output_dir or output_dir)
        return create_zip_archive(segment_output_dir or output_dir)

    for seg in segments:
        seg.update(quality_metrics)

    transcribed_segments = transcribe_audio(segments)
    save_to_csv(transcribed_segments, segment_output_dir)
    
    return create_zip_archive(segment_output_dir)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_to_process = sys.argv[1]
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        process_audio_file(file_to_process, results_dir)
    else:
        print("Bitte geben Sie den Pfad zur Audiodatei an.")
