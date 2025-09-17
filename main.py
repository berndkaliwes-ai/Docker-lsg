import os
import shutil
import csv
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re
import string # Import the string module

# Ensure ffmpeg is in PATH or explicitly set
# AudioSegment.converter = "/usr/bin/ffmpeg" # Not needed if ffmpeg is in PATH

SUPPORTED_FORMATS = [".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"]
TTS_DATASET_DIR = "results/tts_dataset"
WAVS_SUBDIR = "wavs"
METADATA_FILE = "metadata.txt"

def clean_transcription(text):
    """
    Reinigt den transkribierten Text für TTS-Training:
    - Entfernt Satzzeichen.
    - Schreibt Zahlen als Wörter aus (rudimentär).
    """
    # Remove punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('-', ' ') # Replace hyphens with spaces
    text = text.strip()
    
    # Basic number to word conversion (can be expanded)
    num_word_map = {
        '0': 'null', '1': 'eins', '2': 'zwei', '3': 'drei', '4': 'vier', 
        '5': 'fünf', '6': 'sechs', '7': 'sieben', '8': 'acht', '9': 'neun'
    }
    for num, word in num_word_map.items():
        text = text.replace(num, word)

    return text.lower()

def convert_to_wav(file_path, target_sr=16000):
    """
    Konvertiert eine Audiodatei in das .wav-Format und resampelt sie auf die Ziel-Samplerate.
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext.lower() not in SUPPORTED_FORMATS:
        print(f"Fehler: Dateiformat '{file_ext}' wird nicht unterstützt.")
        return None
    
    try:
        audio = AudioSegment.from_file(file_path, format=file_ext[1:])
        
        # Resample if necessary
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        
        # Ensure mono channel
        if audio.channels != 1:
            audio = audio.set_channels(1)

        wav_path = f"{file_name}.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Fehler bei der Konvertierung von '{file_path}': {e}")
        import traceback
        traceback.print_exc()
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

def segment_audio(file_path, original_filename_no_ext, base_output_dir):
    """
    Segmentiert eine .wav-Audiodatei anhand von Stillen und speichert die Segmente.
    Segmente werden in einem globalen WAVS_SUBDIR gespeichert.
    """
    if not file_path or not file_path.lower().endswith(".wav"):
        return [], None

    wavs_output_dir = os.path.join(base_output_dir, WAVS_SUBDIR)
    os.makedirs(wavs_output_dir, exist_ok=True)

    audio = AudioSegment.from_wav(file_path)
    segments = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=250)
    
    segment_data = []
    total_duration = 0
    for i, segment in enumerate(segments):
        segment_number = i + 1
        # Create a unique filename for each segment across all original files
        segment_file_name = f"{original_filename_no_ext}_segment_{segment_number:04d}.wav"
        segment_path = os.path.join(wavs_output_dir, segment_file_name)
        segment.export(segment_path, format="wav")
        
        start_time = total_duration / 1000.0
        end_time = (total_duration + len(segment)) / 1000.0
        total_duration += len(segment)
        
        segment_data.append({
            "original_filename": os.path.basename(file_path), # This is the converted WAV path
            "segment_filename": segment_file_name, # This is the filename for metadata.txt
            "segment_number": segment_number,
            "audio_file_path": segment_path, # Full path to the segment file
            "start_time": start_time,
            "end_time": end_time,
            "duration": len(segment) / 1000.0,
            "transcript": "",
            "error": "",
        })
    return segment_data, wavs_output_dir

def transcribe_audio(segment_data):
    """
    Transkribiert die audio_file-Einträge in den Segmentdaten.
    """
    model = whisper.load_model("base")
    for segment in segment_data:
        if os.path.exists(segment["audio_file_path"]):
            try:
                result = model.transcribe(segment["audio_file_path"])
                segment["transcript"] = result["text"]
            except Exception as e:
                segment["error"] += f"Transcription failed: {e}; "
        else:
            segment["error"] += "Audio file not found; "
    return segment_data

def save_metadata_for_coqui(segment_data, tts_dataset_base_dir):
    """
    Speichert die Segmentdaten im Coqui TTS-Format (metadata.txt).
    """
    metadata_file_path = os.path.join(tts_dataset_base_dir, METADATA_FILE)
    
    with open(metadata_file_path, 'a', newline='', encoding='utf-8') as f:
        for segment in segment_data:
            cleaned_transcript = clean_transcription(segment["transcript"])
            line = f"{segment['segment_filename']}|{cleaned_transcript}\n"
            f.write(line)
    return metadata_file_path

def process_audio_file(file_path, base_output_dir):
    """
    Führt den gesamten Verarbeitungsprozess für eine einzelne Audiodatei aus.
    Gibt ein Dictionary mit dem Verarbeitungsstatus zurück.
    """
    original_filename = os.path.basename(file_path)
    original_filename_no_ext = os.path.splitext(original_filename)[0]
    
    # Ensure the base TTS dataset directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    converted_file = convert_to_wav(file_path)
    if not converted_file:
        print(f"Skipping processing for {original_filename} due to conversion failure.")
        return {"status": "error", "message": "File conversion to WAV failed"}

    quality_metrics = analyze_quality(converted_file)
    
    error_messages = []
    if quality_metrics.get('snr_db', 100) < 20:
        error_messages.append(f"Low SNR ({quality_metrics.get('snr_db', 0):.2f}dB)")
    if quality_metrics.get('clipping_percentage', 0) > 1:
        error_messages.append(f"Clipping detected ({quality_metrics.get('clipping_percentage', 0):.2f}%)")

    # If quality is too low, return an error status
    if error_messages:
        # Clean up the converted file if it was created
        if converted_file != file_path and os.path.exists(converted_file):
            os.remove(converted_file)
        print(f"Skipping processing for {original_filename} due to low quality: {error_messages}")
        return {"status": "error", "message": "; ".join(error_messages)}

    segments, wavs_output_dir = segment_audio(converted_file, original_filename_no_ext, base_output_dir)
    
    if not segments:
        if converted_file != file_path and os.path.exists(converted_file):
            os.remove(converted_file)
        print(f"No segments found for {original_filename}.")
        return {"status": "error", "message": "No voice segments detected"}

    for seg in segments:
        seg.update(quality_metrics)

    transcribed_segments = transcribe_audio(segments)
    save_metadata_for_coqui(transcribed_segments, base_output_dir)
    
    # Clean up the converted_file if it's not the original
    if converted_file != file_path and os.path.exists(converted_file):
        os.remove(converted_file)

    return {"status": "success", "path": os.path.join(base_output_dir, METADATA_FILE)}


def create_zip_archive_of_tts_dataset(tts_dataset_base_dir):
    """
    Erstellt ein ZIP-Archiv des gesamten TTS-Datasets.
    """
    if not os.path.exists(tts_dataset_base_dir):
        return None
    
    zip_file_path = f"{tts_dataset_base_dir}.zip"
    shutil.make_archive(tts_dataset_base_dir, 'zip', tts_dataset_base_dir)
    return zip_file_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_to_process = sys.argv[1]
        # For standalone execution, process into a temporary TTS dataset structure
        temp_tts_dir = "temp_tts_dataset"
        os.makedirs(temp_tts_dir, exist_ok=True)
        process_audio_file(file_to_process, temp_tts_dir)
        create_zip_archive_of_tts_dataset(temp_tts_dir)
    else:
        print("Bitte geben Sie den Pfad zur Audiodatei an.")
