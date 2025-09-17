import os
import shutil
import csv
import whisper
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import re
import string

# --- Constants ---
SUPPORTED_FORMATS = [".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac"]
TTS_DATASET_DIR = "results/tts_dataset"
WAVS_SUBDIR = "wavs"
METADATA_FILE = "metadata.txt"
DETAILED_METADATA_FILE = "metadata_detailed.csv"

# --- Helper Functions ---

def clean_transcription(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('-', ' ')
    text = text.strip()
    num_word_map = {
        '0': 'null', '1': 'eins', '2': 'zwei', '3': 'drei', '4': 'vier',
        '5': 'fünf', '6': 'sechs', '7': 'sieben', '8': 'acht', '9': 'neun'
    }
    for num, word in num_word_map.items():
        text = text.replace(num, f' {word} ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def convert_to_wav(file_path, target_sr=16000):
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext.lower() not in SUPPORTED_FORMATS:
        return None
    try:
        audio = AudioSegment.from_file(file_path, format=file_ext[1:])
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        if audio.channels != 1:
            audio = audio.set_channels(1)
        wav_path = f"{file_name}.wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error during conversion of '{file_path}': {e}")
        return None

def analyze_quality(file_path):
    quality_metrics = {}
    try:
        y, sr = librosa.load(file_path, sr=None)
        clipped_samples = np.sum(np.abs(y) >= 0.99)
        quality_metrics['clipping_percentage'] = (clipped_samples / len(y)) * 100 if len(y) > 0 else 0
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
                quality_metrics['snr_db'] = 10 * np.log10(power_signal / power_noise) if power_noise > 0 else 100
            else:
                quality_metrics['snr_db'] = 0
        else:
            quality_metrics['snr_db'] = 0
    except Exception as e:
        print(f"Error during quality analysis of '{file_path}': {e}")
    return quality_metrics

# --- Segmentation Strategies ---

def segment_audio_by_silence(audio, original_filename_no_ext, base_output_dir):
    wavs_output_dir = os.path.join(base_output_dir, WAVS_SUBDIR)
    os.makedirs(wavs_output_dir, exist_ok=True)
    silence_segments = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=250)
    
    segment_data = []
    for i, segment_audio in enumerate(silence_segments):
        segment_file_name = f"{original_filename_no_ext}_segment_{i+1:04d}.wav"
        segment_path = os.path.join(wavs_output_dir, segment_file_name)
        segment_audio.export(segment_path, format="wav")
        segment_data.append({
            "segment_filename": segment_file_name,
            "audio_file_path": segment_path,
            "transcript": "", "error": "",
            "start_time": sum(len(s) for s in silence_segments[:i]) / 1000.0,
            "end_time": sum(len(s) for s in silence_segments[:i+1]) / 1000.0,
            "duration": len(segment_audio) / 1000.0
        })
    return segment_data

def transcribe_and_segment(audio, audio_path, original_filename_no_ext, base_output_dir, segmentation_mode):
    model = whisper.load_model("base")
    transcription_result = model.transcribe(audio_path, word_timestamps=True)
    
    # Group words into sentences
    sentences = []
    current_sentence = []
    for segment in transcription_result['segments']:
        for word in segment['words']:
            current_sentence.append(word)
            if word['word'].strip().endswith(('.', '!', '?')):
                sentences.append(current_sentence)
                current_sentence = []
    if current_sentence: # Add the last sentence if it doesn't end with punctuation
        sentences.append(current_sentence)

    # Group sentences into paragraphs if required
    if segmentation_mode == 'paragraph':
        paragraphs = []
        current_paragraph = []
        for i, sentence in enumerate(sentences):
            current_paragraph.extend(sentence)
            if i + 1 < len(sentences):
                # Check gap between current sentence end and next sentence start
                gap = sentences[i+1][0]['start'] - sentence[-1]['end']
                if gap > 1.5: # Paragraph break if gap is > 1.5 seconds
                    paragraphs.append(current_paragraph)
                    current_paragraph = []
        if current_paragraph:
            paragraphs.append(current_paragraph)
        # The final segments are the paragraphs
        final_segments = paragraphs
    else: # 'sentence' or 'paragraph'
        # The final segments are the sentences
        final_segments = sentences

    # Create segment data from the final segments (sentences or paragraphs)
    segment_data = []
    wavs_output_dir = os.path.join(base_output_dir, WAVS_SUBDIR)
    os.makedirs(wavs_output_dir, exist_ok=True)
    for i, segment_words in enumerate(final_segments):
        if not segment_words: continue
        start_time = segment_words[0]['start']
        end_time = segment_words[-1]['end']
        transcript = " ".join([word['word'] for word in segment_words]).strip()
        
        segment_audio = audio[start_time*1000:end_time*1000]
        segment_file_name = f"{original_filename_no_ext}_segment_{i+1:04d}.wav"
        segment_path = os.path.join(wavs_output_dir, segment_file_name)
        segment_audio.export(segment_path, format="wav")
        
        segment_data.append({
            "segment_filename": segment_file_name,
            "audio_file_path": segment_path,
            "transcript": transcript,
            "start_time": start_time, "end_time": end_time, "duration": (end_time - start_time),
            "error": ""
        })
    return segment_data

# --- Metadata Saving ---

def save_metadata_for_coqui(segment_data, tts_dataset_base_dir):
    metadata_file_path = os.path.join(tts_dataset_base_dir, METADATA_FILE)
    with open(metadata_file_path, 'a', newline='', encoding='utf-8') as f:
        for segment in segment_data:
            cleaned_transcript = clean_transcription(segment["transcript"])
            if cleaned_transcript:
                f.write(f"{segment['segment_filename']}|{cleaned_transcript}\n")

def save_detailed_metadata(segment_data, tts_dataset_base_dir):
    metadata_file_path = os.path.join(tts_dataset_base_dir, DETAILED_METADATA_FILE)
    file_exists = os.path.isfile(metadata_file_path)
    print(f"DEBUG: save_detailed_metadata called with {len(segment_data)} segments. File exists: {file_exists}")
    with open(metadata_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['segment_filename', 'transcript', 'start_time', 'end_time', 'duration', 'error', 'snr_db', 'clipping_percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for segment in segment_data:
            writer.writerow({key: segment.get(key, '') for key in fieldnames})

# --- Main Processing Orchestrator ---

def process_audio_file(file_path, base_output_dir, processing_mode="transcription", segmentation_mode="silence"):
    original_filename_no_ext = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(base_output_dir, exist_ok=True)

    converted_file = convert_to_wav(file_path)
    if not converted_file:
        return {"status": "error", "message": "File conversion to WAV failed"}

    quality_metrics = analyze_quality(converted_file)
    snr_threshold = 30 if processing_mode == 'voice_cloning' else 20
    error_messages = []
    if quality_metrics.get('snr_db', 100) < snr_threshold:
        error_messages.append(f"Low SNR ({quality_metrics.get('snr_db', 0):.2f}dB) - Required: >{snr_threshold}dB")
    if quality_metrics.get('clipping_percentage', 0) > 1:
        error_messages.append(f"Clipping detected ({quality_metrics.get('clipping_percentage', 0):.2f}%)")

    if error_messages:
        if converted_file != file_path: os.remove(converted_file)
        return {"status": "error", "message": "; ".join(error_messages)}

    # --- Segmentation Dispatch ---
    audio = AudioSegment.from_wav(converted_file)
    segment_data = []
    if segmentation_mode == 'silence':
        silence_segments = segment_audio_by_silence(audio, original_filename_no_ext, base_output_dir)
        # Transcribe each segment individually
        model = whisper.load_model("base")
        for segment in silence_segments:
            try:
                result = model.transcribe(segment["audio_file_path"])
                segment["transcript"] = result["text"]
            except Exception as e:
                segment["error"] += f"Transcription failed: {e}; "
        segment_data = silence_segments
    else: # 'sentence' or 'paragraph'
        segment_data = transcribe_and_segment(audio, converted_file, original_filename_no_ext, base_output_dir, segmentation_mode)

    if not segment_data:
        if converted_file != file_path: os.remove(converted_file)
        return {"status": "error", "message": "No segments could be generated."}

    # --- Finalize ---
    for seg in segment_data:
        seg.update(quality_metrics)
    
    save_metadata_for_coqui(segment_data, base_output_dir)
    save_detailed_metadata(segment_data, base_output_dir)
    
    if converted_file != file_path: os.remove(converted_file)

    return {"status": "success", "path": os.path.join(base_output_dir, METADATA_FILE)}

def create_zip_archive_of_tts_dataset(tts_dataset_base_dir):
    if not os.path.exists(tts_dataset_base_dir):
        return None
    zip_file_path = f"{tts_dataset_base_dir}.zip"
    shutil.make_archive(tts_dataset_base_dir, 'zip', tts_dataset_base_dir)
    return zip_file_path

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_to_process = sys.argv[1]
        process_audio_file(file_to_process, "temp_tts_dataset")
    else:
        print("Bitte geben Sie den Pfad zur Audiodatei an.")