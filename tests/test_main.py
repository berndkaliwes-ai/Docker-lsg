import os
import shutil
import pytest
import csv
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine
import main

# --- Fixtures ---



@pytest.fixture
def mock_whisper_result():
    """Provides a mock of the whisper transcription result with word timestamps."""
    return {
        'text': 'Sentence one. Sentence two? Sentence three! A long pause here. And sentence four.',
        'segments': [
            {
                'words': [
                    {'word': 'Sentence', 'start': 0.1, 'end': 0.5},
                    {'word': 'one.', 'start': 0.6, 'end': 0.9},
                    {'word': 'Sentence', 'start': 1.2, 'end': 1.6},
                    {'word': 'two?', 'start': 1.7, 'end': 2.0},
                    {'word': 'Sentence', 'start': 2.1, 'end': 2.5},
                    {'word': 'three!', 'start': 2.6, 'end': 3.0},
                    # Long pause of 2 seconds for paragraph break
                    {'word': 'A', 'start': 5.0, 'end': 5.1},
                    {'word': 'long', 'start': 5.2, 'end': 5.4},
                    {'word': 'pause', 'start': 5.5, 'end': 5.8},
                    {'word': 'here.', 'start': 5.9, 'end': 6.2},
                    {'word': 'And', 'start': 6.5, 'end': 6.7},
                    {'word': 'sentence', 'start': 6.8, 'end': 7.2},
                    {'word': 'four.', 'start': 7.3, 'end': 7.6},
                ]
            }
        ]
    }

# --- Unit Tests ---

def test_clean_transcription():
    assert main.clean_transcription("Hello World! 123.") == "hello world eins zwei drei"

def test_convert_to_wav_unsupported_format(tmp_path):
    """Tests convert_to_wav with an unsupported file format."""
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("dummy content")
    result = main.convert_to_wav(str(unsupported_file))
    assert result is None

def test_convert_to_wav_conversion_error(monkeypatch, tmp_path):
    """Tests convert_to_wav when an exception occurs during conversion."""
    def mock_from_file(*args, **kwargs):
        raise Exception("Mocked pydub conversion error")
    monkeypatch.setattr(main.AudioSegment, "from_file", mock_from_file)
    
    # Create a dummy supported file to trigger the conversion attempt
    supported_file = tmp_path / "test.mp3"
    supported_file.write_text("dummy content")
    
    with monkeypatch.context() as m:
        m.setattr("builtins.print", lambda *args, **kwargs: None) # Mock print to avoid console output during test
        result = main.convert_to_wav(str(supported_file))
        assert result is None
        # We cannot assert call count on a lambda, but we ensure the path is taken.
        # The coverage report should now correctly identify this line as covered.

def test_analyze_quality_with_signal_and_noise(monkeypatch, tmp_path):
    """Tests analyze_quality with an audio file containing both signal and noise to cover SNR calculation."""
    audio_path = tmp_path / "dummy_audio.wav"
    sr = 16000
    # Create a dummy audio array that librosa.load will return
    # This array will have a signal part and a noise part
    dummy_audio_data = np.concatenate([
        np.sin(np.linspace(0, 2 * np.pi * 440, sr)), # Signal
        0.1 * np.random.randn(sr) # Noise
    ])
    sf.write(audio_path, dummy_audio_data, sr)

    # Mock librosa.load to return our dummy data
    monkeypatch.setattr(main.librosa, 'load', lambda x, sr: (dummy_audio_data, sr))

    # Mock librosa.effects.split to return intervals that ensure both signal and noise parts are identified
    # For simplicity, assume the first half is signal, second half is noise
    monkeypatch.setattr(main.librosa.effects, 'split', 
                        lambda y, top_db: np.array([[0, sr], [sr, 2*sr]])) # Signal from 0 to sr, noise from sr to 2*sr

    # Mock np.mean to ensure power_noise is non-zero
    original_np_mean = np.mean
    def mock_np_mean(arr, *args, **kwargs):
        if np.array_equal(arr, dummy_audio_data[sr:]): # Check if it's the noise part
            return 0.001 # Small non-zero value for noise power
        return original_np_mean(arr, *args, **kwargs)
    monkeypatch.setattr(np, 'mean', mock_np_mean)

    quality_metrics = main.analyze_quality(str(audio_path))
    assert 'snr_db' in quality_metrics
    assert quality_metrics['snr_db'] < 100 # Expect a reasonable SNR, not 100 (which means no noise)
    assert quality_metrics['snr_db'] > -50 # Expect a reasonable SNR, not -inf

def test_analyze_quality_error_handling(monkeypatch, tmp_path):
    """Tests analyze_quality when an exception occurs during quality analysis."""
    audio_path = tmp_path / "dummy_audio.wav"
    audio_path.write_text("dummy content") # Create a dummy file

    def mock_librosa_load(*args, **kwargs):
        raise Exception("Mocked librosa load error")
    monkeypatch.setattr(main.librosa, 'load', mock_librosa_load)

    with monkeypatch.context() as m:
        m.setattr("builtins.print", lambda *args, **kwargs: None) # Mock print to avoid console output during test
        quality_metrics = main.analyze_quality(str(audio_path))
        assert quality_metrics == {} # Expect an empty dictionary if an error occurs
        # We cannot assert call count on a lambda, but we ensure the path is taken.
        # The coverage report should now correctly identify this line as covered.

def test_segment_audio_by_silence(test_assets):
    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.segment_audio_by_silence(audio, "test", test_assets["temp_dir"])
    assert len(segments) > 1

def test_transcribe_and_segment_by_sentence(monkeypatch, test_assets, mock_whisper_result):
    """Tests that the text-based segmentation correctly splits by sentence."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.transcribe_and_segment(audio, test_assets["good"], "test", test_assets["temp_dir"], 'sentence')
    assert len(segments) == 5
    assert "Sentence one" in segments[0]["transcript"]

def test_transcribe_and_segment_by_paragraph(monkeypatch, test_assets, mock_whisper_result):
    """Tests that the text-based segmentation correctly groups sentences into paragraphs."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.transcribe_and_segment(audio, test_assets["good"], "test", test_assets["temp_dir"], 'paragraph')
    assert len(segments) == 2
    assert "Sentence one. Sentence two? Sentence three!" in segments[0]["transcript"]
    assert "A long pause here. And sentence four." in segments[1]["transcript"]

def test_transcribe_and_segment_last_sentence_no_punctuation(monkeypatch, test_assets):
    """Tests transcribe_and_segment when the last sentence does not end with punctuation."""
    mock_result_no_punctuation = {
        'text': 'This is a test sentence without punctuation at the end',
        'segments': [
            {
                'words': [
                    {'word': 'This', 'start': 0.1, 'end': 0.3},
                    {'word': 'is', 'start': 0.4, 'end': 0.5},
                    {'word': 'a', 'start': 0.6, 'end': 0.7},
                    {'word': 'test', 'start': 0.8, 'end': 1.0},
                    {'word': 'sentence', 'start': 1.1, 'end': 1.5},
                    {'word': 'without', 'start': 1.6, 'end': 2.0},
                    {'word': 'punctuation', 'start': 2.1, 'end': 2.7},
                    {'word': 'at', 'start': 2.8, 'end': 2.9},
                    {'word': 'the', 'start': 3.0, 'end': 3.1},
                    {'word': 'end', 'start': 3.2, 'end': 3.5},
                ]
            }
        ]
    }
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_result_no_punctuation
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.transcribe_and_segment(audio, test_assets["good"], "test", test_assets["temp_dir"], 'sentence')
    assert len(segments) == 1
    assert "This is a test sentence without punctuation at the end" in segments[0]["transcript"]

def test_process_audio_file_quality_check_fail_and_cleanup(monkeypatch, tmp_path):
    """Tests process_audio_file when quality check fails and converted file needs cleanup."""
    # Create a dummy non-wav file that will be converted
    dummy_mp3_path = tmp_path / "test.mp3"
    dummy_mp3_path.write_text("dummy mp3 content")

    # Mock convert_to_wav to return a path different from original (simulating conversion)
    mock_wav_path = tmp_path / "test.wav"
    mock_wav_path.write_text("dummy wav content")
    monkeypatch.setattr(main, 'convert_to_wav', lambda x: str(mock_wav_path))

    # Mock analyze_quality to return a failing quality metric
    monkeypatch.setattr(main, 'analyze_quality', lambda x: {'snr_db': 10, 'clipping_percentage': 0})

    # Mock os.remove to track if it's called
    mock_os_remove_called = False
    original_os_remove = os.remove
    def mock_remove(path):
        nonlocal mock_os_remove_called
        mock_os_remove_called = True
        original_os_remove(path)
    monkeypatch.setattr(os, 'remove', mock_remove)

    result = main.process_audio_file(str(dummy_mp3_path), tmp_path)
    assert result["status"] == "error"
    assert "Low SNR" in result["message"]
    assert mock_os_remove_called # Ensure os.remove was called for cleanup

def test_process_audio_file_no_segments_generated(monkeypatch, tmp_path):
    """Tests process_audio_file when no segments are generated."""
    dummy_mp3_path = tmp_path / "test.mp3"
    dummy_mp3_path.write_text("dummy mp3 content")

    mock_wav_path = tmp_path / "test_converted.wav"
    # Create a valid empty WAV file
    AudioSegment.silent(duration=100).export(mock_wav_path, format="wav")

    monkeypatch.setattr(main, 'convert_to_wav', lambda x: str(mock_wav_path))
    monkeypatch.setattr(main, 'analyze_quality', lambda x: {'snr_db': 30, 'clipping_percentage': 0})

    # Mock segmentation functions to return empty lists
    monkeypatch.setattr(main, 'segment_audio_by_silence', lambda *args, **kwargs: [])
    monkeypatch.setattr(main.AudioSegment, 'from_wav', lambda x: AudioSegment.silent(duration=1000)) # Mock AudioSegment.from_wav
    monkeypatch.setattr(main, 'transcribe_and_segment', lambda *args, **kwargs: [])

    # Mock os.remove to track if it's called (for cleanup of converted file)
    mock_os_remove_called = False
    original_os_remove = os.remove
    def mock_remove(path):
        nonlocal mock_os_remove_called
        mock_os_remove_called = True
        original_os_remove(path)
    monkeypatch.setattr(os, 'remove', mock_remove)

    result = main.process_audio_file(str(dummy_mp3_path), tmp_path)
    assert result["status"] == "error"
    assert "No segments could be generated." in result["message"]
    assert mock_os_remove_called # Ensure os.remove was called for cleanup

# --- Integration-Style Tests ---

def test_process_audio_file_mode_silence(monkeypatch, test_assets):
    """Tests the main processing pipeline with silence segmentation."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return {"text": "mocked transcription"}
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='silence')
    assert result["status"] == "success"

def test_process_audio_file_mode_sentence(monkeypatch, test_assets, mock_whisper_result):
    """Tests the main processing pipeline with sentence segmentation."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())
    
    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='sentence')
    assert result["status"] == "success"
    
    # Verify that quality metrics are updated in segment_data
    # This requires re-reading the detailed metadata file
    detailed_metadata_path = os.path.join(test_assets["temp_dir"], main.DETAILED_METADATA_FILE)
    with open(detailed_metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader) # Read the first data row
        assert 'snr_db' in first_row
        assert 'clipping_percentage' in first_row
        assert float(first_row['snr_db']) > 0 # Assuming good quality audio

    with open(os.path.join(test_assets["temp_dir"], main.DETAILED_METADATA_FILE)) as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        assert row_count > 2 # Header + multiple sentences

def test_process_audio_file_mode_paragraph(monkeypatch, test_assets, mock_whisper_result):
    """Tests the main processing pipeline with paragraph segmentation."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    detailed_metadata_path = os.path.join(test_assets["temp_dir"], main.DETAILED_METADATA_FILE)
    if os.path.exists(detailed_metadata_path):
        os.remove(detailed_metadata_path)

    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='paragraph')
    assert result["status"] == "success"

    with open(detailed_metadata_path) as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        assert row_count == 3 # Header + 2 paragraphs

def test_process_audio_file_successful_conversion_cleanup(monkeypatch, tmp_path):
    """Tests that the converted file is cleaned up after successful processing."""
    # Create a dummy non-wav file that will be converted
    dummy_mp3_path = tmp_path / "test.mp3"
    dummy_mp3_path.write_text("dummy mp3 content")

    # Mock convert_to_wav to return a path different from original (simulating conversion)
    mock_wav_path = tmp_path / "test_converted.wav"
    # Create a valid empty WAV file
    AudioSegment.silent(duration=100).export(mock_wav_path, format="wav")
    monkeypatch.setattr(main, 'convert_to_wav', lambda x, target_sr=16000: str(mock_wav_path))

    # Mock analyze_quality to return a passing quality metric
    monkeypatch.setattr(main, 'analyze_quality', lambda x: {'snr_db': 50, 'clipping_percentage': 0})

    # Mock whisper model for transcription
    class MockModel:
        def transcribe(self, *args, **kwargs): return {"text": "mocked transcription", "segments": []}
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    # Mock segmentation to return some segments
    monkeypatch.setattr(main, 'transcribe_and_segment', lambda *args, **kwargs: [
        {"segment_filename": "seg1.wav", "audio_file_path": "path/to/seg1.wav", "transcript": "hello", "error": "", "start_time": 0, "end_time": 1, "duration": 1}
    ])

    # Mock save_metadata_for_coqui and save_detailed_metadata to prevent file I/O issues in test
    monkeypatch.setattr(main, 'save_metadata_for_coqui', lambda *args, **kwargs: None)
    monkeypatch.setattr(main, 'save_detailed_metadata', lambda *args, **kwargs: None)

    # Mock os.remove to track if it's called
    mock_os_remove_called = False
    original_os_remove = os.remove
    def mock_remove(path):
        nonlocal mock_os_remove_called
        mock_os_remove_called = True
        original_os_remove(path)
    monkeypatch.setattr(os, 'remove', mock_remove)

    result = main.process_audio_file(str(dummy_mp3_path), tmp_path)
    assert result["status"] == "success"
    assert mock_os_remove_called # Ensure os.remove was called for cleanup
    assert not os.path.exists(mock_wav_path) # Ensure the converted file is actually removed

def test_process_audio_file_transcription_error_handling(monkeypatch, test_assets):
    """Tests error handling during transcription within process_audio_file."""
    class MockModel:
        def transcribe(self, *args, **kwargs):
            raise Exception("mocked transcription error")
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='silence')
    assert result["status"] == "success" # Should still succeed, but with error in metadata

    detailed_metadata_path = os.path.join(test_assets["temp_dir"], main.DETAILED_METADATA_FILE)
    with open(detailed_metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        assert "Transcription failed: mocked transcription error" in first_row["error"]

def test_group_words_into_sentences(mock_whisper_result):
    sentences = main.group_words_into_sentences(mock_whisper_result)
    assert len(sentences) == 5
    assert "Sentence" in sentences[0][0]['word']
    assert "four." in sentences[4][-1]['word']

def test_group_sentences_into_paragraphs(mock_whisper_result):
    sentences = main.group_words_into_sentences(mock_whisper_result)
    paragraphs = main.group_sentences_into_paragraphs(sentences)
    assert len(paragraphs) == 2
    assert "Sentence one. Sentence two? Sentence three!" in " ".join([w['word'] for s in paragraphs[0] for w in s])
    assert "A long pause here. And sentence four." in " ".join([w['word'] for s in paragraphs[1] for w in s])

def test_create_segments_from_transcription(tmp_path):
    audio = AudioSegment.silent(duration=10000) # 10 seconds of silence
    final_segments = [
        [{'word': 'Hello', 'start': 0.0, 'end': 0.5}, {'word': 'world', 'start': 0.6, 'end': 1.0}],
        [{'word': 'How', 'start': 2.0, 'end': 2.3}, {'word': 'are', 'start': 2.4, 'end': 2.5}, {'word': 'you', 'start': 2.6, 'end': 2.8}]
    ]
    original_filename_no_ext = "test_audio"
    base_output_dir = tmp_path / "results"
    segment_data = main.create_segments_from_transcription(audio, final_segments, original_filename_no_ext, base_output_dir)

    assert len(segment_data) == 2
    assert segment_data[0]["transcript"] == "Hello world"
    assert segment_data[1]["transcript"] == "How are you"
    assert os.path.exists(os.path.join(base_output_dir, main.WAVS_SUBDIR, "test_audio_segment_0001.wav"))

def test_save_metadata_for_coqui(tmp_path):
    tts_dataset_base_dir = tmp_path / "tts_dataset"
    os.makedirs(tts_dataset_base_dir, exist_ok=True)
    segment_data = [
        {"segment_filename": "seg1.wav", "transcript": "Hello world.", "error": ""},
        {"segment_filename": "seg2.wav", "transcript": "This is a test.", "error": ""}
    ]
    main.save_metadata_for_coqui(segment_data, str(tts_dataset_base_dir))
    
    metadata_path = os.path.join(tts_dataset_base_dir, main.METADATA_FILE)
    assert os.path.exists(metadata_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    assert "seg1.wav|hello world\n" in content
    assert "seg2.wav|this is a test\n" in content

def test_save_detailed_metadata(tmp_path):
    tts_dataset_base_dir = tmp_path / "tts_dataset"
    os.makedirs(tts_dataset_base_dir, exist_ok=True)
    segment_data = [
        {"segment_filename": "seg1.wav", "transcript": "Hello world.", "start_time": 0.0, "end_time": 1.0, "duration": 1.0, "error": "", "snr_db": 35.0, "clipping_percentage": 0.1},
        {"segment_filename": "seg2.wav", "transcript": "This is a test.", "start_time": 1.5, "end_time": 2.5, "duration": 1.0, "error": "", "snr_db": 30.0, "clipping_percentage": 0.0}
    ]
    main.save_detailed_metadata(segment_data, str(tts_dataset_base_dir))

    metadata_path = os.path.join(tts_dataset_base_dir, main.DETAILED_METADATA_FILE)
    assert os.path.exists(metadata_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["segment_filename"] == "seg1.wav"
    assert float(rows[0]["snr_db"]) == 35.0

def test_create_zip_archive_of_tts_dataset(tmp_path):
    tts_dataset_base_dir = tmp_path / "tts_dataset"
    os.makedirs(tts_dataset_base_dir / "wavs", exist_ok=True)
    (tts_dataset_base_dir / "wavs" / "audio.wav").write_text("dummy audio")
    (tts_dataset_base_dir / "metadata.txt").write_text("dummy metadata")

    zip_file_path = main.create_zip_archive_of_tts_dataset(str(tts_dataset_base_dir))
    assert zip_file_path is not None
    assert os.path.exists(zip_file_path)
    assert zip_file_path.endswith(".zip")

    # Clean up the created zip file
    os.remove(zip_file_path)
    shutil.rmtree(tts_dataset_base_dir) # Remove the directory as well

def test_create_zip_archive_of_tts_dataset_non_existent_dir(tmp_path):
    non_existent_dir = tmp_path / "non_existent"
    zip_file_path = main.create_zip_archive_of_tts_dataset(str(non_existent_dir))
    assert zip_file_path is None
