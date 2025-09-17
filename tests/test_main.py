import os
import shutil
import pytest
from pydub import AudioSegment
from pydub.generators import Sine
import main

# --- Fixtures ---

@pytest.fixture(scope="module")
def test_assets():
    """Creates a temporary directory and dummy audio files for testing."""
    temp_dir = "temp_test_assets"
    os.makedirs(temp_dir, exist_ok=True)

    sine_generator = Sine(440)
    segment1 = sine_generator.to_audio_segment(duration=1000, volume=-10).fade_in(50).fade_out(50)
    good_audio = segment1 + AudioSegment.silent(duration=600) + segment1
    good_audio_path = os.path.join(temp_dir, "good_audio.wav")
    good_audio.export(good_audio_path, format="wav")

    yield {"temp_dir": temp_dir, "good": good_audio_path}
    
    shutil.rmtree(temp_dir)

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
                    {'word': 'two? ', 'start': 1.7, 'end': 2.0},
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

def test_segment_audio_by_silence(test_assets):
    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.segment_audio_by_silence(audio, "test", test_assets["temp_dir"])
    assert len(segments) > 1

def test_transcribe_and_segment_by_sentence(monkeypatch, test_assets, mock_whisper_result):
    """Tests that the text-based segmentation correctly splits by sentence."""
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "transcribe_and_segment", main.transcribe_and_segment) # Ensure we test the real function
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.transcribe_and_segment(audio, test_assets["good"], "test", test_assets["temp_dir"], 'sentence')
    # Expects 5 sentences from the mock data
    assert len(segments) == 5
    assert "Sentence one" in segments[0]["transcript"]

def test_transcribe_and_segment_by_paragraph(monkeypatch, test_assets, mock_whisper_result):
    """Tests that the text-based segmentation correctly groups sentences into paragraphs."""
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "transcribe_and_segment", main.transcribe_and_segment)
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    audio = AudioSegment.from_wav(test_assets["good"])
    segments = main.transcribe_and_segment(audio, test_assets["good"], "test", test_assets["temp_dir"], 'paragraph')
    # Expects 2 paragraphs from the mock data due to the long pause
    assert len(segments) == 2
    assert "Sentence one. Sentence two? Sentence three!" in segments[0]["transcript"]
    assert "A long pause here. And sentence four." in segments[1]["transcript"]

# --- Integration-Style Tests ---

def test_process_audio_file_mode_silence(monkeypatch, test_assets):
    """Tests the main processing pipeline with silence segmentation."""
    monkeypatch.setattr(main, "transcribe_audio", lambda segments: segments)
    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='silence')
    assert result["status"] == "success"

def test_process_audio_file_mode_sentence(monkeypatch, test_assets, mock_whisper_result):
    """Tests the main processing pipeline with sentence segmentation."""
    class MockModel: 
        def transcribe(self, *args, **kwargs): return mock_whisper_result
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())
    
    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='sentence')
    assert result["status"] == "success"
    # Check if the detailed metadata contains multiple entries
    with open(os.path.join(test_assets["temp_dir"], main.DETAILED_METADATA_FILE)) as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
        assert row_count > 2 # Header + multiple sentences