
import pytest
import os
import csv
from app import app as flask_app
import main
from pydub import AudioSegment

@pytest.fixture
def client():
    """Provides a test client for the Flask application."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

# --- Tests for app.py edge cases ---

def test_upload_empty_file_list(client):
    """Tests submitting the form with an empty file list."""
    response = client.post('/', data={'file[]': []})
    assert response.status_code == 302 # Should redirect

def test_download_file(client, monkeypatch, tmp_path):
    """Tests the file download route."""
    # Create a dummy structure that mimics the app's output
    temp_app_root = tmp_path / "app_root"
    temp_app_root.mkdir()
    
    results_dir = temp_app_root / "results"
    results_dir.mkdir()
    
    tts_dataset_dir = results_dir / "tts_dataset"
    tts_dataset_dir.mkdir()

    # Create the dummy zip file in the 'results' directory
    dummy_zip_path = results_dir / "tts_dataset.zip" # Name of the zip file
    dummy_zip_path.write_text("dummy content")

    # Monkeypatch app.root_path and main.TTS_DATASET_DIR
    monkeypatch.setattr(flask_app, 'root_path', str(temp_app_root))
    monkeypatch.setattr(main, 'TTS_DATASET_DIR', str(tts_dataset_dir))
    monkeypatch.setitem(flask_app.config, 'RESULTS_FOLDER', str(results_dir))
    
    response = client.get(f'/results/{dummy_zip_path.name}')
    assert response.status_code == 200
    assert response.data

# --- Tests for main.py edge cases ---

def test_convert_to_wav_exception(monkeypatch, test_assets):
    """Tests the exception handling in convert_to_wav."""
    def mock_from_file(*args, **kwargs):
        raise Exception("Mocked conversion error")
    monkeypatch.setattr(AudioSegment, "from_file", mock_from_file)
    result = main.convert_to_wav(test_assets["good"])
    assert result is None

def test_analyze_quality_exception(monkeypatch, test_assets):
    """Tests the exception handling in analyze_quality."""
    def mock_load(*args, **kwargs):
        raise Exception("Mocked librosa error")
    monkeypatch.setattr(main.librosa, "load", mock_load)
    result = main.analyze_quality(test_assets["good"])
    assert result == {}

def test_save_detailed_metadata_appends(tmp_path):
    """Tests that detailed metadata is appended without writing a new header."""
    metadata_path = tmp_path / "metadata_detailed.csv"
    segment_data = [{
        'segment_filename': 'test1.wav', 'transcript': 't1', 'start_time': 0, 
        'end_time': 1, 'duration': 1, 'error': ''
    }]
    
    # First call, should write header
    main.save_detailed_metadata(segment_data, str(tmp_path))
    # Second call, should append
    main.save_detailed_metadata(segment_data, str(tmp_path))

    with open(metadata_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3 # 1 header + 2 data rows
        assert rows[0][0] == 'segment_filename'

def test_process_audio_file_silence_transcription_error(monkeypatch, test_assets):
    """Tests an error during transcription in silence segmentation mode."""
    class MockModel:
        def transcribe(self, *args, **kwargs):
            raise Exception("mocked transcription error")
    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: MockModel())

    result = main.process_audio_file(test_assets["good"], test_assets["temp_dir"], segmentation_mode='silence')
    # The process should still complete, but the transcript will be empty and error logged internally
    # This test mainly increases coverage of the except block in the silence path.
    assert result["status"] == "success"
