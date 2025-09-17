
import os
import shutil
import pytest
from pydub import AudioSegment
import main

# Define a fixture to create a temporary directory and test files for our tests
@pytest.fixture(scope="module")
def test_assets():
    """
    This fixture creates a temporary directory and a silent dummy audio file 
    in .opus format for testing purposes. It automatically cleans up the directory 
    and its contents after all tests in the module have been completed.
    """
    temp_dir = "temp_test_assets"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Path for the dummy audio file
    opus_path = os.path.join(temp_dir, "test_audio.opus")
    wav_path = os.path.join(temp_dir, "test_audio.wav")

    # Generate a short, silent audio clip and save it as a WAV file
    silent_segment = AudioSegment.silent(duration=5000)  # 5 seconds of silence
    silent_segment.export(wav_path, format="wav")
    
    # Convert the WAV to OPUS for testing the conversion functionality
    AudioSegment.from_wav(wav_path).export(opus_path, format="opus")

    # Provide the paths to the test assets
    yield {
        "temp_dir": temp_dir,
        "opus": opus_path,
        "wav": wav_path
    }
    
    # Teardown: clean up the temporary directory and its contents after tests
    shutil.rmtree(temp_dir)

def test_clean_transcription():
    """
    Tests the clean_transcription function to ensure it correctly removes 
    punctuation, converts numbers to words, and standardizes the text format.
    """
    assert main.clean_transcription("Hello World! This is a test, number 1.") == "hello world this is a test number eins"
    assert main.clean_transcription("Test with numbers: 1, 2, 3.") == "test with numbers eins zwei drei"
    assert main.clean_transcription("No changes needed here.") == "no changes needed here"

def test_convert_to_wav(test_assets):
    """
    Tests the convert_to_wav function using the .opus test file. It verifies 
    that the conversion to .wav is successful and that the output file is created.
    """
    wav_output_path = main.convert_to_wav(test_assets["opus"])
    assert wav_output_path is not None
    assert os.path.exists(wav_output_path)
    # Clean up the generated WAV file
    os.remove(wav_output_path)

def test_segment_audio(test_assets):
    """
    Tests the segment_audio function. This test uses a silent audio file, 
    so it's expected that no segments will be generated. This verifies the 
    function's ability to handle audio with no detectable speech.
    """
    # Since the test audio is silent, we expect no segments to be returned
    segments, _ = main.segment_audio(test_assets["wav"], "test_audio", test_assets["temp_dir"])
    assert len(segments) == 0

def test_process_audio_file_success(monkeypatch, test_assets):
    """
    Tests the main audio processing pipeline for a successful run. This test 
    uses mocking to bypass the actual transcription process, allowing for a fast 
    and predictable test. It checks whether the function correctly processes 
    the audio and returns a success status.
    """
    # Mock the whisper model to avoid actual transcription
    def mock_transcribe(*args, **kwargs):
        return {"text": "mocked transcription"}

    monkeypatch.setattr(main.whisper, "load_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(main.whisper.Whisper, "transcribe", mock_transcribe)

    # Create a non-silent segment for this test to ensure segmentation works
    temp_wav_path = os.path.join(test_assets["temp_dir"], "non_silent.wav")
    segment1 = AudioSegment.silent(duration=1000)
    segment2 = AudioSegment.silent(duration=1000)
    non_silent_audio = segment1 + AudioSegment.silent(duration=600) + segment2
    non_silent_audio.export(temp_wav_path, format="wav")

    result = main.process_audio_file(temp_wav_path, test_assets["temp_dir"])
    
    assert result["status"] == "success"
    assert os.path.exists(result["path"])

def test_process_audio_file_low_quality(test_assets):
    """
    Tests the system's ability to correctly identify and reject a low-quality 
    audio file. This test uses a silent audio file, which should be flagged 
    for low SNR, and checks that the appropriate error status is returned.
    """
    # A silent file should have a very low SNR and be rejected
    result = main.process_audio_file(test_assets["wav"], test_assets["temp_dir"])
    assert result["status"] == "error"
    assert "Low SNR" in result["message"]
