
import pytest
import os
import shutil
from pydub import AudioSegment
from pydub.generators import Sine

@pytest.fixture(scope="session")
def test_assets():
    """
    Creates a temporary directory and a set of dummy audio files for the entire
    test session. Cleans up the directory after all tests are completed.
    This fixture is available to all test files.
    """
    temp_dir = "temp_test_assets_global"
    os.makedirs(temp_dir, exist_ok=True)

    # Create a good quality file with sound and silence
    sine_generator = Sine(440)
    segment1 = sine_generator.to_audio_segment(duration=1000, volume=-10).fade_in(50).fade_out(50)
    good_audio = segment1 + AudioSegment.silent(duration=600) + segment1
    good_audio_path = os.path.join(temp_dir, "good_audio.wav")
    good_audio.export(good_audio_path, format="wav")

    yield {"temp_dir": temp_dir, "good": good_audio_path}
    
    shutil.rmtree(temp_dir)
