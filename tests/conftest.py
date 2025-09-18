
import pytest
import os
import shutil
from pydub import AudioSegment
from pydub.generators import Sine

@pytest.fixture(scope="function")
def test_assets(tmp_path_factory):
    """
    Creates a temporary directory and a set of dummy audio files for each test function.
    Cleans up the directory after each test function is completed.
    """
    temp_dir = tmp_path_factory.mktemp("temp_test_assets") # Use mktemp for unique dir
    
    # Create a good quality file with sound and silence
    sine_generator = Sine(440)
    segment1 = sine_generator.to_audio_segment(duration=1000, volume=-10).fade_in(50).fade_out(50)
    good_audio = segment1 + AudioSegment.silent(duration=600) + segment1
    good_audio_path = os.path.join(temp_dir, "good_audio.wav")
    good_audio.export(good_audio_path, format="wav")

    yield {"temp_dir": str(temp_dir), "good": str(good_audio_path)} # Return string paths
    
    # tmp_path_factory handles cleanup automatically for function scope
    # No need for shutil.rmtree(temp_dir)
