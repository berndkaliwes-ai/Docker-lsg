import wave
import numpy as np
import subprocess
import os

# Parameters for the WAV file
SAMPLE_RATE = 16000
DURATION = 2  # seconds
CHANNELS = 1
SAMPLE_WIDTH = 2  # 2 bytes for 16-bit audio

WAV_FILE = 'test_audio.wav'
OPUS_FILE = 'test_audio.opus'

# Generate silent audio data
num_samples = int(DURATION * SAMPLE_RATE)
silent_data = np.zeros(num_samples, dtype=np.int16)

# Write the WAV file
with wave.open(WAV_FILE, 'w') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(silent_data.tobytes())

print(f"Generated silent WAV file: {WAV_FILE}")

# Convert WAV to OPUS using ffmpeg
try:
    subprocess.run([
        'ffmpeg',
        '-i', WAV_FILE,
        '-c:a', 'libopus',
        OPUS_FILE,
        '-y'  # Overwrite output file if it exists
    ], check=True, capture_output=True, text=True)
    print(f"Successfully converted to OPUS: {OPUS_FILE}")
    os.remove(WAV_FILE)
    print(f"Removed temporary file: {WAV_FILE}")
except FileNotFoundError:
    print("ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
except subprocess.CalledProcessError as e:
    print(f"Error during ffmpeg conversion:")
    print(f"Stderr: {e.stderr}")
