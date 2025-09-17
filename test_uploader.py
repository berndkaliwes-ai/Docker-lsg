import requests

url = 'http://localhost:5000/'
file_path = 'test_audio.opus'

with open(file_path, 'rb') as f:
    files = {'file[]': (file_path, f, 'audio/opus')}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("File uploaded successfully.")
        # print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
