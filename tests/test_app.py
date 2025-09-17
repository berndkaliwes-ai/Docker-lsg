
import pytest
from app import app as flask_app
import os
import io

@pytest.fixture
def client():
    """
    This fixture configures the Flask application for testing. It sets up a 
    test client that allows for making requests to the application without 
    running a live server. This is the standard way to test Flask applications.
    """
    flask_app.config['TESTING'] = True
    # Ensure the upload folder exists for tests
    os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)
    with flask_app.test_client() as client:
        yield client

def test_index_page(client):
    """
    Tests the main page of the application. It sends a GET request to the 
    root URL and checks for a 200 OK status code, ensuring the page loads correctly.
    """
    response = client.get('/')
    assert response.status_code == 200
    assert b"Upload Your Voice Messages" in response.data

def test_upload_success(client, monkeypatch):
    """
    Tests the file upload functionality with a valid audio file. This test 
    mocks the actual processing function to isolate the web layer. It verifies 
    that the file is correctly received and that the user is redirected to the 
    results page.
    """
    # Mock the processing function to return a success status immediately
    def mock_process_audio_file(*args, **kwargs):
        return {"status": "success", "path": "mock/path/results/tts_dataset/metadata.txt"}
    
    monkeypatch.setattr("main.process_audio_file", mock_process_audio_file)
    monkeypatch.setattr("main.create_zip_archive_of_tts_dataset", lambda *args, **kwargs: "mock_zip.zip")

    # Create a dummy file in memory to upload
    data = {
        'file[]': (io.BytesIO(b"dummy audio data"), 'test.opus')
    }
    
    response = client.post('/', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    assert b"Processing Results" in response.data
    assert b"test.opus" in response.data
    assert b"Processed" in response.data

def test_upload_disallowed_file(client):
    """
    Tests the application's handling of disallowed file types. It attempts to 
    upload a file with an invalid extension and checks that the application 
    correctly rejects it and displays the appropriate status on the results page.
    """
    data = {
        'file[]': (io.BytesIO(b"this is not an audio file"), 'test.txt')
    }
    
    response = client.post('/', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    # The current implementation processes only allowed files, so an invalid file
    # results in an empty list of processed files.
    assert b"No files were processed" in response.data

def test_upload_no_file(client):
    """
    Tests the scenario where the form is submitted without any files. It checks 
    that the application handles this gracefully by redirecting the user back 
    to the upload page.
    """
    response = client.post('/')
    # Expect a redirect back to the same page
    assert response.status_code == 302
