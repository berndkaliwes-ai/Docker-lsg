# WhatsApp Voice Processor für Coqui TTS

Dieses Projekt ist eine Webanwendung zur Verarbeitung von Audiodateien, insbesondere von WhatsApp-Sprachnachrichten. Das Ziel ist es, aus den Eingabedateien ein Trainings-Dataset im LJSpeech-Format für [Coqui TTS](https://coqui.ai/) zu erstellen.

## Funktionen

- **Web-Oberfläche:** Einfacher Upload von mehreren Audiodateien über den Browser.
- **Format-Konvertierung:** Unterstützt gängige Audioformate (.opus, .mp3, .wav, etc.) und konvertiert sie automatisch in das benötigte WAV-Format.
- **Qualitätsprüfung:** Analysiert jede Datei auf Qualität (z.B. Signal-Rausch-Verhältnis, Clipping) und verwirft ungeeignete Dateien mit entsprechendem Feedback.
- **Segmentierung:** Zerlegt die Audiodateien anhand von Sprechpausen in kürzere Segmente.
- **Transkription:** Transkribiert die einzelnen Segmente mithilfe von OpenAI's Whisper.
- **Dataset-Erstellung:** Erstellt ein ZIP-Archiv, das die `wav`-Segmente und eine `metadata.txt`-Datei im Coqui-TTS-Format enthält.

## Voraussetzungen

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (normalerweise in Docker Desktop enthalten)

## Verwendung

Es gibt zwei empfohlene Wege, die Anwendung zu nutzen: über Docker Compose für die Entwicklung oder über einen manuellen Docker-Build für die Produktion.

### 1. Für die Entwicklung (empfohlen)

Diese Methode ist ideal für die lokale Entwicklung, da Änderungen am Quellcode sofort und ohne einen neuen Build wirksam werden.

1.  **Klonen Sie das Repository:**
    ```bash
    git clone <repository-url>
    cd Docker-lsg
    ```

2.  **Starten Sie die Anwendung mit Docker Compose:**
    ```bash
    docker-compose up --build
    ```

3.  Öffnen Sie Ihren Browser und gehen Sie zu `http://localhost:5000`.

Die verarbeiteten Dateien und das finale ZIP-Archiv werden im Ordner `results/` auf Ihrem lokalen System gespeichert.

### 2. Für die Produktion

Diese Methode baut ein eigenständiges Docker-Image, das Sie ausführen können.

1.  **Bauen Sie das Docker-Image:**
    ```bash
    docker build -t voice-processor .
    ```

2.  **Führen Sie den Container aus:**
    Erstellen Sie zuerst einen lokalen Ordner, in dem die Ergebnisse gespeichert werden sollen (z.B. `my_results`).
    ```bash
    mkdir my_results
    docker run -p 5000:5000 -v "$(pwd)/my_results:/app/results" voice-processor
    ```
    Dieser Befehl leitet den Port `5000` weiter und bindet den lokalen Ordner `my_results` in den Container ein, um die Ergebnisdateien zu speichern.

3.  Öffnen Sie Ihren Browser und gehen Sie zu `http://localhost:5000`.

## Ausgabe

Nach dem Hochladen und Verarbeiten der Dateien wird eine Zusammenfassung angezeigt. Sie können ein `.zip`-Archiv herunterladen, das die folgenden Inhalte hat:

```
tts_dataset.zip
├── wavs/
│   ├── file1_segment_0001.wav
│   ├── file1_segment_0002.wav
│   └── ...
└── metadata.txt
```

-   `wavs/`: Enthält alle Audio-Segmente als 16-bit Mono-WAV-Dateien.
-   `metadata.txt`: Enthält die Transkriptionen im `LJSpeech`-Format (`dateiname|transkript`).
