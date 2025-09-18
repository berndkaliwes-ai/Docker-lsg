# syntax=docker/dockerfile:1

# Stufe 1: Der "Builder"
# KORRIGIERT: 'buster' durch das neuere 'bookworm' ersetzt
FROM python:3.8.18-slim-bookworm AS builder

# System-Abhängigkeiten für Audioverarbeitung hinzufügen
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

# Stufe 2: Das finale, schlanke Image
# KORRIGIERT: 'buster' durch das neuere 'bookworm' ersetzt
FROM python:3.8.18-slim-bookworm AS final

# Laufzeit-Abhängigkeiten ebenfalls installieren
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 -m appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]