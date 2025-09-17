FROM python:3.10-slim

# Install dependencies for downloading and extracting ffmpeg
RUN apt-get update && apt-get install -y curl tar xz-utils && apt-get clean

# Download and install a static ffmpeg build
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o ffmpeg.tar.xz && \
    tar -xf ffmpeg.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/bin/ffmpeg && \
    rm -rf ffmpeg.tar.xz ffmpeg-*-amd64-static

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache/pip

# Copy application code
COPY main.py .
COPY app.py .
COPY templates templates

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]