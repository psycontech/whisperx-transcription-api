FROM mcr.microsoft.com/devcontainers/python:2-3.11

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /workspace