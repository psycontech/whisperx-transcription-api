FROM python:3.12 AS builder

# Set build arguments and environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install system build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libpq-dev \
    cmake\
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app


RUN pip install --no-cache-dir --no-deps torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Copy dependency files
COPY requirements_prod.txt  .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements_prod.txt

COPY . .

RUN chmod +x entrypoint.sh

# Command to run the application
ENTRYPOINT [ "./entrypoint.sh" ]