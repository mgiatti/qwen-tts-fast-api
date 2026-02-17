# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies and add deadsnakes PPA for python3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set python3.11 as default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch independently with the correct index URL
RUN pip install --no-cache-dir torch==2.2.2+cu121 torchaudio==2.2.2+cu121 torchvision==0.17.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files (filtered by .dockerignore)
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
