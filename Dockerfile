# Use a Python base image with TensorFlow support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create model directory
RUN mkdir -p model

# Try to pull LFS files, fallback to download if failed
RUN git lfs install && git lfs pull || \
    echo "LFS pull failed, model will be downloaded at runtime"

# Create static/uploads directory
RUN mkdir -p static/uploads

# Expose the port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]