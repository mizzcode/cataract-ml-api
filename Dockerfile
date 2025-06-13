# Use Python base image with TensorFlow support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

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
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/model /app/models /app/static/uploads /app/logs

# Initialize git and try to pull LFS files
RUN git init . || true && \
    git lfs install || true && \
    git lfs pull || echo "LFS pull failed - model will be downloaded at runtime"

# Check if model exists and create debug info
RUN echo "=== BUILD TIME DEBUG ===" && \
    ls -la /app/ && \
    echo "=== Model directory ===" && \
    ls -la /app/model/ || echo "Model directory does not exist" && \
    echo "=== Models directory ===" && \
    ls -la /app/models/ || echo "Models directory does not exist" && \
    echo "=== Looking for .keras files ===" && \
    find /app -name "*.keras" -type f || echo "No .keras files found" && \
    echo "=== Current working directory ===" && \
    pwd && \
    echo "=== Python path ===" && \
    python -c "import sys; print('\\n'.join(sys.path))" && \
    echo "=== TensorFlow version ===" && \
    python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" && \
    echo "=== BUILD DEBUG COMPLETE ==="

# Add a health check script
RUN echo '#!/bin/bash\n\
echo "=== RUNTIME DEBUG ==="\n\
echo "Current directory: $(pwd)"\n\
echo "App directory contents:"\n\
ls -la /app/\n\
echo "Model directory contents:"\n\
ls -la /app/model/ 2>/dev/null || echo "Model directory not found"\n\
echo "Models directory contents:"\n\
ls -la /app/models/ 2>/dev/null || echo "Models directory not found"\n\
echo "Looking for keras files:"\n\
find /app -name "*.keras" -type f 2>/dev/null || echo "No keras files found"\n\
echo "Memory usage:"\n\
free -h\n\
echo "Disk usage:"\n\
df -h\n\
echo "=== END RUNTIME DEBUG ==="\n' > /app/debug.sh && chmod +x /app/debug.sh

# Set proper permissions
RUN chmod -R 755 /app

# Expose the port
EXPOSE 8000

# Add startup script that runs debug and then starts the app
RUN echo '#!/bin/bash\n\
echo "Starting application..."\n\
/app/debug.sh\n\
echo "Starting FastAPI server..."\n\
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info\n' > /app/start.sh && chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]