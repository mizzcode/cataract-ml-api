# Use Python base image with TensorFlow support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV GIT_LFS_AUTH_TOKEN=${GIT_LFS_AUTH_TOKEN}

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

# Initialize Git LFS and pull LFS objects with explicit authentication
RUN echo "=== Git LFS Setup ===" && \
    git init . && \
    git lfs install --force && \
    echo "Verifying GIT_LFS_AUTH_TOKEN: ${GIT_LFS_AUTH_TOKEN}" && \
    [ -n "${GIT_LFS_AUTH_TOKEN}" ] || { echo "ERROR: GIT_LFS_AUTH_TOKEN is not set"; exit 1; } && \
    echo "Setting up Git credentials for LFS..." && \
    git config --global http.extraheader "Authorization: Bearer ${GIT_LFS_AUTH_TOKEN}" && \
    git remote add origin https://github.com/mizzcode/cataract-ml-api.git && \
    echo "Fetching LFS objects from origin main..." && \
    git lfs fetch origin main 2>&1 || { echo "ERROR: git lfs fetch failed"; cat .git/lfs/logs/* || echo "No LFS logs available"; exit 1; } && \
    echo "Checking out LFS objects..." && \
    git lfs checkout 2>&1 || { echo "ERROR: git lfs checkout failed"; cat .git/lfs/logs/* || echo "No LFS logs available"; exit 1; }

# Debug build environment
RUN echo "=== BUILD TIME DEBUG ===" && \
    ls -la /app/ && \
    echo "=== Model directory ===" && \
    ls -la /app/model/ || echo "Model directory does not exist" && \
    echo "=== Looking for .keras files ===" && \
    find /app -name "*.keras" -type f || echo "No .keras files found" && \
    echo "=== Current working directory ===" && \
    pwd && \
    echo "=== Python path ===" && \
    python -c "import sys; print('\\n'.join(sys.path))" && \
    echo "=== TensorFlow version ===" && \
    python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" && \
    echo "=== BUILD DEBUG COMPLETE ==="

# Add health check script
RUN echo '#!/bin/bash\n\
    echo "=== RUNTIME DEBUG ==="\n\
    echo "Current directory: $(pwd)"\n\
    echo "App directory contents:"\n\
    ls -la /app/\n\
    echo "Model directory contents:"\n\
    ls -la /app/model/ 2>/dev/null || echo "Model directory not found"\n\
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

# Startup script
RUN echo '#!/bin/bash\n\
    echo "Starting application..."\n\
    /app/debug.sh\n\
    echo "Starting FastAPI server..."\n\
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info\n' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]