# Use a Python base image with TensorFlow support
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install system dependencies required for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project, including the model files
COPY . .

# Create static/uploads directory for file uploads (if needed)
RUN mkdir -p static/uploads

# Expose the port (default FastAPI port is 8000)
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]