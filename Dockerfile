# Use Python 3.11 slim image for better compatibility with iExec TEE
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for TensorFlow and image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy the trained model (if it exists)
COPY trained-model.keras* ./

# Create directories for input data and outputs
RUN mkdir -p /app/input_data /app/output /iexec_out

# Set environment variables for iExec TEE compatibility
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/trained-model.keras
ENV PORT=8000
ENV IEXEC_OUT=/iexec_out

# Expose the port
EXPOSE 8000

# Create a non-root user for security (required for TEE)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check removed for iExec TEE compatibility

# Start the application
ENTRYPOINT ["python3", "/app/src/app.py"]