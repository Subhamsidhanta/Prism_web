# Backend Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for potential PDF processing optimizations and Rust for tiktoken
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    # Install Rust for building tiktoken
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Ensure Rust is on PATH for pip install
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/processed data/indices models/llm

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]