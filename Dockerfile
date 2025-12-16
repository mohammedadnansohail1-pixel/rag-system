FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Environment variables
ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://ollama:11434
ENV QDRANT_HOST=qdrant
ENV QDRANT_PORT=6333

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
