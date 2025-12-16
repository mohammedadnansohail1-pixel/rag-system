FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create data directory
RUN mkdir -p data/documents data/sample

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
