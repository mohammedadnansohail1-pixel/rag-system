# Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with modular architecture, guardrails, and comprehensive evaluation.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Tests](https://img.shields.io/badge/Tests-171%20passing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Modular Architecture**: Pluggable components with registry pattern
- **Production Guardrails**: Score thresholds, source validation, confidence levels
- **Multiple Loaders**: PDF, TXT, Markdown support
- **Chunking Strategies**: Fixed-size and recursive chunking
- **Local LLM**: Ollama integration (Llama 3.2, nomic-embed-text)
- **Vector Store**: Qdrant for similarity search
- **REST API**: FastAPI with full OpenAPI documentation
- **Evaluation**: Built-in metrics for RAG quality assessment
- **Docker Ready**: One-command deployment

## Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Loaders   │────▶│  Chunkers   │────▶│ Embeddings  │
│ (PDF/TXT/MD)│     │(Fixed/Recur)│     │  (Ollama)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐     ┌──────▼──────┐
│     LLM     │◀────│  Retriever  │◀────│ VectorStore │
│  (Ollama)   │     │   (Dense)   │     │  (Qdrant)   │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
┌──────▼──────┐     ┌─────────────┐
│ Guardrails  │────▶│  Response   │
│ (Validation)│     │ + Confidence│
└─────────────┘     └─────────────┘
```

## Quick Start

### Option 1: Docker (Recommended)
```bash
# GPU support
docker-compose up -d

# CPU only
docker-compose -f docker-compose.cpu.yml up -d

# Access API
curl http://localhost:8000/health
```

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Linux/Mac
# or: rag-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Pull Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2:latest

# Run API
make run
```

## Usage

### REST API
```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/documents", "file_types": [".pdf", ".txt"]}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?"}'
```

### Python SDK
```python
from src.pipeline import ProductionRAGPipeline
from src.embeddings import OllamaEmbeddings
from src.vectorstores import QdrantVectorStore
from src.retrieval import DenseRetriever
from src.generation import OllamaLLM

# Initialize components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = QdrantVectorStore(collection_name="my_docs")
retriever = DenseRetriever(embeddings=embeddings, vectorstore=vectorstore)
llm = OllamaLLM(model="llama3.2:latest")

# Create pipeline
pipeline = ProductionRAGPipeline(
    embeddings=embeddings,
    vectorstore=vectorstore,
    retriever=retriever,
    llm=llm
)

# Ingest documents
pipeline.ingest_directory("data/documents")

# Query with guardrails
response = pipeline.query("What is machine learning?")

print(f"{response.confidence_emoji} Confidence: {response.confidence}")
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
```

## Guardrails

The system includes production guardrails to prevent hallucination:

| Guardrail | Default | Description |
|-----------|---------|-------------|
| `score_threshold` | 0.4 | Minimum retrieval score |
| `min_sources` | 2 | Required quality sources |
| `min_avg_score` | 0.5 | Minimum average relevance |

### Confidence Levels

- 🟢 **HIGH**: 3+ sources with avg score > 0.7
- 🟡 **MEDIUM**: 2+ sources with avg score > 0.5
- 🔴 **LOW**: Insufficient sources or relevance

## Project Structure
```
rag-system/
├── src/
│   ├── api/           # FastAPI REST API
│   ├── core/          # Config and secrets management
│   ├── loaders/       # Document loaders (PDF, TXT, MD)
│   ├── chunkers/      # Text chunking strategies
│   ├── embeddings/    # Embedding providers
│   ├── vectorstores/  # Vector databases
│   ├── retrieval/     # Retrieval strategies
│   ├── generation/    # LLM providers
│   ├── guardrails/    # Production safety
│   ├── pipeline/      # RAG orchestration
│   └── evaluation/    # Quality metrics
├── tests/
│   └── unit/          # 171 unit tests
├── config/            # YAML configuration
├── scripts/           # Demo scripts
├── docker-compose.yml # Docker deployment
└── Makefile          # Common commands
```

## Configuration

Environment variables:
```bash
# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=rag_production

# Guardrails
SCORE_THRESHOLD=0.4
MIN_SOURCES=2
MIN_AVG_SCORE=0.5
```

## Evaluation
```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(pipeline)

# Single query evaluation
result = evaluator.evaluate_query("What is ML?")
print(f"Overall: {result.overall_score:.2f}")
print(f"Faithfulness: {result.faithfulness:.2f}")

# Batch evaluation
results = evaluator.evaluate_batch([
    {"query": "What is gradient descent?"},
    {"query": "Explain neural networks"},
])
print(f"Average score: {results.avg_overall:.2f}")
```

## Testing
```bash
# Run all tests
make test

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

## Commands
```bash
make help          # Show all commands
make install       # Install dependencies
make test          # Run tests
make run           # Start API locally
make demo          # Run interactive demo
make docker-up     # Start with Docker (GPU)
make docker-up-cpu # Start with Docker (CPU)
make docker-down   # Stop Docker services
```

## Tech Stack

- **Framework**: FastAPI
- **LLM**: Ollama (Llama 3.2)
- **Embeddings**: nomic-embed-text
- **Vector DB**: Qdrant
- **Testing**: pytest (171 tests)
- **Containerization**: Docker

## License

MIT License

## Author

Built as a production-grade RAG system demonstrating:
- Modular software architecture
- Registry pattern for extensibility
- Production guardrails for reliability
- Comprehensive testing practices
- Docker deployment readiness
