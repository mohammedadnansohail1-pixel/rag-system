# 🔍 Enterprise RAG System

Production-ready Retrieval-Augmented Generation system with guardrails, built with Python.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Tests](https://img.shields.io/badge/Tests-172%20passing-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **Modular Architecture** - Pluggable loaders, chunkers, embeddings, vectorstores
- **Production Guardrails** - Score thresholds, source validation, confidence levels
- **Multiple Interfaces** - REST API (FastAPI) + Web UI (Streamlit)
- **RAGAS-style Evaluation** - Faithfulness, relevance, context precision metrics
- **Docker Ready** - One-command deployment with GPU support

## 🏗️ Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Loaders   │────▶│  Chunkers   │────▶│ Embeddings  │
│ PDF/TXT/MD  │     │Fixed/Recurs │     │   Ollama    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     LLM     │◀────│  Retrieval  │◀────│ VectorStore │
│   Ollama    │     │    Dense    │     │   Qdrant    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│ Guardrails  │──▶ Confidence: 🟢 HIGH | 🟡 MEDIUM | 🔴 LOW
└─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Ollama installed locally

### Option 1: Docker (Recommended)
```bash
# GPU systems
docker-compose up -d

# CPU only
docker-compose -f docker-compose.cpu.yml up -d

# Pull required models
docker exec rag-ollama ollama pull llama3.2:latest
docker exec rag-ollama ollama pull nomic-embed-text
```

Access:
- API: http://localhost:8000
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Option 2: Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create virtual environment
python -m venv rag-env
source rag-env/bin/activate  # Windows: rag-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start services
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
ollama serve &
ollama pull llama3.2:latest
ollama pull nomic-embed-text

# Run API
uvicorn src.api.main:app --reload

# Or run UI
streamlit run src/ui/app.py
```

## 📖 Usage

### Python SDK
```python
from src.pipeline import ProductionRAGPipeline
from src.embeddings import OllamaEmbeddings
from src.vectorstores import QdrantVectorStore
from src.retrieval import DenseRetriever
from src.generation import OllamaLLM
from src.guardrails import GuardrailsConfig

# Initialize components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = QdrantVectorStore(collection_name="my_docs")
retriever = DenseRetriever(embeddings=embeddings, vectorstore=vectorstore)
llm = OllamaLLM(model="llama3.2:latest")

# Create pipeline with guardrails
pipeline = ProductionRAGPipeline(
    embeddings=embeddings,
    vectorstore=vectorstore,
    retriever=retriever,
    llm=llm,
    guardrails_config=GuardrailsConfig(
        score_threshold=0.4,
        min_sources=2,
        min_avg_score=0.5,
    )
)

# Ingest documents
pipeline.ingest_directory("./documents", file_types=[".pdf", ".txt"])

# Query with confidence
response = pipeline.query("What is machine learning?")
print(f"{response.confidence_emoji} {response.confidence}")
print(f"Answer: {response.answer}")
print(f"Sources: {len(response.sources)}")
```

### REST API
```bash
# Health check
curl http://localhost:8000/health

# Ingest file
curl -X POST http://localhost:8000/ingest/file \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/sample/document.pdf"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is gradient descent?", "top_k": 5}'
```

## 🛡️ Guardrails

The system prevents hallucinations with multiple layers:

| Guard | Default | Description |
|-------|---------|-------------|
| Score Threshold | 0.4 | Min similarity score for chunks |
| Min Sources | 2 | Required quality sources |
| Min Avg Score | 0.5 | Average relevance threshold |

Response confidence levels:
- 🟢 **HIGH**: 3+ sources, avg score ≥ 0.7
- 🟡 **MEDIUM**: 2+ sources, avg score ≥ 0.5
- 🔴 **LOW**: Below thresholds (returns uncertainty)

## 📊 Evaluation
```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(llm)
result = evaluator.evaluate(
    query="What is ML?",
    answer="Machine learning is...",
    contexts=["ML is a subset of AI..."]
)

print(f"Faithfulness: {result.faithfulness:.2f}")
print(f"Relevance: {result.relevance:.2f}")
print(f"Overall: {result.overall_score:.2f}")
```

## 🧪 Testing
```bash
# Run all tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=src --cov-report=html
```

## 📁 Project Structure
```
rag-system/
├── src/
│   ├── core/           # Config, secrets management
│   ├── loaders/        # PDF, TXT, MD loaders
│   ├── chunkers/       # Fixed, recursive chunkers
│   ├── embeddings/     # Ollama embeddings
│   ├── vectorstores/   # Qdrant integration
│   ├── retrieval/      # Dense retriever
│   ├── generation/     # Ollama LLM
│   ├── guardrails/     # Production safety
│   ├── pipeline/       # RAG orchestration
│   ├── evaluation/     # RAGAS metrics
│   ├── api/            # FastAPI endpoints
│   └── ui/             # Streamlit interface
├── tests/
├── config/
├── data/
├── docker-compose.yml
└── requirements.txt
```

## 🔧 Configuration

Environment variables:
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=rag_production
SCORE_THRESHOLD=0.4
MIN_SOURCES=2
MIN_AVG_SCORE=0.5
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Streamlit](https://streamlit.io/) - Web UI
- [RAGAS](https://github.com/explodinggradients/ragas) - Evaluation inspiration
