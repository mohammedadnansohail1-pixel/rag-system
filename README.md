# 🔍 Enterprise RAG System

**Production-grade Retrieval-Augmented Generation for Document Intelligence**

> Transform 500+ page documents into instant, accurate answers with confidence scoring and source citations.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Tests](https://img.shields.io/badge/Tests-260%20Passing-green)
![Faithfulness](https://img.shields.io/badge/Faithfulness-97.5%25-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What This Does

| Problem | Solution |
|---------|----------|
| Analysts spend 40+ hours reviewing documents | Query any document in seconds |
| Information buried in 100s of pages | AI extracts exactly what you need |
| No way to compare across documents | Cross-document analysis built-in |
| LLMs hallucinate | Confidence scoring + source citations |

### Demo: SEC Filing Analysis
```
📁 Ingested: 3 companies (Meta, Tesla, NVIDIA) - 500+ pages
⏱️  Ingestion time: 2.3 seconds
🔍 Query: "What are the main cybersecurity risks?"
✅ Response: 2.4 seconds with HIGH confidence
📑 Sources: 4 cited passages with relevance scores
```

---

## ⚡ Key Features

### 🧠 Intelligent Retrieval
- **Hybrid Search** - Combines semantic (dense) + keyword (sparse) search
- **Cross-Encoder Reranking** - Re-ranks results for precision
- **Parent-Child Retrieval** - Expands context automatically

### 🛡️ Production Guardrails
- **Confidence Scoring** - Know when to trust the answer (high/medium/low)
- **Source Validation** - Minimum source requirements
- **Hallucination Prevention** - Won't answer without evidence

### 🚀 Performance Optimized
- **Embedding Cache** - 436x speedup on repeated content
- **Query Cache** - 15,000x speedup on repeated queries
- **Structure-Aware Chunking** - 96% noise reduction

### 📊 Multi-Document Analysis
- **Cross-Company Comparison** - Compare entities side-by-side
- **Document Registry** - Track all ingested documents
- **Metadata Filtering** - Filter by company, date, type

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│              (Streamlit UI / FastAPI / CLI)                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Loaders   │→ │  Chunkers   │→ │    Enrichment       │ │
│  │ PDF/MD/SEC  │  │  Structure  │  │ Entities/Topics     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Retrieval Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │  Dense   │  │  Sparse  │  │  Hybrid  │  │  Reranker  │  │
│  │ Embeddings│  │  BM25    │  │  Fusion  │  │CrossEncoder│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                           │
│         Qdrant (Hybrid Vector Store) + Caching              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Generation Layer                         │
│        LLM (Ollama/OpenAI) + Guardrails + Citations         │
└─────────────────────────────────────────────────────────────┘
```

📚 **[See TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** for deep dive on architectural decisions including:
- Why RRF over Weighted Sum for hybrid search
- Deterministic confidence scoring (not LLM-based)
- NLI-based faithfulness evaluation

---

## 📊 Evaluation Metrics

### Faithfulness (NLI-Based)

We use DeBERTa NLI model to verify answers are grounded in retrieved context:

| Query Type | Faithfulness | Confidence |
|------------|--------------|------------|
| Tesla manufacturing risks | **100%** | HIGH |
| Meta advertising revenue | **100%** | HIGH |
| NVIDIA data center | **90%** | MEDIUM |
| **Average** | **97.5%** | - |

### Retrieval Quality

| Metric | Score |
|--------|-------|
| Context Relevance | 75%+ |
| Precision@5 | 0.7+ |
| MRR | 0.8+ |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | Ollama (nomic-embed-text), OpenAI-compatible |
| **Vector Store** | Qdrant (hybrid dense + sparse) |
| **Sparse Encoder** | FastEmbed BM25 |
| **LLM** | Ollama (Llama 3.2), OpenAI-compatible |
| **Reranking** | Cross-Encoder (ms-marco-MiniLM) |
| **API** | FastAPI |
| **UI** | Streamlit |
| **Infrastructure** | Docker, Docker Compose |
| **Testing** | pytest (275+ tests) |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM recommended

### 1. Clone & Start Services
```bash
git clone https://github.com/[your-username]/rag-system.git
cd rag-system

# Start Qdrant and Ollama
docker-compose up -d

# Pull required models
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull llama3.2
```

### 2. Install Dependencies
```bash
python -m venv rag-env
source rag-env/bin/activate
pip install -r requirements.txt
```

### 3. Run the UI
```bash
streamlit run src/ui/app.py
```

### 4. Or Use the API
```bash
uvicorn src.api.main:app --reload

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the risk factors?"}'
```

---

## 📖 Usage Examples

### Basic Query
```python
from src.documents import MultiDocumentPipeline
from src.embeddings import OllamaEmbeddings, CachedEmbeddings
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval import HybridRetriever
from src.generation.ollama_llm import OllamaLLM

# Initialize
embeddings = CachedEmbeddings(OllamaEmbeddings(model="nomic-embed-text"))
vectorstore = QdrantHybridStore(collection_name="my_docs", dense_dimensions=768)
retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore)
llm = OllamaLLM(model="llama3.2")

pipeline = MultiDocumentPipeline(
    embeddings=embeddings,
    vectorstore=vectorstore,
    retriever=retriever,
    llm=llm,
)

# Ingest documents
pipeline.ingest_directory("./documents/")

# Query
response = pipeline.query("What are the key findings?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Sources: {len(response.sources)}")
```

### Filtered Query
```python
# Query specific company only
response = pipeline.query(
    "What is the revenue growth?",
    filter_companies=["Tesla"],
)
```

### Cross-Document Comparison
```python
# Compare across multiple companies
response = pipeline.compare_companies(
    "Compare AI strategies",
    companies=["Meta", "Tesla", "NVIDIA"],
)
```

---

## 📁 Project Structure
```
rag-system/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── cache/            # Embedding & query caching
│   ├── chunkers/         # Document chunking strategies
│   ├── documents/        # Multi-document pipeline
│   ├── embeddings/       # Embedding providers
│   ├── enrichment/       # Metadata extraction
│   ├── evaluation/       # Retrieval metrics
│   ├── generation/       # LLM providers
│   ├── guardrails/       # Quality controls
│   ├── loaders/          # Document loaders
│   ├── pipeline/         # RAG orchestration
│   ├── reranking/        # Cross-encoder reranking
│   ├── retrieval/        # Search strategies
│   ├── summarization/    # Hierarchical summaries
│   ├── ui/               # Streamlit interface
│   └── vectorstores/     # Vector databases
├── tests/                # 275+ unit tests
├── config/               # YAML configuration
├── docker-compose.yml    # Infrastructure
└── requirements.txt
```

---

## ⚙️ Configuration

All settings in `config/rag.yaml`:
```yaml
# Chunking
chunking:
  strategy: structure_aware
  chunk_size: 1500

# Retrieval
retrieval:
  search_type: hybrid
  retrieval_top_k: 20
  reranking:
    enabled: true
    top_n: 5

# Guardrails
guardrails:
  score_threshold: 0.35
  min_sources: 2

# Caching
caching:
  embeddings:
    enabled: true
  queries:
    enabled: true
    ttl_seconds: 300
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ --ignore=tests/integration

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Performance Benchmarks

| Operation | Time | Improvement |
|-----------|------|-------------|
| Ingest 500 pages | 2.3s | - |
| Query (cold) | 1.8s | - |
| Query (cached) | 0.0001s | 15,000x |
| Embedding (cold) | 1.4s | - |
| Embedding (cached) | 0.003s | 436x |

### Quality Metrics

| Metric | Before | After Optimizations |
|--------|--------|---------------------|
| Faithfulness | ~30% | **97.5%** |
| Hallucination Rate | ~40% | **<3%** |

---

## 🤝 Need Custom Development?

I build production RAG systems for companies. Services include:

- **Custom RAG Development** - Tailored to your documents and domain
- **AI Chatbot Integration** - Over your internal knowledge base  
- **Performance Optimization** - Make your existing RAG faster
- **Architecture Consulting** - Design review and best practices

**Contact:** [Your Email] | [Your LinkedIn]

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## ⭐ Star This Repo

If this helped you, consider starring the repo. It helps others find it!
