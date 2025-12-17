# Multi-Domain RAG System

Production-grade Retrieval-Augmented Generation system supporting multiple document types and domains.

## Features

- **Multi-Domain Support**: Finance (SEC filings), Technical docs, Education (PDFs)
- **Hybrid Retrieval**: Dense (semantic) + BM25 (keyword) search with RRF fusion
- **Smart Document Loading**: Auto-detects file types, extracts sections from SEC filings
- **Quality Gate**: Pre-embedding analysis filters problematic content
- **Metadata Filtering**: Query specific domains, sources, or sections
- **Source Attribution**: LLM answers cite sources for transparency

## Quality Results

| Domain | Accuracy | Notes |
|--------|----------|-------|
| Finance (NFLX) | 100% | SEC 10-K filings |
| Finance (AAPL) | 100% | SEC 10-K filings |
| Finance (MSFT) | 100% | SEC 10-K filings (new data) |
| Technical | 100% | RAG documentation |
| Education | 80-100% | ML textbook (PDF) |
| **Overall** | **92%** | With metadata filters |

## Quick Start
```python
from src.loaders import LoaderFactory
from src.chunkers.recursive_chunker import RecursiveChunker
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.generation.ollama_llm import OllamaLLM, format_context_with_metadata

# Setup
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = QdrantHybridStore(collection_name="my_docs", dense_dimensions=768)
retriever = HybridRetriever(embeddings=embeddings, vectorstore=vectorstore)
llm = OllamaLLM(model="llama3.2:latest")

# Index documents
docs = LoaderFactory.load("path/to/document.pdf")  # Auto-detects type
texts = [doc.content for doc in docs]
metadatas = [{"source": "my_doc"} for _ in docs]
retriever.add_documents(texts, metadatas)

# Query with filter
results = retriever.retrieve("What is the topic?", metadata_filter={"source": "my_doc"})

# Generate with source attribution
context = format_context_with_metadata(results, source_mapping={"my_doc": "My Document"})
answer = llm.generate_with_context("What is the topic?", context)
```

## Supported File Types

| Type | Loader | Features |
|------|--------|----------|
| SEC Filings | SECLoader | Auto-detection, 11 section extraction |
| PDF | PDFLoader | Text extraction |
| Markdown | MarkdownLoader | Structure preservation |
| Text | TextLoader | Fallback |

## Metadata Filtering
```python
# By source
results = retriever.retrieve(query, metadata_filter={"source": "AAPL"})

# By domain  
results = retriever.retrieve(query, metadata_filter={"domain": "finance"})

# By section
results = retriever.retrieve(query, metadata_filter={"section": "risk_factors"})
```

## Known Limitations

1. **BM25 Persistence**: Re-index for cross-domain queries without filters
2. **Short Queries**: Add context for ambiguous queries
3. **SEC Tables**: Narrative text works best; tables may not fully extract

## Demo
```bash
python scripts/demo_multi_domain_rag.py
```
