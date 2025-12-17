#!/usr/bin/env python
"""Demo end-to-end RAG pipeline with SEC filings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import LoaderFactory
from src.chunkers.recursive_chunker import RecursiveChunker
from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.generation.ollama_llm import OllamaLLM


def setup_components():
    """Initialize all RAG components."""
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768,
    )
    
    vectorstore = QdrantHybridStore(
        collection_name="sec_filings_demo",
        host="localhost",
        port=6333,
        dense_dimensions=768,
        recreate_collection=True,
    )
    
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        sparse_encoder="bm25",
    )
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.2:latest",
        temperature=0.1,
    )
    
    return embeddings, vectorstore, retriever, llm


def index_filings(retriever, filings: list):
    """Index SEC filings."""
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
    
    for ticker, path in filings:
        print(f"\nIndexing {ticker}...")
        docs = LoaderFactory.load(Path(path))  # Auto uses sections
        
        chunks = []
        for doc in docs:
            for chunk in chunker.chunk(doc):
                chunk.metadata['ticker'] = ticker
                chunks.append(chunk)
        
        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        
        ids = retriever.add_documents(texts, metadatas)
        print(f"  Indexed {len(ids)} chunks")


def query_rag(retriever, llm, query: str, ticker: str = None, top_k: int = 3):
    """Run RAG query."""
    meta_filter = {"ticker": ticker} if ticker else None
    
    # Retrieve
    results = retriever.retrieve(query, top_k=top_k, metadata_filter=meta_filter)
    
    # Generate
    context = [r.content for r in results]
    answer = llm.generate_with_context(query, context)
    
    return answer, results


def main():
    print("=" * 70)
    print("RAG PIPELINE DEMO - SEC FILINGS")
    print("=" * 70)
    
    # Setup
    print("\n1. Initializing components...")
    embeddings, vectorstore, retriever, llm = setup_components()
    print("   ✓ Components ready")
    
    # Index
    print("\n2. Indexing SEC filings...")
    filings = [
        ("NFLX", "data/sec_filings/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt"),
        ("AAPL", "data/validation/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/full-submission.txt"),
    ]
    
    # Check files exist
    for ticker, path in filings:
        if not Path(path).exists():
            print(f"   ⚠ {ticker} filing not found at {path}")
            return
    
    index_filings(retriever, filings)
    
    # Query
    print("\n3. Running queries...")
    queries = [
        ("What does Netflix do?", "NFLX"),
        ("What are Apple's main risk factors?", "AAPL"),
        ("How does Netflix make money?", "NFLX"),
    ]
    
    for query, ticker in queries:
        print(f"\n{'='*70}")
        print(f"Q: {query} [filter: {ticker}]")
        print("-" * 70)
        
        answer, results = query_rag(retriever, llm, query, ticker)
        
        sections = [r.metadata.get('section', '?') for r in results]
        print(f"Retrieved: {sections}")
        print(f"\nAnswer:\n{answer[:500]}")
    
    print(f"\n{'='*70}")
    print("✓ Demo complete")
    
    # Interactive mode
    print("\n4. Interactive mode (type 'quit' to exit)")
    print("   Format: <query> | <ticker>  (e.g., 'What is revenue? | AAPL')")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() == 'quit':
                break
            
            if '|' in user_input:
                query, ticker = user_input.split('|', 1)
                query = query.strip()
                ticker = ticker.strip().upper()
            else:
                query = user_input
                ticker = None
            
            answer, results = query_rag(retriever, llm, query, ticker)
            print(f"\n{answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
