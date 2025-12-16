"""End-to-end demo of the RAG pipeline."""

import sys
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("RAG System Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    print("  ✓ Embeddings: nomic-embed-text")
    
    vectorstore = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="rag_demo",
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant (rag_demo)")
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=3
    )
    print("  ✓ Retriever: Dense")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.2:latest",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.2")
    
    # Create pipeline
    print("\n[2/5] Creating pipeline...")
    pipeline = RAGPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        chunker_config={
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
    )
    print("  ✓ Pipeline ready")
    
    # Health check
    print("\n[3/5] Health check...")
    health = pipeline.health_check()
    for component, status in health.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {component}: {'healthy' if status else 'unhealthy'}")
    
    # Ingest documents
    print("\n[4/5] Ingesting documents...")
    count = pipeline.ingest_directory("data/sample", file_types=[".txt"])
    print(f"  ✓ Indexed {count} chunks")
    
    # Query
    print("\n[5/5] Running queries...")
    print("-" * 60)
    
    questions = [
        "What is RAG and how does it work?",
        "What are the different chunking strategies?",
        "What are the benefits of using RAG?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = pipeline.query(q, top_k=3)
        print(f"\nA: {response.answer}")
        print(f"\n[Sources: {len(response.sources)} chunks used]")
        print("-" * 60)
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
