"""Production RAG demo with guardrails."""

import sys
sys.path.insert(0, ".")

from qdrant_client import QdrantClient

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline_production import ProductionRAGPipeline
from src.guardrails.config import GuardrailsConfig


def main():
    print("=" * 60)
    print("Production RAG System with Guardrails")
    print("=" * 60)
    
    # Clean up existing collection
    print("\n[0/5] Cleaning up...")
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.delete_collection("production_demo")
        print("  ✓ Deleted existing collection")
    except:
        print("  ✓ No existing collection")
    
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
        collection_name="production_demo",
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant")
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=10  # Retrieve more, guardrails will filter
    )
    print("  ✓ Retriever: Dense")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.2:latest",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.2")
    
    # Configure guardrails
    guardrails_config = GuardrailsConfig(
        score_threshold=0.4,      # Filter chunks below 40% relevance
        min_sources=2,            # Require at least 2 quality sources
        max_sources=5,            # Use top 5 sources max
        min_avg_score=0.5,        # Average relevance must be 50%+
        require_explicit_uncertainty=True,
        log_filtered_chunks=True,
    )
    print("  ✓ Guardrails configured")
    
    # Create pipeline
    print("\n[2/5] Creating production pipeline...")
    pipeline = ProductionRAGPipeline(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        chunker_config={
            "strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 100
        },
        guardrails_config=guardrails_config,
    )
    print("  ✓ Production pipeline ready")
    
    # Ingest PDF
    print("\n[3/5] Ingesting PDF...")
    count = pipeline.ingest_file("data/books/machine_learning_basics.pdf")
    print(f"  ✓ Indexed {count} chunks")
    
    # Interactive query loop
    print("\n[4/5] Ready for questions!")
    print("=" * 60)
    print("Guardrails active:")
    print(f"  - Score threshold: {guardrails_config.score_threshold}")
    print(f"  - Min sources: {guardrails_config.min_sources}")
    print(f"  - Min avg score: {guardrails_config.min_avg_score}")
    print("=" * 60)
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Q: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\nProcessing with guardrails...")
        response = pipeline.query(query, top_k=10)
        
        # Display response with metadata
        print(f"\n{response.confidence_emoji} Confidence: {response.confidence.upper()}")
        print(f"📊 Avg Relevance: {response.avg_score:.0%}")
        print(f"📚 Sources Used: {len(response.sources)}")
        
        if not response.validation_passed:
            print(f"⚠️  Guardrails: {response.rejection_reason}")
        
        print(f"\nA: {response.answer}")
        
        # Show source scores
        if response.sources:
            print("\n📎 Source Scores:")
            for i, src in enumerate(response.sources, 1):
                print(f"   [{i}] {src['score']:.0%} relevance")
        
        print("-" * 60)


if __name__ == "__main__":
    main()
