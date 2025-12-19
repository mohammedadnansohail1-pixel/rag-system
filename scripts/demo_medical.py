"""Medical RAG demo using MedRAG textbooks corpus."""
import sys
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("Medical RAG Demo - MedRAG Textbooks")
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
        collection_name="medical_textbooks",
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant (medical_textbooks)")
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=5
    )
    print("  ✓ Retriever: Dense (top_k=5)")
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    print("  ✓ LLM: llama3.1:8b")
    
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
    
    if not all(health.values()):
        print("\n✗ Health check failed. Exiting.")
        return
    
    # Ingest medical documents
    print("\n[4/5] Ingesting medical textbooks...")
    print("  (18 textbooks, ~95MB - this may take a few minutes)")
    count = pipeline.ingest_directory("data/medical/textbooks", file_types=[".txt"])
    print(f"  ✓ Indexed {count} chunks")
    
    # Query with medical questions
    print("\n[5/5] Running medical queries...")
    print("-" * 60)
    
    questions = [
        "What are the symptoms of myocardial infarction?",
        "How does the immune system respond to bacterial infection?",
        "What is the pathophysiology of diabetes mellitus?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = pipeline.query(q, top_k=5)
        print(f"\nA: {response.answer}")
        print(f"\n[Sources: {len(response.sources)} chunks from medical textbooks]")
        for i, src in enumerate(response.sources[:2], 1):
            source_file = src.get('metadata', {}).get('source', 'Unknown')
            print(f"  {i}. {source_file}")
        print("-" * 60)
    
    print("\n✓ Medical RAG demo complete!")


if __name__ == "__main__":
    main()
