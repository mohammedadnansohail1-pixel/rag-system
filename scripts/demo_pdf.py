"""Demo with real PDF document."""

import sys
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline

# Delete existing collection for fresh start
from qdrant_client import QdrantClient


def main():
    print("=" * 60)
    print("RAG System - PDF Demo")
    print("=" * 60)
    
    # Clean up existing collection
    print("\n[0/5] Cleaning up...")
    client = QdrantClient(host="localhost", port=6333)
    try:
        client.delete_collection("pdf_demo")
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
        collection_name="pdf_demo",
        dimensions=768
    )
    print("  ✓ VectorStore: Qdrant (pdf_demo)")
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=5
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
            "chunk_size": 1000,
            "chunk_overlap": 100
        }
    )
    print("  ✓ Pipeline ready")
    
    # Ingest PDF
    print("\n[3/5] Ingesting PDF (this may take a minute)...")
    count = pipeline.ingest_file("data/books/machine_learning_basics.pdf")
    print(f"  ✓ Indexed {count} chunks from PDF")
    
    # Interactive query loop
    print("\n[4/5] Ready for questions!")
    print("=" * 60)
    print("Ask questions about the document. Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        print()
        query = input("Q: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        print("\nSearching and generating answer...")
        response = pipeline.query(query, top_k=5)
        
        print(f"\nA: {response.answer}")
        print(f"\n[Used {len(response.sources)} sources]")
        print("-" * 60)


if __name__ == "__main__":
    main()
