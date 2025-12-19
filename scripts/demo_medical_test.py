"""Medical RAG test - small dataset."""
import sys
sys.path.insert(0, ".")

from src.embeddings.ollama_embeddings import OllamaEmbeddings
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.ollama_llm import OllamaLLM
from src.pipeline.rag_pipeline import RAGPipeline


def main():
    print("=" * 60)
    print("Medical RAG Test - First Aid Step1 Only")
    print("=" * 60)
    
    embeddings = OllamaEmbeddings(
        host="http://localhost:11434",
        model="nomic-embed-text",
        dimensions=768
    )
    
    vectorstore = QdrantVectorStore(
        host="localhost",
        port=6333,
        collection_name="medical_subset",
        dimensions=768
    )
    
    retriever = DenseRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
        top_k=5
    )
    
    llm = OllamaLLM(
        host="http://localhost:11434",
        model="llama3.1:8b",
        temperature=0.1
    )
    
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
    
    print("\n[1/3] Ingesting First Aid Step1...")
    count = pipeline.ingest_directory("data/medical/subset", file_types=[".txt"])
    print(f"  ✓ Indexed {count} chunks")
    
    print("\n[2/3] Testing queries...")
    questions = [
        "What are the symptoms of heart failure?",
        "How do you treat hypertension?",
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = pipeline.query(q, top_k=5)
        print(f"A: {response.answer[:500]}...")
        print(f"[Sources: {len(response.sources)}]")
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
