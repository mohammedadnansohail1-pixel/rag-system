"""
Demo: RAG over Google Drive documents.

Shows:
- OAuth2 authentication
- Loading docs from Google Drive
- Indexing with hybrid retrieval
- Q&A with reranking
"""
import os
import sys
from pathlib import Path

# Allow HTTP for local OAuth
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import GoogleDriveLoader
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_hybrid_store import QdrantHybridStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking import RerankerFactory
from src.generation.factory import LLMFactory
from src.pipeline import RAGPipelineV2


def main():
    print("=" * 60)
    print("GOOGLE DRIVE RAG DEMO")
    print("=" * 60)
    
    # Initialize Google Drive loader
    print("\n[1/6] Connecting to Google Drive...")
    gdrive = GoogleDriveLoader()
    
    # List available files
    print("\nAvailable files in your Drive:")
    files = gdrive.list_files(max_results=10)
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f['name']} ({f['mimeType']})")
    
    # Let user choose
    print("\n" + "-" * 60)
    choice = input("Enter file number to load (or 'all' for first 5): ").strip()
    
    if choice.lower() == 'all':
        docs = []
        for f in files[:5]:
            if f['mimeType'] != 'application/vnd.google-apps.folder':
                loaded = gdrive.load_file(f['id'])
                docs.extend(loaded)
                print(f"  Loaded: {f['name']}")
    else:
        idx = int(choice) - 1
        docs = gdrive.load_file(files[idx]['id'])
        print(f"  Loaded: {files[idx]['name']}")
    
    if not docs:
        print("No documents loaded. Exiting.")
        return
    
    print(f"\nTotal documents loaded: {len(docs)}")
    total_chars = sum(len(d.content) for d in docs)
    print(f"Total content: {total_chars:,} characters")
    
    # Initialize RAG components
    print("\n[2/6] Initializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    print("[3/6] Initializing vector store...")
    vectorstore = QdrantHybridStore(
        collection_name="gdrive_demo",
        dense_dimensions=768,
        recreate_collection=True,
    )
    
    print("[4/6] Initializing hybrid retriever...")
    retriever = HybridRetriever(
        embeddings=embeddings,
        vectorstore=vectorstore,
    )
    
    print("[5/6] Initializing reranker...")
    reranker = RerankerFactory.create("cross_encoder")
    
    print("[6/6] Initializing LLM...")
    llm = LLMFactory.create("ollama", model="llama3.2")
    
    # Create pipeline
    pipeline = RAGPipelineV2(
        embeddings=embeddings,
        vectorstore=vectorstore,
        retriever=retriever,
        llm=llm,
        reranker=reranker,
        chunker_config={"strategy": "recursive", "chunk_size": 512},
    )
    
    # Chunk and index documents
    print("\nIndexing documents...")
    from src.chunkers.factory import ChunkerFactory
    chunker = ChunkerFactory.from_config({"strategy": "recursive", "chunk_size": 512})
    
    all_chunks = []
    for doc in docs:
        from src.loaders.base import Document as LoaderDoc
        # Convert to loader Document if needed
        chunks = chunker.chunk(LoaderDoc(content=doc.content, metadata=doc.metadata))
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Add to retriever
    texts = [c.content for c in all_chunks]
    metadatas = [c.metadata for c in all_chunks]
    retriever.add_documents(texts, metadatas)
    print("Indexing complete!")
    
    # Interactive Q&A
    print("\n" + "=" * 60)
    print("ASK QUESTIONS ABOUT YOUR GOOGLE DRIVE DOCUMENTS")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        response = pipeline.query(
            question,
            retrieval_top_k=10,
            rerank_top_n=3,
        )
        
        print(f"\nAssistant: {response.answer}")
        print(f"\nSources ({len(response.sources)}):")
        for s in response.sources:
            name = s['metadata'].get('file_name', 'Unknown')
            print(f"  - {name} (score: {s['score']:.3f})")
        print()
    
    # Cleanup
    print("\nCleaning up...")
    vectorstore._client.delete_collection("gdrive_demo")
    print("Done!")


if __name__ == "__main__":
    main()
