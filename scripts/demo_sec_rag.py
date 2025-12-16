"""
Demo: RAG over SEC 10-K filings.

Tests RAG system on complex financial/legal documents.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import SECLoader
from src.embeddings.factory import EmbeddingsFactory
from src.vectorstores.qdrant_store import QdrantVectorStore
from src.reranking import RerankerFactory
from src.generation.factory import LLMFactory
from src.chunkers.factory import ChunkerFactory


def main():
    print("=" * 60)
    print("SEC 10-K RAG DEMO - Netflix Annual Report")
    print("=" * 60)
    
    # Load SEC filing
    print("\n[1/6] Loading Netflix 10-K filing...")
    loader = SECLoader(download_dir='data/test_docs')
    docs = loader.load(
        'data/test_docs/sec-edgar-filings/NFLX/10-K/0001065280-25-000044/full-submission.txt'
    )
    
    print(f"  Company: {docs[0].metadata.get('company_name', 'Unknown')}")
    print(f"  Filing Date: {docs[0].metadata.get('filing_date', 'Unknown')}")
    print(f"  Characters: {len(docs[0].content):,}")
    
    # Chunk the document
    print("\n[2/6] Chunking document...")
    chunker = ChunkerFactory.from_config({
        "strategy": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    })
    
    chunks = chunker.chunk(docs[0])
    print(f"  Created {len(chunks)} chunks")
    
    # Clean chunks - remove problematic characters
    clean_chunks = []
    for chunk in chunks:
        clean_content = ''.join(c for c in chunk.content if c.isprintable() or c in '\n\t ')
        if len(clean_content) > 50:
            chunk.content = clean_content
            clean_chunks.append(chunk)
    
    print(f"  After cleaning: {len(clean_chunks)} chunks")
    
    # Initialize components
    print("\n[3/6] Initializing embeddings...")
    embeddings = EmbeddingsFactory.create("ollama", model="nomic-embed-text")
    
    print("[4/6] Initializing vector store...")
    vectorstore = QdrantVectorStore(
        collection_name="netflix_10k",
        dimensions=768,
    )
    
    print("[5/6] Initializing reranker...")
    reranker = RerankerFactory.create("cross_encoder")
    
    print("[6/6] Initializing LLM...")
    llm = LLMFactory.create("ollama", model="llama3.2")
    
    # Index chunks
    print(f"\nIndexing {len(clean_chunks)} chunks...")
    
    start = time.perf_counter()
    batch_size = 20
    indexed = 0
    
    for i in range(0, len(clean_chunks), batch_size):
        batch = clean_chunks[i:i + batch_size]
        texts = [c.content for c in batch]
        metadatas = [c.metadata for c in batch]
        
        # Get embeddings
        embeddings_list = []
        valid_texts = []
        valid_metadatas = []
        
        for j, text in enumerate(texts):
            try:
                emb = embeddings.embed_text(text)
                embeddings_list.append(emb)
                valid_texts.append(text)
                valid_metadatas.append(metadatas[j])
            except Exception as e:
                print(f"  Skipping chunk: {str(e)[:50]}")
                continue
        
        if embeddings_list:
            vectorstore.add(
                texts=valid_texts,
                embeddings=embeddings_list,
                metadatas=valid_metadatas,
            )
            indexed += len(embeddings_list)
        
        print(f"  Progress: {indexed}/{len(clean_chunks)}")
    
    index_time = time.perf_counter() - start
    print(f"Indexing completed in {index_time:.1f}s ({indexed} chunks)")
    
    # Query function
    def query_10k(question: str, top_k: int = 5):
        query_emb = embeddings.embed_text(question)
        results = vectorstore.search(query_emb, top_k=top_k * 2)
        
        if not results:
            return "No relevant information found.", []
        
        texts_to_rerank = [r.content for r in results]
        reranked = reranker.rerank(question, texts_to_rerank, top_n=top_k)
        
        context_list = [r.content for r in reranked]
        answer = llm.generate_with_context(question, context_list)
        
        return answer, reranked
    
    # Test queries
    test_queries = [
        "What was Netflix's total revenue in 2024?",
        "What are the main risk factors for Netflix?",
        "How many paid subscribers does Netflix have?",
        "What is Netflix's content investment strategy?",
        "Who are the executive officers of Netflix?",
    ]
    
    print("\n" + "=" * 60)
    print("FINANCIAL ANALYST Q&A")
    print("=" * 60)
    
    for question in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Q: {question}")
        print("─" * 60)
        
        start = time.perf_counter()
        answer, sources = query_10k(question, top_k=3)
        latency = (time.perf_counter() - start) * 1000
        
        display_answer = answer[:600] + "..." if len(answer) > 600 else answer
        print(f"\nA: {display_answer}")
        print(f"\nLatency: {latency:.0f}ms | Sources: {len(sources)}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Ask questions about Netflix's 10-K filing.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer, sources = query_10k(question)
        print(f"\nAssistant: {answer}\n")
    
    # Cleanup
    print("\nCleaning up...")
    vectorstore._client.delete_collection("netflix_10k")
    print("Done!")


if __name__ == "__main__":
    main()
